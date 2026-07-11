"""
视频处理调度服务 — 协调可插拔多模块管线逐帧处理

架构：
  - ProcessingTask + ProcessingConfig 分离：配置可经由 API 传参
  - PipelineOrchestrator：按配置 + 退化检测结果路由各处理模块
  - 每个模块可独立启用/禁用
  - auto_detect=True 时按帧自动调度
"""

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from algorithms import (
    TemporalFogEstimator, dehaze, enhance_frame,
    analyze_frame,
    detect_glare, suppress_glare,
    detect_low_light, enhance_low_light,
)
from services.video_service import (
    get_output_path,
    get_video_info,
    read_frames,
)


# ─── 状态枚举 ────────────────────────────────────────────────────────────


class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── 管线配置（可控参数全集中） ──────────────────────────────────────


@dataclass
class PipelineConfig:
    """管线配置 — 每个字段为何存在均有依据

    enable_*:    模块级开关，false = 跳过该模块
    auto_detect: 如果为 true，根据 analyze_frame 结果自动决定跑哪些模块；
                 如果为 false，以 enable_* 为准强制跑。
    *_threshold: auto_detect 模式下，超过该阈值的退化才触发对应模块
    """

    # 模块开关
    enable_fog: bool = True
    enable_glare: bool = True
    enable_low_light: bool = True
    enable_fusion: bool = True
    enable_enhance: bool = True

    # 自动检测模式
    auto_detect: bool = True
    haze_threshold: float = 0.3
    glare_threshold: float = 0.4
    low_light_threshold: float = 0.35

    # 去雾参数
    omega: float = 0.95
    window_size: int = 15
    percentile: float = 10.0

    # 增强参数
    clahe_clip: float = 2.0
    sharpen_sigma: float = 1.0
    sharpen_strength: float = 0.8
    do_clahe: bool = True
    do_sharpen: bool = True
    do_gamma: bool = True

    # 眩光参数
    glare_strength: float = 1.5
    glare_brightness_threshold: float = 0.93

    # 低光参数
    low_light_strength: float = 1.0
    low_light_threshold: float = 0.18

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """从字典构建（保留默认值）"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ─── 处理任务 ────────────────────────────────────────────────────────────


@dataclass
class ProcessingTask:
    """处理任务状态"""
    video_id: str
    input_path: str
    output_path: str
    config: PipelineConfig = field(default_factory=PipelineConfig)
    status: ProcessingStatus = ProcessingStatus.QUEUED
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    error: str | None = None
    cancel_requested: bool = False

    _progress_callback: Callable | None = None

    def set_progress_callback(self, callback: Callable):
        self._progress_callback = callback

    def update_progress(self, frame_idx: int):
        if self.total_frames > 0:
            self.progress = round(frame_idx / self.total_frames * 100, 1)
            self.current_frame = frame_idx
            if self._progress_callback:
                self._progress_callback(self.video_id, self.progress)


# 全局任务存储
_tasks: dict[str, ProcessingTask] = {}
_tasks_lock = threading.Lock()


def create_task(video_id: str, input_path: str, config: PipelineConfig | None = None, **legacy_params) -> ProcessingTask:
    """创建处理任务

    Args:
        video_id: 视频唯一 ID
        input_path: 输入视频路径
        config: 管线配置（新 API）
        legacy_params: 旧参数格式兼容（将被合入 config）
    """
    info = get_video_info(input_path)
    output_path = get_output_path(video_id)

    # 合并 config 和 legacy_params
    if config is None:
        config = PipelineConfig()
    if legacy_params:
        for k, v in legacy_params.items():
            if hasattr(config, k):
                setattr(config, k, v)

    task = ProcessingTask(
        video_id=video_id,
        input_path=input_path,
        output_path=output_path,
        config=config,
        total_frames=info["total_frames"],
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
    )

    with _tasks_lock:
        _tasks[video_id] = task

    return task


def get_task(video_id: str) -> ProcessingTask | None:
    """获取任务状态"""
    with _tasks_lock:
        return _tasks.get(video_id)


# ─── 管线编排器 ─────────────────────────────────────────────────────────


class PipelineOrchestrator:
    """可插拔管线编排器

    职责：逐帧调用各处理模块，按配置和退化检测结果动态路由。
    每个模块只做一件事，模块间无直接耦合。
    """

    def __init__(self, config: PipelineConfig, height: int, width: int):
        self.config = config
        self.height = height
        self.width = width

        # 有状态模块初始化
        self.fog_estimator = TemporalFogEstimator(
            window_size=config.window_size, percentile=config.percentile
        )
        self.prev_frame: np.ndarray | None = None
        self.prev_frame_for_glare: np.ndarray | None = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧，返回增强结果

        Args:
            frame: (H, W, 3) uint8 BGR 原始帧

        Returns:
            enhanced: (H, W, 3) uint8 BGR
        """
        cfg = self.config
        result = frame.copy()

        # ── Step 0: 退化检测（决定后续调度） ──────────────────────────
        if cfg.auto_detect:
            det = analyze_frame(
                result,
                haze_threshold=cfg.haze_threshold,
                glare_threshold=cfg.glare_threshold,
                low_light_threshold=cfg.low_light_threshold,
            )
            need_haze = cfg.enable_fog and det["haze_score"] >= cfg.haze_threshold
            need_glare = cfg.enable_glare and det["glare_score"] >= cfg.glare_threshold
            need_low_light = cfg.enable_low_light and det["low_light_score"] >= cfg.low_light_threshold
        else:
            # 手动模式：按 enable_* 开关强制执行
            need_haze = cfg.enable_fog
            need_glare = cfg.enable_glare
            need_low_light = cfg.enable_low_light

        # ── Step 1: 去雾（时域雾层估计 + DCP + 融合 + 增强） ─────────
        if need_haze:
            temporal_dehazed = self.fog_estimator.update(result)
            dcp_dehazed = dehaze(result, omega=cfg.omega, use_fast=True)

            if self.prev_frame is not None and self.fog_estimator.is_ready():
                enhanced_dcp = enhance_frame(
                    current_frame=result,
                    previous_frame=self.prev_frame,
                    temporal_frame=temporal_dehazed,
                    dcp_frame=dcp_dehazed,
                    clahe_clip=cfg.clahe_clip,
                    sharpen_sigma=cfg.sharpen_sigma,
                    sharpen_strength=cfg.sharpen_strength,
                    do_clahe=cfg.do_clahe,
                    do_sharpen=cfg.do_sharpen,
                    do_gamma=cfg.do_gamma,
                )
            else:
                # 缓冲区未满时暂用 DCP 结果
                enhanced_dcp = dcp_dehazed
            result = enhanced_dcp
        else:
            # 即使不去雾，也要更新 fog_estimator 的状态
            self.fog_estimator.update(result)

        # ── Step 2: 眩光抑制 ──────────────────────────────────────────
        if need_glare:
            glare_det = detect_glare(
                result,
                brightness_threshold=cfg.glare_brightness_threshold,
            )
            if glare_det["glare_intensity"] > 0.01 and np.any(glare_det["glare_mask"]):
                result = suppress_glare(
                    result,
                    glare_det["glare_mask"],
                    prev_frame=self.prev_frame_for_glare,
                    strength=cfg.glare_strength,
                )

        # ── Step 3: 低光照增强 ────────────────────────────────────────
        if need_low_light:
            result = enhance_low_light(
                result,
                strength=cfg.low_light_strength,
            )

        # ── 更新状态 ──────────────────────────────────────────────────
        self.prev_frame = frame.copy()
        self.prev_frame_for_glare = frame.copy()

        return result


# ─── 处理工作线程 ────────────────────────────────────────────────────────


def _process_video_worker(task: ProcessingTask):
    """后台处理线程 — 使用 PipelineOrchestrator 逐帧处理"""
    try:
        task.status = ProcessingStatus.PROCESSING

        orchestrator = PipelineOrchestrator(task.config, task.height, task.width)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        for frame_idx, frame in read_frames(task.input_path):
            if task.cancel_requested:
                task.status = ProcessingStatus.FAILED
                task.error = "用户取消处理"
                if writer:
                    writer.release()
                return

            if writer is None:
                writer = cv2.VideoWriter(
                    task.output_path, fourcc, task.fps, (task.width, task.height)
                )

            # 通过编排器处理单帧
            enhanced = orchestrator.process_frame(frame)
            writer.write(enhanced)

            task.update_progress(frame_idx + 1)

        if writer:
            writer.release()

        task.status = ProcessingStatus.COMPLETED
        task.progress = 100.0

    except Exception as e:
        task.status = ProcessingStatus.FAILED
        task.error = str(e)


def start_processing(video_id: str) -> bool:
    """开始后台处理"""
    task = get_task(video_id)
    if task is None:
        return False

    thread = threading.Thread(target=_process_video_worker, args=(task,), daemon=True)
    thread.start()
    return True


def cancel_processing(video_id: str) -> bool:
    """取消处理"""
    task = get_task(video_id)
    if task is None:
        return False
    task.cancel_requested = True
    return True
