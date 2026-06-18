"""
视频处理调度服务 — 协调四阶段管线逐帧处理
"""

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from algorithms import TemporalFogEstimator, dehaze, enhance_frame
from services.video_service import (
    get_output_path,
    get_video_info,
    read_frames,
    write_video,
)


class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingTask:
    """处理任务状态"""
    video_id: str
    input_path: str
    output_path: str
    status: ProcessingStatus = ProcessingStatus.QUEUED
    progress: float = 0.0  # 0~100
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    error: str | None = None
    cancel_requested: bool = False

    # 算法参数
    window_size: int = 15
    percentile: float = 10.0
    omega: float = 0.95
    clahe_clip: float = 2.0
    sharpen_sigma: float = 1.0
    sharpen_strength: float = 0.8
    do_clahe: bool = True
    do_sharpen: bool = True
    do_gamma: bool = True

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


def create_task(video_id: str, input_path: str, **params) -> ProcessingTask:
    """创建处理任务"""
    info = get_video_info(input_path)
    output_path = get_output_path(video_id)

    task = ProcessingTask(
        video_id=video_id,
        input_path=input_path,
        output_path=output_path,
        total_frames=info["total_frames"],
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
        **{k: v for k, v in params.items() if hasattr(ProcessingTask, k)},
    )

    with _tasks_lock:
        _tasks[video_id] = task

    return task


def get_task(video_id: str) -> ProcessingTask | None:
    """获取任务状态"""
    with _tasks_lock:
        return _tasks.get(video_id)


def _process_video_worker(task: ProcessingTask):
    """后台处理线程 — 逐帧执行四阶段管线（逐帧写入，不缓存全部帧到内存）"""
    try:
        task.status = ProcessingStatus.PROCESSING

        # 初始化雾层估计器
        fog_estimator = TemporalFogEstimator(
            window_size=task.window_size, percentile=task.percentile
        )

        prev_frame = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        for frame_idx, frame in read_frames(task.input_path):
            if task.cancel_requested:
                task.status = ProcessingStatus.FAILED
                task.error = "用户取消处理"
                if writer:
                    writer.release()
                return

            # 第一次迭代时创建 VideoWriter
            if writer is None:
                writer = cv2.VideoWriter(
                    task.output_path, fourcc, task.fps, (task.width, task.height)
                )

            # Phase 1: 时域雾层估计 + 帧间减法
            temporal_dehazed = fog_estimator.update(frame)

            # Phase 2: DCP 去雾 (快速模式：降采样计算传输率)
            dcp_dehazed = dehaze(frame, omega=task.omega, use_fast=True)

            # Phase 3+4: 融合 + 增强
            if prev_frame is not None and fog_estimator.is_ready():
                enhanced = enhance_frame(
                    current_frame=frame,
                    previous_frame=prev_frame,
                    temporal_frame=temporal_dehazed,
                    dcp_frame=dcp_dehazed,
                    clahe_clip=task.clahe_clip,
                    sharpen_sigma=task.sharpen_sigma,
                    sharpen_strength=task.sharpen_strength,
                    do_clahe=task.do_clahe,
                    do_sharpen=task.do_sharpen,
                    do_gamma=task.do_gamma,
                )
            else:
                # 缓冲区未满时，暂用 DCP 结果直接输出
                enhanced = dcp_dehazed

            writer.write(enhanced)
            prev_frame = frame.copy()

            task.update_progress(frame_idx + 1)

        if writer:
            writer.release()

        task.status = ProcessingStatus.COMPLETED
        task.progress = 100.0

    except Exception as e:
        task.status = ProcessingStatus.FAILED
        task.error = str(e)


def start_processing(video_id: str) -> bool:
    """开始后台处理，返回是否成功启动"""
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
