"""
经典基线对比脚本 (Baseline Comparison) — M8

对比以下方法在同一个视频上的处理效果：
  1. dcp: 仅 DCP 去雾（单帧，无时域信息）
  2. clahe: 仅 CLAHE 增强（在 Lab-L 通道）
  3. retinex: 自定义 Retinex 分解增强
  4. driveclear: 完整 DriveClear 流水线（含去雾+融合+增强+眩光+低光）

输出：各方法的处理视频 + PSNR/SSIM 对比表
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms import dehaze, enhance_frame, TemporalFogEstimator
from scripts.evaluate import evaluate_video


# ─── 各基线处理函数 ──────────────────────────────────────────────────────


def process_dcp_only(input_path: str, output_path: str) -> str:
    """DCP-only 基线：逐帧 DCP 去雾，无时域/融合/增强"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # DCP 快速模式
        result = dehaze(frame, use_fast=True)
        writer.write(result)

    cap.release()
    writer.release()
    return output_path


def process_clahe_only(input_path: str, output_path: str) -> str:
    """CLAHE-only 基线：Lab-L 通道 CLAHE，无去雾/融合"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enh = clahe.apply(l)
        lab_enh = cv2.merge([l_enh, a, b])
        result = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
        writer.write(result)

    cap.release()
    writer.release()
    return output_path


def process_retinex_only(input_path: str, output_path: str) -> str:
    """Retinex 基线：照明层伽马校正 + 反射层 CLAHE

    简化的单尺度 Retinex (SSR)，不使用引导滤波。
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 照明层估计（大核高斯模糊）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernel_size = max(15, min(h, w) // 8)
        if kernel_size % 2 == 0:
            kernel_size += 1
        illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), kernel_size / 3.0)
        illumination = np.clip(illumination / 255.0, 0.01, 1.0)

        # 伽马校正（自适应）
        mean_illum = float(np.mean(illumination))
        gamma = np.clip(0.3 + 0.75 * mean_illum, 0.3, 1.0)
        illum_corrected = np.power(illumination, gamma)

        # 反射层 = 原图 / 照明层（在 RGB 各通道分别操作）
        frame_f32 = frame.astype(np.float32) / 255.0
        reflectance = np.zeros_like(frame_f32)
        for c in range(3):
            reflectance[:, :, c] = frame_f32[:, :, c] / illumination

        # 重建
        result_f32 = reflectance * illum_corrected[:, :, np.newaxis]
        result = np.clip(result_f32 * 255, 0, 255).astype(np.uint8)

        writer.write(result)

    cap.release()
    writer.release()
    return output_path


def process_driveclear(input_path: str, output_path: str, config: dict | None = None) -> str:
    """完整 DriveClear 流水线

    使用 processing_service 中的 PipelineOrchestrator。
    """
    from services.processing_service import PipelineConfig, PipelineOrchestrator

    if config:
        cfg = PipelineConfig.from_dict(config)
    else:
        cfg = PipelineConfig()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    orchestrator = PipelineOrchestrator(cfg, h, w)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = orchestrator.process_frame(frame)
        writer.write(result)

    cap.release()
    writer.release()
    return output_path


# ─── 运行所有基线 ─────────────────────────────────────────────────────────


BASELINE_METHODS = {
    "dcp": process_dcp_only,
    "clahe": process_clahe_only,
    "retinex": process_retinex_only,
    "driveclear": process_driveclear,
}


def run_baselines(
    input_video: str,
    output_dir: str,
    original_video: str | None = None,
) -> dict:
    """运行所有基线并评估

    Args:
        input_video: 输入视频路径
        output_dir: 输出目录
        original_video: 原始（无退化）视频路径，用于全参考评估。None=仅无参考

    Returns:
        {method_name: {psnr_mean, ssim_mean, ...}}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, func in BASELINE_METHODS.items():
        out_path = os.path.join(output_dir, f"{name}.mp4")
        print(f"[{name}] Processing...")
        try:
            func(input_video, out_path)
            print(f"  -> {out_path}")

            if original_video and os.path.exists(original_video):
                eval_result = evaluate_video(original_video, out_path, max_frames=200)
                results[name] = {
                    "psnr": eval_result["psnr_mean"],
                    "ssim": eval_result["ssim_mean"],
                    "psnr_std": eval_result["psnr_std"],
                    "ssim_std": eval_result["ssim_std"],
                }
                print(f"  PSNR={results[name]['psnr']:.2f}, SSIM={results[name]['ssim']:.4f}")
            else:
                results[name] = {"status": "no_reference"}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    # 输出汇总
    summary_path = os.path.join(output_dir, "baseline_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary -> {summary_path}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/baselines.py <input.mp4> <output_dir> [original.mp4]")
        sys.exit(1)

    input_video = sys.argv[1]
    output_dir = sys.argv[2]
    original_video = sys.argv[3] if len(sys.argv) > 3 else None
    run_baselines(input_video, output_dir, original_video)
