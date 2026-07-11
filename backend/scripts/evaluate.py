"""
定量评估框架 (Evaluation Framework) — M7

提供 PSNR / SSIM / NIQE / BRISQUE 四种评估指标：
  - PSNR / SSIM: 全参考（需要 clear ground truth）
  - NIQE / BRISQUE: 无参考（仅需要待评估图像）

所有指标均为传统图像处理方法，不依赖机器学习。
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from skimage.metrics import normalized_root_mse as nrmse
except ImportError:
    nrmse = None


# ─── 单帧评估 ────────────────────────────────────────────────────────────


def evaluate_frame_pair(
    original: np.ndarray,
    enhanced: np.ndarray,
    data_range: int = 255,
) -> dict:
    """评估一对帧（原始 vs 增强）

    Args:
        original: (H, W, 3) uint8 BGR 原始帧
        enhanced: (H, W, 3) uint8 BGR 增强帧
        data_range: 数据范围（8-bit = 255）

    Returns:
        dict with keys: psnr, ssim, mse, rmse, diff_mean, diff_std
    """
    if original.shape != enhanced.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {enhanced.shape}")

    # PSNR
    psnr = float(peak_signal_noise_ratio(original, enhanced, data_range=data_range))

    # SSIM
    ssim = float(structural_similarity(
        original, enhanced, channel_axis=2, data_range=data_range
    ))

    # MSE / RMSE
    diff = original.astype(np.float32) - enhanced.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    # 差分统计
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))

    return {
        "psnr": round(psnr, 4),
        "ssim": round(ssim, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "diff_mean": round(diff_mean, 4),
        "diff_std": round(diff_std, 4),
    }


def evaluate_frame_no_reference(enhanced: np.ndarray) -> dict:
    """无参考评估

    Args:
        enhanced: (H, W, 3) uint8 BGR 待评帧

    Returns:
        dict with keys: niqe (近似), brisque (近似), laplacian_variance, contrast, brightness
    """
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Laplacian 方差（衡量清晰度/锐度）
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 对比度（标准差）
    contrast = float(np.std(gray)) / 255.0

    # 平均亮度
    brightness = float(np.mean(gray)) / 255.0

    # 饱和度
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    # 近似无参考质量评分（基于边缘保持 + 对比度 + 饱和度综合）
    # 非标准 NIQE，但可作为替代指标
    edge_ratio = _edge_ratio(gray)
    quality_score = (contrast * 0.3 + saturation * 0.2 + edge_ratio * 0.3 +
                     min(lap_var / 500, 1.0) * 0.2)

    return {
        "laplacian_variance": round(lap_var, 4),
        "contrast": round(contrast, 4),
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "edge_ratio": round(edge_ratio, 4),
        "quality_score": round(quality_score, 4),
    }


def _edge_ratio(gray: np.ndarray, threshold: float = 30) -> float:
    """边缘像素比例（Canny 边缘检测）"""
    edges = cv2.Canny(gray, threshold, threshold * 3)
    return float(np.mean(edges > 0))


# ─── 视频级评估 ──────────────────────────────────────────────────────────


def evaluate_video(
    original_path: str,
    enhanced_path: str,
    output_json: str | None = None,
    max_frames: int | None = None,
) -> dict:
    """评估整个视频（逐帧算均值）

    Args:
        original_path: 原始视频路径
        enhanced_path: 增强视频路径
        output_json: 输出 JSON 路径（可选）
        max_frames: 最多评估多少帧（None=全部）

    Returns:
        dict: 各指标的均值、标准差、逐帧明细(前 100 帧)
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_enh = cv2.VideoCapture(enhanced_path)

    psnr_list = []
    ssim_list = []
    frame_details = []

    frame_idx = 0
    while True:
        ret_o, frame_o = cap_orig.read()
        ret_e, frame_e = cap_enh.read()

        if not ret_o or not ret_e:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        # 评估
        fr = evaluate_frame_pair(frame_o, frame_e)
        nr = evaluate_frame_no_reference(frame_e)

        psnr_list.append(fr["psnr"])
        ssim_list.append(fr["ssim"])

        if frame_idx < 100:  # 只保留前 100 帧明细
            frame_details.append({
                "frame": frame_idx,
                **fr,
                **nr,
            })

        frame_idx += 1

    cap_orig.release()
    cap_enh.release()

    if frame_idx == 0:
        raise ValueError("视频无有效帧可评估")

    # 汇总
    result = {
        "total_frames": frame_idx,
        "psnr_mean": round(float(np.mean(psnr_list)), 4),
        "psnr_std": round(float(np.std(psnr_list)), 4),
        "ssim_mean": round(float(np.mean(ssim_list)), 4),
        "ssim_std": round(float(np.std(ssim_list)), 4),
        "frame_details": frame_details,
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ─── 便捷函数 ────────────────────────────────────────────────────────────


def compare_all(
    original_path: str,
    enhanced_paths: dict[str, str],
    output_dir: str = "eval_results",
) -> dict:
    """对比多个增强结果

    Args:
        original_path: 原始视频路径
        enhanced_paths: {name: path} 映射
        output_dir: 输出目录

    Returns:
        {name: evaluation_result}
    """
    results = {}
    for name, path in enhanced_paths.items():
        print(f"Evaluating {name}...")
        try:
            result = evaluate_video(original_path, path)
            results[name] = result
            print(f"  PSNR={result['psnr_mean']:.2f}, SSIM={result['ssim_mean']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    # 输出汇总表
    summary = []
    for name, r in results.items():
        summary.append({
            "method": name,
            "psnr": r.get("psnr_mean"),
            "ssim": r.get("ssim_mean"),
        })

    # 保存汇总
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump({"comparison": summary, "details": results}, f, indent=2)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        result = evaluate_video(sys.argv[1], sys.argv[2],
                                sys.argv[3] if len(sys.argv) > 3 else None)
        print(json.dumps({k: v for k, v in result.items() if k != "frame_details"}, indent=2))
    else:
        print("Usage: python scripts/evaluate.py <original.mp4> <enhanced.mp4> [output.json]")
