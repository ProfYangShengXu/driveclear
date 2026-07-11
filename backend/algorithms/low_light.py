"""
低光照增强模块 (Low-Light Enhancement) — M4

使用 Retinex 分解（照明层/反射层分离）增强夜间/暗光行车视频。

方法：
  1. detect_low_light: 亮度统计判断是否低光
  2. enhance_low_light: Retinex 分解 + 照明层伽马校正 + 反射层 CLAHE + 噪声感知增益控制

与传统 Retinex 的区别：
  - 照明层估计使用大核高斯模糊（而非引导滤波，避免 opencv-contrib 依赖）
  - 反射层 CLAHE 仅在 Lab-L 通道操作，避免色偏
  - 噪声感知：在原始信号极低的区域降低增益，避免噪声放大

不依赖任何机器学习模型，纯 OpenCV + NumPy。
"""

import cv2
import numpy as np


# ─── 内部辅助 ────────────────────────────────────────────────────────────


def _estimate_illumination(bgr: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """估计照明层 L_illum

    使用大核高斯模糊估计场景的整体照明分布。
    核大小自适应于图像尺寸（取短边的 1/8，确保足够大以保留色彩边缘）。

    Args:
        bgr: (H, W, 3) uint8 BGR
        kernel_size: 高斯核大小，None=自适应

    Returns:
        illumination: (H, W) float32 [0, 1]，照明层
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = bgr.shape[:2]

    if kernel_size is None:
        # 核大小取短边的 1/8，确保足够大以平滑纹理但保留结构
        kernel_size = max(15, min(h, w) // 8)
        if kernel_size % 2 == 0:
            kernel_size += 1

    sigma = kernel_size / 3.0
    illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    # 归一化到 [0, 1]
    # 使用图像的全局最大最小值（而非本 patch），确保不同亮度水平的帧
    # 在相同尺度下比较。对均匀帧不回退到 0.5，而是使用实际值。
    max_val = 255.0
    illumination = illumination / max_val

    return illumination


def _gamma_correct_illumination(
    illumination: np.ndarray, gamma: float | None = None
) -> np.ndarray:
    """对照明层做伽马校正（暗区提亮，亮区保持）

    伽马值根据照明层均值自适应：均值越低（越暗）→ 伽马越小（提亮越多）

    Args:
        illumination: (H, W) float32 [0, 1]
        gamma: 伽马值，None=自适应

    Returns:
        corrected: (H, W) float32 [0, 1]
    """
    if gamma is None:
        mean_illum = float(np.mean(illumination))
        # 均值 0.1 → gamma=0.3（强提亮）；均值 0.5 → gamma=0.6；均值 0.8 → gamma=0.9
        gamma = 0.3 + 0.75 * mean_illum
        gamma = np.clip(gamma, 0.3, 1.0)

    # 防止除零：加小 epsilon
    corrected = np.power(illumination + 1e-6, gamma)
    return np.clip(corrected, 0, 1)


def _enhance_reflectance(
    bgr: np.ndarray, illumination: np.ndarray, clahe_clip: float = 2.0
) -> np.ndarray:
    """增强反射层（局部对比度）

    在 Lab 色彩空间对 L 通道做 CLAHE，然后除以照明层恢复反射率。

    Args:
        bgr: (H, W, 3) uint8 BGR 原图
        illumination: (H, W) float32 [0, 1]，照明层
        clahe_clip: CLAHE 对比度限制

    Returns:
        enhanced: (H, W, 3) float32 BGR，反射层增强后的结果
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE 增强 L 通道
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch.astype(np.uint8)).astype(np.float32)

    # 重构 Lab → BGR
    lab_enh = cv2.merge([
        np.clip(l_enhanced, 0, 255).astype(np.uint8),
        a_ch.astype(np.uint8),
        b_ch.astype(np.uint8),
    ])
    result = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR).astype(np.float32)
    return result


def _noise_aware_gain(
    original_luminance: np.ndarray,
    gain: np.ndarray,
    noise_threshold: float = 0.05,
) -> np.ndarray:
    """噪声感知增益控制

    在原始信号极低的区域（luminance < noise_threshold），
    增益被抑制以避免放大传感器噪声。

    Args:
        original_luminance: (H, W) float32 [0, 1]，原始帧亮度
        gain: (H, W) float32，增益映射图
        noise_threshold: 噪声阈值，低于此值的区域增益被压低

    Returns:
        adjusted_gain: (H, W) float32
    """
    # 低信号区域的增益抑制因子
    suppress = np.clip(original_luminance / noise_threshold, 0, 1)
    # 使用幂函数使抑制更柔和
    suppress = np.power(suppress, 0.5)
    return gain * suppress


# ─── 公开接口 ────────────────────────────────────────────────────────────


def detect_low_light(frame: np.ndarray, threshold: float = 0.18) -> dict:
    """检测帧是否为低光照场景

    Args:
        frame: (H, W, 3) uint8 BGR
        threshold: 低光判定阈值（mean_luminance < threshold 视为低光）

    Returns:
        dict with keys:
          - is_low_light:    bool
          - mean_luminance:  float [0, 1]
          - low_light_mask:  (H, W) uint8 {0, 255}，暗区像素标记
          - dark_ratio:      float [0, 1]，暗像素占比

    Raises:
        TypeError: 输入类型错误
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame 须为 numpy.ndarray，收到 {type(frame)}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame shape 须为 (H,W,3)，收到 {frame.shape}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    luminance = gray / 255.0

    mean_lum = float(np.mean(luminance))
    dark_mask = (luminance < threshold).astype(np.uint8) * 255
    dark_ratio = float(np.mean(luminance < threshold))

    return {
        "is_low_light": mean_lum < threshold,
        "mean_luminance": round(mean_lum, 4),
        "low_light_mask": dark_mask,
        "dark_ratio": round(dark_ratio, 4),
    }


def enhance_low_light(
    frame: np.ndarray,
    strength: float = 1.0,
    gamma: float | None = None,
    clahe_clip: float = 2.0,
) -> np.ndarray:
    """增强低光照帧

    基于 Retinex 分解：
      1. 估计照明层（大核高斯模糊）
      2. 计算反射层 = 原图 / 照明层
      3. 照明层伽马校正（暗区提亮）
      4. 反射层 CLAHE 增强局部对比度
      5. 重组合并
      6. 噪声感知增益控制

    Args:
        frame: (H, W, 3) uint8 BGR
        strength: 增强强度 [0, +∞)，越大提亮越猛
        gamma: 伽马值，None=自适应（推荐）
        clahe_clip: CLAHE 对比度限制

    Returns:
        enhanced: (H, W, 3) uint8 BGR

    Raises:
        TypeError: 输入类型错误
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame 须为 numpy.ndarray，收到 {type(frame)}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame shape 须为 (H,W,3)，收到 {frame.shape}")

    frame_f32 = frame.astype(np.float32)

    # Step 1: 估计照明层
    illumination = _estimate_illumination(frame)
    # 照明层 clamping 防止除零
    illumination = np.clip(illumination, 0.01, 1.0)

    # Step 2: 计算原始亮度用于噪声感知
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Step 3: 伽马校正照明层
    illum_corrected = _gamma_correct_illumination(illumination, gamma)
    # 应用 strength
    illum_corrected = illum_corrected ** (1.0 / max(strength, 0.1))

    # Step 4: 反射层增强（CLAHE on Lab-L）
    reflectance_enhanced = _enhance_reflectance(frame, illumination, clahe_clip)

    # Step 5: 重组合并
    # Retinex: enhanced = reflectance * corrected_illumination
    # 将 reflectance_enhanced 的亮度映射到校正后的照明水平
    lab_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_orig, a_ch, b_ch = cv2.split(lab_orig)

    lab_enh = cv2.cvtColor(reflectance_enhanced.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    l_enh, _, _ = cv2.split(lab_enh)

    # 新 L = 原始 L × (校正照明 / 原始照明)
    l_new = l_orig * (illum_corrected / illumination)
    # 与 CLAHE 增强的 L 做加权混合（保留细节）
    blend_ratio = 0.3
    l_new = (1 - blend_ratio) * l_new + blend_ratio * l_enh
    l_new = np.clip(l_new, 0, 255)

    # Step 6: 噪声感知
    gain = l_new / (l_orig + 1e-6)
    gain_adj = _noise_aware_gain(gray, gain)
    l_final = l_orig * gain_adj
    l_final = np.clip(l_final, 0, 255)

    # 重构
    lab_out = cv2.merge([
        l_final.astype(np.uint8),
        a_ch.astype(np.uint8),
        b_ch.astype(np.uint8),
    ])
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return result


def low_light_pipeline(
    frame: np.ndarray,
    strength: float = 1.0,
    auto_trigger: bool = True,
    threshold: float = 0.18,
) -> tuple[np.ndarray, dict]:
    """低光检测 + 增强一站式接口

    Args:
        frame: 当前帧
        strength: 增强强度
        auto_trigger: True=仅在低光时增强，False=始终增强
        threshold: 低光判定阈值

    Returns:
        (result_frame, detection_info)
    """
    det = detect_low_light(frame, threshold)

    if auto_trigger and not det["is_low_light"]:
        return frame.copy(), det

    result = enhance_low_light(frame, strength=strength)
    return result, det
