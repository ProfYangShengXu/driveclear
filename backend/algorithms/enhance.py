"""
Phase 3: 自适应融合 + Phase 4: 图像增强

Phase 3 — 融合策略：
  将 Phase1（时域减法）和 Phase2（DCP 去雾）的结果按运动幅值自适应融合。
  运动区域（障碍物）→ 偏重时域减法（保持运动细节）
  静止区域（背景/天空）→ 偏重 DCP（色彩自然，去雾彻底）

Phase 4 — 运动蒙版增强（v2）：
  先计算全图增强版本，再计算增强增量（enhanced - fused），
  用运动图蒙版遮盖该增量：
  - 高运动区域（障碍物）→ 应用完整增强（CLAHE + 锐化 + 伽马）
  - 低运动区域（天空/背景）→ 不做增强，保持融合结果的自然色彩

  所有操作仅在 Lab 色彩空间的 L（亮度）通道上做，a/b 通道不变 → 彻底避免色偏。
"""

import cv2
import numpy as np


# ─── Phase 3: 自适应融合 ──────────────────────────────────────


def compute_motion_map(
    current: np.ndarray, previous: np.ndarray, blur_ksize: int = 15
) -> np.ndarray:
    """计算运动幅值图 M(t)

    基于当前帧与前一帧的灰度差，通过 sigmoid 映射到 [0, 1]。
    高运动区域 → 接近 1

    Args:
        current: 当前原始帧 (H, W, 3) uint8
        previous: 前一帧原始帧 (H, W, 3) uint8
        blur_ksize: 高斯模糊核大小，用于平滑运动图

    Returns:
        motion_map: (H, W) float32 [0, 1]
    """
    gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_prev = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(gray_curr - gray_prev)
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)

    # 归一化：像素差 > 30 视为高运动
    motion_map = np.clip(diff / 30.0, 0, 1)
    return motion_map


def fuse_frames(
    temporal_frame: np.ndarray, dcp_frame: np.ndarray, motion_map: np.ndarray
) -> np.ndarray:
    """自适应融合

    output = sigmoid(motion - 0.3) · temporal + (1 - sigmoid(...)) · dcp

    Args:
        temporal_frame: Phase1 时域减法结果 (H, W, 3) uint8
        dcp_frame: Phase2 DCP 去雾结果 (H, W, 3) uint8
        motion_map: 运动幅值图 (H, W) float32 [0, 1]

    Returns:
        fused: (H, W, 3) uint8
    """
    alpha = 1.0 / (1.0 + np.exp(-10.0 * (motion_map - 0.3)))
    alpha = alpha[:, :, np.newaxis]

    fused = alpha * temporal_frame.astype(np.float32) + (1.0 - alpha) * dcp_frame.astype(np.float32)
    return np.clip(fused, 0, 255).astype(np.uint8)


# ─── Phase 4: 仅亮度通道处理 ────────────────────────────────────


def _split_luminance(bgr: np.ndarray):
    """BGR → (L, a, b)，L 为 float32"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return cv2.split(lab)  # L float32, a float32, b float32


def _merge_luminance(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(L, a, b) → BGR uint8"""
    lab = cv2.merge([np.clip(l, 0, 255).astype(np.uint8),
                     a.astype(np.uint8), b.astype(np.uint8)])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _clahe_luminance(l: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """对亮度通道做 CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(l.astype(np.uint8)).astype(np.float32)


def _unsharp_luminance(l: np.ndarray, sigma: float = 1.0, strength: float = 0.8) -> np.ndarray:
    """对亮度通道做 Unsharp Mask"""
    blurred = cv2.GaussianBlur(l, (0, 0), sigma)
    sharpened = l + strength * (l - blurred)
    return np.clip(sharpened, 0, 255)


def _gamma_luminance(l: np.ndarray, block_size: int = 64) -> np.ndarray:
    """对亮度通道做自适应伽马校正"""
    local_mean = cv2.blur(l, (block_size, block_size)) / 255.0
    gamma_map = 1.4 - 0.8 * local_mean  # [0.6, 1.4]
    corrected = np.power(l / 255.0, gamma_map) * 255.0
    return np.clip(corrected, 0, 255)


# ─── Phase 3+4 完整管线 ──────────────────────────────────────


def enhance_frame(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    temporal_frame: np.ndarray,
    dcp_frame: np.ndarray,
    clahe_clip: float = 2.0,
    sharpen_sigma: float = 1.0,
    sharpen_strength: float = 0.8,
    do_clahe: bool = True,
    do_sharpen: bool = True,
    do_gamma: bool = True,
) -> np.ndarray:
    """Phase 3 + Phase 4 完整处理管线（运动蒙版增强版）

    增强增量仅作用于亮度通道，且只应用于高运动区域（障碍物/车辆/行人）。
    静止区域（天空/背景/远距离）保持融合结果的原始亮度和色彩。

    Args:
        current_frame: 当前原始帧
        previous_frame: 前一帧原始帧
        temporal_frame: Phase1 时域减法结果
        dcp_frame: Phase2 DCP 去雾结果
        clahe_clip: CLAHE 对比度限制（默认 2.0，原 3.0）
        sharpen_sigma: 锐化高斯模糊 σ（默认 1.0，原 1.5）
        sharpen_strength: 锐化强度（默认 0.8，原 1.5）
        do_clahe: 是否做 CLAHE
        do_sharpen: 是否做锐化
        do_gamma: 是否做自适应伽马

    Returns:
        enhanced: (H, W, 3) uint8
    """
    # Phase 3: 自适应融合
    motion_map = compute_motion_map(current_frame, previous_frame)
    fused = fuse_frames(temporal_frame, dcp_frame, motion_map)

    # Phase 4: 仅亮度通道增强
    l_fused, a, b = _split_luminance(fused)

    # 逐级增强 L
    l_enh = l_fused.copy()
    if do_clahe:
        l_enh = _clahe_luminance(l_enh, clahe_clip)
    if do_sharpen:
        l_enh = _unsharp_luminance(l_enh, sharpen_sigma, sharpen_strength)
    if do_gamma:
        l_enh = _gamma_luminance(l_enh)

    # 增强增量 = 增强后L - 融合后L
    delta = l_enh - l_fused

    # 运动蒙版：膨胀模糊 + 提高灵敏度，让蒙版边缘平滑过渡
    mask = cv2.GaussianBlur(motion_map, (31, 31), 0)
    mask = np.clip(mask * 1.5, 0, 1)  # 提高灵敏度，让中等运动也能获得增强

    # 仅对运动区域应用增强增量
    l_out = l_fused + delta * mask
    l_out = np.clip(l_out, 0, 255)

    # 重构
    result = _merge_luminance(l_out, a, b)

    # 自适应去饱和：抑制雾区的过度色彩饱和度
    result = restore_natural_colors(result, current_frame, strength=0.7)
    return result


# ─── 色彩校正：抑制雾区过度饱和度 ────────────────────────────


def restore_natural_colors(
    enhanced: np.ndarray, original: np.ndarray, strength: float = 0.7
) -> np.ndarray:
    """自适应去饱和 — 解决 DCP 恢复导致的雾区色彩失真

    原始帧中雾越重的区域 → 亮度高 + 饱和度低
    DCP 恢复会过度放大这些区域的色彩差异
    这里根据原始帧雾密度估计值，压低恢复结果的饱和度

    Args:
        enhanced: (H, W, 3) uint8 BGR 处理后帧
        original: (H, W, 3) uint8 BGR 原始帧
        strength: 去饱和强度 [0,1]，越大雾区压制越狠

    Returns:
        result: (H, W, 3) uint8 BGR
    """
    hsv_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_orig = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 雾密度估计：亮度高且饱和度低 → 雾重
    v_norm = hsv_orig[:, :, 2] / 255.0
    s_norm = 1.0 - hsv_orig[:, :, 1] / 255.0

    # 雾权重 = 高亮度 × 低饱和度
    fog_weight = np.sqrt(v_norm * s_norm)
    fog_weight = cv2.GaussianBlur(fog_weight, (15, 15), 0)

    # 去饱和：雾区向原始饱和度靠拢，清晰区保持增强后饱和度
    factor = fog_weight * strength
    enhanced_s = hsv_enh[:, :, 1]
    original_s = hsv_orig[:, :, 1]

    hsv_enh[:, :, 1] = (1.0 - factor) * enhanced_s + factor * original_s
    hsv_enh[:, :, 1] = np.clip(hsv_enh[:, :, 1], 0, 255)

    return cv2.cvtColor(hsv_enh.astype(np.uint8), cv2.COLOR_HSV2BGR)
