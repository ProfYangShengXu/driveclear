"""
退化检测器 (Degradation Detector) — M5

基于单帧图像统计量检测三种行车视频退化类型：
  - haze_score:      雾霾浓度估计，使用针对行车场景设计的特征
  - glare_score:     眩光强度估计（亮通道比例 + 高亮连通区域）
  - low_light_score: 低光照程度估计（亮度均值 + 暗像素比例）

设计原则（每个特征的 WHY）：
  1. Haze: 雾天行车视频的核心特征是**中远距离物体被白雾漂白**，
     表现为"亮度偏高 + 饱和度极低 + 垂直方向对比度梯度消失"。
     使用 mid-band washout（画面中段漂白比例）而非全局暗通道均值，
     避免被天空（天然高亮低饱和）和路面（天然低纹理）干扰。

  2. Glare: 眩光是**局部高强度光斑**，通过亮通道连通区域分析定位，
     区分大面积高亮（眩光）和小面积高亮（车灯/反光）。

  3. Low-light: 全局亮度统计 + 暗像素比例，简单可靠。

不依赖任何机器学习模型，纯 OpenCV + NumPy 图像统计。
"""

import cv2
import numpy as np


# ─── 内部辅助函数 ────────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray, center: float = 0.0, slope: float = 8.0) -> np.ndarray:
    """sigmoid 映射，将连续值压到 [0, 1]"""
    return 1.0 / (1.0 + np.exp(-slope * (x - center)))


def _mean_luminance(bgr: np.ndarray) -> float:
    """V 通道（HSV 亮度）均值"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return float(np.mean(hsv[:, :, 2])) / 255.0


def _mean_saturation(bgr: np.ndarray) -> float:
    """S 通道（饱和度）均值"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return float(np.mean(hsv[:, :, 1])) / 255.0


def _contrast(gray: np.ndarray) -> float:
    """全局对比度（标准差）—— 度量整体反差"""
    return float(np.std(gray)) / 255.0


def _texture_density(gray: np.ndarray) -> float:
    """纹理密度 —— Sobel 边缘幅值的归一化均值

    Returns: [0, 1]，越大纹理越丰富
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return float(np.mean(magnitude)) / 255.0


def _dark_channel_mean(bgr: np.ndarray, patch_size: int = 15) -> float:
    """暗通道均值 —— 辅助特征"""
    b, g, r = cv2.split(bgr)
    min_ch = np.minimum(np.minimum(b.astype(np.float32), g.astype(np.float32)), r.astype(np.float32))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_ch, kernel)
    return float(np.mean(dark)) / 255.0


def _bright_channel_mean(bgr: np.ndarray, patch_size: int = 15) -> float:
    """亮通道（max of RGB）均值"""
    b, g, r = cv2.split(bgr)
    max_ch = np.maximum(np.maximum(b.astype(np.float32), g.astype(np.float32)), r.astype(np.float32))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    bright = cv2.dilate(max_ch, kernel)
    return float(np.mean(bright)) / 255.0


# ─── 雾霾专用特征 ────────────────────────────────────────────────────────


def _mid_band_washout(bgr: np.ndarray) -> float:
    """画面中段"漂白"像素比例 —— 核心雾指标

    行车视频垂直分段（针对 16:9 / 4:3 典型行车画面）：
      - 上 35%：天空/远处（天然低饱和，排除不分析）
      - 中 40%：中远距离景物/车辆（雾天会严重漂白）
      - 下 25%：路面/车头（受雾影响最小）

    漂白定义：V > 0.5 且 S < 0.15（亮度中高 + 饱和度极低）
    仅在中段计算以排除天空和路面的干扰。

    设计理由：
      - 天空天然低饱和但饱和度通常 > 0.15（蓝色/灰色天空有色调）
      - 真正雾霾漂白是全局性的，中段景物应有正常色彩
      - 中段 40% 范围确保覆盖"中距离景物"而避开近处路面

    Returns:
        float [0, 1]，越高表示中段漂白越严重 → 雾越浓
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    # 中段：35% ~ 75% 高度范围（避开天空和车头）
    top = int(h * 0.35)
    bot = int(h * 0.75)
    mid_v = hsv[top:bot, :, 2]
    mid_s = hsv[top:bot, :, 1]

    washed = ((mid_v > 0.5) & (mid_s < 0.15)).astype(np.float32)
    return float(np.mean(washed))


def _vertical_contrast_decay(gray: np.ndarray) -> float:
    """垂直对比度衰减 —— 辅助雾指标

    将图像分为 4 个水平条带，计算每个条带的对比度。
    雾天时远处（上方条带）对比度严重损失：bottom_contrast >> top_contrast
    晴日时各条带对比度差异较小。

    Returns:
        decay: float [0, 1]，0=均匀, 1=极度衰减
    """
    h, w = gray.shape
    bands = 4
    bh = h // bands
    contrasts = []
    for i in range(bands):
        band = gray[i * bh : min((i + 1) * bh, h), :]
        contrasts.append(float(np.std(band)))
    # 衰减度 = 1 - (最上条带对比度 / 最下条带对比度)，clamp 到 [0, 1]
    if contrasts[-1] < 1.0:
        return 0.0
    decay = 1.0 - min(contrasts[0] / max(contrasts[-1], 1.0), 1.0)
    return decay


# ─── 眩光专用特征 ────────────────────────────────────────────────────────


def _glare_region_ratio(bgr: np.ndarray, brightness_threshold: float = 0.93) -> float:
    """眩光区域面积比

    策略：
      1. 亮度二值化（V > 0.93）— 阈值高于天空亮度峰值，排除蓝天
      2. 形态学闭运算连接临近高亮
      3. 连通组件分析，只保留面积 > 1% 总像素的组件
         （排除小面积车灯/反光点）

    Returns:
        ratio: [0, 1]，眩光区域占画面比例
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    bright_mask = (v > brightness_threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    total_pixels = bgr.shape[0] * bgr.shape[1]
    min_area = total_pixels * 0.01

    glare_pixels = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            glare_pixels += stats[i, cv2.CC_STAT_AREA]

    return glare_pixels / total_pixels


# ─── 低光专用特征 ────────────────────────────────────────────────────────


def _low_light_pixel_ratio(bgr: np.ndarray, dark_threshold: float = 0.15) -> float:
    """亮度低于暗阈值的像素比例"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    return float(np.mean(v < dark_threshold))


# ─── 公开接口 ────────────────────────────────────────────────────────────


def analyze_frame(
    frame: np.ndarray,
    haze_threshold: float = 0.3,
    glare_threshold: float = 0.4,
    low_light_threshold: float = 0.35,
) -> dict:
    """分析单帧图像的退化类型与程度

    Args:
        frame: (H, W, 3) uint8 BGR 图像
        haze_threshold: 雾判定阈值
        glare_threshold: 眩光判定阈值
        low_light_threshold: 低光判定阈值

    Returns:
        dict with keys:
          - haze_score:      float [0, 1]
          - glare_score:     float [0, 1]
          - low_light_score: float [0, 1]
          - primary_degradation: str | None
          - is_mixed:        bool
          - active_degradations: list[str]
          - details:          dict

    Raises:
        TypeError: 输入类型错误
        ValueError: 输入 shape 错误
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame 须为 numpy.ndarray，收到 {type(frame)}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame shape 须为 (H,W,3)，收到 {frame.shape}")
    if frame.dtype != np.uint8:
        raise TypeError(f"frame dtype 须为 uint8，收到 {frame.dtype}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 雾霾检测 ──────────────────────────────────────────────────────
    washout = _mid_band_washout(frame)
    decay = _vertical_contrast_decay(gray)
    tex = _texture_density(gray)

    # 纹理门控：无纹理区域不是雾
    if tex < 0.03:
        haze_score = 0.0
    else:
        # 综合：中段漂白率 × (1 + 垂直对比度衰减)，sigmoid 映射
        # 纯漂白率在 0~0.7 范围。晴日 <0.2（蓝色天空轻微低饱和），雾天 >0.3
        # 垂直衰减在晴日为 0（上下对比度均匀），雾天 >0.3（远处模糊近处清晰）
        haze_raw = washout * (1.0 + decay)
        haze_score = float(_sigmoid(np.array(haze_raw), center=0.25, slope=10.0))

    # ── 眩光检测 ──────────────────────────────────────────────────────
    bri_mean = _bright_channel_mean(frame)
    glare_ratio = _glare_region_ratio(frame)

    # 眩光特征：亮通道均值高 + 存在显著高亮区域（严格筛选）
    # 使用高阈值（V>0.93）后，glare_ratio 在正常帧中应为 0
    # 眩光区域占比在 1%~20% 之间，glare_area_weight 线性映射
    glare_area_weight = np.clip(glare_ratio * 5.0, 0.0, 1.0) if glare_ratio < 0.2 else 1.0
    glare_raw = bri_mean * glare_area_weight
    glare_score = float(_sigmoid(np.array(glare_raw), center=0.15, slope=15.0))

    # ── 低光检测 ──────────────────────────────────────────────────────
    lum = _mean_luminance(frame)
    dark_ratio = _low_light_pixel_ratio(frame)

    low_light_raw = (1.0 - lum) * dark_ratio
    low_light_score = float(_sigmoid(np.array(low_light_raw), center=0.15, slope=12.0))

    # ── 综合判定 ──────────────────────────────────────────────────────
    scores = {
        "haze": haze_score,
        "glare": glare_score,
        "low_light": low_light_score,
    }
    max_score_val = max(scores.values())
    thresholds = {"haze": haze_threshold, "glare": glare_threshold, "low_light": low_light_threshold}

    primary = None
    for deg in ["haze", "glare", "low_light"]:
        if scores[deg] >= thresholds[deg] and scores[deg] == max_score_val:
            primary = deg
            break

    active_degradations = [
        deg for deg in ["haze", "glare", "low_light"]
        if scores[deg] >= thresholds[deg]
    ]

    return {
        "haze_score": round(haze_score, 4),
        "glare_score": round(glare_score, 4),
        "low_light_score": round(low_light_score, 4),
        "primary_degradation": primary,
        "is_mixed": len(active_degradations) >= 2,
        "active_degradations": active_degradations,
        "details": {
            "mid_band_washout": round(washout, 4),
            "vertical_contrast_decay": round(decay, 4),
            "texture": round(tex, 4),
            "bright_channel_mean": round(bri_mean, 4),
            "glare_region_ratio": round(glare_ratio, 4),
            "luminance": round(lum, 4),
            "dark_pixel_ratio": round(dark_ratio, 4),
            "contrast": round(_contrast(gray), 4),
            "saturation": round(_mean_saturation(frame), 4),
        },
    }
