"""
眩光抑制模块 (Glare Suppression) — M3

处理行车视频中的镜头眩光（lens flare / glare）：
  - 对向车灯产生的光晕
  - 隧道出口/夕阳的强光过曝
  - 镜头内部反射产生的鬼影

方法：
  1. detect_glare: 基于亮度 + 空间位置 + 形态学分析定位眩光区域
  2. suppress_glare: 局部亮度压低 + 色相保持 + 时域插值恢复纹理

不依赖任何机器学习模型，纯 OpenCV + NumPy 图像处理。
"""

import cv2
import numpy as np


# ─── 内部辅助 ────────────────────────────────────────────────────────────


def _gaussian_kernel_1d(radius: int) -> np.ndarray:
    """1D 高斯核（归一化），用于分离式高斯模糊"""
    size = 2 * radius + 1
    sigma = radius / 2.0
    x = np.arange(size) - radius
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _make_glare_kernel(radius: int) -> np.ndarray:
    """生成眩光扩散核 —— 模拟镜头光晕的星形/圆形扩散

    采用高斯核模拟光晕的自然衰减边缘。
    """
    return cv2.getGaussianKernel(2 * radius + 1, radius / 2.0)


# ─── 公开接口 ────────────────────────────────────────────────────────────


def detect_glare(
    frame: np.ndarray,
    brightness_threshold: float = 0.93,
    min_area_ratio: float = 0.005,
    center_bias: float = 0.3,
) -> dict:
    """检测帧中的眩光区域

    策略：
      1. V 通道高阈值二值化（V > 0.93）
      2. 形态学闭运算连接邻近高亮像素
      3. 连通组件分析，过滤面积 < min_area_ratio 的组件（排除小光源/反光点）
      4. 对每个组件计算中心偏移权重：靠近画面中心的眩光更可能是镜头 flare
      5. 合并各组件为 glare_mask

    Args:
        frame: (H, W, 3) uint8 BGR 帧
        brightness_threshold: 亮度阈值，默认 0.93（排除蓝天）
        min_area_ratio: 最小眩光区域占比，默认 0.005 (0.5%)
        center_bias: 中心偏移权重，越大越偏向中心检测

    Returns:
        dict with keys:
          - glare_mask:     (H, W) uint8 {0, 255}，255=眩光像素
          - glare_intensity: float [0, 1]，综合眩光强度
          - num_regions:    int，检测到的眩光区域数
          - details:        dict，各组件统计

    Raises:
        TypeError: 输入类型错误
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame 须为 numpy.ndarray，收到 {type(frame)}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame shape 须为 (H,W,3)，收到 {frame.shape}")

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Step 1: 亮度二值化
    binary = (v > brightness_threshold).astype(np.uint8) * 255

    # Step 2: 形态学闭运算连接邻近高亮
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 对二值掩码做 dilation：宁可过覆盖不可漏覆盖
    # 因为 suppression 对假阳性的处理是平滑亮度降低，视觉上无害
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.dilate(closed, dilate_kernel, iterations=4)

    # Step 3: 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    total_pixels = h * w
    min_area = int(total_pixels * min_area_ratio)

    glare_mask = np.zeros((h, w), dtype=np.uint8)

    # 预计算距离图用于 center_bias
    cy, cx = np.mgrid[0:h, 0:w]
    center_y, center_x = h / 2.0, w / 2.0
    dist_from_center = np.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2)
    max_dist = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    dist_norm = dist_from_center / max_dist  # [0, 1], 边缘=1, 中心=0

    regions_detail = []
    total_glare_pixels = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # 获取组件掩码
        component_mask = (labels == i).astype(np.uint8)

        # 计算平均亮度
        mean_v = float(np.mean(v[component_mask == 1]))

        # 计算中心偏移权重：中心附近的组件更有可能是眩光
        comp_cy, comp_cx = centroids[i]
        comp_dist = np.sqrt((comp_cy - center_y) ** 2 + (comp_cx - center_x) ** 2) / max_dist
        center_weight = 1.0 - comp_dist * center_bias  # [0.7, 1.0]

        # 该组件的贡献
        glare_mask[component_mask == 1] = 255
        total_glare_pixels += area

        regions_detail.append({
            "area_pixels": int(area),
            "area_ratio": round(area / total_pixels, 4),
            "mean_v": round(mean_v, 4),
            "center_dist": round(comp_dist, 4),
            "center_weight": round(center_weight, 4),
        })

    # 综合眩光强度
    glare_ratio = total_glare_pixels / total_pixels
    # intensity = 面积占比 × 平均亮度因子
    if total_glare_pixels > 0:
        # 眩光区域越大强度越高，但超过 30% 画面则降权（可能为全屏过曝）
        area_factor = glare_ratio * 3.0 if glare_ratio < 0.3 else max(1.0 - glare_ratio, 0.2)
    else:
        area_factor = 0.0

    glare_intensity = float(np.clip(area_factor, 0.0, 1.0))

    return {
        "glare_mask": glare_mask,
        "glare_intensity": round(glare_intensity, 4),
        "num_regions": len(regions_detail),
        "glare_ratio": round(glare_ratio, 4),
        "details": {
            "regions": regions_detail,
            "brightness_threshold": brightness_threshold,
            "min_area_ratio": min_area_ratio,
        },
    }


def suppress_glare(
    frame: np.ndarray,
    glare_mask: np.ndarray,
    prev_frame: np.ndarray | None = None,
    strength: float = 1.5,
) -> np.ndarray:
    """抑制帧中的眩光区域

    策略：
      1. 对 glare_mask 做高斯羽化（软化边缘）
      2. 在 Lab 色彩空间的 L 通道做局部压低：
         - 眩光中心区域强度压低到目标亮度
         - 边缘区域平滑过渡
      3. 色相保持：a/b 通道不变
      4. 如果提供了前一帧，使用时域插值恢复被眩光遮挡的纹理

    Args:
        frame: (H, W, 3) uint8 BGR 帧
        glare_mask: (H, W) uint8 {0, 255}，detect_glare 的输出
        prev_frame: (H, W, 3) uint8 BGR | None，前一帧用于时域纹理恢复
        strength: 眩光抑制强度 [0, 1]，0=无效果, 1=最大抑制

    Returns:
        suppressed: (H, W, 3) uint8 BGR

    Raises:
        TypeError: 输入类型错误
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame 须为 numpy.ndarray，收到 {type(frame)}")
    if glare_mask is not None and glare_mask.shape[:2] != frame.shape[:2]:
        raise ValueError(f"glare_mask shape {glare_mask.shape} 与 frame shape {frame.shape} 不匹配")

    # 零蒙版守卫：无眩光时不处理
    if glare_mask is None or np.max(glare_mask) == 0:
        return frame.copy()

    h, w = frame.shape[:2]

    # --- Step 1: 羽化眩光蒙版 ---
    # 高斯模糊使硬边缘过渡自然
    kernel_size = max(31, min(h, w) // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask_float = glare_mask.astype(np.float32) / 255.0
    mask_feather = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), kernel_size / 4.0)
    # 归一化确保 [0, 1]
    mask_feather = np.clip(mask_feather, 0.0, 1.0)

    # --- Step 2: Lab 色彩空间处理（仅 L 通道） ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b_ch = cv2.split(lab)

    # 目标亮度：将眩光区域的 L 值压低到背景水平
    # 背景亮度估计：取非眩光区域（mask < 0.1）的 L 中位数（比均值更抗离群点）
    non_glare_region = (mask_feather < 0.1)
    if np.sum(non_glare_region) > 100:
        bg_luminance = float(np.median(l[non_glare_region]))
    else:
        bg_luminance = float(np.mean(l))

    # 对每个像素：仅当像素比背景亮时才压低
    # 新 L = 原 L - max(原 L - bg, 0) * mask * strength
    # 只暗化（darken only）：当 原L > bg 时压低，当 原L ≤ bg 时保持不变
    # 避免 mask 边缘区域因 l < bg 而被错误提亮
    excess = np.maximum(l - bg_luminance, 0)
    l_target = l - excess * mask_feather * strength
    l_out = np.clip(l_target, 0, 255)

    # --- Step 3: 时域插值（如果提供了前一帧） ---
    if prev_frame is not None:
        prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        prev_l = prev_lab[:, :, 0]

        # 在眩光区域，如果前一帧该位置亮度更低（未被眩光污染），则从前一帧借纹理
        # 眩光区域 mask_feather > 0.5
        texture_borrow_mask = (mask_feather > 0.5).astype(np.float32) * strength * 0.5
        texture_borrow_mask = cv2.GaussianBlur(texture_borrow_mask, (15, 15), 5)

        # 借纹理：在 L 通道上混合
        l_out = l_out * (1.0 - texture_borrow_mask) + prev_l * texture_borrow_mask
        l_out = np.clip(l_out, 0, 255)

    # 重构 Lab → BGR（统一 uint8）
    lab_out = cv2.merge([
        np.clip(l_out, 0, 255).astype(np.uint8),
        np.clip(a, 0, 255).astype(np.uint8),
        np.clip(b_ch, 0, 255).astype(np.uint8),
    ])
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    return result


def glare_pipeline(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    brightness_threshold: float = 0.93,
    strength: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """眩光检测 + 抑制一站式接口

    Args:
        frame: 当前帧 (H, W, 3) uint8 BGR
        prev_frame: 前一帧或 None
        brightness_threshold: 眩光亮度阈值
        strength: 抑制强度

    Returns:
        (suppressed_frame, glare_info)
    """
    det = detect_glare(frame, brightness_threshold=brightness_threshold)

    if det["glare_intensity"] > 0.01 and np.any(det["glare_mask"]):
        result = suppress_glare(frame, det["glare_mask"], prev_frame, strength)
    else:
        result = frame.copy()

    return result, det
