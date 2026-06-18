"""
Phase 2: 暗通道先验 (Dark Channel Prior) 去雾

经典 DCP 去雾算法 [He et al., CVPR 2009]：
  雾天图像模型: I(x) = J(x)·t(x) + A·(1 - t(x))
    I: 观测到的有雾图像
    J: 无雾场景辐射（待恢复）
    t: 传输率 (transmission)
    A: 大气光 (atmospheric light)

  暗通道先验：在无雾户外图像中，至少有一个颜色通道在某些像素上强度极低。
  利用该先验可估计传输率和大气光，进而恢复无雾图像。
"""

import cv2
import numpy as np


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """计算暗通道
    dark(x) = min_{c∈{R,G,B}} ( min_{y∈Ω(x)} I_c(y) )

    Args:
        image: (H, W, 3) 图像
        patch_size: 局部窗口大小

    Returns:
        dark: (H, W) 暗通道图
    """
    b, g, r = cv2.split(image)
    min_channel = np.minimum(np.minimum(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(
    image: np.ndarray, dark: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:
    """估计大气光 A

    取暗通道中最亮 top_percent 个像素，对应原图位置取均值。

    Returns:
        A: (3,) BGR 大气光值
    """
    h, w = dark.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)

    flat_dark = dark.ravel()
    indices = np.argpartition(flat_dark, -num_top)[-num_top:]

    flat_image = image.reshape(-1, 3)
    A = np.mean(flat_image[indices], axis=0)
    return A


def estimate_transmission(
    image: np.ndarray,
    A: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
    use_guided_filter: bool = True,
    guide_radius: int = 40,
    guide_eps: float = 1e-6,
    fast: bool = False,
) -> np.ndarray:
    """估计传输率 t(x)

    t(x) = 1 - ω · min_c( min_y( I_c(y) / A_c ) )

    Args:
        image: (H, W, 3) 原图
        A: (3,) 大气光
        omega: 去雾强度 [0, 1]，越大去雾越彻底，默认 0.95
        patch_size: 暗通道窗口大小
        use_guided_filter: 是否使用引导滤波平滑传输率（保持边缘）
        guide_radius: 引导滤波半径
        guide_eps: 引导滤波正则化系数
        fast: 快速模式 — 降采样 1/4 计算传输率再放大

    Returns:
        transmission: (H, W) 传输率图 [t0, 1.0]
    """
    if fast:
        h, w = image.shape[:2]
        scale = 0.25
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        small_A = A.copy()
        normalized = small.astype(np.float32) / (small_A[np.newaxis, np.newaxis, :] + 1e-6)
        dark_small = dark_channel(normalized, max(patch_size, 7))
        transmission_small = 1.0 - omega * dark_small
        transmission_small = np.clip(transmission_small, 0.05, 1.0)

        if use_guided_filter:
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            try:
                transmission_small = cv2.ximgproc.guidedFilter(
                    guide=gray_small,
                    src=transmission_small.astype(np.float32),
                    radius=max(guide_radius // 4, 5),
                    eps=guide_eps,
                )
            except cv2.error:
                transmission_small = cv2.bilateralFilter(
                    transmission_small.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=10
                )

        transmission = cv2.resize(transmission_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return transmission

    normalized = image.astype(np.float32) / (A[np.newaxis, np.newaxis, :] + 1e-6)
    dark_normalized = dark_channel(normalized, patch_size)
    transmission = 1.0 - omega * dark_normalized

    # 裁剪到合理范围
    transmission = np.clip(transmission, 0.05, 1.0)

    if use_guided_filter:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        try:
            transmission = cv2.ximgproc.guidedFilter(
                guide=gray,
                src=transmission.astype(np.float32),
                radius=guide_radius,
                eps=guide_eps,
            )
        except cv2.error:
            # 若系统没有 ximgproc 模块，回退到双边滤波
            transmission = cv2.bilateralFilter(
                transmission.astype(np.float32), d=9, sigmaColor=50, sigmaSpace=guide_radius
            )

    return transmission


def recover(
    image: np.ndarray, A: np.ndarray, transmission: np.ndarray, t0: float = 0.1
) -> np.ndarray:
    """由雾天图像模型恢复无雾图像

    J(x) = (I(x) - A) / max(t(x), t0) + A

    Args:
        image: (H, W, 3) 有雾图像
        A: (3,) 大气光
        transmission: (H, W) 传输率
        t0: 传输率下界，防除零

    Returns:
        result: (H, W, 3) uint8 去雾图像
    """
    transmission = np.clip(transmission, t0, 1.0)
    result = np.zeros_like(image, dtype=np.float32)

    for c in range(3):
        result[:, :, c] = (
            image[:, :, c].astype(np.float32) - A[c]
        ) / transmission + A[c]

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def dehaze(
    image: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
    use_guided_filter: bool = True,
    use_fast: bool = False,
) -> np.ndarray:
    """暗通道先验去雾完整流程

    Args:
        image: (H, W, 3) uint8 BGR 帧
        omega: 去雾强度 [0, 1]
        patch_size: 暗通道窗口大小
        use_guided_filter: 是否引导滤波平滑传输率
        use_fast: 快速模式 — 降采样计算传输率（快 4~8 倍，质量略有下降）

    Returns:
        dehazed: (H, W, 3) uint8 去雾结果
    """
    dark = dark_channel(image, patch_size)
    A = estimate_atmospheric_light(image, dark)
    transmission = estimate_transmission(
        image, A, omega, patch_size, use_guided_filter, fast=use_fast
    )
    result = recover(image, A, transmission)
    return result
