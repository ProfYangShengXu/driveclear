"""
Phase 1: 时域雾层估计 + 帧间减法

核心思路：
  利用连续帧中雾层相对稳定的特性，估计雾层后从当前帧减去。

快速实现（v2）：
  使用运行式最小值估计（running min）替代原先的 np.percentile。
  对每个像素维护一个运行最小值，以指数衰减方式更新。
  这样避免每帧对 N 帧堆栈做排序，从 O(N log N) 降为 O(1) 每像素。

原理：
  雾增加图像亮度偏移（airlight scattering），运动物体通常比雾暗。
  运行最小值 ≈ 无雾时的最小值，即雾层估计值。
"""

import numpy as np


class TemporalFogEstimator:
    """时域雾层估计器（快速版）

    使用运行式最小值估计雾层，每个新帧以指数衰减方式更新最小值。
    当前帧减去雾层得到时域去雾帧。

    Attributes:
        window_size: 滑动窗口大小（帧数）— 用于预热
        decay: 衰减系数，越小更新越快。默认 0.85
    """

    def __init__(self, window_size: int = 15, percentile: float = 10.0):
        self.window_size = window_size
        # 将 percentile 转换为等效的 decay 系数
        # percentile=10% → 大约保留最近 10-15 帧的影响
        self.decay = 0.90 if percentile >= 10 else 0.95
        self.fog_layer: np.ndarray | None = None
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> np.ndarray:
        """输入新帧，更新雾层估计，返回时域去雾后的帧

        Args:
            frame: (H, W, 3) uint8 BGR 帧

        Returns:
            dehazed_frame: (H, W, 3) uint8 去雾帧
        """
        self.frame_count += 1
        frame_f32 = frame.astype(np.float32)

        if self.fog_layer is None:
            # 首帧：初始化为当前帧
            self.fog_layer = frame_f32.copy()
            return frame

        # 运行式雾层估计：指数衰减取最小值方向
        # fog = decay * fog + (1-decay) * frame 但保持 fog ≤ frame
        # 当物体出现（frame < fog）时快速跟踪；当物体离开（frame > fog）时缓慢恢复
        mask = frame_f32 < self.fog_layer
        self.fog_layer = np.where(
            mask,
            frame_f32,                                    # 新最小值 → 立即更新
            self.decay * self.fog_layer + (1 - self.decay) * frame_f32  # 缓慢衰减
        )

        if self.frame_count < self.window_size:
            return frame

        # 减法去雾
        dehazed = frame_f32 - self.fog_layer
        dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)
        return dehazed

    def reset(self):
        """切换到新场景时重置"""
        self.fog_layer = None
        self.frame_count = 0

    def get_fog_layer(self) -> np.ndarray | None:
        """返回当前雾层估计，尚不可用时返回 None"""
        return self.fog_layer

    def is_ready(self) -> bool:
        """是否已初始化（可开始有效处理）"""
        return self.fog_layer is not None and self.frame_count >= self.window_size
