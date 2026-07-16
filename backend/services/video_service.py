"""
视频 I/O 服务 — 封装 OpenCV 视频读写操作
"""

import uuid
from pathlib import Path

import cv2
import numpy as np

# 临时文件存储目录
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# 支持的视频格式
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# 文件大小限制：500MB
MAX_FILE_SIZE = 500 * 1024 * 1024


def _ensure_dirs():
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def validate_video(file_path: str) -> tuple[bool, str]:
    """验证视频文件是否合法

    Returns:
        (is_valid, error_message)
    """
    path = Path(file_path)

    # 检查扩展名
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"不支持的视频格式: {path.suffix}，支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}"

    # 检查文件大小
    try:
        file_size = path.stat().st_size
    except (FileNotFoundError, OSError) as e:
        return False, f"文件不存在或无法访问: {e}"
    if file_size > MAX_FILE_SIZE:
        return False, f"文件过大（超过 500MB 限制）"

    # 尝试用 OpenCV 打开
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "无法打开视频文件，文件可能已损坏"
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False, "视频文件无法读取第一帧，文件可能已损坏"

    return True, ""


def get_video_info(file_path: str) -> dict:
    """获取视频信息

    Returns:
        dict with keys: fps, width, height, total_frames, duration_sec
    """
    cap = cv2.VideoCapture(str(file_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    return info


def read_frames(file_path: str, start_frame: int = 0, max_frames: int | None = None):
    """逐帧读取视频（生成器），避免全部加载到内存

    Args:
        file_path: 视频路径
        start_frame: 起始帧索引
        max_frames: 最大读取帧数，None 表示读取全部

    Yields:
        (frame_index, frame): 帧索引和 (H, W, 3) uint8 BGR 帧
    """
    cap = cv2.VideoCapture(str(file_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_index, frame
        frame_index += 1
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()


def write_video(
    output_path: str,
    frames: list[np.ndarray] | np.ndarray,
    fps: float,
    width: int,
    height: int,
    fourcc: str = "mp4v",
) -> str:
    """写入视频文件

    Args:
        output_path: 输出路径
        frames: 帧序列 (N, H, W, 3) uint8
        fps: 帧率
        width: 宽度
        height: 高度
        fourcc: 编码器

    Returns:
        output_path: 写入成功的路径
    """
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(output_path, fourcc_code, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
    return output_path


def save_upload(file_bytes: bytes, original_filename: str) -> str:
    """保存上传文件，返回本地路径

    Args:
        file_bytes: 上传的文件字节
        original_filename: 原始文件名（用于推断扩展名）

    Returns:
        保存的文件路径
    """
    _ensure_dirs()
    ext = Path(original_filename).suffix
    video_id = uuid.uuid4().hex
    filename = f"{video_id}{ext}"
    file_path = str(UPLOAD_DIR / filename)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path


def get_output_path(video_id: str) -> str:
    """获取输出视频路径（尚无文件，仅路径）

    Args:
        video_id: 视频唯一 ID

    Returns:
        输出文件路径 (.mp4)
    """
    _ensure_dirs()
    return str(OUTPUT_DIR / f"{video_id}_enhanced.mp4")


def cleanup(video_id: str):
    """清理输入和输出文件"""
    for pattern in [f"{video_id}.*", f"{video_id}_enhanced.*"]:
        for f in UPLOAD_DIR.glob(pattern):
            f.unlink(missing_ok=True)
        for f in OUTPUT_DIR.glob(pattern):
            f.unlink(missing_ok=True)
