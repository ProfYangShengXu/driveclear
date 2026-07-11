"""
fog-drive-enhancer 后端服务
FastAPI 入口 — 提供视频上传/处理/下载/状态查询 REST API
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from services.processing_service import (
    PipelineConfig,
    ProcessingStatus,
    cancel_processing,
    create_task,
    get_task,
    start_processing,
)
from services.video_service import cleanup, save_upload, validate_video

app = FastAPI(
    title="fog-drive-enhancer",
    description="基于时域雾层估计 + 暗通道先验的大雾行车视频去雾增强工具",
    version="0.1.0",
)

# CORS — 允许前端开发服务器访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 健康检查 ────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "fog-drive-enhancer"}


# ─── 视频上传 ────────────────────────────────────────────────


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件

    接受 mp4/avi/mov/mkv/webm 格式，最大 500MB。
    返回 video_id 用于后续操作。
    """
    # 读取文件内容
    contents = await file.read()

    # 保存到临时目录
    file_path = save_upload(contents, file.filename)

    # 验证视频
    is_valid, error_msg = validate_video(file_path)
    if not is_valid:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=error_msg)

    # 提取 video_id（从文件名）
    video_id = Path(file_path).stem

    return {
        "video_id": video_id,
        "filename": file.filename,
        "message": "上传成功",
    }


# ─── 处理控制 ────────────────────────────────────────────────


@app.post("/api/process/{video_id}")
def process_video(
    video_id: str,
    config: dict | None = None,
):
    """开始处理视频

    请求体可选参数（application/json）：
    - 旧参数: window_size, percentile, omega, clahe_clip, sharpen_sigma, sharpen_strength
    - 新参数: enable_fog, enable_glare, enable_low_light, auto_detect, ...
    - 完整配置见 PipelineConfig 定义
    """
    # 查找上传的文件
    upload_dir = Path(__file__).resolve().parent / "uploads"
    video_files = list(upload_dir.glob(f"{video_id}.*"))

    if not video_files:
        raise HTTPException(status_code=404, detail=f"视频 {video_id} 未找到")

    input_path = str(video_files[0])

    # 解析配置
    if config:
        pipeline_config = PipelineConfig.from_dict(config)
    else:
        pipeline_config = PipelineConfig()

    # 创建任务
    task = create_task(video_id, input_path, config=pipeline_config)

    # 启动后台处理
    success = start_processing(video_id)
    if not success:
        raise HTTPException(status_code=500, detail="启动处理失败")

    return {
        "video_id": video_id,
        "status": ProcessingStatus.PROCESSING,
        "message": "处理已开始",
    }


@app.post("/api/process/{video_id}/cancel")
def cancel_video_processing(video_id: str):
    """取消正在进行的处理"""
    success = cancel_processing(video_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"任务 {video_id} 未找到")
    return {"video_id": video_id, "message": "取消请求已发送"}


# ─── 状态查询 ────────────────────────────────────────────────


@app.get("/api/status/{video_id}")
def get_processing_status(video_id: str):
    """查询处理进度和状态"""
    task = get_task(video_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {video_id} 未找到")

    return {
        "video_id": task.video_id,
        "status": task.status.value,
        "progress": task.progress,
        "current_frame": task.current_frame,
        "total_frames": task.total_frames,
        "fps": task.fps,
        "width": task.width,
        "height": task.height,
        "error": task.error,
    }


# ─── 结果下载 ────────────────────────────────────────────────


@app.get("/api/download/{video_id}")
def download_result(video_id: str):
    """下载处理后的视频"""
    task = get_task(video_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {video_id} 未找到")

    if task.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"视频处理未完成，当前状态: {task.status.value}",
        )

    output_path = Path(task.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="处理结果文件未找到")

    return FileResponse(
        path=str(output_path),
        filename=f"enhanced_{task.video_id}.mp4",
        media_type="video/mp4",
    )


# ─── 预览（获取处理后的第一帧作为缩略图） ────────────────────


@app.get("/api/preview/{video_id}")
def get_preview(video_id: str):
    """获取处理结果的第一帧作为预览图"""
    task = get_task(video_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {video_id} 未找到")

    if task.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400, detail=f"视频处理未完成，当前状态: {task.status.value}"
        )

    import cv2

    cap = cv2.VideoCapture(task.output_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="无法读取预览帧")

    import io

    import numpy as np

    ret_bytes, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret_bytes:
        raise HTTPException(status_code=500, detail="生成预览失败")

    from fastapi.responses import StreamingResponse

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


# ─── 清理 ─────────────────────────────────────────────────────


@app.delete("/api/cleanup/{video_id}")
def cleanup_video(video_id: str):
    """清理上传和结果文件"""
    cleanup(video_id)
    return {"video_id": video_id, "message": "已清理"}


# ─── 启动入口 ────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
