/** 视频处理状态 */
export enum ProcessingStatus {
  QUEUED = 'queued',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

/** 上传响应 */
export interface UploadResponse {
  video_id: string;
  filename: string;
  message: string;
}

/** 处理状态响应 */
export interface StatusResponse {
  video_id: string;
  status: ProcessingStatus;
  progress: number;
  current_frame: number;
  total_frames: number;
  fps: number;
  width: number;
  height: number;
  error: string | null;
}

/** 处理参数 */
export interface ProcessParams {
  window_size?: number;
  percentile?: number;
  omega?: number;
  clahe_clip?: number;
  sharpen_sigma?: number;
  sharpen_strength?: number;
}
