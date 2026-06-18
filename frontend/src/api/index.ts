import axios from 'axios';
import type { ProcessParams, StatusResponse, UploadResponse } from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

/**
 * 上传视频文件
 */
export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await api.post<UploadResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000, // 大文件上传延长超时
  });

  return data;
}

/**
 * 开始处理视频
 */
export async function processVideo(
  videoId: string,
  params?: ProcessParams
): Promise<{ video_id: string; status: string; message: string }> {
  const { data } = await api.post(`/process/${videoId}`, params || {});
  return data;
}

/**
 * 查询处理状态
 */
export async function getStatus(
  videoId: string
): Promise<StatusResponse> {
  const { data } = await api.get<StatusResponse>(`/status/${videoId}`);
  return data;
}

/**
 * 取消处理
 */
export async function cancelProcessing(
  videoId: string
): Promise<{ video_id: string; message: string }> {
  const { data } = await api.post(`/process/${videoId}/cancel`);
  return data;
}

/**
 * 获取处理结果预览图 URL
 */
export function getPreviewUrl(videoId: string): string {
  return `/api/preview/${videoId}`;
}

/**
 * 获取处理结果下载 URL
 */
export function getDownloadUrl(videoId: string): string {
  return `/api/download/${videoId}`;
}

/**
 * 清理文件
 */
export async function cleanupVideo(
  videoId: string
): Promise<{ video_id: string; message: string }> {
  const { data } = await api.delete(`/cleanup/${videoId}`);
  return data;
}

/**
 * 轮询处理状态，直到完成或失败
 */
export function pollStatus(
  videoId: string,
  onProgress: (status: StatusResponse) => void,
  onComplete: () => void,
  onError: (error: string) => void,
  intervalMs: number = 1000
): { stop: () => void } {
  let stopped = false;

  const poll = async () => {
    while (!stopped) {
      try {
        const status = await getStatus(videoId);
        onProgress(status);

        if (
          status.status === 'completed' ||
          status.status === 'failed'
        ) {
          if (status.status === 'completed') {
            onComplete();
          } else {
            onError(status.error || '处理失败');
          }
          return;
        }
      } catch (err) {
        if (!stopped) {
          onError('查询状态失败');
        }
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  };

  poll();

  return {
    stop: () => {
      stopped = true;
    },
  };
}
