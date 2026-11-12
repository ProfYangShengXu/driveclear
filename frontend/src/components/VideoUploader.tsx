import { useState } from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { isAxiosError } from 'axios';
import type { UploadFile, UploadProps } from 'antd';
import { uploadVideo } from '../api';

/** 上传大小上限（MB），需与后端 MAX_FILE_SIZE 保持一致 */
const MAX_FILE_SIZE_MB = 500;

const { Dragger } = Upload;

interface VideoUploaderProps {
  onUploadSuccess: (videoId: string, filename: string) => void;
  disabled?: boolean;
}

export function VideoUploader({ onUploadSuccess, disabled }: VideoUploaderProps) {
  const [uploading, setUploading] = useState(false);
  const [fileList, setFileList] = useState<UploadFile[]>([]);

  const handleUpload: UploadProps['customRequest'] = async (options) => {
    const { file, onProgress, onError, onSuccess } = options;

    if (!(file instanceof File)) {
      onError?.(new Error('无效文件'));
      return;
    }

    // 上传前先拦截超大文件，避免整包传完才被后端拒绝
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      const msg = `文件超过 ${MAX_FILE_SIZE_MB}MB 上限`;
      message.error(msg);
      onError?.(new Error(msg));
      return;
    }

    setUploading(true);

    try {
      // antd Dragger 会自行上报真实进度，这里不再伪造 10% 以免进度条回跳
      const result = await uploadVideo(file);
      onProgress?.({ percent: 100 });

      message.success(`视频 "${result.filename}" 上传成功`);
      onSuccess?.(result);
      onUploadSuccess(result.video_id, result.filename);
    } catch (err: unknown) {
      // 优先后端返回的 detail，避免只显示 "status code 400"
      const detail = isAxiosError(err)
        ? (err.response?.data as { detail?: string } | undefined)?.detail
        : undefined;
      const msg = typeof detail === 'string'
        ? detail
        : err instanceof Error ? err.message : '上传失败';
      message.error(msg);
      onError?.(new Error(msg));
    } finally {
      // 统一在 finally 复位，避免任一分支漏掉导致上传区永久禁用
      setUploading(false);
    }
  };

  return (
    <Dragger
      accept=".mp4,.avi,.mov,.mkv,.webm"
      customRequest={handleUpload}
      fileList={fileList}
      onChange={(info) => setFileList(info.fileList.slice(-1))}
      disabled={disabled || uploading}
      showUploadList={false}
    >
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">点击或拖拽视频文件到此区域</p>
      <p className="ant-upload-hint">
        支持 mp4 / avi / mov / mkv / webm 格式，最大 {MAX_FILE_SIZE_MB}MB
      </p>
    </Dragger>
  );
}
