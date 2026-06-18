import { useState } from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd';
import { uploadVideo } from '../api';

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

    setUploading(true);

    try {
      // 模拟进度
      onProgress?.({ percent: 10 });
      const result = await uploadVideo(file);
      onProgress?.({ percent: 100 });

      message.success(`视频 "${result.filename}" 上传成功`);
      onSuccess?.(result);
      setUploading(false);
      onUploadSuccess(result.video_id, result.filename);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '上传失败';
      message.error(msg);
      onError?.(new Error(msg));
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
        支持 mp4 / avi / mov / mkv / webm 格式，最大 500MB
      </p>
    </Dragger>
  );
}
