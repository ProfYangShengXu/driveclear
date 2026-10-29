import { useState, useEffect } from 'react';
import { Card, Segmented } from 'antd';
import { getDownloadUrl } from '../api';


type ViewMode = 'original' | 'enhanced';

interface VideoPlayerProps {
  videoId: string;
  enhanced: boolean;
}

export function VideoPlayer({ videoId, enhanced }: VideoPlayerProps) {
  const [mode, setMode] = useState<ViewMode>('original');

  // 原始视图走上传回放接口，增强视图才走下载接口
  const resultUrl = `/api/upload/${videoId}`;
  const downloadUrl = getDownloadUrl(videoId);
  const videoSrc = mode === 'original' ? resultUrl : downloadUrl;

  // 当处理完成时自动切换到增强视图
  useEffect(() => {
    if (enhanced) {
      setMode('enhanced');
    }
  }, [enhanced]);

  return (
    <Card
      title="视频预览"
      extra={
        <Segmented
          value={mode}
          onChange={(val) => setMode(val as ViewMode)}
          options={[
            { value: 'original', label: '原始' },
            { value: 'enhanced', label: '增强后' },
          ]}
        />
      }
    >
      {mode === 'enhanced' && !enhanced ? (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#999' }}>
          处理完成后将显示增强后的视频
        </div>
      ) : (
        <video
          src={videoSrc}
          controls
          style={{ width: '100%', maxHeight: 480 }}
        />
      )}
    </Card>
  );
}
