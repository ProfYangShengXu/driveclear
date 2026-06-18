import { useRef, useState, useEffect } from 'react';
import { Card, Segmented } from 'antd';
import { getDownloadUrl } from '../api';


type ViewMode = 'original' | 'enhanced' | 'side-by-side';

interface VideoPlayerProps {
  videoId: string;
  enhanced: boolean;
}

export function VideoPlayer({ videoId, enhanced }: VideoPlayerProps) {
  const [mode, setMode] = useState<ViewMode>('original');
  const originalRef = useRef<HTMLVideoElement>(null);
  const enhancedRef = useRef<HTMLVideoElement>(null);

  const downloadUrl = getDownloadUrl(videoId);

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
          disabled={!enhanced && mode === 'enhanced'}
        />
      }
    >
      {mode === 'original' && (
        <video
          ref={originalRef}
          src={downloadUrl}
          controls
          style={{ width: '100%', maxHeight: 480 }}
        />
      )}
      {mode === 'enhanced' && enhanced && (
        <video
          ref={enhancedRef}
          src={downloadUrl}
          controls
          style={{ width: '100%', maxHeight: 480 }}
        />
      )}
      {mode === 'enhanced' && !enhanced && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#999' }}>
          处理完成后将显示增强后的视频
        </div>
      )}
    </Card>
  );
}
