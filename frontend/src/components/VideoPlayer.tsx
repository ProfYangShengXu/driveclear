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
          src={downloadUrl}
          controls
          style={{ width: '100%', maxHeight: 480 }}
        />
      )}
      {mode === 'enhanced' && enhanced && (
        <video
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
