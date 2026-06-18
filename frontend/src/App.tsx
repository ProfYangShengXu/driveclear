import { useState, useCallback, useRef } from 'react';
import { Layout, Typography, Space, message } from 'antd';
import { VideoCameraAddOutlined } from '@ant-design/icons';
import { VideoUploader } from './components/VideoUploader';
import { VideoPlayer } from './components/VideoPlayer';
import { ControlPanel } from './components/ControlPanel';
import { VideoExport } from './components/VideoExport';
import { processVideo, pollStatus, cancelProcessing, cleanupVideo } from './api';
import { ProcessingStatus } from './types';

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;

export default function App() {
  const [videoId, setVideoId] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [enhanced, setEnhanced] = useState(false);
  const pollRef = useRef<{ stop: () => void } | null>(null);

  const handleUploadSuccess = useCallback((id: string, name: string) => {
    setVideoId(id);
    setFilename(name);
    setStatus(null);
    setProgress(0);
    setEnhanced(false);
  }, []);

  const handleStart = useCallback(
    async (params: Record<string, number | boolean>) => {
      if (!videoId) return;

      try {
        // 重置状态
        setStatus(ProcessingStatus.PROCESSING);
        setProgress(0);
        setEnhanced(false);

        // 启动处理
        await processVideo(videoId, params);

        // 开始轮询进度
        if (pollRef.current) {
          pollRef.current.stop();
        }

        pollRef.current = pollStatus(
          videoId,
          (statusResp) => {
            setProgress(statusResp.progress);
            setStatus(statusResp.status as ProcessingStatus);
          },
          () => {
            setStatus(ProcessingStatus.COMPLETED);
            setProgress(100);
            setEnhanced(true);
            message.success('视频处理完成！');
          },
          (error) => {
            setStatus(ProcessingStatus.FAILED);
            message.error(error);
          },
          1000
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : '启动处理失败';
        setStatus(ProcessingStatus.FAILED);
        message.error(msg);
      }
    },
    [videoId]
  );

  const handleCancel = useCallback(async () => {
    if (!videoId) return;
    try {
      await cancelProcessing(videoId);
      if (pollRef.current) {
        pollRef.current.stop();
      }
      setStatus(null);
      setProgress(0);
      message.info('已取消处理');
    } catch {
      message.error('取消失败');
    }
  }, [videoId]);

  const handleCleanup = useCallback(async () => {
    if (!videoId) return;
    try {
      await cleanupVideo(videoId);
      setVideoId(null);
      setFilename(null);
      setStatus(null);
      setProgress(0);
      setEnhanced(false);
      message.success('已清理');
    } catch {
      message.error('清理失败');
    }
  }, [videoId]);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          background: '#001529',
          display: 'flex',
          alignItems: 'center',
          padding: '0 24px',
        }}
      >
        <Space>
          <VideoCameraAddOutlined style={{ fontSize: 24, color: '#fff' }} />
          <Title level={4} style={{ color: '#fff', margin: 0 }}>
            fog-drive-enhancer
          </Title>
          <Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 13 }}>
            雾天行车视频去雾增强工具
          </Text>
        </Space>
      </Header>

      <Content style={{ padding: '24px', maxWidth: 1200, margin: '0 auto', width: '100%' }}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* Step 1: 上传 */}
          {!videoId && (
            <VideoUploader onUploadSuccess={handleUploadSuccess} />
          )}

          {/* Step 2: 处理界面 */}
          {videoId && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <VideoPlayer videoId={videoId} enhanced={enhanced} />
                  <VideoExport
                    videoId={videoId}
                    status={status}
                    onCleanup={handleCleanup}
                  />
                </div>
                <div>
                  <ControlPanel
                    videoId={videoId}
                    status={status}
                    progress={progress}
                    onStart={handleStart}
                    onCancel={handleCancel}
                  />
                  {filename && (
                    <div style={{ marginTop: 8, color: '#888', fontSize: 12 }}>
                      当前视频: {filename}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </Space>
      </Content>

      <Footer style={{ textAlign: 'center', color: '#999', fontSize: 12 }}>
        fog-drive-enhancer v0.1.0 — 基于时域雾层估计 + 暗通道先验的去雾增强工具
      </Footer>
    </Layout>
  );
}
