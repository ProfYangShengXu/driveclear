import { Card, Button, Space, Alert, Typography } from 'antd';
import { DownloadOutlined, DeleteOutlined } from '@ant-design/icons';
import { getDownloadUrl } from '../api';
import { ProcessingStatus } from '../types';

const { Text } = Typography;

interface VideoExportProps {
  videoId: string;
  status: ProcessingStatus | null;
  onCleanup: () => void;
}

export function VideoExport({ videoId, status, onCleanup }: VideoExportProps) {
  const isCompleted = status === ProcessingStatus.COMPLETED;
  const downloadUrl = getDownloadUrl(videoId);

  const handleDownload = () => {
    // 触发浏览器下载
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `enhanced_${videoId}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <Card title="导出">
      {isCompleted ? (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            type="success"
            message="处理结果已就绪"
            showIcon
          />
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            size="large"
            block
            onClick={handleDownload}
          >
            下载增强后的视频
          </Button>
          <Button
            danger
            icon={<DeleteOutlined />}
            block
            onClick={onCleanup}
          >
            清理文件
          </Button>
          <Text type="secondary" style={{ fontSize: 12 }}>
            下载后建议清理服务器上的临时文件以释放空间
          </Text>
        </Space>
      ) : (
        <div style={{ textAlign: 'center', padding: '20px 0', color: '#999' }}>
          处理完成后可在此导出视频
        </div>
      )}
    </Card>
  );
}
