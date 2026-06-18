import { Card, Form, Slider, Button, Space, Progress, Typography, Alert } from 'antd';
import { PlayCircleOutlined, StopOutlined } from '@ant-design/icons';
import { ProcessingStatus } from '../types';

const { Text } = Typography;

interface ControlPanelProps {
  videoId: string;
  status: ProcessingStatus | null;
  progress: number;
  onStart: (params: Record<string, number | boolean>) => void;
  onCancel: () => void;
  disabled?: boolean;
}

const DEFAULT_PARAMS = {
  window_size: 15,
  percentile: 10,
  omega: 0.95,
  clahe_clip: 3.0,
  sharpen_sigma: 1.5,
  sharpen_strength: 1.5,
};

export function ControlPanel({
  status,
  progress,
  onStart,
  onCancel,
  disabled,
}: ControlPanelProps) {
  const [form] = Form.useForm();

  const isProcessing = status === ProcessingStatus.PROCESSING;
  const isCompleted = status === ProcessingStatus.COMPLETED;
  const isFailed = status === ProcessingStatus.FAILED;

  const handleStart = () => {
    const values = form.getFieldsValue();
    onStart({
      ...DEFAULT_PARAMS,
      window_size: values.window_size,
      percentile: values.percentile,
      omega: values.omega,
      clahe_clip: values.clahe_clip,
      sharpen_sigma: values.sharpen_sigma,
      sharpen_strength: values.sharpen_strength,
    });
  };

  return (
    <Card title="处理控制">
      <Form
        form={form}
        layout="vertical"
        initialValues={DEFAULT_PARAMS}
        disabled={isProcessing || disabled}
      >
        <Form.Item label="时域窗口大小 (帧)" name="window_size">
          <Slider min={5} max={60} marks={{ 5: '5', 15: '15', 30: '30', 60: '60' }} />
        </Form.Item>

        <Form.Item label="雾层分位数 (%)" name="percentile">
          <Slider min={1} max={30} marks={{ 1: '1%', 10: '10%', 20: '20%', 30: '30%' }} />
        </Form.Item>

        <Form.Item label="去雾强度 (ω)" name="omega">
          <Slider min={0.5} max={1.0} step={0.05} marks={{ 0.5: '0.5', 0.75: '0.75', 0.95: '0.95', 1.0: '1.0' }} />
        </Form.Item>

        <Form.Item label="CLAHE 对比度" name="clahe_clip">
          <Slider min={0.5} max={5.0} step={0.5} marks={{ 0.5: '0.5', 2: '2', 3: '3', 5: '5' }} />
        </Form.Item>

        <Form.Item label="锐化强度" name="sharpen_strength">
          <Slider min={0} max={3.0} step={0.1} marks={{ 0: '关', 1: '1', 2: '2', 3: '3' }} />
        </Form.Item>

        {isProcessing && (
          <div style={{ marginBottom: 16 }}>
            <Progress percent={Math.round(progress)} />
            <Text type="secondary">正在处理中...</Text>
          </div>
        )}

        {isFailed && (
          <Alert
            type="error"
            message="处理失败"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {isCompleted && (
          <Alert
            type="success"
            message="处理完成！可在预览区查看或导出视频"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <Space>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleStart}
            loading={isProcessing}
            disabled={isProcessing || isCompleted || disabled}
          >
            {isCompleted ? '已完成' : '开始处理'}
          </Button>

          {isProcessing && (
            <Button
              danger
              icon={<StopOutlined />}
              onClick={onCancel}
            >
              取消
            </Button>
          )}
        </Space>
      </Form>
    </Card>
  );
}
