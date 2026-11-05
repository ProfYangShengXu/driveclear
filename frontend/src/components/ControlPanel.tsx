import { Card, Form, Slider, Button, Space, Progress, Typography, Alert } from 'antd';
import { PlayCircleOutlined, StopOutlined } from '@ant-design/icons';
import { ProcessingStatus } from '../types';

const { Text } = Typography;

interface ControlPanelProps {
  status: ProcessingStatus | null;
  progress: number;
  onStart: (params: Record<string, number | boolean>) => void;
  onCancel: () => void;
  disabled?: boolean;
}

/**
 * 默认参数 — 与后端 PipelineConfig 的默认值保持同步，
 * 避免前端初值反向覆盖后端调优后的参数。
 */
const DEFAULT_PARAMS = {
  window_size: 15,
  percentile: 10,
  omega: 0.95,
  clahe_clip: 2.0,
  sharpen_sigma: 1.0,
  sharpen_strength: 0.8,
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
  const isQueued = status === ProcessingStatus.QUEUED;
  // 队列中同样占用后端任务，禁止重复提交
  const isBusy = isProcessing || isQueued;

  const handleStart = () => {
    // 仅回传表单实际渲染的字段，未渲染的字段不覆盖后端默认值
    const values = form.getFieldsValue([
      'window_size',
      'percentile',
      'omega',
      'clahe_clip',
      'sharpen_strength',
    ]);
    onStart({ ...values });
  };

  return (
    <Card title="处理控制">
      <Form
        form={form}
        layout="vertical"
        initialValues={DEFAULT_PARAMS}
        disabled={isBusy || disabled}
      >
        <Form.Item label="时域窗口大小 (帧)" name="window_size">
          <Slider min={5} max={60} marks={{ 5: '5', 15: '15', 30: '30', 60: '60' }} />
        </Form.Item>

        <Form.Item label="雾层分位数 (%)" name="percentile">
          <Slider min={1} max={30} marks={{ 1: '1%', 10: '10%', 20: '20%', 30: '30%' }} />
        </Form.Item>

        <Form.Item label="去雾强度 (ω)" name="omega">
          <Slider
            min={0.5}
            max={1.0}
            step={0.05}
            marks={{ 0.5: '0.5', 0.75: '0.75', 0.95: '0.95', 1.0: '1.0' }}
          />
        </Form.Item>

        <Form.Item label="CLAHE 对比度" name="clahe_clip">
          <Slider
            min={0.5}
            max={5.0}
            step={0.5}
            marks={{ 0.5: '0.5', 2: '2', 3: '3', 5: '5' }}
          />
        </Form.Item>

        <Form.Item label="锐化强度" name="sharpen_strength">
          <Slider min={0} max={3.0} step={0.1} marks={{ 0: '关', 1: '1', 2: '2', 3: '3' }} />
        </Form.Item>

        {isBusy && (
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
            loading={isBusy}
            disabled={isBusy || isCompleted || disabled}
          >
            {isCompleted ? '已完成' : '开始处理'}
          </Button>

          {isBusy && (
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
