import { Spin, Skeleton } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import { designSystem } from '@/styles';

interface LoadingStateProps {
  mode?: 'spin' | 'skeleton' | 'inline';
  size?: 'small' | 'default' | 'large';
  tip?: string;
  rows?: number;
}

export default function LoadingState({ mode = 'spin', size = 'default', tip, rows = 4 }: LoadingStateProps) {
  if (mode === 'skeleton') {
    return <Skeleton active paragraph={{ rows }} />;
  }

  if (mode === 'inline') {
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: designSystem.spacing[2] }}>
        <Spin size="small" indicator={<LoadingOutlined spin />} />
        {tip && <span style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.tertiary }}>{tip}</span>}
      </span>
    );
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '200px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[3] }}>
        <Spin size={size} />
        {tip && <span style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>{tip}</span>}
      </div>
    </div>
  );
}
