import { Empty, Button } from 'antd';
import { PlusOutlined, SearchOutlined, DatabaseOutlined, MessageOutlined } from '@ant-design/icons';

export type EmptyStateType = 'query' | 'conversation' | 'dataset' | 'graph' | 'general';

interface EmptyStateConfig {
  icon: React.ReactNode;
  description: string;
  action?: {
    text: string;
    icon?: React.ReactNode;
    onClick?: () => void;
  };
}

interface EmptyStateProps {
  type?: EmptyStateType;
  description?: string;
  action?: {
    text: string;
    icon?: React.ReactNode;
    onClick?: () => void;
  };
}

const emptyStateConfigs: Record<EmptyStateType, EmptyStateConfig> = {
  query: {
    icon: <SearchOutlined style={{ fontSize: 64, opacity: 0.25 }} />,
    description: 'Enter a query to explore the knowledge graph',
    action: {
      text: 'View Examples',
      icon: <SearchOutlined />,
    },
  },
  conversation: {
    icon: <MessageOutlined style={{ fontSize: 64, opacity: 0.25 }} />,
    description: 'Start a new conversation',
    action: {
      text: 'New Conversation',
      icon: <PlusOutlined />,
    },
  },
  dataset: {
    icon: <DatabaseOutlined style={{ fontSize: 64, opacity: 0.25 }} />,
    description: 'Create a dataset to get started',
    action: {
      text: 'Create Dataset',
      icon: <PlusOutlined />,
    },
  },
  graph: {
    icon: <div style={{ fontSize: '64px' }}>📊</div>,
    description: 'Select a dataset and load the graph',
    action: {
      text: 'Load Graph',
    },
  },
  general: {
    icon: <div style={{ fontSize: '64px' }}>📄</div>,
    description: 'No data available',
  },
};

export default function EmptyState({ type = 'general', description, action }: EmptyStateProps) {
  const config = emptyStateConfigs[type];
  const finalDescription = description || config.description;
  const finalAction = action || config.action;

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '300px' }}>
      <Empty
        image={config.icon}
        description={
          <div style={{ opacity: 0.65 }}>
            {finalDescription}
          </div>
        }
      >
        {finalAction && (
          <Button
            type="primary"
            icon={finalAction.icon}
            onClick={finalAction.onClick}
          >
            {finalAction.text}
          </Button>
        )}
      </Empty>
    </div>
  );
}
