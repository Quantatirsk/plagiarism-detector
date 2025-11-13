import { Result, Button, Collapse } from 'antd';
import { getErrorType, getErrorConfig } from '@/utils/error';

interface ErrorStateProps {
  error?: unknown;
  title?: string;
  message?: string;
  onRetry?: () => void;
  onNavigateLogin?: () => void;
  onNavigateHome?: () => void;
  showDetails?: boolean;
}

export default function ErrorState({
  error,
  title,
  message,
  onRetry,
  onNavigateLogin,
  onNavigateHome,
  showDetails = false,
}: ErrorStateProps) {
  const errorType = error ? getErrorType(error) : 'unknown';
  const config = getErrorConfig(errorType, message);

  const handleAction = () => {
    if (errorType === 'auth' && onNavigateLogin) {
      onNavigateLogin();
    } else if (errorType === 'not-found' && onNavigateHome) {
      onNavigateHome();
    } else if (onRetry) {
      onRetry();
    }
  };

  const errorDetails = error ? (
    typeof error === 'string' ? error : (error as any).message || error.toString()
  ) : null;

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '200px' }}>
      <div style={{ textAlign: 'center', maxWidth: '600px' }}>
        <div style={{ fontSize: '64px', marginBottom: '16px' }}>{config.icon}</div>
        <Result
          status="error"
          title={title || config.title}
          subTitle={message || config.description}
          extra={[
            onRetry || onNavigateLogin || onNavigateHome ? (
              <Button key="action" type="primary" onClick={handleAction}>
                {config.actionText}
              </Button>
            ) : null,
          ].filter(Boolean)}
        />
        {showDetails && errorDetails && (
          <Collapse
            size="small"
            items={[
              {
                key: 'details',
                label: '错误详情',
                children: (
                  <pre style={{ fontSize: '12px', textAlign: 'left', overflow: 'auto', maxHeight: '160px', background: 'rgba(0, 0, 0, 0.04)', padding: '8px', borderRadius: '4px' }}>
                    {errorDetails}
                  </pre>
                ),
              },
            ]}
          />
        )}
      </div>
    </div>
  );
}
