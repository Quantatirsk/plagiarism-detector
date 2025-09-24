import type { ProgressTask } from '@/api/plagiarismApi';
import { cn } from '@/lib/utils';

interface ProgressIndicatorProps {
  task: ProgressTask | null;
  className?: string;
}

export function ProgressIndicator({ task, className }: ProgressIndicatorProps) {
  if (!task || task.status === 'COMPLETED') return null;

  const isRunning = task.status === 'RUNNING';
  const isFailed = task.status === 'FAILED';

  return (
    <div className={cn('inline-flex items-center gap-2', className)}>
      {isRunning && (
        <>
          <div className="h-2 w-2 animate-pulse rounded-full bg-blue-600" />
          <span className="text-sm text-gray-600">
            {task.current_message || 'Processing...'}
          </span>
        </>
      )}
      {isFailed && (
        <>
          <div className="h-2 w-2 rounded-full bg-red-600" />
          <span className="text-sm text-red-600">Failed</span>
        </>
      )}
    </div>
  );
}