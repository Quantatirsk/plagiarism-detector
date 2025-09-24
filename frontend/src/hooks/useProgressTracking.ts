import { useCallback, useEffect, useState } from 'react';
import { plagiarismApi } from '@/api/plagiarismApi';
import type { ProgressTask, ProgressStatus } from '@/api/plagiarismApi';

interface UseProgressTrackingOptions {
  pollingInterval?: number;
  onComplete?: (task: ProgressTask) => void;
  onError?: (task: ProgressTask) => void;
  useSSE?: boolean; // Use Server-Sent Events for real-time updates
}

export function useProgressTracking(
  taskId: string | null,
  options: UseProgressTrackingOptions = {}
) {
  const {
    pollingInterval = 1000,
    onComplete,
    onError,
    useSSE = false,
  } = options;

  const [task, setTask] = useState<ProgressTask | null>(null);
  const [subtasks, setSubtasks] = useState<ProgressTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch task info
  const fetchTask = useCallback(async () => {
    if (!taskId) return;

    try {
      const taskData = await plagiarismApi.getProgressTask(taskId);
      setTask(taskData);

      // Fetch subtasks if any
      if (taskData.sub_tasks.length > 0) {
        const subtasksData = await plagiarismApi.getProgressSubtasks(taskId);
        setSubtasks(subtasksData);
      }

      // Handle completion
      if (taskData.status === 'COMPLETED' && onComplete) {
        onComplete(taskData);
      }

      // Handle error
      if (taskData.status === 'FAILED' && onError) {
        onError(taskData);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [taskId, onComplete, onError]);

  // Polling effect
  useEffect(() => {
    if (!taskId || useSSE) return;

    setLoading(true);
    fetchTask().finally(() => setLoading(false));

    const interval = setInterval(fetchTask, pollingInterval);

    return () => clearInterval(interval);
  }, [taskId, pollingInterval, fetchTask, useSSE]);

  // SSE effect
  useEffect(() => {
    if (!taskId || !useSSE) return;

    const eventSource = plagiarismApi.createProgressEventSource(taskId);

    // 初始获取任务状态
    plagiarismApi.getProgressTask(taskId)
      .then((taskData) => {
        setTask(taskData);

        // 检查是否已完成
        if (taskData.status === 'COMPLETED' && onComplete) {
          onComplete(taskData);
        } else if (taskData.status === 'FAILED' && onError) {
          onError(taskData);
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Unknown error');
      });

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'heartbeat') return;

        if (data.type === 'connected') {
          // 连接成功，不需要再次获取
          return;
        }

        // Update task data
        if (data.task_id === taskId) {
          setTask(data);

          // 处理状态变化
          if (data.status === 'COMPLETED' && onComplete) {
            onComplete(data);
          } else if (data.status === 'FAILED' && onError) {
            onError(data);
          }
        }
      } catch (err) {
        console.error('Failed to parse SSE data:', err);
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE error:', err);
      setError('Failed to connect to progress stream');
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [taskId, useSSE]); // 移除 fetchTask 避免无限循环

  const cancelTask = useCallback(async () => {
    if (!taskId) return;

    try {
      await plagiarismApi.cancelProgressTask(taskId);
      await fetchTask();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel task');
    }
  }, [taskId, fetchTask]);

  return {
    task,
    subtasks,
    loading,
    error,
    cancelTask,
    isComplete: task?.status === 'COMPLETED',
    isFailed: task?.status === 'FAILED',
    isRunning: task?.status === 'RUNNING',
    progress: task?.progress_percent || 0,
  };
}

// Hook for tracking multiple active tasks
export function useActiveProgressTasks(pollingInterval = 2000) {
  const [tasks, setTasks] = useState<ProgressTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchActiveTasks = useCallback(async () => {
    try {
      const activeTasks = await plagiarismApi.getActiveProgressTasks();
      setTasks(activeTasks);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch active tasks');
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchActiveTasks().finally(() => setLoading(false));

    const interval = setInterval(fetchActiveTasks, pollingInterval);

    return () => clearInterval(interval);
  }, [pollingInterval, fetchActiveTasks]);

  return {
    tasks,
    loading,
    error,
    refresh: fetchActiveTasks,
  };
}