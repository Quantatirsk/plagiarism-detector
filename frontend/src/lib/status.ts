import type { DocumentStatus } from '@/api/plagiarismApi';
import type { StatusBadgeTone } from '@/components/ui/status-badge';

export interface StatusMeta {
  label: string;
  tone: StatusBadgeTone;
}

export const DOCUMENT_STATUS_META: Record<DocumentStatus, StatusMeta> = {
  pending: { label: '待处理', tone: 'warning' },
  processing: { label: '处理中', tone: 'info' },
  completed: { label: '已完成', tone: 'success' },
  failed: { label: '处理失败', tone: 'error' },
};

export const JOB_STATUS_META: Record<string, StatusMeta> = {
  draft: { label: '草稿', tone: 'neutral' },
  queued: { label: '排队中', tone: 'warning' },
  running: { label: '执行中', tone: 'info' },
  completed: { label: '已完成', tone: 'success' },
  failed: { label: '失败', tone: 'error' },
};

export const PAIR_STATUS_META: Record<string, StatusMeta> = {
  pending: { label: '排队中', tone: 'warning' },
  running: { label: '执行中', tone: 'info' },
  completed: { label: '已完成', tone: 'success' },
  failed: { label: '失败', tone: 'error' },
  skipped: { label: '已跳过', tone: 'neutral' },
};

export function fallbackStatusMeta(key: string): StatusMeta {
  return { label: key, tone: 'neutral' };
}
