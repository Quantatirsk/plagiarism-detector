import { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { StatusBadge } from '@/components/ui/status-badge';
import { PageShell, PageHeader, PageContent, SectionCard } from '@/components/layout/Page';
import { JOB_STATUS_META, PAIR_STATUS_META, fallbackStatusMeta } from '@/lib/status';
import { TABLE_BASE, TABLE_BODY, TABLE_HEAD } from '@/lib/table';
import type {
  CompareJobSummary,
  ComparePairSummary,
  ProjectSummary,
  DocumentSummary,
} from '@/api/plagiarismApi';

interface ProjectJobPanelProps {
  project: ProjectSummary;
  job: CompareJobSummary;
  onBack: () => void;
  onOpenPair: (pair: ComparePairSummary) => void;
  pairs: ComparePairSummary[];
  pairsLoading: boolean;
  pairsError: string | null;
  onReloadPairs: () => void;
  documentLookup?: Record<number, DocumentSummary>;
}

export function ProjectJobPanel({
  project,
  job,
  onBack,
  onOpenPair,
  pairs,
  pairsLoading,
  pairsError,
  onReloadPairs,
  documentLookup = {},
}: ProjectJobPanelProps) {
  const hasActivePairs = useMemo(
    () => pairs.some((pair) => pair.status === 'pending' || pair.status === 'running'),
    [pairs],
  );

  const jobStatusMeta = JOB_STATUS_META[job.status] ?? fallbackStatusMeta(job.status);

  return (
    <PageShell>
      <PageHeader
        title={
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="text-xs text-muted-foreground hover:text-foreground"
              onClick={onBack}
            >
              ← 返回项目
            </Button>
            <span>{job.name || `任务 #${job.id}`}</span>
          </div>
        }
        subtitle={`所属项目：${project.name || `项目 #${project.id}`}`}
        meta={<StatusBadge tone={jobStatusMeta.tone}>{jobStatusMeta.label}</StatusBadge>}
        actions={
          <Button variant="outline" size="sm" disabled={pairsLoading} onClick={onReloadPairs}>
            刷新结果
          </Button>
        }
      >
        <div className="flex items-center gap-3 text-xs text-muted-foreground sm:text-sm">
          <span>ID：{job.id}</span>
          {hasActivePairs && <span className="inline-flex h-6 items-center rounded-full bg-primary/10 px-2 text-primary">自动刷新中…</span>}
        </div>
      </PageHeader>

      <PageContent className="flex" containerClassName="flex-1 min-h-0">
        <SectionCard padding="none" className="flex flex-col h-full">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3 flex-shrink-0">
            <div className="text-sm font-medium text-muted-foreground">比对结果</div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              {pairsLoading && <span>加载中…</span>}
              {hasActivePairs && !pairsLoading && (
                <span className="inline-flex h-6 items-center rounded-full bg-primary/10 px-2 text-primary">自动刷新中…</span>
              )}
              <Button variant="ghost" size="sm" disabled={pairsLoading} onClick={onReloadPairs}>
                手动刷新
              </Button>
            </div>
          </div>
          <div className="flex-1 min-h-0 overflow-auto">
            {pairsError && (
              <div className="px-5 py-4 text-sm text-destructive">加载比对结果失败：{pairsError}</div>
            )}
            {!pairsLoading && !pairsError && pairs.length === 0 && (
              <div className="px-5 py-6 text-sm text-muted-foreground">该任务尚未生成比对结果。</div>
            )}
            {!pairsLoading && !pairsError && pairs.length > 0 && (
              <table className={TABLE_BASE}>
                <thead className={TABLE_HEAD}>
                  <tr>
                    <th className="h-11 px-5 align-middle">对比文档</th>
                    <th className="h-11 px-5 align-middle">状态</th>
                    <th className="h-11 px-5 align-middle">最高分</th>
                    <th className="h-11 px-5 align-middle text-right">操作</th>
                  </tr>
                </thead>
                <tbody className={TABLE_BODY}>
                  {pairs.map((pair) => {
                    const statusMeta = PAIR_STATUS_META[pair.status] ?? fallbackStatusMeta(pair.status);
                    return (
                      <tr key={pair.id} className="transition hover:bg-muted/60">
                        <td className="px-5 py-3 text-sm font-medium">
                          <span className="truncate" title={`${getDocumentName(pair.left_document_id, documentLookup)} ↔ ${getDocumentName(pair.right_document_id, documentLookup)}`}>
                            {getDocumentName(pair.left_document_id, documentLookup)} ↔ {getDocumentName(pair.right_document_id, documentLookup)}
                          </span>
                        </td>
                        <td className="px-5 py-3 text-xs">
                          <StatusBadge tone={statusMeta.tone}>{statusMeta.label}</StatusBadge>
                        </td>
                        <td className="px-5 py-3 text-xs text-muted-foreground">{formatTopScore(pair.metrics)}</td>
                        <td className="px-5 py-3 text-right text-xs">
                          <Button
                            size="sm"
                            onClick={() => onOpenPair(pair)}
                            disabled={pair.status !== 'completed'}
                          >
                            {pair.status === 'completed' ? '查看详情' : '处理中…'}
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </SectionCard>
      </PageContent>
    </PageShell>
  );
}

function formatTopScore(metrics: ComparePairSummary['metrics']) {
  if (!metrics || metrics.top_score == null) {
    return '—';
  }
  return Number(metrics.top_score).toFixed(3);
}

function getDocumentName(documentId: number, lookup: Record<number, DocumentSummary>): string {
  const doc = lookup[documentId];
  if (!doc) {
    return `文档 #${documentId}`;
  }
  return doc.title || doc.filename || `文档 #${documentId}`;
}
