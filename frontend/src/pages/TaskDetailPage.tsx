import { useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { isAxiosError } from 'axios';
import {
  plagiarismApi,
  type ChunkGranularity,
  type CompareJobSummary,
  type ComparePairSummary,
} from '@/api/plagiarismApi';
import { useCompareJobs, useDocuments, useJobPairs } from '@/hooks/useData';
import { PageShell, PageHeader, PageContent, SectionCard } from '@/components/layout/Page';
import { Button } from '@/components/ui/button';
import { StatusBadge } from '@/components/ui/status-badge';
import { JOB_STATUS_META, PAIR_STATUS_META, fallbackStatusMeta } from '@/lib/status';
import { TABLE_BASE, TABLE_BODY_MUTED, TABLE_HEAD } from '@/lib/table';
import { cn } from '@/lib/utils';

interface CompareJobsPageProps {
  onNavigateDocuments: () => void;
  onOpenPair: (pair: ComparePairSummary, job: CompareJobSummary) => void;
}

const GRANULARITY_LABEL: Record<ChunkGranularity, string> = {
  sentence: '句子',
  paragraph: '段落',
};

export function CompareJobsPage({ onNavigateDocuments, onOpenPair }: CompareJobsPageProps) {
  const { data: jobs, loading: jobsLoading, error: jobsError, reload: reloadJobs } = useCompareJobs();
  const completedDocuments = useDocuments('completed');
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const { data: pairs, loading: pairsLoading, error: pairsError, reload: reloadPairs } = useJobPairs(selectedJobId);

  const selectedJob = useMemo(() => jobs?.find((job) => job.id === selectedJobId) ?? null, [jobs, selectedJobId]);
  const completedDocumentOptions = useMemo(() => completedDocuments.data ?? [], [completedDocuments.data]);
  const completedDocCount = completedDocumentOptions.length;
  const selectedJobStatusMeta = selectedJob
    ? JOB_STATUS_META[selectedJob.status] ?? fallbackStatusMeta(selectedJob.status)
    : null;

  const [leftDocumentId, setLeftDocumentId] = useState<number | ''>('');
  const [rightDocumentId, setRightDocumentId] = useState<number | ''>('');
  const [granularity, setGranularity] = useState<ChunkGranularity>('paragraph');
  const [creatingJob, setCreatingJob] = useState(false);
  const [creatingPair, setCreatingPair] = useState(false);
  const [deletingJobId, setDeletingJobId] = useState<number | null>(null);
  const [deletingPairId, setDeletingPairId] = useState<number | null>(null);

  const hasActivePairs = useMemo(() => pairs?.some((pair) => pair.status === 'pending' || pair.status === 'running') ?? false, [pairs]);

  const handleCreateJob = async () => {
    const name = window.prompt('请输入任务名称（可选）');
    setCreatingJob(true);
    try {
      const job = await plagiarismApi.createCompareJob(0, name ?? null, null);
      reloadJobs();
      setSelectedJobId(job.id);
    } catch (err) {
      console.error(err);
      alert((err as Error).message || '创建任务失败');
    } finally {
      setCreatingJob(false);
    }
  };

  const handleDeleteJob = async (jobId: number) => {
    const confirmed = window.confirm('确定要删除该比对任务吗？相关的配对和结果将一并删除。');
    if (!confirmed) {
      return;
    }
    setDeletingJobId(jobId);
    try {
      await plagiarismApi.deleteJob(jobId);
      if (selectedJobId === jobId) {
        setSelectedJobId(null);
      }
      reloadJobs();
      reloadPairs();
    } catch (err) {
      console.error(err);
      alert((err as Error).message || '删除任务失败');
    } finally {
      setDeletingJobId(null);
    }
  };

  const handleDeletePair = async (pairId: number) => {
    const confirmed = window.confirm('确定要删除该文档配对吗？对应的比对结果将被移除。');
    if (!confirmed) {
      return;
    }
    setDeletingPairId(pairId);
    try {
      await plagiarismApi.deletePair(pairId);
      reloadPairs();
      reloadJobs();
    } catch (err) {
      console.error(err);
      alert((err as Error).message || '删除配对失败');
    } finally {
      setDeletingPairId(null);
    }
  };

  const handleSubmitPair = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedJobId || !leftDocumentId || !rightDocumentId) {
      alert('请先选择任务及左右两个文档');
      return;
    }
    if (leftDocumentId === rightDocumentId) {
      alert('左右文档需要不同，不能选择同一个');
      return;
    }
    setCreatingPair(true);
    let pairSubmitted = false;
    try {
      await plagiarismApi.createPairs(
        selectedJobId,
        [{ left_document_id: Number(leftDocumentId), right_document_id: Number(rightDocumentId) }],
        {
          execute: true,
          granularity,
        },
      );
      pairSubmitted = true;
    } catch (err) {
      console.error(err);
      const isTimeoutError =
        isAxiosError(err) && (err.code === 'ECONNABORTED' || err.message?.toLowerCase().includes('timeout'));
      if (isTimeoutError) {
        pairSubmitted = true;
        window.alert('已开始执行比对，系统将在后台处理，完成后结果会自动更新。');
      } else {
        alert((err as Error).message || '添加配对失败');
      }
    } finally {
      setCreatingPair(false);
      if (pairSubmitted) {
        setLeftDocumentId('');
        setRightDocumentId('');
        reloadPairs();
        reloadJobs();
      }
    }
  };

  const jobCount = jobs?.length ?? 0;

  return (
    <PageShell>
      <PageHeader
        title="比对任务管理"
        subtitle="管理比对任务、维护文档配对并查看结果。"
        meta={
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground sm:text-sm">
            <span>{jobsLoading ? '加载中…' : `任务：${jobCount}`}</span>
            <span className="text-border">•</span>
            <span>文档：{completedDocuments.loading ? '...' : completedDocCount}</span>
            {hasActivePairs && selectedJob && (
              <>
                <span className="text-border">•</span>
                <span className="text-primary">自动刷新中</span>
              </>
            )}
          </div>
        }
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="hover:bg-primary/10 hover:text-primary hover:border-primary" onClick={onNavigateDocuments}>
              返回文档库
            </Button>
            <Button variant="default" size="sm" onClick={handleCreateJob} disabled={creatingJob}>
              {creatingJob ? '正在创建…' : '新建任务'}
            </Button>
          </div>
        }
      />

      <PageContent containerClassName="gap-6">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
          <SectionCard padding="none" className="lg:sticky lg:top-24 lg:h-fit">
            <div className="flex items-center justify-between border-b border-border px-5 py-3">
              <div className="text-sm font-medium text-muted-foreground">
                任务列表
              </div>
              <div className="text-xs text-muted-foreground">
                {jobsLoading ? '加载中…' : `${jobCount} 个任务`}
              </div>
            </div>
            {jobsError && (
              <div className="px-5 py-4 text-sm text-destructive">加载任务失败：{jobsError}</div>
            )}
            {!jobsLoading && !jobsError && jobCount === 0 && (
              <div className="px-5 py-6 text-sm text-muted-foreground">暂无任务，点击右上方按钮创建新任务。</div>
            )}
            {jobCount > 0 && (
              <ul className="max-h-[520px] overflow-auto divide-y divide-border/60">
                {jobs?.map((job) => {
                  const isSelected = job.id === selectedJobId;
                  const statusMeta = JOB_STATUS_META[job.status] ?? fallbackStatusMeta(job.status);
                  return (
                    <li key={job.id}>
                      <div
                        className={cn(
                          'flex cursor-pointer items-start justify-between gap-3 px-5 py-4 text-sm transition hover:bg-muted/40',
                          isSelected ? 'bg-primary/5 font-medium ring-1 ring-primary/20' : 'bg-transparent',
                        )}
                        onClick={() => setSelectedJobId(job.id)}
                      >
                        <div className="flex flex-col gap-1">
                          <span className="truncate">{job.name || `任务 #${job.id}`}</span>
                          <span className="flex items-center gap-2 text-xs text-muted-foreground">
                            状态：
                            <StatusBadge tone={statusMeta.tone}>{statusMeta.label}</StatusBadge>
                          </span>
                          {job.updated_at && (
                            <span className="text-[11px] text-muted-foreground">更新时间：{new Date(job.updated_at).toLocaleString()}</span>
                          )}
                        </div>
                        <button
                          type="button"
                          className="text-xs text-destructive hover:underline disabled:opacity-50"
                          disabled={deletingJobId === job.id}
                          onClick={(event) => {
                            event.stopPropagation();
                            handleDeleteJob(job.id);
                          }}
                        >
                          {deletingJobId === job.id ? '正在删除…' : '删除'}
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </SectionCard>

          <div className="flex min-h-full flex-col gap-6">
            {!selectedJob && (
              <SectionCard>
                <div className="flex flex-col gap-2">
                  <h2 className="text-base font-semibold text-foreground">选择任务以查看详情</h2>
                  <p className="text-sm text-muted-foreground">
                    在左侧选择一个任务后，可添加文档配对、查看配对进展并打开配对详情。
                  </p>
                </div>
              </SectionCard>
            )}

            {selectedJob && (
              <>
                <SectionCard>
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-3">
                        <h2 className="text-base font-semibold text-foreground">{selectedJob.name || `任务 #${selectedJob.id}`}</h2>
                        {selectedJobStatusMeta && (
                          <StatusBadge tone={selectedJobStatusMeta.tone}>{selectedJobStatusMeta.label}</StatusBadge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        任务 ID：{selectedJob.id}
                      </p>
                      {selectedJob.updated_at && (
                        <p className="text-sm text-muted-foreground">更新时间：{new Date(selectedJob.updated_at).toLocaleString()}</p>
                      )}
                    </div>
                    <div className="flex flex-col items-end gap-2 text-right text-xs text-muted-foreground sm:text-sm">
                      <span>配对数量：{pairs?.length ?? 0}</span>
                      <Button variant="outline" size="sm" className="hover:bg-primary/10 hover:text-primary hover:border-primary" onClick={reloadPairs} disabled={pairsLoading}>
                        刷新配对
                      </Button>
                    </div>
                  </div>
                </SectionCard>

                <SectionCard>
                  <div className="flex flex-col gap-5">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h3 className="text-base font-semibold text-foreground">新增文档配对</h3>
                        <p className="text-sm text-muted-foreground">
                          仅可选择已处理完成的文档，新增配对后会立即启动比对。
                        </p>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {completedDocuments.loading
                          ? '已完成文档加载中…'
                          : `可选文档：${completedDocCount} 个`}
                      </div>
                    </div>

                    <form className="flex flex-col gap-4" onSubmit={handleSubmitPair}>
                      <div className="grid gap-4 md:grid-cols-3">
                        <label className="flex flex-col gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          左侧文档
                          <select
                            className="h-9 rounded-md border border-border bg-background px-3 text-sm"
                            value={leftDocumentId}
                            onChange={(event) => setLeftDocumentId(event.target.value ? Number(event.target.value) : '')}
                            disabled={completedDocCount === 0 || completedDocuments.loading}
                          >
                            <option value="">请选择文档…</option>
                            {completedDocumentOptions.map((document) => (
                              <option key={document.id} value={document.id}>
                                {document.title || document.filename || `文档 ${document.id}`}
                              </option>
                            ))}
                          </select>
                        </label>

                        <label className="flex flex-col gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          右侧文档
                          <select
                            className="h-9 rounded-md border border-border bg-background px-3 text-sm"
                            value={rightDocumentId}
                            onChange={(event) => setRightDocumentId(event.target.value ? Number(event.target.value) : '')}
                            disabled={completedDocCount === 0 || completedDocuments.loading}
                          >
                            <option value="">请选择文档…</option>
                            {completedDocumentOptions.map((document) => (
                              <option key={document.id} value={document.id}>
                                {document.title || document.filename || `文档 ${document.id}`}
                              </option>
                            ))}
                          </select>
                        </label>

                        <label className="flex flex-col gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          比对粒度
                          <select
                            className="h-9 rounded-md border border-border bg-background px-3 text-sm"
                            value={granularity}
                            onChange={(event) => setGranularity(event.target.value as ChunkGranularity)}
                          >
                            {Object.entries(GRANULARITY_LABEL).map(([value, label]) => (
                              <option key={value} value={value}>
                                {label}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>

                      <div className="flex flex-wrap items-center gap-3">
                        <Button type="submit" disabled={creatingPair || completedDocCount < 2}>
                          {creatingPair ? '正在添加…' : '添加配对'}
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="hover:bg-primary/10 hover:text-primary"
                          onClick={() => {
                            setLeftDocumentId('');
                            setRightDocumentId('');
                          }}
                        >
                          重置选择
                        </Button>
                        {completedDocCount < 2 && !completedDocuments.loading && (
                          <span className="text-xs text-muted-foreground">至少需要 2 个已完成文档才能创建配对。</span>
                        )}
                      </div>
                    </form>
                  </div>
                </SectionCard>

                <SectionCard padding="none">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3">
                    <div className="flex items-center gap-3 text-sm font-medium text-muted-foreground">
                      <span>配对列表</span>
                      <span className="text-xs text-muted-foreground">{pairs?.length ?? 0} 条</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      {pairsLoading && <span>加载中…</span>}
                      <Button variant="ghost" size="sm" className="hover:bg-primary/10 hover:text-primary" onClick={reloadPairs} disabled={pairsLoading}>
                        刷新
                      </Button>
                    </div>
                  </div>

                  <div className="max-h-[420px] overflow-auto">
                    <table className={TABLE_BASE}>
                      <thead className={TABLE_HEAD}>
                        <tr>
                          <th className="h-11 px-5 align-middle">对比文档</th>
                          <th className="h-11 px-5 align-middle">状态</th>
                          <th className="h-11 px-5 align-middle">最高分</th>
                          <th className="h-11 px-5 align-middle">操作</th>
                        </tr>
                      </thead>
                      <tbody className={TABLE_BODY_MUTED}>
                        {pairsLoading && (
                          <tr>
                            <td colSpan={4} className="px-5 py-4 text-center text-sm text-muted-foreground">
                              正在加载配对…
                            </td>
                          </tr>
                        )}
                        {pairsError && !pairsLoading && (
                          <tr>
                            <td colSpan={4} className="px-5 py-4 text-center text-sm text-destructive">
                              加载配对失败：{pairsError}
                            </td>
                          </tr>
                        )}
                        {!pairsLoading && !pairsError && (!pairs || pairs.length === 0) && (
                          <tr>
                            <td colSpan={4} className="px-5 py-4 text-center text-sm text-muted-foreground">
                              该任务尚未添加配对，可通过上方表单新增。
                            </td>
                          </tr>
                        )}
                        {pairs?.map((pair) => {
                          const statusMeta = PAIR_STATUS_META[pair.status] ?? fallbackStatusMeta(pair.status);
                          return (
                            <tr key={pair.id} className="transition hover:bg-muted/60">
                              <td className="px-5 py-3 text-xs">
                                L#{pair.left_document_id} ↔ R#{pair.right_document_id}
                              </td>
                              <td className="px-5 py-3 text-xs">
                                <StatusBadge tone={statusMeta.tone}>{statusMeta.label}</StatusBadge>
                              </td>
                              <td className="px-5 py-3 text-xs text-muted-foreground">
                                {pair.metrics?.top_score != null ? Number(pair.metrics.top_score).toFixed(3) : '—'}
                              </td>
                              <td className="px-5 py-3 text-xs">
                                <div className="flex items-center gap-3">
                                  <button
                                    className="text-primary hover:underline disabled:opacity-50"
                                    disabled={pair.status !== 'completed'}
                                    onClick={() => {
                                      if (selectedJob) {
                                        onOpenPair(pair, selectedJob);
                                      }
                                    }}
                                  >
                                    查看详情
                                  </button>
                                  <button
                                    className="text-destructive hover:underline disabled:opacity-50"
                                    disabled={deletingPairId === pair.id}
                                    onClick={() => handleDeletePair(pair.id)}
                                  >
                                    {deletingPairId === pair.id ? '正在删除…' : '删除'}
                                  </button>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </SectionCard>
              </>
            )}
          </div>
        </div>
      </PageContent>
    </PageShell>
  );
}
