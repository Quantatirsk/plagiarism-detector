import { useEffect, useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import { plagiarismApi, type CompareJobSummary, type ProjectSummary } from '@/api/plagiarismApi';
import { useDocuments, useCompareJobs } from '@/hooks/useData';
import { PageShell, PageHeader, PageContent, SectionCard } from '@/components/layout/Page';
import { Button } from '@/components/ui/button';
import { StatusBadge } from '@/components/ui/status-badge';
import { DOCUMENT_STATUS_META, JOB_STATUS_META, fallbackStatusMeta } from '@/lib/status';
import { TABLE_BASE, TABLE_BODY, TABLE_HEAD } from '@/lib/table';

interface ProjectDetailPanelProps {
  project: ProjectSummary;
  onBack: () => void;
  onOpenJob: (job: CompareJobSummary) => void;
}


export function ProjectDetailPanel({ project, onBack, onOpenJob }: ProjectDetailPanelProps) {
  const [uploading, setUploading] = useState(false);
  const documentState = useDocuments({ projectId: project.id });
  const jobsState = useCompareJobs(project.id);
  const [runningComparisons, setRunningComparisons] = useState(false);

  const documents = documentState.data ?? [];
  const jobs = jobsState.data ?? [];

  const hasActiveJobs = useMemo(
    () => jobs.some((job) => job.status === 'queued' || job.status === 'running'),
    [jobs],
  );

  useEffect(() => {
    if (!hasActiveJobs) {
      return;
    }
    const timer = window.setInterval(() => {
      jobsState.reload();
    }, 5000);
    return () => {
      window.clearInterval(timer);
    };
  }, [hasActiveJobs, jobsState.reload]);

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) {
      return;
    }
    setUploading(true);
    try {
      await plagiarismApi.uploadDocuments(project.id, Array.from(event.target.files));
      documentState.reload();
      jobsState.reload();
    } catch (error) {
      console.error(error);
      alert((error as Error).message || '上传失败');
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const handleRunComparisons = async () => {
    setRunningComparisons(true);
    try {
      await plagiarismApi.runProjectComparisons(project.id);
      jobsState.reload();
    } catch (error) {
      console.error(error);
      alert((error as Error).message || '运行比对失败');
    } finally {
      setRunningComparisons(false);
    }
  };

  const completedDocs = documents.filter((document) => document.status === 'completed').length;
  const projectStats = {
    totalDocs: documents.length,
    completedDocs,
    pendingDocs: documents.length - completedDocs,
    jobsTotal: jobs.length,
  };

  const highlightStats = [
    { label: '文档总数', value: projectStats.totalDocs },
    { label: '已完成', value: projectStats.completedDocs },
    { label: '待处理', value: projectStats.pendingDocs },
    { label: '比对任务', value: projectStats.jobsTotal },
  ];

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
              ← 返回列表
            </Button>
            <span>{project.name || `项目 #${project.id}`}</span>
          </div>
        }
        subtitle={project.description || '暂无描述'}
        meta={
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground sm:text-sm">
            <span>文档：{projectStats.completedDocs}/{projectStats.totalDocs}</span>
            <span className="text-border">•</span>
            <span>任务：{projectStats.jobsTotal}</span>
            <span className="text-border">•</span>
            <span>创建：{new Date(project.created_at).toLocaleDateString()}</span>
          </div>
        }
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                documentState.reload();
                jobsState.reload();
              }}
              disabled={documentState.loading || jobsState.loading}
            >
              刷新数据
            </Button>
            <Button
              size="sm"
              onClick={handleRunComparisons}
              disabled={runningComparisons || projectStats.totalDocs < 2}
            >
              {runningComparisons ? '正在执行…' : '运行比对'}
            </Button>
          </div>
        }
      />

      <PageContent className="flex flex-col" containerClassName="flex flex-col flex-1 min-h-0 gap-4 py-4">
        <SectionCard className="flex-shrink-0">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <div className="space-y-5">
              <div className="space-y-1.5">
                <h2 className="text-base font-semibold leading-6 text-foreground">上传文档</h2>
                <p className="text-sm text-muted-foreground">将文件上传至该项目后，系统会自动解析并在需要时生成比对任务。</p>
              </div>
              <dl className="grid gap-3 text-sm text-muted-foreground sm:grid-cols-4">
                {highlightStats.map((item) => (
                  <div
                    key={item.label}
                    className="rounded-lg border border-border/60 bg-background px-3 py-3"
                  >
                    <dt className="text-xs uppercase tracking-wide text-muted-foreground/80">{item.label}</dt>
                    <dd className="mt-2 text-lg font-semibold text-foreground">{item.value}</dd>
                  </div>
                ))}
              </dl>
              <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                <span>支持单个文件最大 100 MB，建议上传可解析的文本或 PDF。</span>
                {hasActiveJobs && <span className="inline-flex h-6 items-center rounded-full bg-primary/10 px-2 text-primary">活跃任务自动刷新中…</span>}
              </div>
            </div>
            <label className="flex min-h-[180px] flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-background/70 px-6 py-10 text-sm text-muted-foreground transition hover:border-primary/40 hover:bg-primary/5">
              <span className="text-sm font-medium text-foreground">
                {uploading ? '正在上传…' : '点击或拖拽选择文件'}
              </span>
              <span className="text-xs text-muted-foreground">支持多选，上传后会自动开始处理</span>
              <input type="file" multiple className="sr-only" onChange={handleUpload} disabled={uploading} />
            </label>
          </div>
        </SectionCard>

        <SectionCard padding="none" className="flex flex-col flex-1 min-h-0">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3 flex-shrink-0">
            <div className="flex items-center gap-3 text-sm font-medium text-muted-foreground">
              <span>项目文档</span>
              <span className="text-xs text-muted-foreground">共 {documents.length} 个</span>
            </div>
            {documentState.loading && <span className="text-xs text-muted-foreground">加载中…</span>}
          </div>
          <div className="flex-1 min-h-0 overflow-auto">
            {documentState.error && (
              <div className="px-5 py-4 text-sm text-destructive">加载文档列表失败：{documentState.error}</div>
            )}
            {!documentState.loading && !documentState.error && documents.length === 0 && (
              <div className="px-5 py-6 text-sm text-muted-foreground">当前项目还没有上传任何文档。</div>
            )}
            {!documentState.loading && !documentState.error && documents.length > 0 && (
              <table className={TABLE_BASE}>
                <thead className={TABLE_HEAD}>
                  <tr>
                    <th className="h-11 px-5 align-middle">文档</th>
                    <th className="h-11 px-5 align-middle">状态</th>
                    <th className="h-11 px-5 align-middle">语言</th>
                    <th className="h-11 px-5 align-middle">更新时间</th>
                  </tr>
                </thead>
                <tbody className={TABLE_BODY}>
                  {documents.map((document) => {
                    const statusMeta = DOCUMENT_STATUS_META[document.status];
                    return (
                      <tr key={document.id} className="transition hover:bg-muted/60">
                        <td className="px-5 py-3 text-sm font-medium">
                          {document.title || document.filename || `文档 #${document.id}`}
                        </td>
                        <td className="px-5 py-3 text-xs">
                          <StatusBadge tone={statusMeta.tone}>{statusMeta.label}</StatusBadge>
                        </td>
                        <td className="px-5 py-3 text-xs text-muted-foreground">{document.language || '—'}</td>
                        <td className="px-5 py-3 text-xs text-muted-foreground">{new Date(document.updated_at).toLocaleString()}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </SectionCard>

        <SectionCard padding="none" className="flex flex-col flex-1 min-h-0">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3 flex-shrink-0">
            <div className="flex items-center gap-3 text-sm font-medium text-muted-foreground">
              <span>比对任务</span>
              <span className="text-xs text-muted-foreground">共 {jobs.length} 个</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              {hasActiveJobs && <span className="inline-flex h-6 items-center rounded-full bg-primary/10 px-2 text-primary">自动刷新中…</span>}
              {jobsState.loading && <span>加载中…</span>}
              <Button variant="ghost" size="sm" disabled={jobsState.loading} onClick={jobsState.reload}>
                刷新
              </Button>
            </div>
          </div>
          <div className="flex-1 min-h-0 overflow-auto">
            {jobsState.error && (
              <div className="px-5 py-4 text-sm text-destructive">加载任务失败：{jobsState.error}</div>
            )}
            {!jobsState.loading && !jobsState.error && jobs.length === 0 && (
              <div className="px-5 py-6 text-sm text-muted-foreground">尚未创建任何比对任务，上传足够文档后可自动生成。</div>
            )}
            {!jobsState.loading && !jobsState.error && jobs.length > 0 && (
              <table className={TABLE_BASE}>
                <thead className={TABLE_HEAD}>
                  <tr>
                    <th className="h-11 px-5 align-middle">任务</th>
                    <th className="h-11 px-5 align-middle">状态</th>
                    <th className="h-11 px-5 align-middle">更新时间</th>
                    <th className="h-11 px-5 align-middle text-right">操作</th>
                  </tr>
                </thead>
                <tbody className={TABLE_BODY}>
                  {jobs.map((job) => {
                    const statusMeta = JOB_STATUS_META[job.status] ?? fallbackStatusMeta(job.status);
                    return (
                      <tr key={job.id} className="transition hover:bg-muted/60">
                        <td className="px-5 py-3 text-sm font-medium">{job.name || `任务 #${job.id}`}</td>
                        <td className="px-5 py-3 text-xs">
                          <StatusBadge tone={statusMeta.tone}>{statusMeta.label}</StatusBadge>
                        </td>
                        <td className="px-5 py-3 text-xs text-muted-foreground">
                          {job.updated_at ? new Date(job.updated_at).toLocaleString() : '—'}
                        </td>
                        <td className="px-5 py-3 text-right text-xs">
                          <Button variant="outline" size="sm" onClick={() => onOpenJob(job)}>
                            查看任务
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
