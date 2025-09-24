import { useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import { plagiarismApi, type DocumentStatus, type DocumentSummary } from '@/api/plagiarismApi';
import { useDocuments } from '@/hooks/useData';
import { PageShell, PageHeader, PageContent, SectionCard } from '@/components/layout/Page';
import { Button } from '@/components/ui/button';
import { StatusBadge } from '@/components/ui/status-badge';
import { DOCUMENT_STATUS_META } from '@/lib/status';
import { TABLE_BASE, TABLE_BODY_MUTED, TABLE_HEAD } from '@/lib/table';
import { cn } from '@/lib/utils';

interface DocumentLibraryPageProps {
  onNavigateJobs: () => void;
}

export function DocumentLibraryPage({ onNavigateJobs }: DocumentLibraryPageProps) {
  const [statusFilter, setStatusFilter] = useState<DocumentStatus | 'all'>('all');
  const [uploading, setUploading] = useState(false);
  const { data, loading, error, reload } = useDocuments(statusFilter === 'all' ? undefined : statusFilter);

  const documents = useMemo(() => data ?? [], [data]);
  const documentCount = documents.length;

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      return;
    }
    setUploading(true);
    try {
      await plagiarismApi.uploadDocuments(0, Array.from(files));
      reload();
    } catch (err) {
      console.error(err);
      alert((err as Error).message || '上传失败');
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const handleDelete = async (documentId: number, summary: DocumentSummary) => {
    const confirmed = window.confirm(`确定要删除文档「${summary.title || summary.filename || documentId}」吗？该操作会移除所有相关数据。`);
    if (!confirmed) {
      return;
    }
    try {
      await plagiarismApi.deleteDocument(documentId);
      reload();
    } catch (err) {
      console.error(err);
      alert((err as Error).message || 'Delete failed');
    }
  };

  return (
    <PageShell>
      <PageHeader
        title="文档库"
        subtitle="上传文档、查看处理进度，并为后续比对做准备。"
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="hover:bg-primary/10 hover:text-primary hover:border-primary" onClick={onNavigateJobs}>
              前往比对任务
            </Button>
            <Button asChild variant="default" size="sm" disabled={uploading}>
              <label className="cursor-pointer">
                {uploading ? '正在上传…' : '上传文档'}
                <input type="file" multiple className="sr-only" onChange={handleUpload} />
              </label>
            </Button>
          </div>
        }
      />

      <PageContent>
        <SectionCard>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-col gap-1">
              <h2 className="text-base font-semibold text-foreground">状态筛选</h2>
              <p className="text-sm text-muted-foreground">按处理进度快速定位文档，共 {documentCount} 个结果。</p>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {loading && <span>加载中…</span>}
              <Button variant="ghost" size="sm" className="hover:bg-primary/10 hover:text-primary" onClick={reload} disabled={loading}>
                刷新列表
              </Button>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {(['all', 'pending', 'processing', 'completed', 'failed'] as const).map((value) => (
              <button
                key={value}
                className={cn(
                  'rounded-full border px-3 py-1 text-xs transition',
                  statusFilter === value
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border text-muted-foreground hover:text-foreground',
                )}
                onClick={() => setStatusFilter(value)}
              >
                {value === 'all' ? '全部' : DOCUMENT_STATUS_META[value].label}
              </button>
            ))}
          </div>
        </SectionCard>

        <SectionCard padding="none">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="text-sm font-medium text-muted-foreground">文档列表</div>
            <div className="text-xs text-muted-foreground">{documentCount} 条记录</div>
          </div>
          <div className="max-h-[460px] overflow-auto">
            {loading && (
              <div className="px-5 py-6 text-sm text-muted-foreground">正在加载文档…</div>
            )}

            {error && !loading && (
              <div className="px-5 py-4 text-sm text-destructive">加载文档失败：{error}</div>
            )}

            {!loading && !error && documentCount === 0 && (
              <div className="px-5 py-6 text-sm text-muted-foreground">暂无任何文档，点击右上角按钮上传文件。</div>
            )}

            {!loading && !error && documentCount > 0 && (
              <table className={TABLE_BASE}>
                <thead className={TABLE_HEAD}>
                  <tr>
                    <th className="h-11 px-5 align-middle">文档</th>
                    <th className="h-11 px-5 align-middle">语言</th>
                    <th className="h-11 px-5 align-middle">段落数</th>
                    <th className="h-11 px-5 align-middle">句子数</th>
                    <th className="h-11 px-5 align-middle">状态</th>
                    <th className="h-11 px-5 align-middle">更新时间</th>
                    <th className="h-11 px-5 align-middle text-right">操作</th>
                  </tr>
                </thead>
                <tbody className={TABLE_BODY_MUTED}>
                  {documents.map((summary) => (
                    <tr key={summary.id} className="transition hover:bg-muted/60">
                      <td className="px-5 py-3">
                        <div className="flex flex-col">
                          <span className="font-medium">{summary.title || summary.filename || `文档 #${summary.id}`}</span>
                          <span className="text-xs text-muted-foreground">ID：{summary.id}</span>
                        </div>
                      </td>
                      <td className="px-5 py-3 text-xs uppercase text-muted-foreground">{summary.language || '—'}</td>
                      <td className="px-5 py-3 text-xs">{summary.paragraph_count}</td>
                      <td className="px-5 py-3 text-xs">{summary.sentence_count}</td>
                      <td className="px-5 py-3 text-xs">
                        <StatusBadge tone={DOCUMENT_STATUS_META[summary.status].tone}>
                          {DOCUMENT_STATUS_META[summary.status].label}
                        </StatusBadge>
                      </td>
                      <td className="px-5 py-3 text-xs text-muted-foreground">{new Date(summary.updated_at).toLocaleString()}</td>
                      <td className="px-5 py-3 text-right text-xs">
                        <button
                          className="text-destructive hover:underline"
                          onClick={() => handleDelete(summary.id, summary)}
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </SectionCard>
      </PageContent>
    </PageShell>
  );
}
