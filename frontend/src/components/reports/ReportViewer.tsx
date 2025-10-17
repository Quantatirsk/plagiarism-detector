/**
 * ReportViewer - 报告查看器主组件
 * 支持文档、对比和项目三种类型的报告展示
 */
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  FileText,
  Download,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Clock,
  Eye,
  Settings
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { MarkdownRenderer } from '@/components/markdown';
import { saveAs } from 'file-saver';
import { renderToStaticMarkup } from 'react-dom/server';

// 报告相关类型定义
export interface ReportData {
  id: string;
  type: 'document' | 'comparison' | 'project';
  title: string;
  summary: string;
  content: Record<string, any>;
  data: any;
  generated_at: string;
  language: string;
  export_formats: string[];
}

export interface ReportProgress {
  task_id: string;
  progress: number;
  stage: string;
  message: string;
  estimated_remaining?: number;
  error?: string;
}

interface ReportViewerProps {
  report?: ReportData;
  progress?: ReportProgress;
  isGenerating?: boolean;
  className?: string;
}

export function ReportViewer({
  report,
  progress,
  isGenerating = false,
  className
}: ReportViewerProps) {
  const [activeSection, setActiveSection] = useState<string>('content');

  // 自动滚动到新内容
  useEffect(() => {
    if (report && activeSection === 'content') {
      const contentElement = document.getElementById('report-content');
      if (contentElement) {
        contentElement.scrollTop = contentElement.scrollHeight;
      }
    }
  }, [report?.content, activeSection]);

  // 获取报告类型显示信息
  const getReportTypeInfo = (type: string) => {
    const typeMap = {
      document: { label: '文档报告', icon: FileText, color: 'bg-blue-500' },
      comparison: { label: '对比报告', icon: Eye, color: 'bg-green-500' },
      project: { label: '项目报告', icon: Settings, color: 'bg-purple-500' }
    };
    return typeMap[type as keyof typeof typeMap] || typeMap.document;
  };

  // 获取进度阶段显示文本
  const getProgressStageText = (stage: string) => {
    const stageMap: Record<string, string> = {
      initializing: '初始化中...',
      data_collection: '收集数据中...',
      template_preparation: '准备模板中...',
      llm_generation: '生成报告内容中...',
      content_processing: '处理报告内容中...',
      report_finalization: '完成报告生成中...',
      completed: '生成完成',
      error: '生成失败'
    };
    return stageMap[stage] || stage;
  };

  // 渲染生成进度
  const renderProgress = () => {
    if (!isGenerating && !progress) return null;

    const progressPercent = progress ? Math.round(progress.progress * 100) : 0;
    const isError = progress?.error || progress?.stage === 'error';

    return (
      <Card className="mb-6">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2">
            {isError ? (
              <AlertCircle className="h-5 w-5 text-red-500" />
            ) : (
              <RefreshCw className={cn("h-5 w-5", isGenerating && "animate-spin")} />
            )}
            报告生成进度
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isError ? (
            <Alert variant="destructive">
              <AlertDescription>
                生成失败: {progress?.error || '未知错误'}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span>{getProgressStageText(progress?.stage || 'initializing')}</span>
                <span>{progressPercent}%</span>
              </div>
              <Progress value={progressPercent} className="w-full" />
              {progress?.message && (
                <p className="text-sm text-muted-foreground">{progress.message}</p>
              )}
              {progress?.estimated_remaining && (
                <p className="text-sm text-muted-foreground flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  预计剩余 {progress.estimated_remaining} 秒
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  // 渲染报告头部
    const renderReportHeader = () => {
      if (!report) return null;

    const typeInfo = getReportTypeInfo(report.type);
    const TypeIcon = typeInfo.icon;

    return (
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div className={cn("p-2 rounded-lg", typeInfo.color)}>
                <TypeIcon className="h-6 w-6 text-white" />
              </div>
              <div>
                <CardTitle className="text-xl mb-2">{report.title}</CardTitle>
                <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                  <Badge variant="secondary">{typeInfo.label}</Badge>
                  <Badge variant="outline">{report.language === 'zh' ? '中文' : 'English'}</Badge>
                  <span>{new Date(report.generated_at).toLocaleString('zh-CN')}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleExportHtml(report)}
                disabled={isGenerating}
              >
                <Download className="h-4 w-4 mr-1" />
                下载报告
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>
    );
  };

  // 渲染报告导航
  const renderNavigation = () => {
    if (!report) return null;

    const sections = [
      { id: 'content', label: '详细内容', icon: Eye },
      { id: 'data', label: '原始数据', icon: Settings }
    ];

    return (
      <div className="flex gap-1 mb-6 p-1 bg-muted rounded-lg">
        {sections.map((section) => {
          const SectionIcon = section.icon;
          const isActive = activeSection === section.id;

          return (
            <Button
              key={section.id}
              variant={isActive ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveSection(section.id)}
              className="flex-1"
            >
              <SectionIcon className="h-4 w-4 mr-2" />
              {section.label}
            </Button>
          );
        })}
      </div>
    );
  };

  // 渲染报告内容
  const renderReportContent = () => {
    if (!report) return null;

    const renderContent = () => {
      const content = report.content;
      const fullContent = content?.full_content || content?.generated_content || '';
      const structuredReport = content?.structured_report;
      const segmentsTableHtml = structuredReport?.summary?.segments_table_html;
      const highlightedSegments = segmentsTableHtml ? enhanceSimilarityTable(segmentsTableHtml) : '';
      const datasetWarning = structuredReport?.summary?.large_dataset_warning
        || structuredReport?.large_dataset_warning;

      return (
        <Card>
          <CardHeader>
            <CardTitle>详细报告内容</CardTitle>
          </CardHeader>
          <CardContent>
            {report.summary && (
              <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 p-4">
                <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-primary">
                  <CheckCircle className="h-4 w-4" />
                  概览摘要
                </div>
                <MarkdownRenderer content={report.summary} className="prose prose-sm text-muted-foreground" />
              </div>
            )}
            {datasetWarning && (
              <div className="mb-4 rounded-lg border border-orange-300 bg-orange-100/70 px-3 py-3 text-sm text-orange-900">
                {datasetWarning}
              </div>
            )}
            <div
              id="report-content"
              className="prose prose-sm max-w-none"
            >
              {fullContent ? (
                <MarkdownRenderer content={fullContent} />
              ) : (
                <p className="text-muted-foreground">暂无详细内容</p>
              )}
            </div>

            {highlightedSegments && (
              <div className="mt-6">
                <h3 className="mb-2 text-sm font-semibold text-foreground">高相似片段列表 (≥ 0.8)</h3>
                <div className="rounded-lg border border-border bg-muted/40 p-2 text-sm">
                  <div
                    className="markdown-renderer"
                    dangerouslySetInnerHTML={{ __html: highlightedSegments }}
                  />
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      );
    };

    const renderData = () => (
      <Card>
        <CardHeader>
          <CardTitle>数据详情</CardTitle>
          <p className="text-sm text-muted-foreground">报告生成所使用的原始数据</p>
        </CardHeader>
        <CardContent>
          <div className="bg-muted p-4 rounded-lg overflow-auto max-h-96">
            <pre className="text-sm">
              {JSON.stringify(report.data, null, 2)}
            </pre>
          </div>
        </CardContent>
      </Card>
    );

    switch (activeSection) {
      case 'content':
        return renderContent();
      case 'data':
        return renderData();
      default:
        return renderContent();
    }
  };

  return (
    <div className={cn("space-y-6", className)}>
      {renderProgress()}
      {renderReportHeader()}
      {report && (
        <>
          {renderNavigation()}
          {renderReportContent()}
        </>
      )}

      {!report && !isGenerating && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">暂无报告</h3>
            <p className="text-muted-foreground text-center">
              请选择文档或项目生成报告
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function handleExportHtml(report: ReportData) {
  const { title, content, generated_at, language } = report;
  const structured = content?.structured_report;
  const fullContent = content?.full_content || '';

  const rawSegments = structured?.summary?.segments_table_html ?? '';
  const htmlSegments = rawSegments ? enhanceSimilarityTable(rawSegments) : '';
  const structuredSectionsHtml = structured ? buildStructuredSections(report, structured) : '';
  const datasetWarning = structured?.summary?.large_dataset_warning || structured?.large_dataset_warning;
  const safeTitle = title.replace(/[^\w\u4e00-\u9fa5]+/g, '-');

  const summaryHtml = report.summary ? markdownToHtml(report.summary) : '<p>暂无摘要</p>';
  const fullContentHtml = fullContent ? markdownToHtml(fullContent) : '<p>暂无详细内容</p>';

  const htmlDocument = `<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>${escapeHtml(title)}</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --muted: #e2e8f0;
      --text: #1f2937;
      --text-subtle: #64748b;
      --accent: #2563eb;
      --accent-soft: rgba(37, 99, 235, 0.08);
      --divider: rgba(148, 163, 184, 0.35);
      font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      padding: clamp(24px, 3vw, 48px) clamp(16px, 4vw, 56px) clamp(40px, 6vw, 72px);
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
      width: 100vw;
    }
    .page-shell {
      width: min(1400px, calc(100vw - clamp(32px, 6vw, 112px)));
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: clamp(32px, 3.5vw, 52px);
    }
    header {
      background: var(--surface);
      padding: clamp(32px, 5.5vw, 56px) clamp(28px, 5vw, 60px);
      border-bottom: 1px solid var(--divider);
      box-shadow: 0 20px 40px rgba(15, 23, 42, 0.06);
      position: relative;
      overflow: hidden;
      width: 100%;
      border-radius: 24px;
    }
    header::after {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(37, 99, 235, 0.08), rgba(79, 70, 229, 0.05));
      pointer-events: none;
    }
    header .content {
      position: relative;
      z-index: 1;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.36em;
      font-size: 12px;
      color: var(--text-subtle);
      margin-bottom: 18px;
    }
    h1 {
      font-size: 32px;
      font-weight: 600;
      margin: 0;
      color: var(--text);
    }
    .meta {
      margin-top: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 14px;
      color: var(--text-subtle);
    }
    main {
      width: 100%;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: clamp(32px, 4.5vw, 56px);
    }
    .section {
      background: var(--surface);
      border-radius: 20px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      padding: clamp(28px, 3.8vw, 46px);
      box-shadow: 0 28px 60px rgba(15, 23, 42, 0.08);
      position: relative;
      overflow: hidden;
      width: 100%;
    }
    .section::before {
      content: '';
      position: absolute;
      top: 0;
      left: 32px;
      width: 80px;
      height: 4px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(37,99,235,0.9), rgba(14,165,233,0.9));
    }
    .section h2 {
      margin-top: 0;
      margin-bottom: 18px;
      font-size: 20px;
      letter-spacing: 0.02em;
      font-weight: 600;
    }
    .section h3 {
      margin-top: 24px;
      margin-bottom: 12px;
      font-size: 16px;
      font-weight: 600;
    }
    p {
      color: #1e293b;
      margin-bottom: 1rem;
    }
    blockquote {
      margin: 1.5rem 0;
      padding: 1rem 1.5rem;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      color: var(--text);
      border-radius: 0 16px 16px 0;
    }
    code {
      background: rgba(148, 163, 184, 0.2);
      padding: 0.15rem 0.4rem;
      border-radius: 0.4rem;
      font-size: 0.9em;
    }
    pre {
      background: rgba(15, 23, 42, 0.92);
      color: #f8fafc;
      padding: 1.25rem 1.5rem;
      border-radius: 16px;
      overflow: auto;
      font-size: 0.95em;
    }
    .section > * {
      width: 100%;
    }
    .markdown-renderer,
    .summary-panel,
    .table-scroll {
      width: 100%;
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.25rem 0;
      font-size: 0.95em;
      table-layout: fixed;
      border: 1px solid rgba(226, 232, 240, 0.9);
    }
    thead th {
      background: rgba(37, 99, 235, 0.08);
      color: #0f172a;
      font-weight: 600;
      border-bottom: 1px solid rgba(226, 232, 240, 0.9);
    }
    th,
    td {
      border-right: 1px solid rgba(226, 232, 240, 0.9);
      padding: 12px 16px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }
    th:last-child,
    td:last-child {
      border-right: none;
    }
    .summary-panel {
      border-radius: 18px;
      border: 1px solid rgba(37, 99, 235, 0.25);
      background: linear-gradient(180deg, rgba(37,99,235,0.12), rgba(37,99,235,0.02));
      padding: 24px 28px;
      margin-bottom: 24px;
      width: 100%;
    }
    .dataset-warning {
      margin-bottom: 18px;
      padding: 0.85rem 1rem;
      border-radius: 14px;
      border: 1px solid rgba(249, 115, 22, 0.4);
      background: rgba(249, 115, 22, 0.12);
      color: #7c2d12;
      font-size: 0.9rem;
      line-height: 1.5;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: clamp(16px, 3vw, 28px);
      margin-top: clamp(12px, 2vw, 18px);
    }
    .stat-card {
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.04), rgba(15, 23, 42, 0.02));
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 18px;
      padding: clamp(18px, 3vw, 28px);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .stat-label {
      font-size: 0.78rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: rgba(15, 23, 42, 0.55);
    }
    .stat-value {
      font-weight: 600;
      font-size: clamp(1.25rem, 2.9vw, 1.7rem);
      color: var(--text);
    }
    .stat-subtext {
      font-size: 0.8rem;
      color: var(--text-subtle);
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: clamp(12px, 2.5vw, 20px);
      margin-top: clamp(18px, 2.8vw, 26px);
    }
    .detail-card {
      border: 1px solid rgba(226, 232, 240, 0.8);
      border-radius: 16px;
      padding: clamp(16px, 2.6vw, 22px);
      background: rgba(255, 255, 255, 0.92);
      box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.15);
    }
    .detail-label {
      font-size: 0.85rem;
      color: rgba(15, 23, 42, 0.65);
      margin-bottom: 6px;
    }
    .detail-value {
      font-weight: 600;
      color: var(--text);
    }
    .risk-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.05em;
    }
    .risk-pill.low {
      background: rgba(34, 197, 94, 0.16);
      color: #14532d;
    }
    .risk-pill.moderate {
      background: rgba(250, 204, 21, 0.18);
      color: #78350f;
    }
    .risk-pill.high {
      background: rgba(249, 115, 22, 0.18);
      color: #7c2d12;
    }
    .risk-pill.critical {
      background: rgba(239, 68, 68, 0.2);
      color: #7f1d1d;
    }
    .list-stack {
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-top: 20px;
    }
    .list-stack li {
      margin-left: 1.25rem;
      color: var(--text-subtle);
    }
    .table-title {
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: rgba(15, 23, 42, 0.6);
      margin-bottom: 12px;
    }
    .summary-panel h3 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--accent);
    }
    @media print {
      body {
        background: #ffffff;
      }
      header {
        box-shadow: none;
      }
      .section {
        box-shadow: none;
        break-inside: avoid;
        page-break-inside: avoid;
      }
    }
  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-mll67QQ0KC5xRys8D0nWWsMl0zW34NLG8RpjYsY9P+8pcbpucW0V1mFQ0b8x2W2H" crossorigin="anonymous" />
</head>
<body>
  <div class="page-shell">
    <header>
      <div class="content">
        <div class="eyebrow">Strategic Similarity Intelligence</div>
        <h1>${escapeHtml(title)}</h1>
        <div class="meta">
          <span>生成时间：${escapeHtml(new Date(generated_at || Date.now()).toLocaleString('zh-CN'))}</span>
          <span>报告语言：${language === 'zh' ? '中文' : 'English'}</span>
        </div>
      </div>
    </header>

    <main>
      <section class="section">
        <div class="summary-panel">
          <h3>Executive Brief</h3>
          <div>${summaryHtml}</div>
        </div>
        <h2>详细洞察</h2>
        ${datasetWarning ? `<div class="dataset-warning">${escapeHtml(datasetWarning)}</div>` : ''}
        <div>${fullContentHtml}</div>
      </section>

      ${htmlSegments ? `<section class="section"><h2>高相似片段（≥ 0.8）</h2><div class="table-scroll">${htmlSegments}</div></section>` : ''}

      ${structuredSectionsHtml || (structured ? `<section class="section"><h2>结构化数据总览</h2><pre>${escapeHtml(JSON.stringify(structured, null, 2))}</pre></section>` : '')}
    </main>
  </div>
</body>
</html>`;

  try {
    const blob = new Blob([htmlDocument], { type: 'text/html;charset=utf-8' });
    saveAs(blob, `${safeTitle || 'report'}.html`);
  } catch (error) {
    console.error('导出报告失败', error);
  }
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function markdownToHtml(markdown: string): string {
  return renderToStaticMarkup(
    <MarkdownRenderer content={markdown} className="markdown-renderer" />
  );
}

function enhanceSimilarityTable(html: string): string {
  if (!html?.trim()) {
    return html;
  }

  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const rows = Array.from(doc.querySelectorAll('tbody tr'));

    rows.forEach((row) => {
      const scoreCell = row.querySelector('td:nth-child(2)');
      if (!scoreCell) {
        return;
      }

      const rawText = scoreCell.textContent?.trim().replace('%', '') ?? '';
      const score = Number(rawText) / 100;
      if (Number.isNaN(score)) {
        return;
      }

      if (score >= 0.9) {
        row.classList.add('score-critical');
        row.setAttribute('style', combineStyles(row.getAttribute('style'), 'background: rgba(239, 68, 68, 0.18); color: #7f1d1d;'));
      } else if (score >= 0.85) {
        row.classList.add('score-high');
        row.setAttribute('style', combineStyles(row.getAttribute('style'), 'background: rgba(249, 115, 22, 0.16);'));
      } else if (score >= 0.8) {
        row.classList.add('score-elevated');
        row.setAttribute('style', combineStyles(row.getAttribute('style'), 'background: rgba(250, 204, 21, 0.14);'));
      }
    });

    const tbody = doc.querySelector('tbody');
    if (tbody) {
      return tbody.parentElement?.outerHTML ?? html;
    }
    return doc.body.innerHTML || html;
  } catch (error) {
    console.warn('Failed to enhance similarity table', error);
    return html;
  }
}

function combineStyles(existing: string | null, incoming: string): string {
  const base = existing?.trim().replace(/;\s*$/, '') ?? '';
  return base ? `${base}; ${incoming}` : incoming;
}

function buildStructuredSections(report: ReportData, structured: any): string {
  if (!structured) {
    return '';
  }

  switch (report.type) {
    case 'document':
      return buildDocumentSections(structured);
    case 'comparison':
      return buildComparisonSections(structured);
    case 'project':
      return buildProjectSections(structured);
    default:
      return '';
  }
}

function buildDocumentSections(structured: any): string {
  const meta = structured.document ?? {};
  const summary = structured.summary ?? {};
  const statistics = summary.statistics ?? {};
  const sources = structured.top_similarity_sources ?? [];

  const statCards = buildStatGrid([
    { label: '总体相似度', value: formatPercent(meta.total_similarity_score) },
    { label: '风险等级', value: renderRiskPill(meta.risk_level), html: true },
    { label: '触发片段数量', value: formatNumber(summary.segments_above_threshold) },
    { label: '监测阈值', value: formatPercent(summary.segment_threshold) }
  ]);

  const statsDetail = buildDetailGrid(statistics);

  const sourcesTable = buildSourcesTable(sources, '相似来源');

  return [
    `<section class="section">
      <h2>文档风险概览</h2>
      ${statCards}
      ${statsDetail}
    </section>`,
    sourcesTable
  ].filter(Boolean).join('');
}

function buildComparisonSections(structured: any): string {
  const metrics = structured.similarity_metrics ?? {};
  const leftDoc = structured.documents?.left?.title ?? '文档A';
  const rightDoc = structured.documents?.right?.title ?? '文档B';
  const highlights = structured.side_by_side_highlights ?? [];
  const summary = structured.summary ?? {};
  const datasetWarning = summary.large_dataset_warning || structured.large_dataset_warning;

  const statCards = buildStatGrid([
    { label: `${leftDoc} → ${rightDoc}`, value: formatPercent(metrics.a_to_b) },
    { label: `${rightDoc} → ${leftDoc}`, value: formatPercent(metrics.b_to_a) },
    { label: '共同内容占比', value: formatPercent(metrics.common_similarity) },
    { label: `${leftDoc} 独有内容`, value: formatPercent(metrics.unique_a_ratio) },
    { label: `${rightDoc} 独有内容`, value: formatPercent(metrics.unique_b_ratio) }
  ]);

  const highlightList = buildHighlightList(highlights, leftDoc, rightDoc);
  const warningBlock = datasetWarning
    ? `<div class="dataset-warning" style="margin-top:18px;">${escapeHtml(datasetWarning)}</div>`
    : '';

  return `
    <section class="section">
      <h2>对比指标概览</h2>
      ${statCards}
      ${warningBlock}
      ${highlightList}
    </section>
  `;
}

function buildProjectSections(structured: any): string {
  const stats = structured.statistics ?? {};
  const highRiskDocs = structured.high_risk_documents ?? [];
  const recommendations = structured.recommendations ?? [];
  const anomalies = structured.anomalies ?? [];

  const statCards = buildStatGrid([
    { label: '项目文档数', value: formatNumber(stats.total_documents) },
    { label: '比较总数', value: formatNumber(stats.total_comparisons) },
    { label: '平均雷同度', value: formatPercent(stats.average_similarity) },
    { label: '高风险文档', value: formatNumber(stats.high_risk_count) }
  ]);

  const statsDetail = buildDetailGrid(stats);

  const highRiskTable = buildHighRiskTable(highRiskDocs);
  const recommendationList = Array.isArray(recommendations) && recommendations.length
    ? `<section class="section">
        <h2>合规建议</h2>
        <ul class="list-stack">
          ${recommendations.map((item: string) => `<li>${escapeHtml(item)}</li>`).join('')}
        </ul>
      </section>`
    : '';

  const anomaliesList = Array.isArray(anomalies) && anomalies.length
    ? `<section class="section">
        <h2>异常预警</h2>
        <ul class="list-stack">
          ${anomalies.map((item: any) => `<li><strong>${escapeHtml((item?.type ?? '未知')).toUpperCase()}</strong> — ${escapeHtml(item?.description ?? '未提供说明')} (${escapeHtml(item?.severity ?? 'normal')})</li>`).join('')}
        </ul>
      </section>`
    : '';

  return [
    `<section class="section">
      <h2>项目雷同风险总览</h2>
      ${statCards}
      ${statsDetail}
    </section>`,
    highRiskTable,
    recommendationList,
    anomaliesList
  ].filter(Boolean).join('');
}

function buildStatGrid(cards: Array<{ label: string; value: string; html?: boolean; subtext?: string }>): string {
  if (!cards.length) {
    return '';
  }

  const content = cards
    .map((card) => `
      <div class="stat-card">
        <div class="stat-label">${escapeHtml(card.label)}</div>
        <div class="stat-value">${card.html ? card.value : escapeHtml(card.value)}</div>
        ${card.subtext ? `<div class="stat-subtext">${escapeHtml(card.subtext)}</div>` : ''}
      </div>
    `)
    .join('');

  return `<div class="stat-grid">${content}</div>`;
}

function buildDetailGrid(stats: Record<string, any>): string {
  const entries = Object.entries(stats ?? {}).filter(([_, value]) => value !== undefined && value !== null);
  if (!entries.length) {
    return '';
  }

  const content = entries
    .map(([key, value]) => `
      <div class="detail-card">
        <div class="detail-label">${escapeHtml(formatKeyLabel(key))}</div>
        <div class="detail-value">${escapeHtml(formatMetricValue(key, value))}</div>
      </div>
    `)
    .join('');

  return `<div class="detail-grid">${content}</div>`;
}

function buildSourcesTable(sources: any[], title: string): string {
  if (!Array.isArray(sources) || sources.length === 0) {
    return '';
  }

  const rows = sources
    .map((source: any, index: number) => {
      const score = Number(source?.similarity_score ?? NaN);
      const scoreClass = classForScore(score);
      const rowStyle = rowStyleForScore(score);
      return `
        <tr class="${scoreClass}" style="${rowStyle}">
          <td>${index + 1}</td>
          <td>${escapeHtml(source?.document_title ?? '-')}</td>
          <td>${formatPercent(score)}</td>
          <td>${formatNumber(source?.match_count)}</td>
          <td>${formatNumber(source?.total_text_length)}</td>
        </tr>
      `;
    })
    .join('');

  return `
    <section class="section">
      <h2>${escapeHtml(title)}</h2>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>来源文档</th>
              <th>相似度</th>
              <th>匹配数量</th>
              <th>覆盖文本长度</th>
            </tr>
          </thead>
          <tbody>
            ${rows}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function buildHighRiskTable(docs: any[]): string {
  if (!Array.isArray(docs) || docs.length === 0) {
    return '';
  }

  const rows = docs
    .map((item: any, index: number) => {
      const meta = item?.document ?? {};
      const summary = item?.summary ?? {};
      const score = Number(meta?.total_similarity_score ?? NaN);
      const scoreClass = classForScore(score);
      const rowStyle = rowStyleForScore(score);
      return `
        <tr class="${scoreClass}" style="${rowStyle}">
          <td>${index + 1}</td>
          <td>${escapeHtml(meta?.title ?? '-')}</td>
          <td>${renderRiskPill(meta?.risk_level)}</td>
          <td>${formatPercent(score)}</td>
          <td>${formatNumber(summary?.segments_above_threshold)}</td>
        </tr>
      `;
    })
    .join('');

  return `
    <section class="section">
      <h2>高风险投标文件</h2>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>文档</th>
              <th>风险等级</th>
              <th>总体相似度</th>
              <th>触发片段</th>
            </tr>
          </thead>
          <tbody>
            ${rows}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function buildHighlightList(highlights: any[], leftLabel: string, rightLabel: string): string {
  if (!Array.isArray(highlights) || highlights.length === 0) {
    return '';
  }

  const items = highlights.slice(0, 6).map((item: any, idx: number) => {
    const sim = formatPercent(item?.similarity);
    const left = escapeHtml(item?.text_a ?? `${leftLabel} 段落`);
    const right = escapeHtml(item?.text_b ?? `${rightLabel} 段落`);
    return `<li><strong>#${idx + 1} · ${sim}</strong> — ${left} ↔ ${right}</li>`;
  }).join('');

  return `
    <div>
      <div class="table-title">并排关键片段</div>
      <ul class="list-stack">${items}</ul>
    </div>
  `;
}

function formatPercent(value: number | string | null | undefined, digits = 1): string {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  const numeric = typeof value === 'string' ? Number(value) : value;
  if (Number.isNaN(numeric)) {
    return '-';
  }
  const resolved = Math.abs(numeric) <= 1 ? numeric * 100 : numeric;
  return `${resolved.toFixed(digits)}%`;
}

function formatNumber(value: any): string {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  if (typeof value === 'number') {
    return new Intl.NumberFormat('zh-CN').format(value);
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value);
  }
  return new Intl.NumberFormat('zh-CN').format(numeric);
}

function formatMetricValue(key: string, value: any): string {
  if (typeof value === 'number') {
    if (/ratio|similar/i.test(key)) {
      return formatPercent(value);
    }
    return formatNumber(value);
  }
  return String(value ?? '-');
}

function formatKeyLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function renderRiskPill(level: string | undefined): string {
  if (!level) {
    return '<span class="risk-pill moderate">未知</span>';
  }
  const normalized = String(level).toLowerCase();
  const label = formatRiskLabel(normalized);
  return `<span class="risk-pill ${normalized}">${escapeHtml(label)}</span>`;
}

function formatRiskLabel(level: string): string {
  switch (level) {
    case 'low':
      return '低风险';
    case 'moderate':
      return '中等风险';
    case 'high':
      return '高风险';
    case 'critical':
      return '严重风险';
    default:
      return level;
  }
}

function classForScore(score: number): string {
  if (!Number.isFinite(score)) {
    return '';
  }
  if (score >= 0.9) {
    return 'score-critical';
  }
  if (score >= 0.85) {
    return 'score-high';
  }
  if (score >= 0.8) {
    return 'score-elevated';
  }
  return '';
}

function rowStyleForScore(score: number): string {
  if (!Number.isFinite(score)) {
    return '';
  }
  if (score >= 0.9) {
    return 'background: rgba(239, 68, 68, 0.18); color: #7f1d1d;';
  }
  if (score >= 0.85) {
    return 'background: rgba(249, 115, 22, 0.16);';
  }
  if (score >= 0.8) {
    return 'background: rgba(250, 204, 21, 0.14);';
  }
  return '';
}
