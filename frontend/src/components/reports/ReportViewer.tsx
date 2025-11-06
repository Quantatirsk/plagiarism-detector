/**
 * ReportViewer - 报告查看器主组件
 * 支持文档、对比和项目三种类型的报告展示
 */
import { useState, useEffect } from 'react';
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
import { generateProfessionalWordDocument, type WordReportData } from './ProfessionalWordExport';

// 报告内容结构化类型
export interface ReportContent {
  full_content?: string;
  generated_content?: string;
  structured_report?: {
    summary?: {
      segments_table_html?: string;
      large_dataset_warning?: string;
    };
    large_dataset_warning?: string;
  };
  // 保留扩展性
  [key: string]: unknown;
}

// 报告相关类型定义
export interface ReportData {
  id: string;
  type: 'document' | 'comparison' | 'project';
  title: string;
  summary: string;
  content: ReportContent;
  data: unknown;
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
      <Card className="mb-4">
        <CardHeader className="pb-3 pt-4">
          <CardTitle className="flex items-center gap-2 text-sm">
            {isError ? (
              <AlertCircle className="h-4 w-4 text-red-500" />
            ) : (
              <RefreshCw className={cn("h-4 w-4", isGenerating && "animate-spin")} />
            )}
            报告生成进度
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4">
          {isError ? (
            <Alert variant="destructive" className="py-2">
              <AlertDescription className="text-xs">
                生成失败: {progress?.error || '未知错误'}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{getProgressStageText(progress?.stage || 'initializing')}</span>
                <span className="font-mono font-semibold">{progressPercent}%</span>
              </div>
              <Progress value={progressPercent} className="w-full h-1.5" />
              {progress?.message && (
                <p className="text-xs text-muted-foreground">{progress.message}</p>
              )}
              {progress?.estimated_remaining && (
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" />
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
      <Card className="mb-4">
        <CardHeader className="py-4">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3">
              <div className={cn("p-2 rounded-lg", typeInfo.color)}>
                <TypeIcon className="h-5 w-5 text-white" />
              </div>
              <div>
                <CardTitle className="text-lg mb-1.5">{report.title}</CardTitle>
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="secondary" className="text-[10px] h-5">{typeInfo.label}</Badge>
                  <Badge variant="outline" className="text-[10px] h-5">{report.language === 'zh' ? '中文' : 'English'}</Badge>
                  <span className="text-[11px]">{new Date(report.generated_at).toLocaleString('zh-CN')}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs"
                onClick={() => handleExportWord(report)}
                disabled={isGenerating}
              >
                <Download className="h-3.5 w-3.5 mr-1" />
                导出报告
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
      <div className="flex gap-1 mb-4 p-1 bg-muted/50 rounded-lg">
        {sections.map((section) => {
          const SectionIcon = section.icon;
          const isActive = activeSection === section.id;

          return (
            <Button
              key={section.id}
              variant={isActive ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveSection(section.id)}
              className="flex-1 h-7 text-xs"
            >
              <SectionIcon className="h-3.5 w-3.5 mr-1.5" />
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
      const highlightedSegments = segmentsTableHtml || '';
      const datasetWarning = structuredReport?.summary?.large_dataset_warning
        || structuredReport?.large_dataset_warning;

      return (
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">详细报告内容</CardTitle>
          </CardHeader>
          <CardContent className="pb-4">
            {report.summary && (
              <div className="mb-4 rounded-lg border border-primary/30 bg-primary/5 p-3">
                <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.24em] text-primary">
                  <CheckCircle className="h-3.5 w-3.5" />
                  概览摘要
                </div>
                <MarkdownRenderer content={report.summary} className="prose prose-sm text-muted-foreground [&>*]:text-xs [&>*]:leading-relaxed" />
              </div>
            )}
            {datasetWarning && (
              <div className="mb-3 rounded-lg border border-orange-300 bg-orange-100/70 px-2.5 py-2 text-xs text-orange-900">
                {datasetWarning}
              </div>
            )}
            <div
              id="report-content"
              className="prose prose-sm max-w-none [&>*]:text-sm [&>*]:leading-relaxed"
            >
              {fullContent ? (
                <MarkdownRenderer content={fullContent} />
              ) : (
                <p className="text-muted-foreground text-xs">暂无详细内容</p>
              )}
            </div>

            {highlightedSegments && (
              <div className="mt-4">
                <h3 className="mb-1.5 text-xs font-semibold text-foreground uppercase tracking-wide">高相似片段列表 (≥ 0.8)</h3>
                <div className="rounded-lg border border-border bg-muted/40 p-2 text-xs">
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
        <CardHeader className="py-3">
          <CardTitle className="text-base">数据详情</CardTitle>
          <p className="text-xs text-muted-foreground">报告生成所使用的原始数据</p>
        </CardHeader>
        <CardContent className="pb-4">
          <div className="bg-muted p-3 rounded-lg overflow-auto max-h-96">
            <pre className="text-xs">
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
    <div className={cn("space-y-4", className)}>
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
          <CardContent className="flex flex-col items-center justify-center py-10">
            <FileText className="h-10 w-10 text-muted-foreground mb-3" />
            <h3 className="text-base font-medium mb-1.5">暂无报告</h3>
            <p className="text-muted-foreground text-center text-xs">
              请选择文档或项目生成报告
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

async function handleExportWord(report: ReportData) {
  const { title, content, generated_at, language } = report;
  const structured = content?.structured_report;
  const fullContent = content?.full_content || '';
  const datasetWarning = structured?.summary?.large_dataset_warning || structured?.large_dataset_warning;
  const safeTitle = title.replace(/[^\w\u4e00-\u9fa5]+/g, '-');

  // 直接使用原始 HTML，Word 导出模块会自行解析
  const wordData: WordReportData = {
    title,
    summary: report.summary || '暂无摘要',
    fullContent: fullContent || '暂无详细内容',
    htmlSegments: structured?.summary?.segments_table_html,
    structuredSectionsHtml: undefined, // Word 导出暂不使用此字段
    datasetWarning,
    generated_at,
    language
  };

  try {
    const blob = await generateProfessionalWordDocument(wordData);
    saveAs(blob, `${safeTitle || 'report'}.docx`);
  } catch (error) {
    console.error('导出 Word 报告失败', error);
  }
}
