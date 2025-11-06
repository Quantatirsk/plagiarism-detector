/**
 * ReportsPage - 报告生成和查看页面
 * 集成报告生成器和查看器，支持文档、对比和项目三种报告类型
 */
import { useEffect, useState } from 'react';
import { PageShell, PageHeader, PageContent } from '@/components/layout/Page';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ReportGenerator, type ReportConfig } from '@/components/reports/ReportGenerator';
import { ReportViewer, type ReportData, type ReportProgress, type ReportContent } from '@/components/reports/ReportViewer';
import { FileText, ArrowLeft, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getApiUrl } from '@/config';
import { plagiarismApi, type ProjectSummary, type DocumentSummary, type CompareJobSummary } from '@/api/plagiarismApi';

interface ReportsPageProps {
  mode: 'project' | 'job' | 'document';
  project?: ProjectSummary;
  job?: CompareJobSummary;
  document?: DocumentSummary;
  availableDocuments?: Array<{ id: string; title: string }>;
  availableProjects?: Array<{ id: string; name: string }>;
  onBack: () => void;
  className?: string;
}

export function ReportsPage({
  mode,
  project,
  job,
  document: documentProp,
  availableDocuments = [],
  availableProjects = [],
  onBack,
  className
}: ReportsPageProps) {
  const [view, setView] = useState<'generator' | 'viewer'>('generator');
  const [currentReport, setCurrentReport] = useState<ReportData | null>(null);
  const [reportProgress, setReportProgress] = useState<ReportProgress | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [modelOptions, setModelOptions] = useState<Array<{ id: string; name: string }>>([]);

  useEffect(() => {
    let cancelled = false;

    const fetchModels = async () => {
      try {
        const models = await plagiarismApi.listLLMModels();
        if (cancelled) return;

        const formatted = models.map(model => ({
          id: model.id,
          name: model.root || model.id
        }));

        setModelOptions(formatted);
      } catch (error) {
        if (cancelled) return;

        console.error('加载模型列表失败:', error);
        setModelOptions([
          { id: 'google/gemini-2.5-flash-lite', name: 'google/gemini-2.5-flash-lite' }
        ]);
      }
    };

    fetchModels();

    return () => {
      cancelled = true;
    };
  }, []);

  // 获取页面标题和描述
  const getPageInfo = () => {
    switch (mode) {
      case 'project':
        return {
          title: `${project?.name || '项目'} - 生成报告`,
          description: '为整个项目生成学术诚信分析报告',
          icon: Settings,
          badge: '项目报告'
        };
      case 'job':
        return {
          title: `${job?.name || '任务'} - 生成报告`,
          description: '为对比任务生成详细分析报告',
          icon: FileText,
          badge: '对比报告'
        };
      case 'document':
        return {
          title: `${documentProp?.title || '文档'} - 生成报告`,
          description: '为单个文档生成抄袭检测报告',
          icon: FileText,
          badge: '文档报告'
        };
    }
  };

  const pageInfo = getPageInfo();
  const Icon = pageInfo.icon;

  const stageMessages: Record<string, string> = {
    initializing: '正在初始化报告生成...',
    data_collection: '正在收集分析数据...',
    template_preparation: '准备报告模板...',
    llm_generation: '正在生成报告内容...',
    content_processing: '正在处理报告内容...',
    report_finalization: '正在完成报告生成...',
    completed: '报告生成完成',
    error: '报告生成失败'
  };

  // 处理报告生成
  const handleGenerate = async (config: ReportConfig) => {
    const taskId = `task_${Date.now()}`;

    const normalizeReport = (raw: unknown): ReportData => {
      // Type narrowing for unknown input
      if (!raw || typeof raw !== 'object') {
        throw new Error('Invalid report data: not an object');
      }

      const data = raw as Record<string, unknown>;

      return {
        id: typeof data.id === 'string' ? data.id : '',
        type: (data.type === 'document' || data.type === 'comparison' || data.type === 'project')
          ? data.type
          : 'document',
        title: typeof data.title === 'string' ? data.title : '',
        summary: typeof data.summary === 'string' ? data.summary : '',
        content: (data.content && typeof data.content === 'object') ? data.content as ReportContent : {},
        data: data.data,
        generated_at: typeof data.generated_at === 'string'
          ? data.generated_at
          : data.generated_at
            ? new Date(data.generated_at as string | number).toISOString()
            : new Date().toISOString(),
        language: typeof data.language === 'string' ? data.language : 'zh',
        export_formats: Array.isArray(data.export_formats)
          ? (data.export_formats as string[])
          : ['html', 'pdf', 'json']
      };
    };

    const updateProgress = (partial: Partial<ReportProgress>) => {
      setReportProgress(prev => {
        const nextStage = partial.stage ?? prev?.stage ?? 'initializing';
        const nextMessage = partial.message
          ?? prev?.message
          ?? stageMessages[nextStage] 
          ?? stageMessages.initializing;

        return {
          task_id: taskId,
          progress: partial.progress ?? prev?.progress ?? 0,
          stage: nextStage,
          message: nextMessage,
          ...(partial.error !== undefined || prev?.error !== undefined
            ? { error: partial.error ?? prev?.error }
            : {}),
          ...(partial.estimated_remaining !== undefined || prev?.estimated_remaining !== undefined
            ? { estimated_remaining: partial.estimated_remaining ?? prev?.estimated_remaining }
            : {})
        };
      });
    };

    setIsGenerating(true);
    setView('viewer');
    setCurrentReport(null);
    updateProgress({
      progress: 0,
      stage: 'initializing',
      message: stageMessages.initializing
    });

    try {
      // 根据报告类型调用对应的API端点
      const endpoint = config.type === 'project'
        ? 'api/v1/reports/project'
        : config.type === 'comparison'
        ? 'api/v1/reports/comparison'
        : 'api/v1/reports/document';

      const response = await fetch(getApiUrl(endpoint), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.stream_response ? { Accept: 'text/event-stream' } : {})
        },
        body: JSON.stringify(config)
      });

      if (!response.ok) {
        throw new Error(`报告生成失败: ${response.statusText}`);
      }

      if (config.stream_response) {
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('无法获取响应流');
        }

        const decoder = new TextDecoder();
        let buffer = '';
        let isDone = false;
        let finalReport: ReportData | null = null;
        let streamError: Error | null = null;

        const processLine = (rawLine: string) => {
          if (isDone) {
            return;
          }

          const line = rawLine.trim();
          if (!line.startsWith('data:')) {
            return;
          }

          const payload = line.slice(5).trim();
          if (!payload) {
            return;
          }

          if (payload === '[DONE]') {
            isDone = true;
            return;
          }

          try {
            const event = JSON.parse(payload);
            switch (event.type) {
              case 'progress': {
                const stage = event.stage ?? 'llm_generation';
                updateProgress({
                  progress: typeof event.progress === 'number' ? event.progress : undefined,
                  stage,
                  message: stageMessages[stage] || stageMessages.llm_generation
                });
                break;
              }
              case 'content': {
                updateProgress({
                  stage: 'llm_generation',
                  message: stageMessages.llm_generation
                });
                break;
              }
              case 'completed': {
                finalReport = normalizeReport(event.report);
                updateProgress({
                  progress: 1,
                  stage: 'completed',
                  message: stageMessages.completed
                });
                break;
              }
              case 'error': {
                const message = event.message || '报告生成失败';
                updateProgress({
                  progress: 0,
                  stage: 'error',
                  message: stageMessages.error,
                  error: message
                });
                streamError = new Error(message);
                isDone = true;
                break;
              }
              default:
                break;
            }
          } catch (err) {
            console.warn('解析报告生成流事件失败:', err);
          }
        };

        const flushBuffer = () => {
          let newlineIndex = buffer.indexOf('\n');
          while (newlineIndex !== -1) {
            const line = buffer.slice(0, newlineIndex);
            buffer = buffer.slice(newlineIndex + 1);
            processLine(line);
            if (isDone) {
              return;
            }
            newlineIndex = buffer.indexOf('\n');
          }
        };

        while (!isDone) {
          const { done, value } = await reader.read();
          if (done) {
            buffer += decoder.decode();
            flushBuffer();
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          flushBuffer();
          if (isDone) {
            break;
          }
        }

        if (buffer.trim()) {
          processLine(buffer);
        }

        if (streamError) {
          throw streamError;
        }

        if (!finalReport) {
          throw new Error('报告生成未完成: 未收到完成事件');
        }

        setCurrentReport(finalReport);
      } else {
        const data = await response.json();
        const report = normalizeReport(data);
        setCurrentReport(report);
        updateProgress({
          progress: 1,
          stage: 'completed',
          message: stageMessages.completed
        });
      }

    } catch (error) {
      console.error('报告生成失败:', error);
      const message = error instanceof Error ? error.message : '未知错误';
      updateProgress({
        progress: 0,
        stage: 'error',
        message: stageMessages.error,
        error: message
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const headerTitle = (
    <div className="flex items-center gap-3">
      <Button variant="ghost" size="sm" onClick={onBack} className="px-2 h-7">
        <ArrowLeft className="h-3.5 w-3.5 mr-1.5" />
        返回
      </Button>
      <div className="flex items-center gap-2.5">
        <div className="p-1.5 rounded-lg bg-primary/10">
          <Icon className="h-5 w-5 text-primary" />
        </div>
        <div className="flex items-center gap-2 min-w-0">
          <span className="truncate text-sm font-semibold sm:text-base">{pageInfo.title}</span>
          <Badge variant="secondary" className="text-[10px] h-5">{pageInfo.badge}</Badge>
        </div>
      </div>
    </div>
  );

  const headerActions = currentReport ? (
    <div className="flex items-center gap-1.5">
      <Button
        variant={view === 'generator' ? 'default' : 'outline'}
        size="sm"
        onClick={() => setView('generator')}
        className="h-7 text-xs"
      >
        <Settings className="h-3.5 w-3.5 mr-1.5" />
        配置
      </Button>
      <Button
        variant={view === 'viewer' ? 'default' : 'outline'}
        size="sm"
        onClick={() => setView('viewer')}
        className="h-7 text-xs"
      >
        <FileText className="h-3.5 w-3.5 mr-1.5" />
        查看
      </Button>
    </div>
  ) : null;

  return (
    <PageShell className={cn("space-y-4", className)}>
      <PageHeader title={headerTitle} subtitle={pageInfo.description} actions={headerActions} />

      <PageContent>
        {view === 'generator' && (
          <ReportGenerator
            onGenerate={handleGenerate}
            onCancel={() => setView('viewer')}
            isGenerating={isGenerating}
            availableDocuments={availableDocuments}
            availableProjects={availableProjects}
            availableModels={modelOptions}
          />
        )}

        {view === 'viewer' && (
          <ReportViewer
            report={currentReport || undefined}
            progress={reportProgress || undefined}
            isGenerating={isGenerating}
          />
        )}
      </PageContent>
    </PageShell>
  );
}
