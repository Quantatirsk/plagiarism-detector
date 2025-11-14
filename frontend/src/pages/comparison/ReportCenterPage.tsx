/**
 * ReportCenterPage - 报告中心页
 *
 * 功能：
 * - 生成文档抄袭检测报告 / 文档对比分析报告
 * - 查看报告生成历史
 * - 实时查看报告生成进度
 */

import { useMemo } from 'react';
import { App, Card, Statistic, Button, Alert, Progress, Space, Typography, Empty } from 'antd';
import {
  FileTextOutlined,
  DownloadOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  FolderOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { ReportGenerator, type ReportConfig } from '@/components/reports/ReportGenerator';
import { ReportViewer } from '@/components/reports/ReportViewer';
import { exportReportToWord } from '@/components/reports/reportExportUtils';
import { useDocuments, useProjects } from '@/hooks/useData';
import { designSystem } from '@/styles/DesignSystem';
import { getApiUrl } from '@/config';
import { useReportStore, type ReportData, type ReportProgress } from '@/store/reportStore';
import { useComparisonStore } from '@/store/comparisonStore';

const { Text } = Typography;

// 报告生成阶段消息
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

// 右侧栏组件（供父组件使用）
export function ReportCenterSidebar() {
  const { currentReport, reportProgress, isGenerating } = useReportStore();
  const { selectedProjectId } = useComparisonStore();
  const { data: projects } = useProjects();
  const { data: documents } = useDocuments(selectedProjectId ? { projectId: selectedProjectId } : undefined);

  const currentProject = projects?.find(p => p.id === selectedProjectId);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[2] }}>
      {/* 当前项目 */}
      {currentProject ? (
        <Card size="small" title="当前项目" style={{ borderRadius: designSystem.borderRadius.lg }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2] }}>
            <FolderOutlined style={{ fontSize: 20, color: designSystem.colors.primary[500] }} />
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <div style={{
                fontWeight: designSystem.typography.fontWeight.medium,
                fontSize: designSystem.typography.fontSize.sm,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}>
                {currentProject.name}
              </div>
              {currentProject.description && (
                <div style={{
                  fontSize: designSystem.typography.fontSize.xs,
                  color: designSystem.semantic.text.secondary,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>
                  {currentProject.description}
                </div>
              )}
            </div>
          </div>
        </Card>
      ) : (
        <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
          <Alert
            message="未选择项目"
            description="请先在项目管理中选择一个项目"
            type="warning"
            showIcon
            icon={<WarningOutlined />}
            style={{ fontSize: designSystem.typography.fontSize.xs }}
          />
        </Card>
      )}

      {/* 统计信息 */}
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <Statistic
          title="项目文档"
          value={documents?.length ?? 0}
          prefix={<FileTextOutlined />}
        />
      </Card>

      {/* 生成进度 */}
      {(isGenerating || reportProgress) && (
        <Card size="small" title="生成进度" style={{ borderRadius: designSystem.borderRadius.lg }}>
          {reportProgress?.error || reportProgress?.stage === 'error' ? (
            <Alert
              message="生成失败"
              description={reportProgress?.error || '未知错误'}
              type="error"
              showIcon
              icon={<ExclamationCircleOutlined />}
              style={{ fontSize: designSystem.typography.fontSize.xs }}
            />
          ) : (
            <Space direction="vertical" style={{ width: '100%' }} size={parseInt(designSystem.spacing[2])}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: designSystem.spacing[1],
                fontSize: designSystem.typography.fontSize.xs
              }}>
                <SyncOutlined spin={isGenerating} />
                <Text type="secondary">
                  {getProgressStageText(reportProgress?.stage || 'initializing')}
                </Text>
              </div>
              <Progress
                percent={reportProgress ? Math.round(reportProgress.progress * 100) : 0}
                size="small"
                strokeColor={designSystem.colors.primary[500]}
              />
              {reportProgress?.estimated_remaining && (
                <Text type="secondary" style={{
                  fontSize: designSystem.typography.fontSize.xs,
                  display: 'flex',
                  alignItems: 'center',
                  gap: designSystem.spacing[1]
                }}>
                  <ClockCircleOutlined />
                  剩余 {reportProgress.estimated_remaining}s
                </Text>
              )}
            </Space>
          )}
        </Card>
      )}

      {/* 导出按钮 */}
      {currentReport && (
        <Button
          icon={<DownloadOutlined />}
          onClick={() => currentReport && exportReportToWord(currentReport)}
          disabled={isGenerating}
          block
        >
          导出报告
        </Button>
      )}
    </div>
  );
}

export default function ReportCenterPage() {
  // ==================== 状态管理 ====================
  const { message } = App.useApp();
  const { currentReport, isGenerating, setCurrentReport, setReportProgress, setIsGenerating } = useReportStore();
  const { selectedProjectId } = useComparisonStore();
  const { data: projects } = useProjects();

  // 根据选中的项目加载文档
  const { data: documents } = useDocuments(selectedProjectId ? { projectId: selectedProjectId } : undefined);
  const { data: models } = { data: [
    { id: 'google/gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash' },
  ]};

  const currentProject = projects?.find(p => p.id === selectedProjectId);

  // ==================== 数据转换 ====================
  const availableDocuments = useMemo(() => {
    return (documents || []).map(doc => ({
      id: String(doc.id),
      title: doc.title || doc.filename || `文档 #${doc.id}`
    }));
  }, [documents]);

  const availableModels = useMemo(() => {
    return (models || []).map(model => ({
      id: model.id,
      name: model.name
    }));
  }, [models]);

  // ==================== 操作函数 ====================
  const handleGenerateReport = async (config: ReportConfig) => {
    const taskId = `task_${Date.now()}`;

    const normalizeReport = (raw: any): ReportData => ({
      id: raw.id,
      type: raw.type,
      title: raw.title,
      summary: raw.summary,
      content: raw.content ?? {},
      data: raw.data,
      generated_at: typeof raw.generated_at === 'string'
        ? raw.generated_at
        : raw.generated_at
          ? new Date(raw.generated_at).toISOString()
          : new Date().toISOString(),
      language: raw.language || 'zh',
      export_formats: raw.export_formats ?? ['html', 'pdf', 'json']
    });

    const updateProgress = (partial: Partial<ReportProgress>) => {
      const nextStage = partial.stage ?? 'initializing';
      const nextMessage = partial.message ?? stageMessages[nextStage] ?? stageMessages.initializing;

      setReportProgress({
        task_id: taskId,
        progress: partial.progress ?? 0,
        stage: nextStage,
        message: nextMessage,
        ...(partial.error !== undefined ? { error: partial.error } : {}),
        ...(partial.estimated_remaining !== undefined ? { estimated_remaining: partial.estimated_remaining } : {})
      });
    };

    setIsGenerating(true);
    setCurrentReport(null);
    updateProgress({
      progress: 0,
      stage: 'initializing',
      message: stageMessages.initializing
    });

    try {
      // 根据报告类型调用对应的API端点
      const endpoint = config.type === 'comparison'
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
        let streamedContent = ''; // 累积流式内容

        const processLine = (rawLine: string) => {
          if (isDone) return;

          const line = rawLine.trim();
          if (!line.startsWith('data:')) return;

          const payload = line.slice(5).trim();
          if (!payload) return;

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
                // 累积内容并实时更新报告
                const chunk = event.chunk || '';
                streamedContent += chunk;

                // 实时更新报告显示
                setCurrentReport(prev => {
                  const baseReport = prev || {
                    id: taskId,
                    type: config.type,
                    title: config.type === 'comparison'
                      ? '文档对比分析报告'
                      : '文档抄袭检测报告',
                    summary: '',
                    content: {},
                    data: {},
                    generated_at: new Date().toISOString(),
                    language: 'zh',
                    export_formats: ['docx']
                  };

                  return {
                    ...baseReport,
                    content: {
                      ...baseReport.content,
                      full_content: streamedContent,
                      generated_content: streamedContent
                    }
                  };
                });

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
                const errMsg = event.message || '报告生成失败';
                updateProgress({
                  progress: 0,
                  stage: 'error',
                  message: stageMessages.error,
                  error: errMsg
                });
                streamError = new Error(errMsg);
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
            if (isDone) return;
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
          if (isDone) break;
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
        message.success('报告生成成功');
      } else {
        const data = await response.json();
        const report = normalizeReport(data);
        setCurrentReport(report);
        updateProgress({
          progress: 1,
          stage: 'completed',
          message: stageMessages.completed
        });
        message.success('报告生成成功');
      }

    } catch (error) {
      console.error('报告生成失败:', error);
      const errMsg = error instanceof Error ? error.message : '未知错误';
      updateProgress({
        progress: 0,
        stage: 'error',
        message: stageMessages.error,
        error: errMsg
      });
      message.error(errMsg);
    } finally {
      setIsGenerating(false);
    }
  };

  // ==================== 渲染 ====================

  // 如果没有选中项目，显示提示
  if (!selectedProjectId || !currentProject) {
    return (
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: designSystem.spacing[6],
        }}
      >
        <Empty
          image={<FolderOutlined style={{ fontSize: 64, color: designSystem.colors.neutral[400] }} />}
          description={
            <div>
              <div style={{
                fontSize: designSystem.typography.fontSize.base,
                fontWeight: designSystem.typography.fontWeight.medium,
                marginBottom: designSystem.spacing[2],
              }}>
                未选择项目
              </div>
              <div style={{
                fontSize: designSystem.typography.fontSize.sm,
                color: designSystem.semantic.text.secondary,
              }}>
                报告生成需要基于项目，请先在"项目管理"中选择一个项目
              </div>
            </div>
          }
        />
      </div>
    );
  }

  // 如果项目没有文档，显示提示
  if (!documents || documents.length === 0) {
    return (
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: designSystem.spacing[6],
        }}
      >
        <Empty
          image={<FileTextOutlined style={{ fontSize: 64, color: designSystem.colors.neutral[400] }} />}
          description={
            <div>
              <div style={{
                fontSize: designSystem.typography.fontSize.base,
                fontWeight: designSystem.typography.fontWeight.medium,
                marginBottom: designSystem.spacing[2],
              }}>
                项目暂无文档
              </div>
              <div style={{
                fontSize: designSystem.typography.fontSize.sm,
                color: designSystem.semantic.text.secondary,
              }}>
                当前项目"{currentProject.name}"中还没有文档，请先上传文档
              </div>
            </div>
          }
        />
      </div>
    );
  }

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[1],
        minHeight: 0,
      }}
    >
      {/* 工具栏：项目信息 + 生成器 */}
      <Card
        size="small"
        style={{
          flexShrink: 0,
          borderRadius: designSystem.borderRadius.lg,
        }}
        styles={{
          body: {
            padding: `${designSystem.spacing[1]} ${designSystem.spacing[2]}`,
            display: 'flex',
            alignItems: 'center',
            gap: designSystem.spacing[2],
          }
        }}
      >
        {/* 项目标签 */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: designSystem.spacing[1],
          padding: `${designSystem.spacing[1]} ${designSystem.spacing[2]}`,
          backgroundColor: designSystem.colors.primary[50],
          borderRadius: designSystem.borderRadius.md,
          border: `1px solid ${designSystem.colors.primary[200]}`,
        }}>
          <FolderOutlined style={{ fontSize: 12, color: designSystem.colors.primary[600] }} />
          <span style={{
            fontSize: designSystem.typography.fontSize.xs,
            color: designSystem.colors.primary[700],
            fontWeight: designSystem.typography.fontWeight.medium,
          }}>
            {currentProject.name}
          </span>
        </div>

        {/* 分隔线 */}
        <div style={{
          width: 1,
          height: 20,
          backgroundColor: designSystem.colors.neutral[200],
        }} />

        {/* 报告生成器 */}
        <div style={{ flex: 1 }}>
          <ReportGenerator
            availableDocuments={availableDocuments}
            availableModels={availableModels}
            onGenerate={handleGenerateReport}
            isGenerating={isGenerating}
          />
        </div>
      </Card>

      {/* 报告查看器 */}
      <Card
        size="small"
        style={{
          flex: 1,
          borderRadius: designSystem.borderRadius.lg,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
        }}
        styles={{
          body: {
            flex: 1,
            overflow: 'auto',
            padding: designSystem.spacing[2],
            minHeight: 0,
          }
        }}
      >
        <ReportViewer report={currentReport ?? undefined} />
      </Card>
    </div>
  );
}
