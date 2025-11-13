/**
 * ReportViewer - 报告查看器主组件
 * 支持文档、对比和项目三种类型的报告展示
 */
import { useState, useEffect } from 'react';
import { Card, Button, Tag, Progress, Alert, Space, Typography, Segmented } from 'antd';
import {
  FileTextOutlined,
  DownloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  EyeOutlined,
  SettingOutlined,
  SyncOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { MarkdownRenderer } from '@/components/markdown';
import { saveAs } from 'file-saver';
import { generateProfessionalWordDocument, type WordReportData } from './ProfessionalWordExport';
import { designSystem } from '@/styles/DesignSystem';

const { Title, Text } = Typography;

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
      document: {
        label: '文档报告',
        icon: FileTextOutlined,
        color: designSystem.colors.primary[500]
      },
      comparison: {
        label: '对比报告',
        icon: EyeOutlined,
        color: designSystem.colors.success[500]
      },
      project: {
        label: '项目报告',
        icon: SettingOutlined,
        color: designSystem.colors.warning[500]
      }
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
      <Card style={{ marginBottom: designSystem.spacing[4] }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2] }}>
            {isError ? (
              <CloseCircleOutlined style={{ fontSize: 16, color: designSystem.colors.error[500] }} />
            ) : (
              <SyncOutlined spin={isGenerating} style={{ fontSize: 16 }} />
            )}
            <Text strong style={{ fontSize: designSystem.typography.fontSize.sm }}>
              报告生成进度
            </Text>
          </div>

          {isError ? (
            <Alert
              message={`生成失败: ${progress?.error || '未知错误'}`}
              type="error"
              showIcon
              icon={<ExclamationCircleOutlined />}
              style={{ fontSize: designSystem.typography.fontSize.xs }}
            />
          ) : (
            <Space direction="vertical" style={{ width: '100%' }} size={parseInt(designSystem.spacing[3])}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: designSystem.typography.fontSize.xs
              }}>
                <Text type="secondary">
                  {getProgressStageText(progress?.stage || 'initializing')}
                </Text>
                <Text strong style={{ fontFamily: 'monospace' }}>
                  {progressPercent}%
                </Text>
              </div>
              <Progress
                percent={progressPercent}
                showInfo={false}
                strokeColor={designSystem.colors.primary[500]}
              />
              {progress?.message && (
                <Text type="secondary" style={{ fontSize: designSystem.typography.fontSize.xs }}>
                  {progress.message}
                </Text>
              )}
              {progress?.estimated_remaining && (
                <Text type="secondary" style={{
                  fontSize: designSystem.typography.fontSize.xs,
                  display: 'flex',
                  alignItems: 'center',
                  gap: designSystem.spacing[1]
                }}>
                  <ClockCircleOutlined />
                  预计剩余 {progress.estimated_remaining} 秒
                </Text>
              )}
            </Space>
          )}
        </Space>
      </Card>
    );
  };

  // 渲染报告头部
  const renderReportHeader = () => {
    if (!report) return null;

    const typeInfo = getReportTypeInfo(report.type);
    const TypeIcon = typeInfo.icon;

    return (
      <Card style={{ marginBottom: designSystem.spacing[4] }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start'
        }}>
          <div style={{ display: 'flex', gap: designSystem.spacing[3] }}>
            <div style={{
              padding: designSystem.spacing[2],
              borderRadius: designSystem.borderRadius.lg,
              backgroundColor: typeInfo.color,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <TypeIcon style={{ fontSize: 20, color: '#fff' }} />
            </div>
            <div>
              <Title level={4} style={{ marginBottom: designSystem.spacing[1] }}>
                {report.title}
              </Title>
              <Space size="small" wrap>
                <Tag color="blue" style={{ fontSize: 10 }}>
                  {typeInfo.label}
                </Tag>
                <Tag style={{ fontSize: 10 }}>
                  {report.language === 'zh' ? '中文' : 'English'}
                </Tag>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {new Date(report.generated_at).toLocaleString('zh-CN')}
                </Text>
              </Space>
            </div>
          </div>
          <Button
            icon={<DownloadOutlined />}
            onClick={() => handleExportWord(report)}
            disabled={isGenerating}
            size="small"
          >
            导出报告
          </Button>
        </div>
      </Card>
    );
  };

  // 渲染报告导航
  const renderNavigation = () => {
    if (!report) return null;

    return (
      <div style={{ marginBottom: designSystem.spacing[4] }}>
        <Segmented
          value={activeSection}
          onChange={(value) => setActiveSection(value as string)}
          options={[
            {
              value: 'content',
              label: (
                <Space>
                  <EyeOutlined />
                  <span>详细内容</span>
                </Space>
              )
            },
            {
              value: 'data',
              label: (
                <Space>
                  <SettingOutlined />
                  <span>原始数据</span>
                </Space>
              )
            }
          ]}
          block
        />
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
        <Card
          title={
            <Text strong style={{ fontSize: designSystem.typography.fontSize.base }}>
              详细报告内容
            </Text>
          }
        >
          {report.summary && (
            <div style={{
              marginBottom: designSystem.spacing[4],
              padding: designSystem.spacing[3],
              borderRadius: designSystem.borderRadius.lg,
              border: `1px solid ${designSystem.colors.primary[300]}`,
              backgroundColor: `${designSystem.colors.primary[50]}`
            }}>
              <div style={{
                marginBottom: designSystem.spacing[2],
                display: 'flex',
                alignItems: 'center',
                gap: designSystem.spacing[2],
                fontSize: 10,
                fontWeight: designSystem.typography.fontWeight.semibold,
                textTransform: 'uppercase',
                letterSpacing: '0.24em',
                color: designSystem.colors.primary[600]
              }}>
                <CheckCircleOutlined style={{ fontSize: 14 }} />
                概览摘要
              </div>
              <div style={{ fontSize: designSystem.typography.fontSize.xs }}>
                <MarkdownRenderer
                  content={report.summary}
                  className="prose prose-sm"
                />
              </div>
            </div>
          )}

          {datasetWarning && (
            <Alert
              message={datasetWarning}
              type="warning"
              showIcon
              style={{
                marginBottom: designSystem.spacing[3],
                fontSize: designSystem.typography.fontSize.xs
              }}
            />
          )}

          <div
            id="report-content"
            className="prose prose-sm max-w-none"
            style={{ fontSize: designSystem.typography.fontSize.sm }}
          >
            {fullContent ? (
              <MarkdownRenderer content={fullContent} />
            ) : (
              <Text type="secondary" style={{ fontSize: designSystem.typography.fontSize.xs }}>
                暂无详细内容
              </Text>
            )}
          </div>

          {highlightedSegments && (
            <div style={{ marginTop: designSystem.spacing[4] }}>
              <Title
                level={5}
                style={{
                  marginBottom: designSystem.spacing[2],
                  fontSize: designSystem.typography.fontSize.xs,
                  fontWeight: designSystem.typography.fontWeight.semibold,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}
              >
                高相似片段列表 (≥ 0.8)
              </Title>
              <div style={{
                padding: designSystem.spacing[2],
                borderRadius: designSystem.borderRadius.lg,
                border: `1px solid ${designSystem.colors.neutral[200]}`,
                backgroundColor: designSystem.colors.neutral[50],
                fontSize: designSystem.typography.fontSize.xs
              }}>
                <div
                  className="markdown-renderer"
                  dangerouslySetInnerHTML={{ __html: highlightedSegments }}
                />
              </div>
            </div>
          )}
        </Card>
      );
    };

    const renderData = () => (
      <Card
        title={
          <Space direction="vertical" size={0}>
            <Text strong style={{ fontSize: designSystem.typography.fontSize.base }}>
              数据详情
            </Text>
            <Text type="secondary" style={{ fontSize: designSystem.typography.fontSize.xs }}>
              报告生成所使用的原始数据
            </Text>
          </Space>
        }
      >
        <div style={{
          padding: designSystem.spacing[3],
          borderRadius: designSystem.borderRadius.lg,
          backgroundColor: designSystem.colors.neutral[50],
          overflow: 'auto',
          maxHeight: 384
        }}>
          <pre style={{
            fontSize: designSystem.typography.fontSize.xs,
            margin: 0
          }}>
            {JSON.stringify(report.data, null, 2)}
          </pre>
        </div>
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
    <div className={className} style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[4] }}>
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
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: `${designSystem.spacing[10]} 0`
          }}>
            <FileTextOutlined style={{
              fontSize: 40,
              color: designSystem.colors.neutral[400],
              marginBottom: designSystem.spacing[3]
            }} />
            <Title level={5} style={{ marginBottom: designSystem.spacing[2] }}>
              暂无报告
            </Title>
            <Text type="secondary" style={{
              fontSize: designSystem.typography.fontSize.xs,
              textAlign: 'center'
            }}>
              请选择文档或项目生成报告
            </Text>
          </div>
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
