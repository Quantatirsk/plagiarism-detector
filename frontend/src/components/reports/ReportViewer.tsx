/**
 * ReportViewer - 报告查看器主组件
 * 支持文档、对比和项目三种类型的报告展示
 */
import { useState, useEffect } from 'react';
import { Tag, Typography, Segmented, Alert } from 'antd';
import {
  FileTextOutlined,
  CheckCircleOutlined,
  EyeOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { MarkdownRenderer } from '@/components/markdown';
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
  className?: string;
  // 移除 progress 和 isGenerating，这些将在父组件的右侧栏处理
}

export function ReportViewer({
  report,
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
  }, [report, activeSection]);

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

  // 进度相关逻辑已移至父组件的右侧栏

  // 渲染报告内容
  const renderReportContent = () => {
    if (!report) return null;

    const typeInfo = getReportTypeInfo(report.type);
    const TypeIcon = typeInfo.icon;

    const renderContentSection = () => {
      const content = report.content;
      const fullContent = content?.full_content || content?.generated_content || '';
      const structuredReport = content?.structured_report;
      const segmentsTableHtml = structuredReport?.summary?.segments_table_html;
      const highlightedSegments = segmentsTableHtml || '';
      const datasetWarning = structuredReport?.summary?.large_dataset_warning
        || structuredReport?.large_dataset_warning;

      return (
        <>
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

          {/* 报告内容 - 保留概览摘要的蓝色框样式 */}
          <div
            id="report-content"
            style={{
              marginBottom: designSystem.spacing[4],
              padding: designSystem.spacing[3],
              borderRadius: designSystem.borderRadius.lg,
              border: `1px solid ${designSystem.colors.primary[300]}`,
              backgroundColor: `${designSystem.colors.primary[50]}`
            }}
          >
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
              分析报告
            </div>
            <div className="prose prose-sm max-w-none" style={{ fontSize: designSystem.typography.fontSize.sm }}>
              {fullContent ? (
                <MarkdownRenderer content={fullContent} />
              ) : (
                <Text type="secondary" style={{ fontSize: designSystem.typography.fontSize.sm }}>
                  报告生成中...
                </Text>
              )}
            </div>
          </div>

          {/* 高相似片段列表 */}
          {highlightedSegments && (
            <div style={{
              marginTop: designSystem.spacing[4],
              marginBottom: designSystem.spacing[3]
            }}>
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
                padding: designSystem.spacing[3],
                borderRadius: designSystem.borderRadius.lg,
                border: `1px solid ${designSystem.colors.neutral[200]}`,
                backgroundColor: designSystem.colors.neutral[50],
                fontSize: designSystem.typography.fontSize.sm  // 12px - 表格内容使用 sm 字号
              }}>
                <div
                  className="markdown-renderer"
                  dangerouslySetInnerHTML={{ __html: highlightedSegments }}
                />
              </div>
            </div>
          )}
        </>
      );
    };

    const renderDataSection = () => (
      <div style={{
        flex: 1,
        padding: designSystem.spacing[2],
        borderRadius: designSystem.borderRadius.lg,
        backgroundColor: designSystem.colors.neutral[50],
        overflow: 'auto',
        minHeight: 0,
      }}>
        <pre style={{
          fontSize: designSystem.typography.fontSize.xs,
          margin: 0,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word'
        }}>
          {JSON.stringify(report.data, null, 2)}
        </pre>
      </div>
    );

    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
        {/* 单行工具栏：标题 + 信息 + 切换 */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: designSystem.spacing[3],
          paddingBottom: designSystem.spacing[2],
          borderBottom: `1px solid ${designSystem.colors.neutral[200]}`,
          flexShrink: 0,
        }}>
          {/* 左侧：图标 + 标题 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2], flex: 1, minWidth: 0 }}>
            <TypeIcon style={{ fontSize: 16, color: typeInfo.color, flexShrink: 0 }} />
            <Text strong style={{
              fontSize: designSystem.typography.fontSize.base,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}>
              {report.title}
            </Text>
          </div>

          {/* 右侧：标签 + 时间 + 切换 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2], flexShrink: 0 }}>
            <Tag color="blue" style={{ fontSize: 10, margin: 0 }}>
              {typeInfo.label}
            </Tag>
            <Text type="secondary" style={{ fontSize: designSystem.typography.fontSize.xs }}>
              {new Date(report.generated_at).toLocaleString('zh-CN')}
            </Text>
            <Segmented
              value={activeSection}
              onChange={(value) => setActiveSection(value as string)}
              options={[
                { label: '详细', value: 'content' },
                { label: '数据', value: 'data' },
              ]}
              size="small"
            />
          </div>
        </div>

        {/* 内容区域 */}
        <div style={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          padding: designSystem.spacing[3],
          overflow: 'auto'
        }}>
          {activeSection === 'content' ? renderContentSection() : renderDataSection()}
        </div>
      </div>
    );
  };

  return (
    <div className={className} style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {report ? (
        renderReportContent()
      ) : (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          flex: 1,
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
      )}
    </div>
  );
}
