/**
 * 报告中心页
 *
 * 功能：
 * - 生成文档/对比/项目报告
 * - 查看报告生成历史
 * - 实时查看报告生成进度
 */

import { useState, useMemo } from 'react';
import { Card, Segmented, message, Statistic, Row, Col } from 'antd';
import { FileTextOutlined, SwapOutlined, FolderOutlined } from '@ant-design/icons';
import PageLayout from '@/layout/PageLayout';
import { ReportGenerator, type ReportConfig } from '@/components/reports/ReportGenerator';
import { ReportViewer, type ReportData, type ReportProgress } from '@/components/reports/ReportViewer';
import { useProjects, useDocuments } from '@/hooks/useData';
import { designSystem } from '@/styles/DesignSystem';

type ReportType = 'document' | 'comparison' | 'project';

export default function ReportsPage() {
  // ==================== 状态管理 ====================
  const [reportType, setReportType] = useState<ReportType>('document');
  const [currentReport, setCurrentReport] = useState<ReportData | null>(null);
  const [reportProgress, setReportProgress] = useState<ReportProgress | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const { data: projects } = useProjects();
  const { data: documents } = useDocuments();
  const { data: models } = { data: [
    { id: 'google/gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash' },
  ]};

  // UI 状态
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);

  // ==================== 数据转换 ====================
  const availableDocuments = useMemo(() => {
    return (documents || []).map(doc => ({
      id: String(doc.id),
      title: doc.title || doc.filename || `文档 #${doc.id}`
    }));
  }, [documents]);

  const availableProjects = useMemo(() => {
    return (projects || []).map(proj => ({
      id: String(proj.id),
      name: proj.name || `项目 #${proj.id}`
    }));
  }, [projects]);

  const availableModels = useMemo(() => {
    return (models || []).map(model => ({
      id: model.id,
      name: model.name
    }));
  }, [models]);

  // ==================== 操作函数 ====================
  const handleGenerateReport = async (config: ReportConfig) => {
    setIsGenerating(true);
    setReportProgress({
      task_id: 'temp',
      progress: 0,
      stage: 'initializing',
      message: '开始生成报告...'
    });

    try {
      // TODO: 实现报告生成 API
      console.log('生成报告:', config);

      // 模拟延迟
      await new Promise(resolve => setTimeout(resolve, 1000));

      // 模拟报告数据
      setCurrentReport({
        id: 'temp',
        type: config.type,
        title: `${config.type === 'document' ? '文档' : config.type === 'comparison' ? '对比' : '项目'}报告`,
        summary: '报告已生成',
        content: { full_content: '报告内容' },
        data: {},
        generated_at: new Date().toISOString(),
        language: config.language,
        export_formats: ['docx']
      });

      message.success('报告生成成功');
    } catch (err) {
      console.error(err);
      message.error((err as Error).message || '报告生成失败');
    } finally {
      setIsGenerating(false);
      setReportProgress(null);
    }
  };

  // ==================== 布局区域 ====================

  // topBar 工具栏
  const topBar = (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: designSystem.spacing[1],
      padding: designSystem.spacing[1],
      width: '100%'
    }}>
      <Segmented
        value={reportType}
        onChange={(value) => setReportType(value as ReportType)}
        options={[
          {
            label: '文档报告',
            value: 'document',
            icon: <FileTextOutlined />
          },
          {
            label: '对比报告',
            value: 'comparison',
            icon: <SwapOutlined />
          },
          {
            label: '项目报告',
            value: 'project',
            icon: <FolderOutlined />
          }
        ]}
      />
    </div>
  );

  // 左侧快捷信息面板
  const leftSidebar = (
    <>
      <Card
        size="small"
        title="报告类型说明"
        style={{
          marginBottom: designSystem.spacing[1],
          borderRadius: designSystem.borderRadius.lg,
        }}
      >
        <div style={{ fontSize: designSystem.typography.fontSize.xs, color: designSystem.semantic.text.secondary }}>
          {reportType === 'document' && (
            <>
              <p>分析单个文档在数据库中的抄袭情况，包括：</p>
              <ul style={{ paddingLeft: 20, margin: `${designSystem.spacing[1]} 0` }}>
                <li>相似片段检测</li>
                <li>相似度评分</li>
                <li>改进建议</li>
              </ul>
            </>
          )}
          {reportType === 'comparison' && (
            <>
              <p>深入分析两个文档之间的相似性：</p>
              <ul style={{ paddingLeft: 20, margin: `${designSystem.spacing[1]} 0` }}>
                <li>逐段对比分析</li>
                <li>匹配可视化</li>
                <li>差异性评估</li>
              </ul>
            </>
          )}
          {reportType === 'project' && (
            <>
              <p>对项目内所有文档进行宏观分析：</p>
              <ul style={{ paddingLeft: 20, margin: `${designSystem.spacing[1]} 0` }}>
                <li>整体相似度分布</li>
                <li>异常文档识别</li>
                <li>网络关系图谱</li>
              </ul>
            </>
          )}
        </div>
      </Card>

      <Card
        size="small"
        title="统计"
        style={{
          borderRadius: designSystem.borderRadius.lg,
        }}
      >
        <Row gutter={[8, 8]}>
          <Col span={24}>
            <Statistic
              title="可用文档"
              value={availableDocuments.length}
              prefix={<FileTextOutlined />}
              valueStyle={{ fontSize: designSystem.typography.fontSize.lg }}
            />
          </Col>
          <Col span={24}>
            <Statistic
              title="可用项目"
              value={availableProjects.length}
              prefix={<FolderOutlined />}
              valueStyle={{ fontSize: designSystem.typography.fontSize.lg }}
            />
          </Col>
        </Row>
      </Card>
    </>
  );

  // 右侧快捷操作面板
  const rightSidebar = (
    <Card
      size="small"
      title="生成进度"
      style={{
        borderRadius: designSystem.borderRadius.lg,
      }}
    >
      {isGenerating && reportProgress ? (
        <div style={{ fontSize: designSystem.typography.fontSize.xs }}>
          <div style={{ marginBottom: designSystem.spacing[2] }}>
            <div style={{ color: designSystem.semantic.text.secondary }}>
              当前阶段:
            </div>
            <div style={{ color: designSystem.semantic.text.primary, fontWeight: designSystem.typography.fontWeight.medium }}>
              {reportProgress.stage}
            </div>
          </div>
          <div>
            <div style={{ color: designSystem.semantic.text.secondary }}>
              进度:
            </div>
            <div style={{ color: designSystem.semantic.text.primary }}>
              {Math.round(reportProgress.progress * 100)}%
            </div>
          </div>
        </div>
      ) : (
        <div style={{
          textAlign: 'center',
          padding: designSystem.spacing[4],
          color: designSystem.semantic.text.tertiary,
          fontSize: designSystem.typography.fontSize.sm
        }}>
          未在生成报告
        </div>
      )}
    </Card>
  );

  // 底部状态栏
  const bottomBar = (
    <>
      <span>报告类型: {reportType === 'document' ? '文档报告' : reportType === 'comparison' ? '对比报告' : '项目报告'}</span>
      {isGenerating && <span>生成中...</span>}
    </>
  );

  // ==================== 渲染 ====================
  return (
    <PageLayout
      topBar={topBar}
      leftSidebar={leftSidebar}
      leftDefaultCollapsed={leftCollapsed}
      onLeftCollapsedChange={setLeftCollapsed}
      rightSidebar={rightSidebar}
      rightDefaultCollapsed={rightCollapsed}
      onRightCollapsedChange={setRightCollapsed}
      bottomBar={bottomBar}
    >
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        background: designSystem.semantic.surface.base,
        padding: designSystem.spacing[1],
      }}>
        {!currentReport ? (
          <ReportGenerator
            onGenerate={handleGenerateReport}
            isGenerating={isGenerating}
            availableDocuments={availableDocuments}
            availableProjects={availableProjects}
            availableModels={availableModels}
          />
        ) : (
          <ReportViewer
            report={currentReport}
            progress={reportProgress || undefined}
            isGenerating={isGenerating}
          />
        )}
      </div>
    </PageLayout>
  );
}
