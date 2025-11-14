/**
 * ContextSidebar - 动态右侧栏
 * 根据当前选中的内容显示不同的上下文信息
 */

import { Card, Statistic, Row, Col, Button, Tag, Spin } from 'antd';
import { FileTextOutlined, UnorderedListOutlined, EyeOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { useProjects } from '@/hooks/useData';
import { plagiarismApi, type DocumentSummary, type CompareJobSummary } from '@/api/plagiarismApi';
import { DOCUMENT_STATUS_META, JOB_STATUS_META, fallbackStatusMeta } from '@/lib/status';
import { designSystem } from '@/styles/DesignSystem';
import { useState } from 'react';

interface ContextSidebarProps {
  documentState: {
    data: DocumentSummary[] | null;
    loading: boolean;
    error: string | null;
    reload: () => void;
  };
  jobsState: {
    data: CompareJobSummary[] | null;
    loading: boolean;
    error: string | null;
    reload: () => void;
  };
}

export default function ContextSidebar({ documentState, jobsState }: ContextSidebarProps) {
  const navigate = useNavigate();
  const {
    selectedProjectId,
    selectedDocumentId,
    selectedTaskId,
    activeTab,
  } = useWorkspaceStore();

  const { data: projects } = useProjects();
  const [loadingPairs, setLoadingPairs] = useState(false);

  const selectedProject = projects?.find((p) => p.id === selectedProjectId);
  const documents = documentState.data ?? [];
  const jobs = jobsState.data ?? [];

  const selectedDocument = documents.find((d) => d.id === selectedDocumentId);
  const selectedTask = jobs.find((j) => j.id === selectedTaskId);

  const colorMap = {
    info: 'blue',
    warning: 'orange',
    success: 'green',
    error: 'red',
    neutral: 'default',
  } as const;

  // ==================== 交互函数 ====================

  const handleViewTaskResults = async () => {
    if (!selectedTaskId) return;

    setLoadingPairs(true);
    try {
      const pairs = await plagiarismApi.listPairs(selectedTaskId);
      if (pairs.length > 0) {
        // 导航到第一个pair的全屏对比页
        navigate(`/comparison/results/${pairs[0].id}`);
      }
    } catch (error) {
      console.error('Failed to load pairs:', error);
    } finally {
      setLoadingPairs(false);
    }
  };

  // ==================== 渲染 ====================

  if (!selectedProjectId) {
    return (
      <div
        style={{
          textAlign: 'center',
          padding: designSystem.spacing[4],
          color: designSystem.semantic.text.tertiary,
        }}
      >
        请选择项目
      </div>
    );
  }

  if (!selectedProject) {
    return (
      <div style={{ padding: designSystem.spacing[4] }}>
        <Spin tip="加载中..." />
      </div>
    );
  }

  const completedDocs = documents.filter((doc) => doc.status === 'completed').length;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[2],
        height: '100%',
        overflow: 'auto',
      }}
    >
      {/* 项目统计 */}
      <Card
        size="small"
        title="项目统计"
        style={{ borderRadius: designSystem.borderRadius.lg }}
      >
        <Row gutter={[8, 8]}>
          <Col span={24}>
            <Statistic
              title="文档总数"
              value={documents.length}
              prefix={<FileTextOutlined />}
              valueStyle={{ fontSize: designSystem.typography.fontSize.lg }}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="已完成"
              value={completedDocs}
              valueStyle={{
                fontSize: designSystem.typography.fontSize.base,
                color: designSystem.colors.success[500],
              }}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="待处理"
              value={documents.length - completedDocs}
              valueStyle={{ fontSize: designSystem.typography.fontSize.base }}
            />
          </Col>
          <Col span={24}>
            <Statistic
              title="比对任务"
              value={jobs.length}
              prefix={<UnorderedListOutlined />}
              valueStyle={{ fontSize: designSystem.typography.fontSize.lg }}
            />
          </Col>
        </Row>
      </Card>

      {/* 文档Tab：显示选中文档详情 */}
      {activeTab === 'documents' && selectedDocument && (
        <Card
          size="small"
          title="文档详情"
          style={{ borderRadius: designSystem.borderRadius.lg }}
        >
          <div style={{ fontSize: designSystem.typography.fontSize.xs }}>
            <div style={{ marginBottom: designSystem.spacing[2] }}>
              <div style={{ color: designSystem.semantic.text.secondary, marginBottom: designSystem.spacing[1] }}>
                文档名称:
              </div>
              <div style={{ fontWeight: designSystem.typography.fontWeight.medium }}>
                {selectedDocument.title || selectedDocument.filename}
              </div>
            </div>
            <div style={{ marginBottom: designSystem.spacing[2] }}>
              <div style={{ color: designSystem.semantic.text.secondary, marginBottom: designSystem.spacing[1] }}>
                状态:
              </div>
              <Tag color={colorMap[DOCUMENT_STATUS_META[selectedDocument.status].tone]}>
                {DOCUMENT_STATUS_META[selectedDocument.status].label}
              </Tag>
            </div>
            {selectedDocument.language && (
              <div style={{ marginBottom: designSystem.spacing[2] }}>
                <div style={{ color: designSystem.semantic.text.secondary, marginBottom: designSystem.spacing[1] }}>
                  语言:
                </div>
                <div>{selectedDocument.language}</div>
              </div>
            )}
            <div>
              <div style={{ color: designSystem.semantic.text.secondary, marginBottom: designSystem.spacing[1] }}>
                更新时间:
              </div>
              <div>{new Date(selectedDocument.updated_at).toLocaleString()}</div>
            </div>
          </div>
        </Card>
      )}

      {/* 任务Tab：显示选中任务详情 */}
      {activeTab === 'tasks' && selectedTask && (
        <Card
          size="small"
          title="任务详情"
          style={{ borderRadius: designSystem.borderRadius.lg }}
        >
          <div style={{ fontSize: designSystem.typography.fontSize.xs }}>
            <div style={{ marginBottom: designSystem.spacing[2] }}>
              <div style={{ color: designSystem.semantic.text.secondary, marginBottom: designSystem.spacing[1] }}>
                任务名称:
              </div>
              <div style={{ fontWeight: designSystem.typography.fontWeight.medium }}>
                {selectedTask.name || `任务 #${selectedTask.id}`}
              </div>
            </div>
            <div style={{ marginBottom: designSystem.spacing[2] }}>
              <div style={{ color: designSystem.semantic.text.secondary, marginBottom: designSystem.spacing[1] }}>
                状态:
              </div>
              <Tag
                color={
                  colorMap[
                    (JOB_STATUS_META[selectedTask.status] ?? fallbackStatusMeta(selectedTask.status))
                      .tone
                  ]
                }
              >
                {(JOB_STATUS_META[selectedTask.status] ?? fallbackStatusMeta(selectedTask.status)).label}
              </Tag>
            </div>
            <div style={{ marginBottom: designSystem.spacing[3] }}>
              <div style={{ color: designSystem.semantic.text.secondary, marginBottom: designSystem.spacing[1] }}>
                更新时间:
              </div>
              <div>
                {selectedTask.updated_at
                  ? new Date(selectedTask.updated_at).toLocaleString()
                  : '—'}
              </div>
            </div>
            {selectedTask.status === 'completed' && (
              <Button
                size="small"
                block
                type="primary"
                icon={<EyeOutlined />}
                onClick={handleViewTaskResults}
                loading={loadingPairs}
              >
                查看对比结果
              </Button>
            )}
          </div>
        </Card>
      )}

      {/* 项目说明 */}
      <Card
        size="small"
        title="项目说明"
        style={{ borderRadius: designSystem.borderRadius.lg }}
      >
        <div
          style={{
            fontSize: designSystem.typography.fontSize.xs,
            color: designSystem.semantic.text.secondary,
          }}
        >
          {selectedProject.description || '暂无描述'}
        </div>
      </Card>
    </div>
  );
}
