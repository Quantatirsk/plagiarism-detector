/**
 * TasksTab - 任务管理Tab
 * 复用 ProjectDetailPanel 的任务列表和对比功能
 */

import { useState, useEffect, useMemo } from 'react';
import { Card, Table, Tag, Button, message } from 'antd';
import { PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { plagiarismApi, type ProjectSummary, type CompareJobSummary } from '@/api/plagiarismApi';
import { useCompareJobs, useDocuments } from '@/hooks/useData';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { JOB_STATUS_META, fallbackStatusMeta } from '@/lib/status';
import { useProgressTracking } from '@/hooks/useProgressTracking';
import { ProgressIndicator } from '@/components/progress/ProgressIndicator';
import { designSystem } from '@/styles/DesignSystem';

interface TasksTabProps {
  project: ProjectSummary;
}

export default function TasksTab({ project }: TasksTabProps) {
  // ==================== 状态管理 ====================
  const [comparisonTaskId, setComparisonTaskId] = useState<string | null>(null);
  const [runningComparisons, setRunningComparisons] = useState(false);
  const { selectedTaskId, openTask, setActiveTab } = useWorkspaceStore();

  const jobsState = useCompareJobs(project.id);
  const documentState = useDocuments({ projectId: project.id });
  const jobs = jobsState.data ?? [];
  const documents = documentState.data ?? [];

  // Progress tracking for comparisons
  const comparisonProgress = useProgressTracking(comparisonTaskId, {
    useSSE: true,
    onComplete: () => {
      setComparisonTaskId(null);
      setRunningComparisons(false);
      jobsState.reload();
    },
    onError: () => {
      setComparisonTaskId(null);
      setRunningComparisons(false);
      message.error('比对任务失败');
    },
  });

  // Auto-refresh for active jobs
  const hasActiveJobs = useMemo(
    () => jobs.some((job) => job.status === 'queued' || job.status === 'running'),
    [jobs],
  );

  useEffect(() => {
    if (!hasActiveJobs) return;
    const timer = window.setInterval(() => {
      jobsState.reload();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [hasActiveJobs, jobsState.reload]);

  // ==================== 操作函数 ====================
  const handleRunComparisons = async () => {
    setRunningComparisons(true);
    try {
      const response = await plagiarismApi.runProjectComparisons(project.id);
      setComparisonTaskId(response.task_id);
    } catch (error) {
      console.error(error);
      message.error((error as Error).message || '运行比对失败');
      setRunningComparisons(false);
    }
  };

  const colorMap = {
    info: 'blue',
    warning: 'orange',
    success: 'green',
    error: 'red',
    neutral: 'default',
  } as const;

  // ==================== 表格列定义 ====================
  const columns: ColumnsType<CompareJobSummary> = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record) => (
        <span style={{ fontWeight: designSystem.typography.fontWeight.medium }}>
          {name || `任务 #${record.id}`}
        </span>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const statusMeta = JOB_STATUS_META[status] ?? fallbackStatusMeta(status);
        return <Tag color={colorMap[statusMeta.tone]}>{statusMeta.label}</Tag>;
      },
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      width: 180,
      render: (date: string) => (
        <span style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>
          {date ? new Date(date).toLocaleString() : '—'}
        </span>
      ),
    },
  ];

  const completedDocs = documents.filter((doc) => doc.status === 'completed').length;

  // ==================== 交互函数 ====================
  const handleTaskClick = (job: CompareJobSummary) => {
    // 选中任务
    openTask(job.id);
  };

  const handleTaskDoubleClick = (job: CompareJobSummary) => {
    // 双击已完成任务，切换到 compare tab 查看结果
    if (job.status === 'completed') {
      openTask(job.id);
      setActiveTab('compare');
    }
  };

  // ==================== 渲染 ====================
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[2],
        height: '100%',
      }}
    >
      {/* 操作区域 */}
      <Card
        size="small"
        title="比对操作"
        style={{ borderRadius: designSystem.borderRadius.lg }}
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={() => jobsState.reload()}
            loading={jobsState.loading}
            size="small"
          >
            刷新
          </Button>
        }
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[2] }}>
          <div style={{ fontSize: designSystem.typography.fontSize.xs, color: designSystem.semantic.text.secondary }}>
            项目包含 {documents.length} 个文档，其中 {completedDocs} 个已完成处理，可以运行比对任务
          </div>
          <div style={{ display: 'flex', gap: designSystem.spacing[2], alignItems: 'center' }}>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleRunComparisons}
              disabled={runningComparisons || documents.length < 2}
              loading={runningComparisons}
            >
              运行比对
            </Button>
            {comparisonProgress.task && (
              <div style={{ flex: 1 }}>
                <ProgressIndicator task={comparisonProgress.task} />
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* 任务列表 */}
      <Card
        title={`比对任务 (${jobs.length})`}
        style={{
          borderRadius: designSystem.borderRadius.lg,
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
        }}
        bodyStyle={{
          padding: parseInt(designSystem.spacing[1]),
          flex: 1,
          overflow: 'hidden',
        }}
      >
        <Table
          size="small"
          rowKey="id"
          columns={columns}
          dataSource={jobs}
          loading={jobsState.loading}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
            size: 'small',
          }}
          onRow={(record) => ({
            onClick: () => handleTaskClick(record),
            onDoubleClick: () => handleTaskDoubleClick(record),
            style: {
              cursor: 'pointer',
              backgroundColor: selectedTaskId === record.id ? designSystem.colors.primary[50] : undefined
            },
          })}
          scroll={{ y: 'calc(100vh - 520px)' }}
        />
      </Card>
    </div>
  );
}
