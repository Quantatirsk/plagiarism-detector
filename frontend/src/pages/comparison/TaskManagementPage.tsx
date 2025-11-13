/**
 * TaskManagementPage - 比对任务管理页
 *
 * 功能：
 * - 项目选择器
 * - 任务列表（每行显示"查看结果"按钮）
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Select, Space, Empty, Statistic, Descriptions } from 'antd';
import { FolderOutlined, CheckCircleOutlined, SyncOutlined, CloseCircleOutlined, UnorderedListOutlined } from '@ant-design/icons';
import { useComparisonStore } from '@/store/comparisonStore';
import { useProjects } from '@/hooks/useData';
import { plagiarismApi } from '@/api/plagiarismApi';
import TasksTab from '@/pages/workspace/TasksTab';
import { designSystem } from '@/styles/DesignSystem';

// 右侧栏组件
export function TaskManagementSidebar() {
  const { selectedProjectId } = useComparisonStore();
  const { data: projects } = useProjects();
  const [taskStats, setTaskStats] = useState({
    total: 0,
    completed: 0,
    running: 0,
    failed: 0
  });

  // 加载任务统计数据
  useEffect(() => {
    if (!selectedProjectId) {
      setTaskStats({ total: 0, completed: 0, running: 0, failed: 0 });
      return;
    }

    let mounted = true;
    const loadTaskStats = async () => {
      try {
        const jobs = await plagiarismApi.listCompareJobs(selectedProjectId);
        if (mounted) {
          setTaskStats({
            total: jobs.length,
            completed: jobs.filter(j => j.status === 'completed').length,
            running: jobs.filter(j => j.status === 'running').length,
            failed: jobs.filter(j => j.status === 'failed').length
          });
        }
      } catch (err) {
        if (mounted) {
          setTaskStats({ total: 0, completed: 0, running: 0, failed: 0 });
        }
      }
    };
    loadTaskStats();
    return () => { mounted = false; };
  }, [selectedProjectId]);

  const selectedProject = projects?.find((p) => p.id === selectedProjectId);

  if (!selectedProject) {
    return (
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="请先选择项目"
        />
      </Card>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[2] }}>
      <Card
        title="项目信息"
        size="small"
        style={{ borderRadius: designSystem.borderRadius.lg }}
      >
        <Descriptions column={1} size="small">
          <Descriptions.Item label="项目名称">{selectedProject.name}</Descriptions.Item>
          {selectedProject.description && (
            <Descriptions.Item label="描述">{selectedProject.description}</Descriptions.Item>
          )}
        </Descriptions>
      </Card>
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <Statistic
          title="总任务数"
          value={taskStats.total}
          prefix={<UnorderedListOutlined />}
        />
      </Card>
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <Statistic
          title="已完成"
          value={taskStats.completed}
          prefix={<CheckCircleOutlined />}
          valueStyle={{ color: '#52c41a' }}
        />
      </Card>
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <Statistic
          title="进行中"
          value={taskStats.running}
          prefix={<SyncOutlined spin />}
          valueStyle={{ color: designSystem.colors.primary[500] }}
        />
      </Card>
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <Statistic
          title="失败"
          value={taskStats.failed}
          prefix={<CloseCircleOutlined />}
          valueStyle={{ color: '#ff4d4f' }}
        />
      </Card>
    </div>
  );
}

export default function TaskManagementPage() {
  const navigate = useNavigate();
  const { selectedProjectId, selectProject } = useComparisonStore();
  const { data: projects, loading: projectsLoading } = useProjects();

  // 如果没有项目，引导用户创建
  useEffect(() => {
    if (!projectsLoading && projects && projects.length === 0) {
      navigate('/comparison/projects');
    }
  }, [projects, projectsLoading, navigate]);

  const selectedProject = projects?.find((p) => p.id === selectedProjectId);

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[1],
      }}
    >
      {/* 项目选择器 */}
      <Card
        size="small"
        style={{ borderRadius: designSystem.borderRadius.lg }}
      >
        <Space>
          <FolderOutlined style={{ fontSize: designSystem.iconSizes.md }} />
          <span style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>
            选择项目:
          </span>
          <Select
            style={{ minWidth: 200 }}
            placeholder="请选择项目"
            value={selectedProjectId}
            onChange={(value) => selectProject(value)}
            loading={projectsLoading}
            options={projects?.map((p) => ({
              value: p.id,
              label: p.name,
            }))}
          />
        </Space>
      </Card>

      {/* 任务列表 */}
      {!selectedProject ? (
        <Card style={{ flex: 1, borderRadius: designSystem.borderRadius.lg }}>
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="请先选择一个项目"
          />
        </Card>
      ) : (
        <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <TasksTab project={selectedProject} />
        </div>
      )}
    </div>
  );
}
