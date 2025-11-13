/**
 * DocumentManagementPage - 文档管理页
 *
 * 复用 DocumentsTab 组件，添加项目选择器
 */

import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Select, Space, Empty } from 'antd';
import { FolderOutlined } from '@ant-design/icons';
import { useComparisonStore } from '@/store/comparisonStore';
import { useProjects } from '@/hooks/useData';
import DocumentsTab from '@/pages/workspace/DocumentsTab';
import { designSystem } from '@/styles/DesignSystem';

export default function DocumentManagementPage() {
  const navigate = useNavigate();
  const { selectedProjectId, selectProject } = useComparisonStore();
  const { data: projects, loading: projectsLoading } = useProjects();

  // 如果没有项目，引导用户创建
  useEffect(() => {
    if (!projectsLoading && projects && projects.length === 0) {
      // 没有项目，跳转到项目管理页
      navigate('/comparison/projects');
    }
  }, [projects, projectsLoading, navigate]);

  const selectedProject = projects?.find((p) => p.id === selectedProjectId);

  return (
    <div
      style={{
        padding: designSystem.spacing[3],
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[2],
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

      {/* 文档列表 */}
      {!selectedProject ? (
        <Card style={{ flex: 1, borderRadius: designSystem.borderRadius.lg }}>
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="请先选择一个项目"
          />
        </Card>
      ) : (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <DocumentsTab project={selectedProject} />
        </div>
      )}
    </div>
  );
}
