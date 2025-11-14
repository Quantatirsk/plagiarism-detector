/**
 * WorkspacePage - 统一工作区页面
 *
 * 功能：
 * - 项目管理（左侧栏导航）
 * - 文档管理（Tab 1）
 * - 任务管理（Tab 2）
 * - 文档对比（Tab 3）
 * - 动态右侧栏（上下文信息）
 */

import { useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Breadcrumb, Tabs } from 'antd';
import { HomeOutlined, FolderOutlined } from '@ant-design/icons';
import PageLayout from '@/layout/PageLayout';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { useProjects, useDocuments, useCompareJobs } from '@/hooks/useData';
import { designSystem } from '@/styles/DesignSystem';

// 子组件（稍后实现）
import ProjectListSidebar from './workspace/ProjectListSidebar';
import DocumentsTab from './workspace/DocumentsTab';
import TasksTab from './workspace/TasksTab';
import CompareTab from './workspace/CompareTab';
import ContextSidebar from './workspace/ContextSidebar';

export default function WorkspacePage() {
  const { projectId } = useParams<{ projectId?: string }>();
  const navigate = useNavigate();

  // ==================== 全局状态 ====================
  const {
    selectedProjectId,
    activeTab,
    leftCollapsed,
    rightCollapsed,
    selectProject,
    setActiveTab,
    setLeftCollapsed,
    setRightCollapsed,
  } = useWorkspaceStore();

  // ==================== 数据加载 ====================
  const { data: projects } = useProjects();

  // Fix: Memoize filter object to prevent infinite re-renders
  const documentFilter = useMemo(
    () => (selectedProjectId ? { projectId: selectedProjectId } : undefined),
    [selectedProjectId]
  );
  const documentState = useDocuments(documentFilter);
  const jobsState = useCompareJobs(selectedProjectId ?? undefined);

  // ==================== URL 同步 ====================
  // URL参数 → Store状态
  useEffect(() => {
    if (projectId) {
      const id = Number(projectId);
      if (!isNaN(id) && id !== selectedProjectId) {
        selectProject(id);
      }
    }
  }, [projectId, selectedProjectId, selectProject]);

  // Store状态 → URL
  useEffect(() => {
    if (selectedProjectId) {
      const targetPath = `/workspace/${selectedProjectId}`;
      if (window.location.pathname !== targetPath) {
        navigate(targetPath, { replace: true });
      }
    } else {
      if (window.location.pathname !== '/workspace') {
        navigate('/workspace', { replace: true });
      }
    }
  }, [selectedProjectId, navigate]);

  // ==================== 数据获取 ====================
  const selectedProject = projects?.find((p) => p.id === selectedProjectId);

  // ==================== 布局区域 ====================

  // topBar: 面包屑 + 操作按钮
  const topBar = (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: designSystem.spacing[1],
        padding: designSystem.spacing[1],
        width: '100%',
      }}
    >
      <Breadcrumb
        items={[
          {
            title: (
              <span style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[1] }}>
                <HomeOutlined />
                <span>工作区</span>
              </span>
            ),
          },
          ...(selectedProject
            ? [
                {
                  title: (
                    <span style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[1] }}>
                      <FolderOutlined />
                      <span>{selectedProject.name}</span>
                    </span>
                  ),
                },
              ]
            : []),
        ]}
      />
    </div>
  );

  // leftSidebar: 项目列表（二级导航）
  const leftSidebar = <ProjectListSidebar />;

  // mainContent: Tabs 系统
  const mainContent = selectedProject ? (
    <Tabs
      activeKey={activeTab}
      onChange={(key) => setActiveTab(key as typeof activeTab)}
      items={[
        {
          key: 'documents',
          label: '文档',
          children: <DocumentsTab project={selectedProject} documentState={documentState} />,
        },
        {
          key: 'tasks',
          label: '任务',
          children: <TasksTab project={selectedProject} jobsState={jobsState} />,
        },
        {
          key: 'compare',
          label: '对比',
          children: <CompareTab />,
        },
      ]}
      style={{
        height: '100%',
      }}
    />
  ) : (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: designSystem.semantic.text.tertiary,
      }}
    >
      请在左侧选择或创建项目
    </div>
  );

  // rightSidebar: 动态上下文信息
  const rightSidebar = <ContextSidebar documentState={documentState} jobsState={jobsState} />;

  // bottomBar: 状态信息
  const bottomBar = (
    <>
      {selectedProject && (
        <>
          <span>项目: {selectedProject.name}</span>
          <span>当前: {activeTab === 'documents' ? '文档管理' : activeTab === 'tasks' ? '任务管理' : '文档对比'}</span>
        </>
      )}
      {!selectedProject && <span>未选择项目</span>}
    </>
  );

  // ==================== 渲染 ====================
  return (
    <PageLayout
      topBar={topBar}
      leftSidebar={leftSidebar}
      leftSidebarWidth="280px"
      leftDefaultCollapsed={leftCollapsed}
      onLeftCollapsedChange={setLeftCollapsed}
      rightSidebar={rightSidebar}
      rightDefaultCollapsed={rightCollapsed}
      onRightCollapsedChange={setRightCollapsed}
      bottomBar={bottomBar}
    >
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          background: designSystem.semantic.surface.base,
          padding: designSystem.spacing[1],
        }}
      >
        {mainContent}
      </div>
    </PageLayout>
  );
}
