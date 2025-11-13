/**
 * ComparisonSystemPage - 文档比对系统统一入口
 *
 * 使用 PageLayout 提供统一的导航和布局：
 * - 左侧栏：功能子菜单
 * - 主区域：根据路由渲染功能页（Outlet）
 * - 右侧栏：根据模块动态显示相关信息
 */

import { useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Menu, Breadcrumb } from 'antd';
import {
  FolderOutlined,
  UnorderedListOutlined,
  FileSearchOutlined,
} from '@ant-design/icons';
import PageLayout from '@/layout/PageLayout';
import { useComparisonStore, type ComparisonModule } from '@/store/comparisonStore';
import { ProjectManagementSidebar } from './ProjectManagementPage';
import { TaskManagementSidebar } from './TaskManagementPage';
import { ReportCenterSidebar } from './ReportCenterPage';
import { designSystem } from '@/styles/DesignSystem';

// 功能模块配置
const MODULE_CONFIG = [
  {
    key: 'projects' as ComparisonModule,
    icon: <FolderOutlined />,
    label: '项目管理',
    path: '/comparison/projects',
  },
  {
    key: 'tasks' as ComparisonModule,
    icon: <UnorderedListOutlined />,
    label: '比对任务',
    path: '/comparison/tasks',
  },
  {
    key: 'reports' as ComparisonModule,
    icon: <FileSearchOutlined />,
    label: '报告中心',
    path: '/comparison/reports',
  },
];

export default function ComparisonSystemPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const {
    selectedProjectId,
    activeModule,
    leftCollapsed,
    rightCollapsed,
    setActiveModule,
    setLeftCollapsed,
    setRightCollapsed,
  } = useComparisonStore();

  // 根据当前路径更新 activeModule
  useEffect(() => {
    const path = location.pathname;
    const module = MODULE_CONFIG.find((m) => path.startsWith(m.path));
    if (module && module.key !== activeModule) {
      setActiveModule(module.key);
    }
  }, [location.pathname, activeModule, setActiveModule]);

  // 左侧功能菜单
  const leftSidebar = (
    <Menu
      mode="inline"
      selectedKeys={[activeModule]}
      items={MODULE_CONFIG.map((module) => ({
        key: module.key,
        icon: module.icon,
        label: module.label,
      }))}
      onClick={({ key }) => {
        const module = MODULE_CONFIG.find((m) => m.key === key);
        if (module) {
          navigate(module.path);
        }
      }}
      style={{ borderRight: 'none', height: '100%' }}
    />
  );

  // 顶部面包屑
  const getCurrentModuleLabel = () => {
    return MODULE_CONFIG.find((m) => m.key === activeModule)?.label || '文档比对系统';
  };

  const topBar = (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: designSystem.spacing[1],
        width: '100%',
      }}
    >
      <Breadcrumb
        items={[
          {
            title: '文档比对系统',
          },
          {
            title: getCurrentModuleLabel(),
          },
        ]}
      />
    </div>
  );

  // 底部状态栏
  const bottomBar = (
    <>
      {selectedProjectId && <span>当前项目: #{selectedProjectId}</span>}
      <span>模块: {getCurrentModuleLabel()}</span>
    </>
  );

  // 右侧栏（根据模块动态显示）
  const rightSidebar = (() => {
    switch (activeModule) {
      case 'projects':
        return <ProjectManagementSidebar />;
      case 'tasks':
        return <TaskManagementSidebar />;
      case 'reports':
        return <ReportCenterSidebar />;
      default:
        return undefined;
    }
  })();

  return (
    <PageLayout
      topBar={topBar}
      leftSidebar={leftSidebar}
      leftSidebarWidth={designSystem.sidebarSystem.leftWidth}
      leftDefaultCollapsed={leftCollapsed}
      onLeftCollapsedChange={setLeftCollapsed}
      rightSidebar={rightSidebar}
      rightDefaultCollapsed={rightCollapsed}
      onRightCollapsedChange={setRightCollapsed}
      bottomBar={bottomBar}
      contentPadding={designSystem.spacing[1]}
    >
      <Outlet />
    </PageLayout>
  );
}
