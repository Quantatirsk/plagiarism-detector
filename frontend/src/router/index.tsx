/**
 * Router Configuration
 *
 * 使用 React Router v6 配置路由
 */

import { createBrowserRouter, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { LoadingState } from '@/components/common';

// ==================== 组件懒加载 ====================

const WorkspacePage = lazy(() => import('@/pages/WorkspacePage'));
const ReportsPage = lazy(() => import('@/pages/ReportsPage'));
const ComparePage = lazy(() => import('@/pages/ComparePage'));

// 文档比对系统页面
const ComparisonSystemPage = lazy(() => import('@/pages/comparison/ComparisonSystemPage'));
const ProjectManagementPage = lazy(() => import('@/pages/comparison/ProjectManagementPage'));
const TaskManagementPage = lazy(() => import('@/pages/comparison/TaskManagementPage'));
const ReportCenterPage = lazy(() => import('@/pages/comparison/ReportCenterPage'));

// ==================== 布局组件 ====================

import MainLayout from '@/layout/MainLayout';

/**
 * Loading 组件 - 懒加载时的占位符
 */
function PageLoading() {
  return (
    <div style={{ height: '100vh' }}>
      <LoadingState mode="skeleton" rows={10} />
    </div>
  );
}

// ==================== 路由配置 ====================

export const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: <Navigate to="/comparison" replace />,
      },
      // 文档比对系统（新架构）
      {
        path: 'comparison',
        element: (
          <Suspense fallback={<PageLoading />}>
            <ComparisonSystemPage />
          </Suspense>
        ),
        children: [
          {
            index: true,
            element: <Navigate to="/comparison/projects" replace />,
          },
          {
            path: 'projects',
            element: (
              <Suspense fallback={<PageLoading />}>
                <ProjectManagementPage />
              </Suspense>
            ),
          },
          {
            path: 'tasks',
            element: (
              <Suspense fallback={<PageLoading />}>
                <TaskManagementPage />
              </Suspense>
            ),
          },
          {
            path: 'reports',
            element: (
              <Suspense fallback={<PageLoading />}>
                <ReportCenterPage />
              </Suspense>
            ),
          },
        ],
      },
      // 旧路由（保留用于向后兼容，后续可删除）
      {
        path: 'workspace',
        children: [
          {
            index: true,
            element: (
              <Suspense fallback={<PageLoading />}>
                <WorkspacePage />
              </Suspense>
            ),
          },
          {
            path: ':projectId',
            element: (
              <Suspense fallback={<PageLoading />}>
                <WorkspacePage />
              </Suspense>
            ),
          },
        ],
      },
      {
        path: 'reports',
        element: (
          <Suspense fallback={<PageLoading />}>
            <ReportsPage />
          </Suspense>
        ),
      },
    ],
  },
  // 全屏对比页面（独立路由，脱离 MainLayout）
  {
    path: '/comparison/results/:pairId',
    element: (
      <Suspense fallback={<PageLoading />}>
        <ComparePage />
      </Suspense>
    ),
  },
  // 兼容旧对比页路由
  {
    path: '/compare/:pairId',
    element: (
      <Suspense fallback={<PageLoading />}>
        <ComparePage />
      </Suspense>
    ),
  },
  {
    path: '*',
    element: <Navigate to="/comparison" replace />,
  },
]);
