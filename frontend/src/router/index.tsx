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
        element: <Navigate to="/workspace" replace />,
      },
      // 新的统一工作区
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
      // 报告中心（独立功能）
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
  // 全屏对比页面（脱离 MainLayout）
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
    element: <Navigate to="/workspace" replace />,
  },
]);
