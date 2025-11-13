/**
 * PageLayout - 通用页面布局组件
 *
 * 布局结构：
 * - 顶部工具栏
 * - 左侧边栏（可折叠）
 * - 主内容区
 * - 右侧边栏（可折叠）
 * - 底部状态栏
 */

import { useState, useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Button } from 'antd';
import { MenuFoldOutlined, MenuUnfoldOutlined } from '@ant-design/icons';
import { designSystem } from '@/styles/DesignSystem';
import { useResponsive } from '@/hooks/useResponsive';

const STORAGE_KEY = 'page-layout-sidebar-state';

interface SidebarState {
  left: boolean;
  right: boolean;
}

const loadSidebarState = (): SidebarState | null => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
};

const saveSidebarState = (state: SidebarState) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // 忽略存储失败
  }
};

interface PageLayoutProps {
  // 顶部工具栏
  topBar: ReactNode;
  topBarHeight?: string;

  // 左侧边栏
  leftSidebar?: ReactNode;
  leftSidebarWidth?: string;
  leftSidebarPadding?: string;  // 左侧边栏内边距
  leftDefaultCollapsed?: boolean;
  onLeftCollapsedChange?: (collapsed: boolean) => void;

  // 主内容区
  children: ReactNode;
  contentPadding?: string;  // 主内容区内边距

  // 右侧边栏
  rightSidebar?: ReactNode;
  rightSidebarWidth?: string;
  rightSidebarPadding?: string;  // 右侧边栏内边距
  rightDefaultCollapsed?: boolean;
  onRightCollapsedChange?: (collapsed: boolean) => void;

  // 底部状态栏
  bottomBar: ReactNode;
}

export default function PageLayout({
  topBar,
  topBarHeight = designSystem.heights.toolbar,
  leftSidebar,
  leftSidebarWidth = designSystem.sidebarSystem.leftWidth,
  leftSidebarPadding = designSystem.spacing[1],  // 默认 8px
  leftDefaultCollapsed = false,
  onLeftCollapsedChange,
  children,
  contentPadding,  // 主内容区不设置默认 padding，由页面自行控制
  rightSidebar,
  rightSidebarWidth = designSystem.sidebarSystem.rightWidth,
  rightSidebarPadding = designSystem.spacing[1],  // 默认 8px
  rightDefaultCollapsed = false,
  onRightCollapsedChange,
  bottomBar,
}: PageLayoutProps) {
  // 响应式断点：移动端/平板自动折叠侧边栏
  const { isDesktop, isMobile } = useResponsive();

  // 初始化：localStorage 优先级最高（避免闪烁）
  const savedState = loadSidebarState();
  const [leftCollapsed, setLeftCollapsed] = useState(() => {
    // 移动端强制折叠
    if (typeof window !== 'undefined' && window.innerWidth < 768) return true;
    // localStorage 优先
    if (savedState?.left !== undefined) return savedState.left;
    // 最后才用 props
    return leftDefaultCollapsed;
  });
  const [rightCollapsed, setRightCollapsed] = useState(() => {
    if (typeof window !== 'undefined' && window.innerWidth < 768) return true;
    if (savedState?.right !== undefined) return savedState.right;
    return rightDefaultCollapsed;
  });

  // 记住用户在宽屏时的手动设置状态（用于窗口拉宽后恢复）
  const leftUserPreference = useRef(savedState?.left ?? leftDefaultCollapsed);
  const rightUserPreference = useRef(savedState?.right ?? rightDefaultCollapsed);

  // 记录上一次的窗口宽度状态，用于检测真正的尺寸变化
  const prevIsWideEnough = useRef<boolean | null>(null);

  // 处理折叠状态变化
  const handleLeftCollapse = (collapsed: boolean, isUserAction = false) => {
    setLeftCollapsed(collapsed);
    onLeftCollapsedChange?.(collapsed);

    // 如果是用户手动操作且窗口足够宽，保存偏好
    if (isUserAction && isDesktop) {
      leftUserPreference.current = collapsed;
    }
  };

  const handleRightCollapse = (collapsed: boolean, isUserAction = false) => {
    setRightCollapsed(collapsed);
    onRightCollapsedChange?.(collapsed);

    // 如果是用户手动操作且窗口足够宽，保存偏好
    if (isUserAction && isDesktop) {
      rightUserPreference.current = collapsed;
    }
  };

  // 初始化时同步实际状态到父组件（一次性）
  useEffect(() => {
    onLeftCollapsedChange?.(leftCollapsed);
    onRightCollapsedChange?.(rightCollapsed);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // 持久化状态到 localStorage（仅桌面端）
  useEffect(() => {
    if (isDesktop) {
      saveSidebarState({ left: leftCollapsed, right: rightCollapsed });
    }
  }, [leftCollapsed, rightCollapsed, isDesktop]);

  // 响应式折叠/展开侧边栏（仅处理窗口尺寸变化，不处理路由切换）
  useEffect(() => {
    // 初始化
    if (prevIsWideEnough.current === null) {
      prevIsWideEnough.current = isDesktop;
      return;
    }

    const wasWideEnough = prevIsWideEnough.current;

    // 仅在窗口尺寸真正变化时触发（避免路由切换时的闪烁）
    if (wasWideEnough !== isDesktop) {
      // 窗口从宽变窄：保存当前状态并自动折叠
      if (wasWideEnough && !isDesktop) {
        leftUserPreference.current = leftCollapsed;
        rightUserPreference.current = rightCollapsed;
        handleLeftCollapse(true);
        handleRightCollapse(true);
      }

      // 窗口从窄变宽：恢复用户偏好
      if (!wasWideEnough && isDesktop) {
        handleLeftCollapse(leftUserPreference.current);
        handleRightCollapse(rightUserPreference.current);
      }

      prevIsWideEnough.current = isDesktop;
    }
  }, [isDesktop, leftCollapsed, rightCollapsed]);

  // 监听外部状态变化已移除：localStorage 优先级最高，避免闪烁

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: isMobile ? 'auto' : '100%',
        minHeight: isMobile ? '100vh' : 0,
        backgroundColor: designSystem.semantic.surface.base,
      }}
    >
      {/* 顶部工具栏 */}
      <div
        style={{
          minHeight: topBarHeight,
          flexShrink: 0,
          backgroundColor: designSystem.semantic.surface.base,
          display: 'flex',
          alignItems: 'center',
          overflow: 'auto',  // 防止内容溢出，支持横向滚动
          zIndex: 10,
        }}
      >
        {topBar}
      </div>

      {/* 主体区域 */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        {/* 左侧边栏 */}
        {leftSidebar && !leftCollapsed && (
          <div
            style={{
              width: leftSidebarWidth,
              minWidth: designSystem.sidebarSystem.leftMinWidth,
              maxWidth: designSystem.sidebarSystem.leftMaxWidth,
              flexShrink: 0,
              backgroundColor: designSystem.semantic.surface.base,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <div style={{ flex: 1, overflow: 'auto', padding: leftSidebarPadding }}>
              {leftSidebar}
            </div>
          </div>
        )}

        {/* 主内容区 */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          minWidth: 0,
          ...(contentPadding ? { padding: contentPadding } : {})
        }}>
          {children}
        </div>

        {/* 右侧边栏 */}
        {rightSidebar && !rightCollapsed && (
          <div
            style={{
              flexBasis: rightSidebarWidth,
              minWidth: designSystem.sidebarSystem.rightMinWidth,
              maxWidth: designSystem.sidebarSystem.rightMaxWidth,
              flexShrink: 1,
              flexGrow: 0,
              backgroundColor: designSystem.semantic.surface.base,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <div style={{ flex: 1, overflow: 'auto', padding: rightSidebarPadding }}>
              {rightSidebar}
            </div>
          </div>
        )}
      </div>

      {/* 底部状态栏 */}
      {!isMobile && (
        <div
          style={{
            flexShrink: 0,
            boxShadow: designSystem.shadows.xs,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: designSystem.spacing[1],  // 8px 最紧凑布局
            fontSize: designSystem.typography.fontSize.sm,
            gap: designSystem.spacing[1],  // 8px
            backgroundColor: designSystem.semantic.surface.base,
            margin: `0 -${designSystem.spacing[1]} -${designSystem.spacing[1]} -${designSystem.spacing[1]}`,  // 抵消父容器 padding，贴合底部和两边
          }}
        >
          {/* 左侧：页面传入的内容 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[3] }}>
            {bottomBar}
          </div>

          {/* 右侧：统一的侧栏控制按钮 */}
          {(leftSidebar || rightSidebar) && (
            <Button
              type="text"
              size="small"
              icon={
                (leftCollapsed && rightCollapsed) ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />
              }
              onClick={() => {
                const nextState = leftCollapsed && rightCollapsed ? false : true;
                handleLeftCollapse(nextState, true);
                handleRightCollapse(nextState, true);
              }}
              style={{
                fontSize: designSystem.componentFontSize.button,
                color: (leftCollapsed && rightCollapsed)
                  ? designSystem.semantic.text.tertiary
                  : designSystem.colors.primary[500],
                display: 'flex',
                alignItems: 'center',
                gap: designSystem.spacing[0.5],
              }}
            >
              <span style={{ fontSize: designSystem.componentFontSize.button }}>
                侧栏
              </span>
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
