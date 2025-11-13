import type { ReactNode } from 'react';
import { designSystem } from '@/styles/DesignSystem';

/**
 * DisplayLayout - 全屏展示布局组件
 *
 * 用于大屏展示、数据可视化等需要最大化内容区域的场景
 * 提供可选的顶部操作栏，支持自定义内容区 padding
 */

interface DisplayLayoutProps {
  /** 主内容区 */
  children: ReactNode;

  /** 可选的顶部操作栏（返回、关闭、全屏切换等） */
  topBar?: ReactNode;

  /** 主内容区 padding，默认 0（适合大屏展示） */
  contentPadding?: string;

  /** 背景色，默认白色 */
  backgroundColor?: string;
}

export default function DisplayLayout({
  children,
  topBar,
  contentPadding = '0',
  backgroundColor = designSystem.semantic.surface.base,
}: DisplayLayoutProps) {
  const topBarHeight = designSystem.heights.header; // 56px

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100%',
        backgroundColor,
        overflow: 'hidden',
      }}
    >
      {/* ========== 顶部操作栏 ========== */}
      {topBar && (
        <div
          style={{
            height: topBarHeight,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            padding: `0 ${designSystem.spacing[3]}`,
            borderBottom: `1px solid ${designSystem.semantic.border.light}`,
            backgroundColor: designSystem.semantic.surface.base,
          }}
        >
          {topBar}
        </div>
      )}

      {/* ========== 主内容区 ========== */}
      <div
        style={{
          flex: 1,
          overflow: 'auto',
          padding: contentPadding,
          minHeight: 0, // 防止 flex 子元素溢出
        }}
      >
        {children}
      </div>
    </div>
  );
}
