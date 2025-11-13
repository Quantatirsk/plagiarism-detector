/**
 * useResponsive - 响应式断点 Hook
 * 基于 Ant Design Grid.useBreakpoint
 */

import { Grid } from 'antd';

export interface ResponsiveInfo {
  isMobile: boolean;    // < 768px
  isTablet: boolean;    // 768px ~ 992px
  isDesktop: boolean;   // ≥ 992px
  isWide: boolean;      // ≥ 1200px
  screens: Partial<Record<'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl', boolean>>;
}

export function useResponsive(): ResponsiveInfo {
  const screens = Grid.useBreakpoint();

  return {
    isMobile: !screens.md,
    isTablet: !!screens.md && !screens.lg,
    isDesktop: !!screens.lg,
    isWide: !!screens.xl,
    screens,
  };
}
