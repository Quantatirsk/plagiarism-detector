/**
 * 通用样式常量
 * 提供常用的 CSS 样式对象，避免重复定义
 */

import type { CSSProperties } from 'react';
import { designSystem } from './DesignSystem';

/**
 * Flex 布局相关样式
 */
export const flexStyles = {
  // 水平居中
  centerX: {
    display: 'flex',
    justifyContent: 'center',
  } as CSSProperties,

  // 垂直居中
  centerY: {
    display: 'flex',
    alignItems: 'center',
  } as CSSProperties,

  // 完全居中
  center: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  } as CSSProperties,

  // 两端对齐
  spaceBetween: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  } as CSSProperties,

  // 列布局
  column: {
    display: 'flex',
    flexDirection: 'column',
  } as CSSProperties,
};

/**
 * 文本样式
 */
export const textStyles = {
  // 文本截断（单行）
  ellipsis: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  } as CSSProperties,

  // 多行文本截断
  ellipsisMultiline: (lines: number) => ({
    display: '-webkit-box',
    WebkitLineClamp: lines,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  } as CSSProperties),

  // 主标题
  heading: {
    fontSize: designSystem.typography.fontSize['2xl'],
    fontWeight: designSystem.typography.fontWeight.bold,
    lineHeight: designSystem.typography.lineHeight.tight,
  } as CSSProperties,

  // 副标题
  subheading: {
    fontSize: designSystem.typography.fontSize.lg,
    fontWeight: designSystem.typography.fontWeight.semibold,
    lineHeight: designSystem.typography.lineHeight.normal,
  } as CSSProperties,

  // 正文
  body: {
    fontSize: designSystem.typography.fontSize.base,
    fontWeight: designSystem.typography.fontWeight.normal,
    lineHeight: designSystem.typography.lineHeight.normal,
  } as CSSProperties,

  // 小字（次要文字）
  small: {
    fontSize: designSystem.typography.fontSize.sm,
  } as CSSProperties,

  // 提示文字（辅助文字）
  hint: {
    fontSize: designSystem.typography.fontSize.xs,
  } as CSSProperties,
};

/**
 * 页面容器样式
 */
export const pageContainer = {
  background: designSystem.cardSystem.pageBackground,
  minHeight: '100vh',
  padding: designSystem.spacing[6],
} as CSSProperties;
