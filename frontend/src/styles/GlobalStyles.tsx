/**
 * 全局样式组件
 * 从设计系统配置中自动生成全局 CSS 样式
 * 用于无法通过 ConfigProvider 配置的样式（如 hover 动画效果）
 */

import { designSystem } from './DesignSystem';

export function GlobalStyles() {
  const styles = `
    /* 全局滚动条样式 - 统一设计 */
    /* 配置来源: designSystem.scrollbarSystem */
    * {
      scrollbar-width: thin;  /* Firefox */
      scrollbar-color: ${designSystem.scrollbarSystem.thumbColor} ${designSystem.scrollbarSystem.trackColor};  /* Firefox: thumb track */
    }

    /* WebKit 浏览器 (Chrome, Safari, Edge) */
    *::-webkit-scrollbar {
      width: ${designSystem.scrollbarSystem.width};  /* 垂直滚动条宽度 */
      height: ${designSystem.scrollbarSystem.height};  /* 水平滚动条高度 */
    }

    *::-webkit-scrollbar-track {
      background: ${designSystem.scrollbarSystem.trackColor};  /* 轨道背景 */
    }

    *::-webkit-scrollbar-thumb {
      background-color: ${designSystem.scrollbarSystem.thumbColor};  /* 滑块颜色 */
      border-radius: ${designSystem.scrollbarSystem.borderRadius};  /* 圆角 */
      transition: background-color ${designSystem.transitions.fast};  /* 平滑过渡 */
    }

    *::-webkit-scrollbar-thumb:hover {
      background-color: ${designSystem.scrollbarSystem.thumbHoverColor};  /* hover 时变为品牌色 */
    }

    /* 确保滚动条尺寸不变 */
    *::-webkit-scrollbar-button {
      display: none;  /* 隐藏滚动条箭头按钮 */
    }

    *::-webkit-scrollbar-corner {
      background: transparent;  /* 滚动条交角透明 */
    }

    /* 全局卡片样式 - hover 动效（无法通过 token 配置）*/
    /* 配置来源: designSystem.cardSystem */
    .ant-card {
      border: 1px solid ${designSystem.semantic.border.light} !important;
      transition: box-shadow ${designSystem.transitions.default}, transform ${designSystem.transitions.default} !important;
    }

    .ant-card:hover {
      box-shadow: ${designSystem.cardSystem.shadowHover} !important;
      transform: translateY(${designSystem.cardSystem.hover.translateY}) !important;
    }

    /* 卡片操作栏细节样式 - 部分无法通过 token 配置 */
    .ant-card-actions {
      border-top: 1px solid ${designSystem.semantic.border.light} !important;
      padding: 0 !important;
    }

    .ant-card-actions > li {
      padding: 0;
    }

    .ant-card-actions > li > span {
      font-size: ${designSystem.componentFontSize.button} !important;  /* 13px - 保留覆盖默认字体 */
      line-height: 1.2;
      padding: 0;
      display: inline-flex;
      align-items: center;
      cursor: pointer;
    }

    .ant-card-actions > li > span > .anticon {
      font-size: ${designSystem.iconSizes.xs} !important;  /* 12px - 精致小图标 */
    }

    /* Table容器卡片样式 - 统一Table组件外观 */
    /* 配置来源: designSystem.tableSystem */
    .ant-table-wrapper > .ant-card {
      border-radius: ${designSystem.tableSystem.containerBorderRadius} !important;
      box-shadow: ${designSystem.tableSystem.containerShadow} !important;
      transition: box-shadow ${designSystem.tableSystem.hoverTransition}, transform ${designSystem.tableSystem.hoverTransition} !important;
    }

    .ant-table-wrapper > .ant-card:hover {
      box-shadow: ${designSystem.tableSystem.containerShadowHover} !important;
      transform: translateY(${designSystem.tableSystem.hoverTranslateY}) !important;
    }

    /* Table充分利用空间 */
    .ant-table-wrapper {
      display: flex !important;
      flex-direction: column !important;
      height: 100%;
    }

    .ant-spin-nested-loading,
    .ant-spin-container {
      display: flex !important;
      flex-direction: column !important;
      height: 100%;
    }

    /* Table组件样式 - 布局配置（无法通过 token 配置）*/
    /* 配置来源: designSystem.tableSystem */
    .ant-table {
      flex: 1 !important;
      display: flex !important;
      flex-direction: column !important;
      overflow: hidden;
    }

    .ant-table-container {
      flex: 1 !important;
      display: flex !important;
      flex-direction: column !important;
      overflow: hidden;
    }

    .ant-table-body {
      flex: 1 !important;
      overflow-y: auto !important;
    }

    .ant-table-pagination {
      flex-shrink: 0 !important;
      margin-top: ${designSystem.spacing[2]} !important;
    }

    /* 分页器样式已通过 App.tsx theme.components.Pagination 配置 */

    /* Table 单元格布局细节（无法完全通过 token 配置）*/
    .ant-table-thead > tr > th {
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
    }

    .ant-table-tbody > tr > td {
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
      max-width: 0 !important;
    }

    /* Layout组件样式 */
    /* 配置来源: designSystem.semantic.surface */
    .ant-layout-header {
      background-color: ${designSystem.semantic.surface.base} !important;
    }

    .ant-layout-content {
      background-color: ${designSystem.semantic.surface.base} !important;
    }

    .ant-layout-sider {
      background-color: ${designSystem.semantic.surface.base} !important;
    }

    /* Tooltip 额外样式（部分无法通过 token 配置）*/
    /* 配置来源: designSystem.tooltipSystem */
    .ant-tooltip {
      max-width: ${designSystem.tooltipSystem.maxWidth} !important;
    }

    .ant-tooltip-inner {
      padding: ${designSystem.tooltipSystem.paddingBlock} ${designSystem.tooltipSystem.paddingInline} !important;
      box-shadow: ${designSystem.tooltipSystem.boxShadow} !important;
      min-height: auto !important;
    }

    /* 全局组件字体统一 - 语义化配置 */
    /* 按钮组件 */
    .ant-btn {
      font-size: ${designSystem.componentFontSize.button} !important;
    }

    /* 输入框组件 */
    .ant-input,
    .ant-input-number-input {
      font-size: ${designSystem.componentFontSize.input} !important;
    }

    /* 下拉选择器（展示和选项统一） */
    .ant-select-selection-item,
    .ant-select-item {
      font-size: ${designSystem.componentFontSize.select} !important;
    }

    /* Checkbox & Radio */
    .ant-checkbox-wrapper,
    .ant-radio-wrapper {
      font-size: ${designSystem.componentFontSize.checkbox} !important;
    }

    /* Tag 标签 */
    .ant-tag {
      font-size: ${designSystem.componentFontSize.tag} !important;
    }

    /* Badge 徽标 */
    .ant-badge {
      font-size: ${designSystem.componentFontSize.badge} !important;
    }

    /* 表单标签 */
    .ant-form-item-label > label {
      font-size: ${designSystem.componentFontSize.formLabel} !important;
    }

    /* List 列表 */
    .ant-list-item {
      font-size: ${designSystem.componentFontSize.list} !important;
    }

    /* Menu 菜单样式已通过 App.tsx theme.components.Menu 配置 */

    /* ==================== 弹窗间距系统 ==================== */
    /* 注意: Modal/Drawer 的 padding 已通过 App.tsx theme.components 配置 */
    /* 此处保留用户自主构建 div 的 spacing[1] 约定 */

    /* ==================== 无障碍支持（WCAG 2.1）==================== */

    /* 焦点样式 - 确保键盘用户能清晰看到焦点 */
    button:focus-visible,
    a:focus-visible,
    [tabindex]:focus-visible {
      outline: none;
      box-shadow: ${designSystem.a11y.focusRingStyle};
      outline-offset: ${designSystem.a11y.focusRingOffset};
    }

    /* 移除鼠标点击时的焦点环（但保留键盘焦点环） */
    button:focus:not(:focus-visible),
    a:focus:not(:focus-visible) {
      outline: none;
      box-shadow: none;
    }

    /* ==================== 用户动画偏好 ==================== */

    /* 为禁用动画的用户提供极短时长 */
    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: ${designSystem.a11y.reducedMotionDuration} !important;
        animation-iteration-count: 1 !important;
        transition-duration: ${designSystem.a11y.reducedMotionDuration} !important;
        scroll-behavior: auto !important;
      }
    }

    /* ==================== 平滑滚动（优先级较低）==================== */

    /* 仅在用户未禁用动画时启用平滑滚动 */
    @media (prefers-reduced-motion: no-preference) {
      html {
        scroll-behavior: smooth;
      }
    }
  `;

  return <style dangerouslySetInnerHTML={{ __html: styles }} />;
}
