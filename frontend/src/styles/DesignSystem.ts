/**
 * Ant Design Template - Design System
 *
 * 设计原则：
 * - 基于 8px 网格系统（Apple iOS 标准）
 * - 主色调：#005BAC
 * - Apple级现代化设计语言
 * - 统一配置驱动，避免硬编码
 */

// ==================== 色彩系统 ====================

export const colors = {
  // 主色 - 基于 #005BAC
  primary: {
    50: '#E6F0F9',
    100: '#B3D4ED',
    200: '#80B8E0',
    300: '#4D9CD4',
    400: '#2680C7',
    500: '#005BAC', // 主色
    600: '#004F96',
    700: '#004380',
    800: '#00376A',
    900: '#002B54',
  },

  // 中性色（Apple级7级精简系统）
  neutral: {
    0: '#FFFFFF',    // 纯白 - 卡片、面板
    50: '#F5F5F7',   // Apple Gray - 页面背景（更冷、更干净）
    100: '#E8E8ED',  // 分隔线、禁用背景
    200: '#D1D1D6',  // 边框、次要分隔
    400: '#8E8E93',  // 次要文字、图标
    600: '#48484A',  // 正文、主要图标
    900: '#1C1C1E',  // 标题、强调文字
  },

  // 功能色
  success: '#10B981',
  warning: '#F59E0B',
  error: '#EF4444',
  info: '#2680C7', // 基于主色 primary.400
} as const;

// 语义化颜色（Apple级语义系统）
export const semantic = {
  surface: {
    base: colors.neutral[0],         // 卡片、面板、悬浮元素
    background: colors.neutral[50],  // 页面背景
    elevated: colors.neutral[0],     // 提升元素（Modal、Popover）
    overlay: 'rgba(0, 0, 0, 0.4)',  // 蒙层
  },
  border: {
    light: colors.neutral[100],      // 浅色边框、分隔线
    medium: colors.neutral[200],     // 标准边框
  },
  text: {
    primary: colors.neutral[900],    // 标题、主要内容
    secondary: colors.neutral[600],  // 正文、次要内容
    tertiary: colors.neutral[400],   // 辅助文字、禁用状态
    inverse: colors.neutral[0],      // 反色文字
  },
  interactive: {
    primary: colors.primary[500],         // 主按钮
    primaryHover: colors.primary[600],    // 主按钮悬停
    secondary: colors.neutral[100],       // 次要按钮背景
    secondaryHover: colors.neutral[200],  // 次要按钮悬停
  },
} as const;

// ==================== 间距系统（Apple级8px基础网格）====================

export const spacing = {
  0: '0',
  0.25: '2px',   // 微小间距（标签内部、微调）
  0.5: '4px',    // 极小间距（图标与文字、紧凑布局）
  0.75: '6px',   // 小间距（卡片操作栏）
  1: '8px',      // 最小间距（iOS列表内元素）
  2: '12px',     // 紧凑间距（标签、小按钮）
  3: '16px',     // 标准间距（卡片内元素、表单间距）
  4: '20px',     // 舒适间距（卡片padding）
  5: '24px',     // 大间距（卡片之间、区块内部）
  6: '32px',     // 区域间距（页面section之间）
  8: '40px',     // 大区域间距（页面顶部、底部）
  10: '48px',    // 超大区域间距（页面板块）
  12: '64px',    // 页面级间距（hero区域）
  16: '80px',    // 特大页面间距
} as const;

// ==================== 高度系统（8px网格对齐）====================

export const heights = {
  // 导航与工具栏
  header: '56px',       // 主标题栏（iOS标准）
  toolbar: '48px',      // 工具栏（增加到48px，更易触控）
  breadcrumb: '32px',   // 面包屑

  // 输入组件（iOS最小触控44px）
  inputSm: '32px',      // 小输入框（从24px增加）
  inputMd: '40px',      // 标准输入框（从32px增加）
  inputLg: '48px',      // 大输入框（从40px增加）

  // 按钮（符合iOS 44pt最小触控标准）
  buttonSm: '32px',     // 小按钮（从24px增加）
  buttonMd: '40px',     // 标准按钮（从32px增加）
  buttonLg: '48px',     // 大按钮（从40px增加）

  // 列表与表格
  listItem: '44px',     // 列表项（iOS标准触控高度）
  tableRow: '48px',     // 表格行（从40px增加）
  statusBar: '32px',    // 状态栏（从24px增加）
  tabBar: '48px',       // 标签栏（从40px增加）

  // 触控目标（新增）
  minTouch: '44px',     // iOS最小触控目标
} as const;

// ==================== 字体系统（iOS标准对齐）====================

export const typography = {
  // 字号（专业工具类网站 - 紧凑统一）
  fontSize: {
    xs: '11px',    // 极小文字（脚注、特殊标注）
    sm: '12px',    // 辅助信息（Tag、Badge、分页器）
    base: '13px',  // 核心字号（表格、按钮、表单、所有主要交互组件）
    md: '15px',    // 正文（长文阅读、描述性内容）
    lg: '20px',    // Title 3
    xl: '22px',    // Title 2
    '2xl': '28px', // Title 1
    '3xl': '34px', // Large Title
    '4xl': '40px', // 特大标题（自定义）
  },

  // 字重（SF Pro字体标准）
  fontWeight: {
    light: 300,    // 新增：轻字重
    normal: 400,   // Regular
    medium: 500,   // Medium
    semibold: 600, // Semibold
    bold: 700,     // Bold
  },

  // 行高（优化可读性）
  lineHeight: {
    tight: 1.2,      // 大标题专用
    snug: 1.375,     // 副标题、按钮（新增）
    normal: 1.5,     // 正文
    relaxed: 1.625,  // 长文阅读
  },

  // 字体家族
  fontFamily: {
    sans: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
    mono: '"SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace',
  },
} as const;

// ==================== 圆角系统（紧凑精致设计）====================

export const borderRadius = {
  none: '0',
  sm: '2px',    // 小元素（Tag、Badge）
  md: '4px',    // 标准按钮、输入框
  lg: '6px',    // 卡片
  xl: '8px',    // Modal、Sheet
  '2xl': '12px', // 大卡片、面板
  '3xl': '18px', // Hero区域
  full: '9999px', // 圆形
} as const;

// ==================== 阴影系统（Apple级多层叠加）====================

export const shadows = {
  none: 'none',

  // 卡片阴影（多层叠加，更立体）
  xs: '0 1px 2px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.04)',
  sm: '0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 4px rgba(0, 0, 0, 0.04)',
  md: '0 4px 16px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.06)',
  lg: '0 8px 24px rgba(0, 0, 0, 0.15), 0 4px 12px rgba(0, 0, 0, 0.08)',
  xl: '0 16px 48px rgba(0, 0, 0, 0.18), 0 8px 24px rgba(0, 0, 0, 0.10)',
  '2xl': '0 24px 64px rgba(0, 0, 0, 0.20), 0 12px 32px rgba(0, 0, 0, 0.12)',

  // 特殊用途阴影
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)', // 内阴影
  focus: '0 0 0 4px rgba(0, 91, 172, 0.15)',       // 焦点环（使用你的品牌色）
} as const;

// ==================== 卡片系统（Apple级现代设计）====================

export const cardSystem = {
  // 卡片圆角（增大到10px）
  borderRadius: borderRadius.lg,  // 10px

  // 卡片尺寸
  minWidth: '240px',              // 卡片最小宽度（用于 grid 布局）

  // 卡片背景色
  background: colors.neutral[0],   // 纯白

  // 卡片阴影（轻量精致设计）
  shadow: shadows.xs,              // 默认：轻量阴影
  shadowHover: shadows.sm,         // 悬停：适度提升
  shadowElevated: shadows.md,      // 提升：明显分层

  // 卡片内边距（增加到20px，大卡片24px）
  padding: spacing[4],             // 20px（从16px增加）
  paddingLarge: spacing[5],        // 24px（大卡片专用）
  paddingCompact: spacing[3],      // 16px（紧凑卡片）

  // 元素间距
  gap: spacing[3],                 // 16px
  gapSmall: spacing[2],           // 12px（紧凑元素）

  // 卡片外边距
  margin: spacing[5],              // 24px（从16px增加）
  marginCompact: spacing[3],       // 16px（紧凑布局）

  // 页面背景色（Apple Gray - 更干净）
  pageBackground: colors.neutral[50],  // #F5F5F7

  // Hover交互效果
  hover: {
    translateY: '0',              // 禁用卡片上浮效果
    scale: 1,                     // 禁用放大效果
  },
} as const;

// ==================== Tooltip 组件系统 ====================

export const tooltipSystem = {
  // 字体大小由 componentFontSize.tooltip 统一管理

  // 内边距
  paddingBlock: spacing[1],              // 8px 上下
  paddingInline: spacing[2],             // 12px 左右

  // 背景与边框
  background: 'rgba(0, 0, 0, 0.85)',     // 半透明黑色
  borderRadius: borderRadius.sm,          // 4px

  // 阴影
  boxShadow: shadows.md,

  // 最大宽度
  maxWidth: '300px',

  // 过渡动画
  transition: '150ms cubic-bezier(0.4, 0, 0.2, 1)',  // 快速动画
} as const;

// ==================== Table 组件系统 ====================

export const tableSystem = {
  // Table容器卡片样式
  containerBorderRadius: borderRadius.lg,  // 10px
  containerBackground: colors.neutral[0],
  containerShadow: shadows.sm,
  containerShadowHover: shadows.md,
  containerPadding: spacing[4],             // 20px（与卡片一致）

  // Hover效果
  hoverTranslateY: '0',                     // 禁用上浮效果
  hoverTransition: '200ms cubic-bezier(0.4, 0, 0.2, 1)', // 过渡时间

  // Table内部样式
  rowHoverBackground: colors.neutral[50],   // 行hover背景
  headerBackground: colors.neutral[0],      // 表头背景（白色）
  borderColor: colors.neutral[100],         // 边框颜色
} as const;

// ==================== 组件字体系统（语义化统一配置）====================

export const componentFontSize = {
  // 全局基础字号（Ant Design ConfigProvider token.fontSize）
  global: typography.fontSize.base,  // 13px - 所有组件默认继承

  // 表格系统
  tableHeader: typography.fontSize.base,    // 13px - 表头
  tableCell: typography.fontSize.base,      // 13px - 单元格

  // 交互组件
  button: typography.fontSize.base,         // 13px - 按钮
  input: typography.fontSize.base,          // 13px - 输入框
  select: typography.fontSize.base,         // 13px - 下拉选择（展示和选项统一）
  checkbox: typography.fontSize.base,       // 13px - 复选框
  radio: typography.fontSize.base,          // 13px - 单选框

  // 辅助组件（较小字号）
  tag: typography.fontSize.sm,              // 12px - 标签
  badge: typography.fontSize.sm,            // 12px - 徽标
  pagination: typography.fontSize.sm,       // 12px - 分页器
  tooltip: typography.fontSize.base,        // 13px - 提示框（保持清晰）

  // 布局组件
  menu: typography.fontSize.base,           // 13px - 菜单
  list: typography.fontSize.base,           // 13px - 列表
  formLabel: typography.fontSize.base,      // 13px - 表单标签
} as const;

// ==================== 断点系统 ====================

export const breakpoints = {
  mobile: '640px',
  tablet: '768px',
  threeColumn: '900px',  // PageLayout 自动折叠断点
  laptop: '1024px',
  desktop: '1280px',
  wide: '1536px',
} as const;

// ==================== Z-Index 层级 ====================

export const zIndex = {
  base: 0,
  dropdown: 1000,
  sticky: 1020,
  fixed: 1030,
  modalBackdrop: 1040,
  modal: 1050,
  popover: 1060,
  tooltip: 1070,
} as const;

// ==================== 模糊效果（Apple毛玻璃）====================

export const blur = {
  none: '0',
  sm: 'blur(8px)',   // 轻微模糊
  md: 'blur(16px)',  // 标准模糊
  lg: 'blur(24px)',  // 强模糊
  xl: 'blur(40px)',  // 超强模糊（背景虚化）
} as const;

// ==================== 不透明度层级 ====================

export const opacity = {
  disabled: 0.38,    // 禁用状态
  inactive: 0.50,    // 非活跃元素
  hover: 0.08,       // 悬停遮罩
  focus: 0.12,       // 焦点遮罩
  selected: 0.16,    // 选中遮罩
  overlay: 0.40,     // 蒙层
} as const;

// ==================== 动画曲线系统（Material Design & Apple 标准）====================

export const easings = {
  // 标准曲线（Material Design）
  standard: 'cubic-bezier(0.4, 0, 0.2, 1)',      // 最常用的缓动曲线

  // 分解曲线
  decelerate: 'cubic-bezier(0.0, 0, 0.2, 1)',    // 减速（元素进入视图）
  accelerate: 'cubic-bezier(0.4, 0, 1, 1)',      // 加速（元素离开视图）
  sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',         // 快速敏锐（临时元素）

  // Apple 风格曲线
  appleEase: 'cubic-bezier(0.25, 0.1, 0.25, 1)', // Apple 经典缓动
  appleSpring: 'cubic-bezier(0.5, 1.5, 0.5, 1)', // 弹性效果（谨慎使用）

  // 特殊曲线
  linear: 'linear',                               // 线性（用于颜色/不透明度）
  easeIn: 'ease-in',                             // 内置缓入
  easeOut: 'ease-out',                           // 内置缓出
  easeInOut: 'ease-in-out',                      // 内置缓入缓出
} as const;

// ==================== 无障碍支持（WCAG 2.1 标准）====================

export const a11y = {
  // 最小触控目标尺寸（WCAG 2.1 AAA）
  minTouchTarget: '44px',         // iOS Human Interface Guidelines
  minTouchTargetWeb: '48px',      // Material Design 推荐

  // 焦点环样式
  focusRingWidth: '2px',          // 焦点边框宽度
  focusRingOffset: '2px',         // 焦点边框偏移
  focusRingColor: colors.primary[500],  // 焦点颜色（使用品牌色）
  focusRingStyle: `0 0 0 2px ${colors.primary[500]}, 0 0 0 4px ${colors.primary[100]}`,  // 双环焦点样式

  // 色彩对比度（参考值，实际需测试）
  contrastRatios: {
    normalText: 4.5,    // WCAG AA 标准（普通文字）
    largeText: 3.0,     // WCAG AA 标准（大文字 18px+ bold 或 24px+）
    graphical: 3.0,     // 图形对象和 UI 组件
  },

  // 动画偏好（尊重用户系统设置）
  respectReducedMotion: true,  // 标记，配合 CSS prefers-reduced-motion 使用
  reducedMotionDuration: '0.01ms',  // 为禁用动画的用户提供极短时长
} as const;

// ==================== 响应式文字缩放（Fluid Typography）====================

export const fluidTypography = {
  // 视口范围（用于计算 clamp）
  minViewport: 320,   // 最小视口宽度（px）
  maxViewport: 1920,  // 最大视口宽度（px）

  // 基础字号缩放范围
  baseFontSize: {
    min: 13,  // 移动端基础字号（px）
    max: 14,  // 桌面端基础字号（px）
  },

  // 标题字号缩放范围
  headingScale: {
    h1: { min: 28, max: 40 },  // Title 1 Large
    h2: { min: 24, max: 34 },  // Title 1
    h3: { min: 20, max: 28 },  // Title 2
    h4: { min: 18, max: 22 },  // Title 3
    h5: { min: 15, max: 20 },  // Headline
    h6: { min: 13, max: 16 },  // Subheadline
  },

  // 生成 CSS clamp 函数的辅助计算
  // clamp(minSize, minSize + (maxSize - minSize) * ((100vw - minVW) / (maxVW - minVW)), maxSize)
  generateClamp: (minSize: number, maxSize: number, minVW = 320, maxVW = 1920) => {
    const slope = (maxSize - minSize) / (maxVW - minVW);
    const yAxisIntersection = minSize - slope * minVW;
    return `clamp(${minSize}px, ${yAxisIntersection.toFixed(4)}px + ${(slope * 100).toFixed(4)}vw, ${maxSize}px)`;
  },
} as const;

// ==================== 滚动条系统 ====================

export const scrollbarSystem = {
  // 尺寸
  width: '4px',
  height: '4px',
  borderRadius: '2px',

  // 颜色（直接引用）
  thumbColor: semantic.text.tertiary,
  thumbHoverColor: colors.primary[500],
  trackColor: 'transparent',
} as const;

// ==================== 侧边栏系统 ====================

export const sidebarSystem = {
  // 左侧栏
  leftWidth: '240px',
  leftMinWidth: '220px',
  leftMaxWidth: '260px',

  // 右侧栏
  rightWidth: '258px',
  rightMinWidth: '240px',
  rightMaxWidth: '280px',

  // 折叠宽度（与 header 高度一致，更统一）
  collapsedWidth: '56px',

  // 响应式断点（引用 breakpoints）
  autoCollapseBreakpoint: breakpoints.threeColumn,
} as const;

// ==================== 输入组件宽度系统 ====================

export const inputWidths = {
  xs: 80,
  sm: 120,
  md: 200,
  lg: 240,
  xl: 320,
  search: 240,  // 搜索框标准宽度
  select: 120,  // 下拉框标准宽度
} as const;

// ==================== 表格列宽预设 ====================

export const tableColumnWidths = {
  checkbox: 48,         // 复选框列
  index: 60,            // 序号列
  action: 150,          // 操作列
  actionCompact: 120,   // 紧凑操作列
  status: 70,           // 状态列
  type: 70,             // 类型列
  date: 130,            // 日期列
  datetime: 160,        // 日期时间列
  number: 85,           // 数字列
  name: 120,            // 名称列
  description: 200,     // 描述列
} as const;

// ==================== 图标尺寸系统 ====================

export const iconSizes = {
  xs: '12px',
  sm: '14px',
  base: '16px',
  md: '18px',
  lg: '20px',
  xl: '24px',
  '2xl': '32px',
} as const;

// ==================== 按钮尺寸预设 ====================

export const buttonSizes = {
  iconButton: {
    width: '32px',
    height: '32px',
  },
  iconButtonLarge: {
    width: '40px',
    height: '40px',
  },
} as const;

// ==================== 头像/容器尺寸系统 ====================

export const avatarSizes = {
  xs: '24px',
  sm: '32px',
  md: '40px',
  lg: '48px',
  xl: '64px',
} as const;

// ==================== Grid Gutter 预设 ====================

export const gridGutter = {
  none: 0,
  xs: 8,
  sm: 12,
  default: 16,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

// ==================== 动画/过渡系统 ====================

export const transitions = {
  // 时长
  duration: {
    instant: '100ms',
    fast: '150ms',
    base: '200ms',
    slow: '300ms',
    slower: '500ms',
  },

  // 缓动函数（Apple级）
  easing: {
    linear: 'linear',
    ease: 'cubic-bezier(0.4, 0, 0.2, 1)',        // 标准
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',        // 进入
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',       // 离开
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',   // 进出
    spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)', // 弹性
    bounce: 'cubic-bezier(0.68, -0.55, 0.27, 1.55)', // 弹跳
  },

  // 常用组合
  default: '200ms cubic-bezier(0.4, 0, 0.2, 1)',
  fast: '150ms cubic-bezier(0.4, 0, 0.2, 1)',
  slow: '300ms cubic-bezier(0.4, 0, 0.2, 1)',
  spring: '400ms cubic-bezier(0.34, 1.56, 0.64, 1)',
} as const;

// ==================== 完整设计系统导出 ====================

export const designSystem = {
  colors,
  semantic,
  spacing,
  heights,
  typography,
  borderRadius,
  shadows,
  cardSystem,
  tooltipSystem,
  tableSystem,
  componentFontSize,  // 组件字体系统
  breakpoints,
  zIndex,
  blur,
  opacity,
  easings,             // 动画曲线系统
  a11y,                // 无障碍支持
  fluidTypography,     // 响应式文字缩放
  scrollbarSystem,     // 滚动条系统
  sidebarSystem,       // 侧边栏系统
  inputWidths,         // 输入组件宽度
  tableColumnWidths,   // 表格列宽预设
  iconSizes,           // 图标尺寸
  buttonSizes,         // 按钮尺寸
  avatarSizes,         // 头像尺寸
  gridGutter,          // Grid gutter
  transitions,         // 过渡动画
} as const;

export default designSystem;
