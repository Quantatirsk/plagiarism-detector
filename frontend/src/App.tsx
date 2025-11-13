import { ConfigProvider, theme, App as AntdApp } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { RouterProvider } from 'react-router-dom';
import { router } from './router';
import { designSystem } from './styles';
import { GlobalStyles } from './styles/GlobalStyles';

function App() {
  return (
    <>
      <GlobalStyles />
      <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: theme.defaultAlgorithm,  // 默认主题
        token: {
          // 品牌色配置
          colorPrimary: designSystem.colors.primary[500],
          colorLink: designSystem.colors.primary[500],
          colorLinkHover: designSystem.colors.primary[600],

          // 全局圆角（从 design-system 统一配置）
          borderRadius: parseInt(designSystem.borderRadius.md),      // 4px - 按钮、输入框、菜单
          borderRadiusLG: parseInt(designSystem.borderRadius.lg),    // 6px - 卡片
          borderRadiusSM: parseInt(designSystem.borderRadius.sm),    // 2px - 小元素

          // 字号（从 design-system 统一配置）
          fontSize: parseInt(designSystem.componentFontSize.global),  // 13px - 全局基础字号
          fontSizeHeading1: parseInt(designSystem.typography.fontSize['2xl']),  // 28px
          fontSizeHeading2: parseInt(designSystem.typography.fontSize.xl),      // 22px
          fontSizeHeading3: parseInt(designSystem.typography.fontSize.lg),      // 20px

          // 全局间距配置（Margin）- 从 design-system 统一配置
          marginXXS: parseInt(designSystem.spacing[0.25]),  // 2px - 微小外边距
          marginXS: parseInt(designSystem.spacing[0.5]),    // 4px - 极小外边距
          marginSM: parseInt(designSystem.spacing[1]),      // 8px - 小外边距
          margin: parseInt(designSystem.spacing[2]),        // 12px - 基础外边距
          marginLG: parseInt(designSystem.spacing[3]),      // 16px - 大外边距
          marginXL: parseInt(designSystem.spacing[5]),      // 24px - 超大外边距

          // 全局间距配置（Padding）- 从 design-system 统一配置
          paddingXXS: parseInt(designSystem.spacing[0.5]),  // 4px - 超超小内边距
          paddingXS: parseInt(designSystem.spacing[1]),     // 8px - 超小内边距
          paddingSM: parseInt(designSystem.spacing[2]),     // 12px - 小内边距
          padding: parseInt(designSystem.spacing[3]),       // 16px - 基础内边距
          paddingLG: parseInt(designSystem.spacing[5]),     // 24px - 大内边距
          paddingXL: parseInt(designSystem.spacing[6]),     // 32px - 超大内边距
        },
        // 组件级样式覆盖 - 从 design-system 统一配置
        components: {
          // Card 卡片组件
          Card: {
            borderRadiusLG: parseInt(designSystem.cardSystem.borderRadius),  // 6px
            paddingLG: parseInt(designSystem.spacing[2]),  // 12px - 卡片紧凑 padding
            boxShadow: designSystem.cardSystem.shadow,
            actionsLiMargin: `${designSystem.spacing[0.5]} 0`,  // 2px 上下
            actionsBg: designSystem.semantic.surface.base,
          },
          // Table 表格组件
          Table: {
            borderRadiusLG: parseInt(designSystem.tableSystem.containerBorderRadius),  // 6px
            cellPaddingBlock: parseInt(designSystem.spacing[1]),  // 8px 垂直边距
            cellPaddingInline: parseInt(designSystem.spacing[2]),  // 12px 水平边距
            headerBg: designSystem.tableSystem.headerBackground,
            headerColor: designSystem.semantic.text.primary,
            borderColor: designSystem.tableSystem.borderColor,
            rowHoverBg: designSystem.tableSystem.rowHoverBackground,
            fontSize: parseInt(designSystem.componentFontSize.tableCell),  // 13px
          },
          // Modal 弹窗组件
          Modal: {
            contentBg: designSystem.semantic.surface.elevated,
            headerBg: designSystem.semantic.surface.elevated,
            paddingLG: parseInt(designSystem.spacing[3]),  // 16px - 容器层边距
            paddingMD: parseInt(designSystem.spacing[3]),  // 16px
            paddingContentHorizontalLG: parseInt(designSystem.spacing[3]),
            borderRadiusLG: parseInt(designSystem.borderRadius.lg),  // 6px
          },
          // Drawer 侧边栏组件
          Drawer: {
            paddingLG: parseInt(designSystem.spacing[3]),  // 16px - 容器层边距
            colorBgElevated: designSystem.semantic.surface.elevated,
          },
          // Pagination 分页组件
          Pagination: {
            fontSize: parseInt(designSystem.componentFontSize.pagination),  // 13px
            itemSize: 28,  // 分页项尺寸
          },
          // Menu 菜单组件
          Menu: {
            itemPaddingInline: parseInt(designSystem.spacing[3]),  // 16px - 左右内边距
            fontSize: parseInt(designSystem.componentFontSize.menu),  // 13px
            itemHeight: 40,
            collapsedWidth: parseInt(designSystem.sidebarSystem.collapsedWidth),  // 56px - 与 Sider 一致
          },
          // Tooltip 提示组件
          Tooltip: {
            paddingXS: parseInt(designSystem.tooltipSystem.paddingBlock),  // 6px
            fontSize: parseInt(designSystem.componentFontSize.tooltip),  // 12px
            borderRadius: parseInt(designSystem.tooltipSystem.borderRadius),  // 4px
            colorBgSpotlight: designSystem.tooltipSystem.background,
          },
        },
      }}
      >
        <AntdApp>
          <RouterProvider router={router} />
        </AntdApp>
      </ConfigProvider>
    </>
  );
}

export default App;
