/**
 * MainLayout - 主布局组件
 *
 * 功能：
 * - 侧边栏导航
 * - 顶部栏
 * - 内容区域（使用 Outlet 渲染子路由）
 */

import { useState, useEffect } from 'react';
import { Layout, Menu, Button, Drawer, Tooltip } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  ProjectOutlined,
  FileTextOutlined,
  MenuOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';
import { designSystem } from '@/styles/DesignSystem';
import { useResponsive } from '@/hooks/useResponsive';

const { Header, Sider, Content } = Layout;

export default function MainLayout() {
  const navigate = useNavigate();
  const location = useLocation();

  // 响应式：移动端/平板自动折叠
  const { isDesktop, isMobile } = useResponsive();

  const [collapsed, setCollapsed] = useState(() => !isDesktop);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [tooltipOpen, setTooltipOpen] = useState(false);

  useEffect(() => {
    if (!isDesktop) {
      setCollapsed(true);
    }
  }, [isDesktop]);

  // 点击菜单栏以外区域自动折叠侧边栏
  const handleOutsideClick = () => {
    if (!collapsed) {
      setCollapsed(true);
    }
  };

  // 菜单项配置
  const menuItems = [
    {
      key: '/workspace',
      icon: <ProjectOutlined />,
      label: '项目工作区',
    },
    {
      key: '/reports',
      icon: <FileTextOutlined />,
      label: '报告中心',
    },
  ];

  // 智能匹配选中项（支持子路由）
  const selectedKey = (() => {
    const path = location.pathname;
    if (path.startsWith('/workspace')) return '/workspace';
    if (path === '/' || path === '') return '/workspace';
    return path;
  })();

  // 根据路由获取页面标题
  const getPageTitle = () => {
    const path = location.pathname;
    if (path.startsWith('/workspace')) return '项目工作区';
    if (path === '/' || path === '') return '项目工作区';
    const menuItem = menuItems.find(item => item.key === path);
    return menuItem?.label || '论文查重系统';
  };

  const menuContent = (
    <>
      <div
        style={{
          height: designSystem.heights.header,
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'space-between',
          paddingLeft: collapsed ? 0 : designSystem.spacing[6],
          paddingRight: collapsed ? 0 : designSystem.spacing[2],
        }}
      >
        <span
          style={{
            fontSize: designSystem.typography.fontSize.lg,
            fontWeight: designSystem.typography.fontWeight.semibold,
            color: designSystem.colors.primary[500],
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            opacity: collapsed ? 0 : 1,
            width: collapsed ? 0 : 'auto',
            transition: `opacity ${designSystem.transitions.fast}, width ${designSystem.transitions.default}`,
            transitionDelay: collapsed ? '0ms' : '150ms', // 展开时延迟显示文字
          }}
        >
          查重系统
        </span>
        <Tooltip
          title={collapsed ? '展开菜单' : '折叠菜单'}
          placement="right"
          open={tooltipOpen}
          onOpenChange={setTooltipOpen}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => {
              setCollapsed(!collapsed);
              setTooltipOpen(false);
            }}
            style={{
              width: designSystem.buttonSizes.iconButton.width,
              height: designSystem.buttonSizes.iconButton.height,
            }}
          />
        </Tooltip>
      </div>
      <Menu
        mode="inline"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onClick={({ key }) => {
          navigate(key);
          setMobileDrawerOpen(false);
        }}
        inlineCollapsed={collapsed}
        style={{ borderRight: 'none' }}
      />
    </>
  );

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 桌面侧边栏（移动端隐藏）*/}
      {!isMobile && (
        <Sider
          theme="light"
          width={parseInt(designSystem.sidebarSystem.leftWidth)}
          collapsedWidth={parseInt(designSystem.sidebarSystem.collapsedWidth)}
          collapsed={collapsed}
          trigger={null}
          style={{
            boxShadow: designSystem.cardSystem.shadow,
            overflow: 'auto',
            height: '100vh',
            position: 'fixed',
            left: 0,
            top: 0,
            bottom: 0,
            zIndex: 100,
          }}
        >
          {menuContent}
        </Sider>
      )}

      {/* 移动端抽屉 */}
      <Drawer
        title="菜单"
        placement="left"
        open={mobileDrawerOpen}
        onClose={() => setMobileDrawerOpen(false)}
      >
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={({ key }) => {
            navigate(key);
            setMobileDrawerOpen(false);
          }}
          style={{ borderRight: 'none' }}
        />
      </Drawer>

      <Layout style={{
        marginLeft: isMobile ? 0 : (collapsed ? parseInt(designSystem.sidebarSystem.collapsedWidth) : parseInt(designSystem.sidebarSystem.leftWidth)),
        transition: 'margin-left 0.2s',
        minHeight: isMobile ? '100vh' : '100%',
      }}>
        <Header
          onClick={handleOutsideClick}
          style={{
            backgroundColor: designSystem.semantic.surface.base,
            boxShadow: designSystem.cardSystem.shadow,
            padding: `0 ${designSystem.spacing[6]}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            height: designSystem.heights.header,
            position: 'sticky',
            top: 0,
            zIndex: 50,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[4] }}>
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => setMobileDrawerOpen(true)}
              style={{ display: isMobile ? 'inline-flex' : 'none' }}
            />
            <span
              style={{
                fontSize: designSystem.typography.fontSize.lg,
                fontWeight: designSystem.typography.fontWeight.semibold,
                color: designSystem.semantic.text.primary,
              }}
            >
              {getPageTitle()}
            </span>
          </div>
        </Header>

        <Content
          onClick={handleOutsideClick}
          style={{
            height: `calc(100vh - ${designSystem.heights.header})`,
            overflow: 'auto',
            padding: designSystem.spacing[1],  // 8px 最紧凑布局
          }}
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
