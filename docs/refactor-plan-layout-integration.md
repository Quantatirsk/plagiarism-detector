# 文档比对系统布局集成重构方案

## 1. 需求概述

### 1.1 当前问题
- 系统名称："查重系统"（需要更新）
- 路由结构：功能分散在多个平级路由（`/workspace`, `/reports`, `/compare/:pairId`）
- 布局模式：WorkspacePage 使用复杂的三栏布局 + Tab切换
- 导航体验：ComparePage 为独立全屏页面，脱离主布局

### 1.2 目标方案
- **系统重命名**：统一命名为"文档比对系统"（或用户指定的新名称）
- **集成到 MainLayout**：作为一个一级菜单项
- **使用 PageLayout**：左侧栏为功能子菜单，主区域为对应功能页面
- **统一导航**：所有功能在同一布局体系下切换

---

## 2. 架构设计

### 2.1 当前架构

```
MainLayout (一级导航)
├─ /workspace (项目工作区)
│  └─ WorkspacePage
│      ├─ 左侧栏：项目列表 (ProjectListSidebar)
│      ├─ 主区域：Tabs
│      │  ├─ Tab 1: DocumentsTab (文档管理)
│      │  ├─ Tab 2: TasksTab (任务管理)
│      │  └─ Tab 3: CompareTab (对比结果预览)
│      └─ 右侧栏：ContextSidebar (上下文信息)
├─ /reports (报告中心)
│  └─ ReportsPage
└─ （独立路由，脱离 MainLayout）
    └─ /compare/:pairId
        └─ ComparePage (全屏对比页)
```

### 2.2 目标架构

```
MainLayout (一级导航)
├─ /comparison (文档比对系统)
│  └─ ComparisonSystemPage (统一入口)
│      └─ PageLayout
│          ├─ topBar: 面包屑导航
│          ├─ leftSidebar: 功能子菜单 (Menu)
│          │  ├─ 项目管理
│          │  ├─ 文档管理
│          │  ├─ 比对任务
│          │  ├─ 结果查看
│          │  └─ 报告中心
│          ├─ mainContent: 根据路由渲染功能页
│          │  ├─ /comparison/projects → ProjectManagementPage
│          │  ├─ /comparison/documents → DocumentManagementPage
│          │  ├─ /comparison/tasks → TaskManagementPage
│          │  ├─ /comparison/results → ResultViewPage
│          │  ├─ /comparison/results/:pairId → ResultDetailPage (PlanComparePage)
│          │  └─ /comparison/reports → ReportCenterPage
│          └─ bottomBar: 状态栏信息
└─ 其他一级菜单...
```

---

## 3. 功能模块划分

### 3.1 项目管理 (`/comparison/projects`)
**来源**：WorkspacePage 左侧栏 (ProjectListSidebar)

**功能**：
- 项目列表展示
- 创建新项目
- 项目选择
- 项目统计信息

**UI结构**：
- 主区域：项目列表卡片
- 操作栏：创建项目按钮

### 3.2 文档管理 (`/comparison/documents`)
**来源**：WorkspacePage > DocumentsTab

**功能**：
- 文档列表（需要先选择项目）
- 上传文档
- 文档处理进度
- 文档详情查看

**UI结构**：
- 顶部：项目选择器 + 上传按钮
- 主区域：文档表格
- 右侧（可选）：文档详情

### 3.3 比对任务 (`/comparison/tasks`)
**来源**：WorkspacePage > TasksTab

**功能**：
- 任务列表（需要先选择项目）
- 运行比对
- 任务进度追踪
- 任务状态查看

**UI结构**：
- 顶部：项目选择器 + 运行比对按钮
- 主区域：任务表格
- 右侧（可选）：任务详情

### 3.4 结果查看 (`/comparison/results`)
**来源**：WorkspacePage > CompareTab + ComparePage

**功能**：
- **列表页** (`/comparison/results`)：对比结果卡片预览（原 CompareTab）
- **详情页** (`/comparison/results/:pairId`)：全屏对比视图（原 PlanComparePage）

**UI结构（列表页）**：
- 顶部：项目选择器 + 任务筛选
- 主区域：对比结果卡片网格
- 点击卡片 → 导航到详情页

**UI结构（详情页）**：
- 使用 PageLayout 全屏布局
- 左侧栏：匹配列表
- 主区域：左右文档对比
- 右侧栏（可选）：匹配信息

### 3.5 报告中心 (`/comparison/reports`)
**来源**：ReportsPage

**功能**：
- 报告生成
- 报告查看
- 报告导出

**UI结构**：保持原有布局

---

## 4. 路由设计

### 4.1 路由结构

```typescript
{
  path: '/',
  element: <MainLayout />,
  children: [
    {
      path: 'comparison',
      element: <ComparisonSystemPage />,  // 统一入口
      children: [
        {
          index: true,
          element: <Navigate to="/comparison/projects" replace />,
        },
        {
          path: 'projects',
          element: <ProjectManagementPage />,
        },
        {
          path: 'documents',
          element: <DocumentManagementPage />,
        },
        {
          path: 'tasks',
          element: <TaskManagementPage />,
        },
        {
          path: 'results',
          children: [
            {
              index: true,
              element: <ResultViewPage />,  // 列表页
            },
            {
              path: ':pairId',
              element: <ResultDetailPage />,  // 详情页（PlanComparePage）
            },
          ],
        },
        {
          path: 'reports',
          element: <ReportCenterPage />,
        },
      ],
    },
    // 其他路由...
  ],
}
```

### 4.2 路由守卫

某些页面需要前置条件：
- **文档管理/任务管理/结果查看**：需要先选择项目
- 如果未选择项目，重定向到项目管理页

---

## 5. 状态管理

### 5.1 Store 重命名

```typescript
// 从 workspaceStore.ts 重命名为 comparisonStore.ts
export const useComparisonStore = create<ComparisonState>()(...)
```

### 5.2 状态结构

```typescript
interface ComparisonState {
  // 选中状态
  selectedProjectId: number | null;
  selectedDocumentId: number | null;
  selectedTaskId: number | null;
  selectedPairId: number | null;

  // 当前活动的功能模块（子菜单）
  activeModule: 'projects' | 'documents' | 'tasks' | 'results' | 'reports';

  // PageLayout 侧边栏状态
  leftCollapsed: boolean;
  rightCollapsed: boolean;

  // Actions
  selectProject: (id: number | null) => void;
  selectDocument: (id: number | null) => void;
  selectTask: (id: number | null) => void;
  selectPair: (id: number | null) => void;
  setActiveModule: (module: string) => void;
  // ...
}
```

---

## 6. 组件设计

### 6.1 ComparisonSystemPage（统一入口）

```typescript
export default function ComparisonSystemPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { selectedProjectId, activeModule, leftCollapsed, rightCollapsed } = useComparisonStore();

  // 左侧功能菜单
  const menuItems = [
    { key: 'projects', icon: <FolderOutlined />, label: '项目管理', path: '/comparison/projects' },
    { key: 'documents', icon: <FileTextOutlined />, label: '文档管理', path: '/comparison/documents' },
    { key: 'tasks', icon: <UnorderedListOutlined />, label: '比对任务', path: '/comparison/tasks' },
    { key: 'results', icon: <EyeOutlined />, label: '结果查看', path: '/comparison/results' },
    { key: 'reports', icon: <FileSearchOutlined />, label: '报告中心', path: '/comparison/reports' },
  ];

  const leftSidebar = (
    <Menu
      mode="inline"
      selectedKeys={[activeModule]}
      items={menuItems}
      onClick={({ key }) => {
        const item = menuItems.find(m => m.key === key);
        if (item) navigate(item.path);
      }}
    />
  );

  const topBar = (
    <Breadcrumb>
      <Breadcrumb.Item>文档比对系统</Breadcrumb.Item>
      <Breadcrumb.Item>{/* 当前模块名称 */}</Breadcrumb.Item>
    </Breadcrumb>
  );

  const bottomBar = (
    <>
      {selectedProjectId && <span>当前项目: #{selectedProjectId}</span>}
    </>
  );

  return (
    <PageLayout
      topBar={topBar}
      leftSidebar={leftSidebar}
      leftDefaultCollapsed={leftCollapsed}
      onLeftCollapsedChange={setLeftCollapsed}
      rightDefaultCollapsed={rightCollapsed}
      onRightCollapsedChange={setRightCollapsed}
      bottomBar={bottomBar}
    >
      <Outlet />  {/* 渲染子路由 */}
    </PageLayout>
  );
}
```

### 6.2 各功能页面

**ProjectManagementPage**：
- 复用 WorkspacePage 的 ProjectListSidebar 逻辑
- 改为全屏卡片布局

**DocumentManagementPage**：
- 复用 DocumentsTab 组件
- 顶部添加项目选择器

**TaskManagementPage**：
- 复用 TasksTab 组件
- 顶部添加项目选择器

**ResultViewPage**：
- 复用 CompareTab 组件（卡片网格）

**ResultDetailPage**：
- 直接渲染 PlanComparePage
- PlanComparePage 已经是完整的 PageLayout 布局
- **注意**：需要处理嵌套 PageLayout 的问题

**ReportCenterPage**：
- 复用 ReportsPage 内容

---

## 7. 实施步骤

### 阶段 1：准备工作
1. 确认系统新名称（默认：文档比对系统）
2. 创建 `/pages/comparison/` 目录
3. 创建 `comparisonStore.ts`（从 workspaceStore 迁移）

### 阶段 2：创建统一入口
1. 创建 `ComparisonSystemPage.tsx`
2. 实现 PageLayout + 左侧菜单
3. 配置子路由 Outlet

### 阶段 3：拆分功能页面
1. **项目管理页**：
   - 创建 `ProjectManagementPage.tsx`
   - 从 WorkspacePage 提取项目列表逻辑
2. **文档管理页**：
   - 创建 `DocumentManagementPage.tsx`
   - 复用 DocumentsTab，添加项目选择器
3. **任务管理页**：
   - 创建 `TaskManagementPage.tsx`
   - 复用 TasksTab，添加项目选择器
4. **结果查看页**：
   - 创建 `ResultViewPage.tsx`（列表）
   - 创建 `ResultDetailPage.tsx`（详情，集成 PlanComparePage）
5. **报告中心页**：
   - 创建 `ReportCenterPage.tsx`
   - 复用 ReportsPage 内容

### 阶段 4：更新路由
1. 修改 `router/index.tsx`
2. 添加 `/comparison/*` 嵌套路由
3. 移除旧的平级路由

### 阶段 5：更新 MainLayout
1. 更新系统名称
2. 修改菜单项配置
3. 添加"文档比对系统"菜单

### 阶段 6：处理 ResultDetailPage
**问题**：PlanComparePage 内部已经使用了 PageLayout，会导致嵌套

**解决方案 A**：拆分 PlanComparePage
- 提取核心对比逻辑到独立组件
- ResultDetailPage 自己构建 PageLayout

**解决方案 B**：让 ResultDetailPage 全屏显示
- ResultDetailPage 不使用 ComparisonSystemPage 的布局
- 直接渲染 PlanComparePage（独立路由）

**推荐方案 B**，保持对比页的全屏体验。

### 阶段 7：清理旧代码
1. 废弃 WorkspacePage.tsx
2. 移除独立的 ComparePage 路由
3. 更新所有导航链接
4. 更新 workspaceStore 引用

### 阶段 8：测试
1. TypeScript 类型检查
2. 路由导航测试
3. 状态管理测试
4. UI/UX 测试

---

## 8. 关键问题和决策

### Q1: 系统新名称是什么？
**待确认**：默认使用"文档比对系统"

### Q2: ResultDetailPage（对比详情）如何集成？
**方案 B**（推荐）：
- `/comparison/results/:pairId` 为独立路由，直接渲染 PlanComparePage
- 不嵌套在 ComparisonSystemPage 的 PageLayout 中
- 保持全屏对比体验

**路由调整**：
```typescript
{
  path: 'results',
  children: [
    {
      index: true,
      element: <ResultViewPage />,  // 嵌套在 ComparisonSystemPage
    },
    {
      path: ':pairId',
      element: <ResultDetailPage />,  // 独立全屏，不嵌套
    },
  ],
}
```

### Q3: 项目选择器如何实现？
某些页面（文档/任务/结果）需要先选择项目：
- **方案**：在各页面顶部添加项目选择下拉框
- **状态同步**：使用 comparisonStore 的 selectedProjectId
- **路由参数**：可选，也可以通过 `/comparison/documents?projectId=123` 传递

### Q4: 是否需要右侧栏？
**建议**：
- 大部分页面不需要右侧栏
- ResultDetailPage（对比详情）可以保留右侧栏显示匹配信息
- 通过 `rightDefaultCollapsed={true}` 默认折叠

---

## 9. 文件变更清单

### 新增文件
```
frontend/src/
├─ pages/comparison/
│  ├─ ComparisonSystemPage.tsx        # 统一入口
│  ├─ ProjectManagementPage.tsx       # 项目管理
│  ├─ DocumentManagementPage.tsx      # 文档管理
│  ├─ TaskManagementPage.tsx          # 任务管理
│  ├─ ResultViewPage.tsx              # 结果列表
│  ├─ ResultDetailPage.tsx            # 结果详情
│  └─ ReportCenterPage.tsx            # 报告中心
├─ store/
│  └─ comparisonStore.ts              # 状态管理（重命名）
└─ docs/
   └─ refactor-plan-layout-integration.md  # 本文档
```

### 修改文件
```
frontend/src/
├─ layout/MainLayout.tsx              # 更新菜单和系统名称
├─ router/index.tsx                   # 重构路由结构
└─ pages/workspace/                   # 组件复用和调整
   ├─ ProjectListSidebar.tsx          # 复用到项目管理页
   ├─ DocumentsTab.tsx                # 复用到文档管理页
   ├─ TasksTab.tsx                    # 复用到任务管理页
   └─ CompareTab.tsx                  # 复用到结果查看页
```

### 废弃文件
```
frontend/src/pages/
├─ WorkspacePage.tsx                  # 拆分到各功能页
├─ ComparePage.tsx                    # 集成到 ResultDetailPage
└─ store/workspaceStore.ts            # 重命名为 comparisonStore.ts
```

---

## 10. 风险评估

### 高风险
- **状态管理迁移**：workspaceStore → comparisonStore，需要确保所有引用更新
- **路由重定向**：旧路由需要重定向到新路由，避免 404

### 中风险
- **PlanComparePage 集成**：嵌套 PageLayout 可能导致样式问题
- **项目选择逻辑**：需要在多个页面同步项目选择状态

### 低风险
- 组件复用：现有 Tab 组件逻辑清晰，易于提取

---

## 11. 回滚计划

如果重构出现严重问题：
1. 保留旧代码的备份分支
2. 可以快速回滚到 WorkspacePage 架构
3. 建议分阶段提交，便于局部回滚

---

## 12. 后续优化

完成基础重构后的优化方向：
1. **性能优化**：懒加载各功能页面
2. **用户体验**：添加页面切换动画
3. **功能增强**：添加快捷导航、最近访问等
4. **响应式设计**：优化移动端菜单布局

---

## 13. 需要用户确认的问题

### 问题 1：系统新名称
**当前名称**："查重系统"
**建议名称**："文档比对系统" 或 "文档相似度检测系统"
**用户决定**：【待确认】

### 问题 2：结果详情页布局
**选项 A**：嵌套在 ComparisonSystemPage 的 PageLayout 中（需要拆分 PlanComparePage）
**选项 B**：独立全屏渲染（保持当前 PlanComparePage 不变）
**建议**：选项 B
**用户决定**：【待确认】

### 问题 3：项目选择器位置
**选项 A**：在 ComparisonSystemPage 的 topBar 全局显示
**选项 B**：在各功能页面独立显示
**建议**：选项 B（更灵活）
**用户决定**：【待确认】

### 问题 4：是否保留报告中心为独立模块
**选项 A**：集成到文档比对系统的子菜单
**选项 B**：保持为 MainLayout 的独立一级菜单
**建议**：选项 A（统一到文档比对系统）
**用户决定**：【待确认】

---

## 附录：UI 示意图

### 目标布局
```
┌─────────────────────────────────────────────────────────┐
│ MainLayout Header: 文档比对系统                          │
├──────┬──────────────────────────────────────────────────┤
│ Main │ PageLayout (ComparisonSystemPage)                │
│ Nav  │ ┌─────────────────────────────────────────────┐ │
│      │ │ TopBar: 面包屑                              │ │
│ • 文 │ ├──────┬──────────────────────────────────────┤ │
│   档 │ │ Left │ MainContent                         │ │
│   比 │ │ Menu │                                     │ │
│   对 │ │      │  (根据路由渲染功能页)                │ │
│   系 │ │ • 项│                                     │ │
│   统 │ │   目│  ProjectManagementPage              │ │
│      │ │ • 文│  DocumentManagementPage             │ │
│      │ │   档│  TaskManagementPage                 │ │
│      │ │ • 任│  ResultViewPage                     │ │
│      │ │   务│  ReportCenterPage                   │ │
│      │ │ • 结│                                     │ │
│      │ │   果│                                     │ │
│      │ │ • 报│                                     │ │
│      │ │   告│                                     │ │
│      │ ├──────┴──────────────────────────────────────┤ │
│      │ │ BottomBar: 状态信息                        │ │
│      │ └─────────────────────────────────────────────┘ │
└──────┴──────────────────────────────────────────────────┘
```

---

**文档版本**：v1.0
**创建时间**：2025-11-13
**状态**：待用户确认
