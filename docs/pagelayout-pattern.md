# PageLayout 三栏联动模式

本项目采用了一种特殊的 PageLayout 使用模式，实现了「左侧功能导航 + 中间主内容 + 右侧联动信息」的三栏联动布局。

## 核心设计理念

### 设计目标

- **统一体验**：所有子功能共享一致的导航和布局结构
- **动态联动**：右侧栏根据当前功能和操作实时响应
- **状态共享**：通过全局状态管理实现跨页面数据同步
- **避免嵌套**：子功能页面不使用 PageLayout，避免双层布局问题

### 三栏职责划分

```
┌────────────────────────────────────────────────────────┐
│                    TopBar (面包屑)                      │
├──────────┬──────────────────────────┬─────────────────┤
│          │                          │                 │
│  左侧栏   │        中间主内容         │    右侧栏        │
│          │                          │                 │
│ 功能菜单  │   当前功能页面内容         │  当前功能的      │
│ (固定)   │   (通过 Outlet 渲染)      │  联动信息        │
│          │                          │  (动态切换)      │
│          │                          │                 │
├──────────┴──────────────────────────┴─────────────────┤
│                  BottomBar (状态栏)                     │
└────────────────────────────────────────────────────────┘
```

**左侧栏**：功能导航菜单（项目管理、比对任务、报告中心）
**中间区域**：当前功能的页面内容（通过路由 Outlet 渲染）
**右侧栏**：根据当前功能和页面状态动态显示相关信息

---

## 实现架构

### 1. 容器页面（使用 PageLayout）

**文件**：`src/pages/comparison/ComparisonSystemPage.tsx`

**职责**：
- 提供统一的 PageLayout 布局框架
- 管理左侧功能菜单
- 根据当前模块动态切换右侧栏
- 通过 `<Outlet />` 渲染子功能页面

**关键代码**：
```tsx
export default function ComparisonSystemPage() {
  const { activeModule, leftCollapsed, rightCollapsed } = useComparisonStore();

  // 左侧：功能菜单（固定）
  const leftSidebar = (
    <Menu
      mode="inline"
      selectedKeys={[activeModule]}
      items={MODULE_CONFIG}
      onClick={({ key }) => navigate(MODULE_CONFIG.find(m => m.key === key)?.path)}
    />
  );

  // 右侧：根据当前模块动态切换
  const rightSidebar = (() => {
    switch (activeModule) {
      case 'projects':
        return <ProjectManagementSidebar />;
      case 'tasks':
        return <TaskManagementSidebar />;
      case 'reports':
        return <ReportCenterSidebar />;
      default:
        return undefined;
    }
  })();

  return (
    <PageLayout
      topBar={topBar}
      leftSidebar={leftSidebar}
      leftDefaultCollapsed={leftCollapsed}
      onLeftCollapsedChange={setLeftCollapsed}
      rightSidebar={rightSidebar}
      rightDefaultCollapsed={rightCollapsed}
      onRightCollapsedChange={setRightCollapsed}
      bottomBar={bottomBar}
      contentPadding={designSystem.spacing[1]}
    >
      <Outlet />  {/* 子功能页面在此渲染 */}
    </PageLayout>
  );
}
```

### 2. 子功能页面（不使用 PageLayout）

**文件**：
- `src/pages/comparison/ProjectManagementPage.tsx`
- `src/pages/comparison/TaskManagementPage.tsx`
- `src/pages/comparison/ReportCenterPage.tsx`

**职责**：
- 实现具体功能的页面内容
- 导出对应的 Sidebar 组件供容器页面使用
- 通过 store 更新状态，触发右侧栏联动

**关键模式**：
```tsx
// ========== 导出 Sidebar 组件 ==========
export function ProjectManagementSidebar() {
  const { selectedProjectId } = useComparisonStore();
  const { data: projects } = useProjects();

  // 根据当前页面状态显示相关信息
  const selectedProject = projects?.find(p => p.id === selectedProjectId);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing[2] }}>
      <Card><Statistic title="总项目数" value={projects?.length ?? 0} /></Card>
      {selectedProject && (
        <Card><Descriptions>
          <Descriptions.Item label="项目名称">{selectedProject.name}</Descriptions.Item>
        </Descriptions></Card>
      )}
    </div>
  );
}

// ========== 页面主组件 ==========
export default function ProjectManagementPage() {
  const { selectedProjectId, selectProject } = useComparisonStore();

  const handleSelectProject = (projectId: number) => {
    selectProject(projectId);  // 更新状态 → 触发右侧栏联动
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: spacing[1] }}>
      {/* 页面内容 */}
    </div>
  );
}
```

### 3. 全局状态管理

**文件**：`src/store/comparisonStore.ts`

**职责**：
- 管理跨页面的共享状态（选中的项目、文档、任务等）
- 管理当前激活的功能模块（activeModule）
- 管理侧边栏折叠状态
- 持久化状态到 localStorage

**关键状态**：
```tsx
interface ComparisonState {
  // 选中状态
  selectedProjectId: number | null;
  selectedDocumentId: number | null;
  selectedTaskId: number | null;

  // 当前模块
  activeModule: 'projects' | 'tasks' | 'reports';

  // 侧边栏状态
  leftCollapsed: boolean;
  rightCollapsed: boolean;

  // Actions
  selectProject: (id: number | null) => void;
  setActiveModule: (module: ComparisonModule) => void;
  setLeftCollapsed: (collapsed: boolean) => void;
  setRightCollapsed: (collapsed: boolean) => void;
}
```

---

## 联动机制详解

### 场景 1：切换功能模块

**用户操作**：点击左侧菜单的「比对任务」
**联动流程**：

```
1. 用户点击菜单项
   ↓
2. 容器页面响应 onClick 事件
   ↓
3. navigate('/comparison/tasks')
   ↓
4. 路由变化，useEffect 监听到路径变化
   ↓
5. setActiveModule('tasks')
   ↓
6. 右侧栏动态切换：TaskManagementSidebar 渲染
   ↓
7. Outlet 渲染 TaskManagementPage 内容
```

**关键代码**：
```tsx
// ComparisonSystemPage.tsx
useEffect(() => {
  const path = location.pathname;
  const module = MODULE_CONFIG.find(m => path.startsWith(m.path));
  if (module && module.key !== activeModule) {
    setActiveModule(module.key);  // 更新模块状态
  }
}, [location.pathname]);

// 右侧栏动态切换
const rightSidebar = (() => {
  switch (activeModule) {
    case 'tasks':
      return <TaskManagementSidebar />;
    // ...
  }
})();
```

### 场景 2：页面内操作触发联动

**用户操作**：在项目管理页选中某个项目
**联动流程**：

```
1. 用户点击项目卡片
   ↓
2. ProjectManagementPage 响应点击事件
   ↓
3. selectProject(projectId)  // 更新 store 状态
   ↓
4. ProjectManagementSidebar 监听到 selectedProjectId 变化
   ↓
5. 右侧栏实时更新显示该项目的详细信息
```

**关键代码**：
```tsx
// ProjectManagementPage.tsx
const handleSelectProject = (projectId: number) => {
  selectProject(projectId);  // 触发联动
};

// ProjectManagementSidebar.tsx
export function ProjectManagementSidebar() {
  const { selectedProjectId } = useComparisonStore();  // 响应状态变化
  const { data: projects } = useProjects();

  const selectedProject = projects?.find(p => p.id === selectedProjectId);

  // 根据 selectedProject 渲染不同内容
  return selectedProject ? (
    <Card>
      <Descriptions>
        <Descriptions.Item label="项目名称">{selectedProject.name}</Descriptions.Item>
      </Descriptions>
    </Card>
  ) : (
    <Empty description="请选择项目" />
  );
}
```

### 场景 3：跨页面状态共享

**场景**：在「项目管理」页选中项目后，切换到「比对任务」页，任务列表自动筛选该项目的任务

**实现**：
```tsx
// ProjectManagementPage.tsx
const handleSelectProject = (projectId: number) => {
  selectProject(projectId);  // 更新全局状态
};

// TaskManagementPage.tsx
export default function TaskManagementPage() {
  const { selectedProjectId } = useComparisonStore();  // 读取全局状态

  // 根据 selectedProjectId 筛选任务
  const filteredTasks = useMemo(() => {
    return tasks.filter(task => task.projectId === selectedProjectId);
  }, [tasks, selectedProjectId]);

  return <TaskList tasks={filteredTasks} />;
}

// TaskManagementSidebar.tsx
export function TaskManagementSidebar() {
  const { selectedProjectId } = useComparisonStore();

  // 显示当前项目的任务统计
  return (
    <Card>
      <Statistic title="当前项目任务数" value={taskCount} />
    </Card>
  );
}
```

---

## 设计优势

### 1. 避免布局嵌套问题

**传统做法（错误）**：
```tsx
// 容器页
<PageLayout leftSidebar={menu} bottomBar={<></>}>
  <Outlet />
</PageLayout>

// 子页面
<PageLayout topBar={toolbar} bottomBar={stats}>
  {content}
</PageLayout>
```

**问题**：双层 bottomBar，空容器显示

**本项目做法（正确）**：
```tsx
// 容器页：使用 PageLayout
<PageLayout leftSidebar={menu} rightSidebar={dynamicSidebar} bottomBar={bar}>
  <Outlet />
</PageLayout>

// 子页面：不使用 PageLayout
<div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
  {content}
</div>
```

### 2. 统一的用户体验

- 用户在任何子功能页面都能看到一致的导航结构
- 左侧菜单始终可见，方便快速切换功能
- 右侧栏实时显示相关信息，无需额外点击

### 3. 状态管理清晰

- 通过 Zustand store 集中管理跨页面状态
- 状态变化自动触发联动，无需手动同步
- 持久化到 localStorage，刷新页面不丢失状态

### 4. 高度解耦

- 容器页面只负责布局和路由
- 子功能页面专注于业务逻辑
- Sidebar 组件独立导出，可自由组合

---

## 开发规范

### 新增子功能页面

1. **创建页面组件**（不使用 PageLayout）
```tsx
// src/pages/comparison/NewFeaturePage.tsx
export default function NewFeaturePage() {
  const { someState, updateState } = useComparisonStore();

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: spacing[1] }}>
      {/* 页面内容 */}
    </div>
  );
}
```

2. **导出对应的 Sidebar 组件**
```tsx
// NewFeaturePage.tsx
export function NewFeatureSidebar() {
  const { someState } = useComparisonStore();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing[2] }}>
      <Card>
        <Statistic title="相关统计" value={someValue} />
      </Card>
    </div>
  );
}
```

3. **在容器页面添加配置**
```tsx
// ComparisonSystemPage.tsx
const MODULE_CONFIG = [
  // ... 已有配置
  {
    key: 'new-feature',
    icon: <NewIcon />,
    label: '新功能',
    path: '/comparison/new-feature',
  },
];

// 右侧栏动态切换
const rightSidebar = (() => {
  switch (activeModule) {
    // ... 已有分支
    case 'new-feature':
      return <NewFeatureSidebar />;
  }
})();
```

4. **添加路由配置**
```tsx
// src/router/index.tsx
{
  path: 'comparison',
  element: <ComparisonSystemPage />,
  children: [
    // ... 已有路由
    {
      path: 'new-feature',
      element: <NewFeaturePage />,
    },
  ],
}
```

### Sidebar 组件开发规范

1. **独立数据获取**：Sidebar 组件自己负责数据加载，不依赖页面组件传递
```tsx
export function MySidebar() {
  const { selectedId } = useComparisonStore();
  const { data } = useSomeData(selectedId);  // 自己加载数据

  return <div>{/* 渲染 */}</div>;
}
```

2. **响应式状态监听**：使用 store 中的状态，实现自动联动
```tsx
export function MySidebar() {
  const { selectedId } = useComparisonStore();  // 监听状态变化

  useEffect(() => {
    // selectedId 变化时自动重新加载数据
    loadData(selectedId);
  }, [selectedId]);

  return <div>{/* 渲染 */}</div>;
}
```

3. **条件渲染**：根据状态显示不同内容
```tsx
export function MySidebar() {
  const { selectedId } = useComparisonStore();

  if (!selectedId) {
    return <Empty description="请先选择项目" />;
  }

  return (
    <div>
      <Card>详细信息</Card>
    </div>
  );
}
```

---

## 常见问题

### Q1：为什么子页面不使用 PageLayout？

**A**：避免布局嵌套问题。如果子页面也使用 PageLayout，会导致：
- 双层 bottomBar（容器的 + 子页面的）
- 空容器时显示多余的侧边栏占位
- 滚动条嵌套，用户体验差

### Q2：右侧栏数据如何保持最新？

**A**：通过两种机制保证：
1. **状态监听**：Sidebar 组件通过 `useComparisonStore()` 监听状态变化
2. **自动刷新**：使用 React Query 的数据加载 hooks（如 `useProjects()`），自动缓存和刷新数据

### Q3：如何在页面间共享数据？

**A**：通过 `comparisonStore` 共享状态：
```tsx
// 页面 A
const { selectProject } = useComparisonStore();
selectProject(123);  // 设置选中的项目

// 页面 B
const { selectedProjectId } = useComparisonStore();
// 自动获取到页面 A 设置的 projectId = 123
```

### Q4：子页面中的多个容器如何保持左右对齐？

**A**：所有布局容器都用 Card 包裹，但避免 Card 嵌套 Card。

**核心原则：**
1. **所有布局容器都用 Card 包裹** - 保持左右边距一致
2. **避免 Card 嵌套 Card** - 如果只有一个主要内容，不需要外层 Card
3. **内容卡片可以嵌套** - Tabs 或可滚动区域内部的内容卡片可以嵌套

**正确做法：**
```tsx
// ✅ 多个布局容器，每个都用 Card
export default function ListPage() {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: spacing[1] }}>
      {/* 顶部工具栏 */}
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <Input.Search />
      </Card>

      {/* 主内容区 - 表格 */}
      <Card
        size="small"
        style={{ flex: 1, borderRadius: designSystem.borderRadius.lg, minHeight: 0 }}
        styles={{ body: { padding: spacing[1], overflow: 'hidden' } }}
      >
        <Table />
      </Card>
    </div>
  );
}

// ✅ Tabs 内的内容卡片可以嵌套
export default function DetailPage() {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: spacing[1] }}>
      {/* 顶部工具栏 */}
      <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
        <div>操作按钮</div>
      </Card>

      {/* 主内容区 - Tabs */}
      <Card size="small" style={{ flex: 1, borderRadius: designSystem.borderRadius.lg, minHeight: 0 }}>
        <Tabs items={[
          {
            key: 'basic',
            label: '基本信息',
            children: (
              <div style={{ padding: spacing[1] }}>
                {/* 内容卡片，不是布局嵌套 */}
                <Card size="small" title="详细信息"><Descriptions /></Card>
                <Card size="small" title="统计信息"><Statistics /></Card>
              </div>
            ),
          },
        ]} />
      </Card>
    </div>
  );
}
```

**错误做法：**
```tsx
// ❌ 主内容区没有 Card 包裹
<div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: spacing[1] }}>
  <Card size="small"><Input.Search /></Card>

  {/* 直接用 div，左右边距与上面的 Card 不一致 */}
  <div style={{ flex: 1, padding: spacing[1] }}>
    <Table />
  </div>
</div>

// ❌ Card 嵌套 Card（布局层面）
<div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: spacing[1] }}>
  <Card size="small"><Input.Search /></Card>

  {/* 外层 Card */}
  <Card size="small" style={{ flex: 1 }}>
    {/* 内层 Card - 多余的嵌套 */}
    <Card size="small">
      <Table />
    </Card>
  </Card>
</div>
```

**关键点**：
1. **布局容器都用 Card** - 保持左右边距一致
2. **避免布局 Card 嵌套** - 不要在 Card 里再包一层 Card
3. **内容卡片可以嵌套** - Tabs/可滚动区域内的卡片是内容，不是布局嵌套

### Q5：Sidebar 组件何时重新渲染？

**A**：当以下情况发生时：
1. `useComparisonStore()` 中监听的状态变化（如 `selectedProjectId` 变化）
2. React Query 数据更新（如 `useProjects()` 返回新数据）
3. 父组件 `ComparisonSystemPage` 重新渲染

---

## 总结

本项目的 PageLayout 三栏联动模式通过以下设计实现了高效的多功能页面管理：

1. **容器页面（ComparisonSystemPage）** 提供统一的布局框架和导航
2. **子功能页面** 专注业务逻辑，不使用 PageLayout
3. **动态 Sidebar** 根据当前模块和页面状态实时联动
4. **全局状态管理（comparisonStore）** 实现跨页面数据共享

这种模式既保持了统一的用户体验，又避免了布局嵌套问题，适合需要多个子功能模块协同工作的复杂应用场景。

---

## 参考文件

- 容器页面：`src/pages/comparison/ComparisonSystemPage.tsx`
- 子功能页面：`src/pages/comparison/ProjectManagementPage.tsx`
- 状态管理：`src/store/comparisonStore.ts`
- 路由配置：`src/router/index.tsx`
