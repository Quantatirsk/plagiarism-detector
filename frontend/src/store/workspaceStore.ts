/**
 * 工作区状态管理
 *
 * 管理项目工作区的全局状态，包括：
 * - 选中的项目、文档、任务、文档对
 * - 当前激活的Tab
 * - 侧边栏折叠状态
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type WorkspaceTab = 'documents' | 'tasks' | 'compare';

interface WorkspaceState {
  // ==================== 选中状态 ====================
  selectedProjectId: number | null;
  selectedDocumentId: number | null;
  selectedTaskId: number | null;
  selectedPairId: number | null;

  // ==================== 视图状态 ====================
  activeTab: WorkspaceTab;

  // ==================== 侧边栏状态 ====================
  leftCollapsed: boolean;
  rightCollapsed: boolean;

  // ==================== Actions ====================

  // 选择操作
  selectProject: (id: number | null) => void;
  selectDocument: (id: number | null) => void;
  selectTask: (id: number | null) => void;
  selectPair: (id: number | null) => void;

  // Tab切换
  setActiveTab: (tab: WorkspaceTab) => void;

  // 侧边栏操作
  setLeftCollapsed: (collapsed: boolean) => void;
  setRightCollapsed: (collapsed: boolean) => void;

  // 复合操作
  openProject: (projectId: number) => void;
  openTask: (taskId: number) => void;
  openCompare: (pairId: number, taskId?: number) => void;

  // 重置
  reset: () => void;
  resetSelection: () => void;
}

const initialState = {
  selectedProjectId: null,
  selectedDocumentId: null,
  selectedTaskId: null,
  selectedPairId: null,
  activeTab: 'documents' as WorkspaceTab,
  leftCollapsed: false,
  rightCollapsed: false,
};

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // ==================== 选择操作 ====================
      selectProject: (id) => set({ selectedProjectId: id }),
      selectDocument: (id) => set({ selectedDocumentId: id }),
      selectTask: (id) => set({ selectedTaskId: id }),
      selectPair: (id) => set({ selectedPairId: id }),

      // ==================== Tab切换 ====================
      setActiveTab: (tab) => set({ activeTab: tab }),

      // ==================== 侧边栏操作 ====================
      setLeftCollapsed: (collapsed) => set({ leftCollapsed: collapsed }),
      setRightCollapsed: (collapsed) => set({ rightCollapsed: collapsed }),

      // ==================== 复合操作 ====================

      /**
       * 打开项目 - 切换到文档Tab，清除子级选择
       */
      openProject: (projectId) => {
        set({
          selectedProjectId: projectId,
          activeTab: 'documents',
          selectedDocumentId: null,
          selectedTaskId: null,
          selectedPairId: null,
        });
      },

      /**
       * 打开任务 - 切换到任务Tab，清除文档对选择
       */
      openTask: (taskId) => {
        set({
          selectedTaskId: taskId,
          activeTab: 'tasks',
          selectedPairId: null,
        });
      },

      /**
       * 打开对比视图 - 切换到对比Tab
       */
      openCompare: (pairId, taskId) => {
        set({
          selectedPairId: pairId,
          activeTab: 'compare',
          ...(taskId && { selectedTaskId: taskId }),
        });
      },

      // ==================== 重置 ====================

      /**
       * 完全重置到初始状态
       */
      reset: () => set(initialState),

      /**
       * 重置选择状态（保留侧边栏状态）
       */
      resetSelection: () => {
        const { leftCollapsed, rightCollapsed } = get();
        set({
          ...initialState,
          leftCollapsed,
          rightCollapsed,
        });
      },
    }),
    {
      name: 'workspace-storage',
      // 只持久化部分状态
      partialize: (state) => ({
        selectedProjectId: state.selectedProjectId,
        activeTab: state.activeTab,
        leftCollapsed: state.leftCollapsed,
        rightCollapsed: state.rightCollapsed,
      }),
    }
  )
);

/**
 * 选择器 - 用于性能优化
 */
export const workspaceSelectors = {
  // 获取选中的项目ID
  getSelectedProjectId: (state: WorkspaceState) => state.selectedProjectId,

  // 获取当前Tab
  getActiveTab: (state: WorkspaceState) => state.activeTab,

  // 检查是否有项目被选中
  hasSelectedProject: (state: WorkspaceState) => state.selectedProjectId !== null,

  // 检查是否处于对比模式
  isCompareMode: (state: WorkspaceState) =>
    state.activeTab === 'compare' && state.selectedPairId !== null,
};
