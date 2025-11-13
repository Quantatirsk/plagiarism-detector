/**
 * 文档比对系统状态管理
 *
 * 管理文档比对系统的全局状态，包括：
 * - 选中的项目、文档、任务、文档对
 * - 当前激活的功能模块
 * - 侧边栏折叠状态
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type ComparisonModule = 'projects' | 'tasks' | 'reports';

interface ComparisonState {
  // ==================== 选中状态 ====================
  selectedProjectId: number | null;
  selectedDocumentId: number | null;
  selectedTaskId: number | null;
  selectedPairId: number | null;

  // ==================== 视图状态 ====================
  activeModule: ComparisonModule;

  // ==================== 侧边栏状态 ====================
  leftCollapsed: boolean;
  rightCollapsed: boolean;

  // ==================== Actions ====================

  // 选择操作
  selectProject: (id: number | null) => void;
  selectDocument: (id: number | null) => void;
  selectTask: (id: number | null) => void;
  selectPair: (id: number | null) => void;

  // 模块切换
  setActiveModule: (module: ComparisonModule) => void;

  // 侧边栏操作
  setLeftCollapsed: (collapsed: boolean) => void;
  setRightCollapsed: (collapsed: boolean) => void;

  // 重置
  reset: () => void;
  resetSelection: () => void;
}

const initialState = {
  selectedProjectId: null,
  selectedDocumentId: null,
  selectedTaskId: null,
  selectedPairId: null,
  activeModule: 'projects' as ComparisonModule,
  leftCollapsed: false,
  rightCollapsed: false,
};

export const useComparisonStore = create<ComparisonState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // ==================== 选择操作 ====================
      selectProject: (id) => set({ selectedProjectId: id }),
      selectDocument: (id) => set({ selectedDocumentId: id }),
      selectTask: (id) => set({ selectedTaskId: id }),
      selectPair: (id) => set({ selectedPairId: id }),

      // ==================== 模块切换 ====================
      setActiveModule: (module) => set({ activeModule: module }),

      // ==================== 侧边栏操作 ====================
      setLeftCollapsed: (collapsed) => set({ leftCollapsed: collapsed }),
      setRightCollapsed: (collapsed) => set({ rightCollapsed: collapsed }),

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
      name: 'comparison-storage',
      // 只持久化部分状态
      partialize: (state) => ({
        selectedProjectId: state.selectedProjectId,
        activeModule: state.activeModule,
        leftCollapsed: state.leftCollapsed,
        rightCollapsed: state.rightCollapsed,
      }),
    }
  )
);

/**
 * 选择器 - 用于性能优化
 */
export const comparisonSelectors = {
  // 获取选中的项目ID
  getSelectedProjectId: (state: ComparisonState) => state.selectedProjectId,

  // 获取当前模块
  getActiveModule: (state: ComparisonState) => state.activeModule,

  // 检查是否有项目被选中
  hasSelectedProject: (state: ComparisonState) => state.selectedProjectId !== null,
};
