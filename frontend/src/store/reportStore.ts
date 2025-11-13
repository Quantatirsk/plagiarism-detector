/**
 * 报告中心状态管理
 */
import { create } from 'zustand';

export interface ReportProgress {
  task_id: string;
  progress: number;
  stage: string;
  message: string;
  estimated_remaining?: number;
  error?: string;
}

export interface ReportData {
  id: string;
  type: 'document' | 'comparison' | 'project';
  title: string;
  summary: string;
  content: any;
  data: unknown;
  generated_at: string;
  language: string;
  export_formats: string[];
}

interface ReportStore {
  currentReport: ReportData | null;
  reportProgress: ReportProgress | null;
  isGenerating: boolean;

  setCurrentReport: (report: ReportData | null | ((prev: ReportData | null) => ReportData | null)) => void;
  setReportProgress: (progress: ReportProgress | null) => void;
  setIsGenerating: (isGenerating: boolean) => void;
}

export const useReportStore = create<ReportStore>((set) => ({
  currentReport: null,
  reportProgress: null,
  isGenerating: false,

  setCurrentReport: (report) => set((state) => ({
    currentReport: typeof report === 'function' ? report(state.currentReport) : report
  })),
  setReportProgress: (progress) => set({ reportProgress: progress }),
  setIsGenerating: (isGenerating) => set({ isGenerating }),
}));
