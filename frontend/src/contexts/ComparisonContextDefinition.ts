/**
 * ComparisonContext definition
 * 独立文件以支持 React Fast Refresh
 */
import { createContext } from 'react';
import type { UseDataReturn } from '@/hooks/useData';
import type { ProjectSummary } from '@/api/plagiarismApi';

export interface ComparisonContextValue {
  projectsState: UseDataReturn<ProjectSummary[]>;
}

export const ComparisonContext = createContext<ComparisonContextValue | null>(null);
