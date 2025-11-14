/**
 * ComparisonProvider - 文档比对系统共享数据上下文 Provider
 * 避免重复请求，提升性能
 */

import type { ReactNode } from 'react';
import { useProjects } from '@/hooks/useData';
import { ComparisonContext } from './ComparisonContextDefinition';

export function ComparisonProvider({ children }: { children: ReactNode }) {
  const projectsState = useProjects();

  return (
    <ComparisonContext.Provider value={{ projectsState }}>
      {children}
    </ComparisonContext.Provider>
  );
}
