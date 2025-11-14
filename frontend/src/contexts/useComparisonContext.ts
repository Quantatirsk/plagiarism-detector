/**
 * useComparisonContext hook
 * 独立文件以支持 React Fast Refresh
 */
import { useContext } from 'react';
import { ComparisonContext, type ComparisonContextValue } from './ComparisonContextDefinition';

export function useComparisonContext(): ComparisonContextValue {
  const context = useContext(ComparisonContext);
  if (!context) {
    throw new Error('useComparisonContext must be used within ComparisonProvider');
  }
  return context;
}
