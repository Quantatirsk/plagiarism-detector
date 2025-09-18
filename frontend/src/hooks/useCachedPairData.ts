import { useCallback, useEffect, useRef, useState } from 'react';
import { plagiarismApi, type PairReport, type DocumentDetail, type ComparePairSummary } from '@/api/plagiarismApi';

interface CachedPairData {
  report: PairReport | null;
  leftDocument: DocumentDetail | null;
  rightDocument: DocumentDetail | null;
  loading: boolean;
  error: string | null;
}

interface CacheEntry {
  data: CachedPairData;
  timestamp: number;
}

const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export function useCachedPairData(currentPairId: number | null, adjacentPairIds: number[] = []) {
  const cache = useRef<Map<number, CacheEntry>>(new Map());
  const [currentData, setCurrentData] = useState<CachedPairData>({
    report: null,
    leftDocument: null,
    rightDocument: null,
    loading: false,
    error: null,
  });

  const fetchPairData = useCallback(async (pairId: number): Promise<CachedPairData> => {
    try {
      const report = await plagiarismApi.getPairReport(pairId);
      const [leftDocument, rightDocument] = await Promise.all([
        plagiarismApi.getDocument(report.left_document_id),
        plagiarismApi.getDocument(report.right_document_id),
      ]);

      return {
        report,
        leftDocument,
        rightDocument,
        loading: false,
        error: null,
      };
    } catch (error) {
      return {
        report: null,
        leftDocument: null,
        rightDocument: null,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load data',
      };
    }
  }, []);

  const getCachedOrFetch = useCallback(async (pairId: number, updateCurrent = false): Promise<CachedPairData> => {
    const now = Date.now();
    const cached = cache.current.get(pairId);

    if (cached && now - cached.timestamp < CACHE_TTL) {
      if (updateCurrent) {
        setCurrentData(cached.data);
      }
      return cached.data;
    }

    if (updateCurrent) {
      setCurrentData(prev => ({ ...prev, loading: true, error: null }));
    }

    const data = await fetchPairData(pairId);
    cache.current.set(pairId, { data, timestamp: now });

    if (updateCurrent) {
      setCurrentData(data);
    }

    return data;
  }, [fetchPairData]);

  // Load current pair data
  useEffect(() => {
    if (!currentPairId) {
      setCurrentData({
        report: null,
        leftDocument: null,
        rightDocument: null,
        loading: false,
        error: null,
      });
      return;
    }

    getCachedOrFetch(currentPairId, true);
  }, [currentPairId, getCachedOrFetch]);

  // Pre-fetch adjacent pairs
  useEffect(() => {
    adjacentPairIds.forEach(pairId => {
      if (pairId !== currentPairId) {
        getCachedOrFetch(pairId, false);
      }
    });
  }, [adjacentPairIds, currentPairId, getCachedOrFetch]);

  const clearCache = useCallback(() => {
    cache.current.clear();
  }, []);

  const reload = useCallback(() => {
    if (currentPairId) {
      cache.current.delete(currentPairId);
      getCachedOrFetch(currentPairId, true);
    }
  }, [currentPairId, getCachedOrFetch]);

  return {
    ...currentData,
    clearCache,
    reload,
  };
}