import { useCallback, useEffect, useState } from 'react';
import {
  plagiarismApi,
  type DocumentStatus,
  type DocumentSummary,
  type DocumentDetail,
  type CompareJobSummary,
  type ComparePairSummary,
  type PairReport,
  type ProjectSummary,
} from '@/api/plagiarismApi';

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

type AsyncHookReturn<T> = AsyncState<T> & { reload: () => void };

function createInitialState<T>(): AsyncState<T> {
  return { data: null, loading: true, error: null };
}

function createErrorState<T>(error: unknown): AsyncState<T> {
  return {
    data: null,
    loading: false,
    error: error instanceof Error ? error.message : String(error),
  };
}

type DocumentFilter = DocumentStatus | { status?: DocumentStatus; projectId?: number } | undefined;

export function useDocuments(filter?: DocumentFilter): AsyncHookReturn<DocumentSummary[]> {
  const [state, setState] = useState<AsyncState<DocumentSummary[]>>(createInitialState);

  const status = typeof filter === 'string' ? filter : filter?.status;
  const projectId = typeof filter === 'object' && filter !== null ? filter.projectId : undefined;

  const load = useCallback(() => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    plagiarismApi
      .listDocuments({ status, projectId })
      .then((items) => {
        setState({ data: items, loading: false, error: null });
      })
      .catch((error) => {
        setState(createErrorState(error));
      });
  }, [projectId, status]);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}

export function useCompareJobs(projectId?: number | null): AsyncHookReturn<CompareJobSummary[]> {
  const [state, setState] = useState<AsyncState<CompareJobSummary[]>>(createInitialState);

  const load = useCallback(() => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    const request = projectId
      ? plagiarismApi.listProjectJobs(projectId)
      : plagiarismApi.listCompareJobs();
    request
      .then((items) => {
        setState({ data: items, loading: false, error: null });
      })
      .catch((error) => {
        setState(createErrorState(error));
      });
  }, [projectId]);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}

export function useJobPairs(jobId: number | null): AsyncHookReturn<ComparePairSummary[]> {
  const [state, setState] = useState<AsyncState<ComparePairSummary[]>>({ data: null, loading: Boolean(jobId), error: null });

  const load = useCallback(() => {
    if (!jobId) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    setState((prev) => ({ ...prev, loading: true, error: null }));
    plagiarismApi
      .listPairs(jobId)
      .then((items) => {
        setState({ data: items, loading: false, error: null });
      })
      .catch((error) => {
        setState(createErrorState(error));
      });
  }, [jobId]);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}

export function usePairReport(pairId: number | null): AsyncHookReturn<PairReport> {
  const [state, setState] = useState<AsyncState<PairReport>>({ data: null, loading: Boolean(pairId), error: null });

  const load = useCallback(() => {
    if (!pairId) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    setState((prev) => ({ ...prev, loading: true, error: null }));
    plagiarismApi
      .getPairReport(pairId)
      .then((report) => {
        setState({ data: report, loading: false, error: null });
      })
      .catch((error) => {
        setState(createErrorState(error));
      });
  }, [pairId]);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}

export function useDocumentDetail(documentId: number | null): AsyncHookReturn<DocumentDetail> {
  const [state, setState] = useState<AsyncState<DocumentDetail>>({ data: null, loading: Boolean(documentId), error: null });

  const load = useCallback(() => {
    if (!documentId) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    setState((prev) => ({ ...prev, loading: true, error: null }));
    plagiarismApi
      .getDocument(documentId)
      .then((detail) => {
        setState({ data: detail, loading: false, error: null });
      })
      .catch((error) => {
        setState(createErrorState(error));
      });
  }, [documentId]);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}
export function useProjects(): AsyncHookReturn<ProjectSummary[]> {
  const [state, setState] = useState<AsyncState<ProjectSummary[]>>(createInitialState);

  const load = useCallback(() => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    plagiarismApi
      .listProjects()
      .then((items) => {
        setState({ data: items, loading: false, error: null });
      })
      .catch((error) => {
        setState(createErrorState(error));
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}
