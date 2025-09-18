import axios from 'axios';

const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || '';

const apiClient = axios.create({
  baseURL: API_BASE_URL || '/',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface DocumentSummary {
  id: number;
  project_id: number;
  title: string | null;
  filename: string | null;
  source: string | null;
  language: string | null;
  status: DocumentStatus;
  paragraph_count: number;
  sentence_count: number;
  char_count: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentDetail extends DocumentSummary {
  processed_text?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface ProjectSummary {
  id: number;
  name: string | null;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export type CompareJobStatus = 'draft' | 'queued' | 'running' | 'completed' | 'failed';

export interface CompareJobSummary {
  id: number;
  project_id: number;
  name: string | null;
  status: CompareJobStatus;
  created_at: string | null;
  updated_at: string | null;
  config: Record<string, unknown> | null;
}

export type ComparePairStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface PairMetrics {
  top_score?: number | null;
  [metric: string]: number | null | undefined;
}

export interface ComparePairSummary {
  id: number;
  job_id: number;
  left_document_id: number;
  right_document_id: number;
  status: ComparePairStatus;
  metrics: PairMetrics | null;
}

export interface MatchGroupModel {
  id: number;
  left_chunk_id: number;
  right_chunk_id: number;
  final_score: number | null;
  semantic_score: number | null;
  lexical_overlap: number | null;
  cross_score: number | null;
  alignment_ratio: number | null;
  span_count: number;
  match_count: number;
  paragraph_spans?: SpanJson[] | null;
  document_spans?: SpanJson[] | null;
}

export interface MatchDetailModel {
  group_id: number;
  left_chunk_id: number;
  right_chunk_id: number;
  final_score: number | null;
  semantic_score: number | null;
  lexical_overlap: number | null;
  cross_score: number | null;
  spans?: SpanJson[] | null;
}

export interface SpanJson {
  left_start: number;
  left_end: number;
  right_start: number;
  right_end: number;
}

export interface PairReport {
  pair: ComparePairSummary;
  left_document_id: number;
  right_document_id: number;
  groups: MatchGroupModel[];
  details: MatchDetailModel[];
}

export interface PipelineOptions {
  lexical_shingle_size?: number;
  lexical_threshold?: number;
  semantic_threshold?: number;
  final_threshold?: number;
  top_k?: number;
  max_candidates?: number;
  cross_encoder_top_k?: number;
  cross_encoder_threshold?: number;
}

export type ChunkGranularity = 'sentence' | 'paragraph';

// API methods
export const plagiarismApi = {
  async checkHealth(): Promise<{ status: string }> {
    const response = await apiClient.get('/api/v1/health');
    return response.data;
  },

  async uploadDocuments(projectId: number, files: File[], source?: string): Promise<DocumentSummary[]> {
    if (!files.length) {
      return [];
    }
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('project_id', String(projectId));
    if (source) {
      formData.append('source', source);
    }
    const response = await apiClient.post<{ items: DocumentSummary[] }>(
      '/api/v1/documents',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data.items;
  },

  async listDocuments(options?: { status?: DocumentStatus; projectId?: number }): Promise<DocumentSummary[]> {
    const params: Record<string, unknown> = {};
    if (options?.status) {
      params.status = options.status;
    }
    if (options?.projectId) {
      params.project_id = options.projectId;
    }
    const response = await apiClient.get<{ items: DocumentSummary[] }>('/api/v1/documents', {
      params: Object.keys(params).length ? params : undefined,
    });
    return response.data.items;
  },

  async getDocument(documentId: number): Promise<DocumentDetail> {
    const response = await apiClient.get<{ document: DocumentSummary; processed_text?: string | null; metadata?: Record<string, unknown> | null }>(
      `/api/v1/documents/${documentId}`
    );
    const { document, processed_text, metadata } = response.data;
    return {
      ...document,
      processed_text,
      metadata,
    };
  },

  async deleteDocument(documentId: number): Promise<void> {
    await apiClient.delete(`/api/v1/documents/${documentId}`);
  },

  async createCompareJob(projectId: number, name?: string | null, config?: Record<string, unknown> | null): Promise<CompareJobSummary> {
    const response = await apiClient.post<CompareJobSummary>('/api/v1/compare-jobs', {
      project_id: projectId,
      name: name ?? null,
      config: config ?? null,
    });
    return response.data;
  },

  async listCompareJobs(projectId?: number): Promise<CompareJobSummary[]> {
    const response = await apiClient.get<{ items: CompareJobSummary[] }>('/api/v1/compare-jobs', {
      params: projectId ? { project_id: projectId } : undefined,
    });
    return response.data.items;
  },

  async createPairs(
    jobId: number,
    pairs: Array<{ left_document_id: number; right_document_id: number }>,
    options?: { execute?: boolean; pipeline?: PipelineOptions; granularity?: ChunkGranularity }
  ): Promise<ComparePairSummary[]> {
    const payload = {
      pairs,
      execute: options?.execute ?? true,
      pipeline: options?.pipeline,
      granularity: options?.granularity ?? 'paragraph',
    };
    const response = await apiClient.post<{ items: ComparePairSummary[] }>(
      `/api/v1/compare-jobs/${jobId}/pairs`,
      payload
    );
    return response.data.items;
  },

  async deletePair(pairId: number): Promise<void> {
    await apiClient.delete(`/api/v1/compare-jobs/pairs/${pairId}`);
  },

  async deleteJob(jobId: number): Promise<void> {
    await apiClient.delete(`/api/v1/compare-jobs/${jobId}`);
  },

  async listProjects(): Promise<ProjectSummary[]> {
    const response = await apiClient.get<{ items: ProjectSummary[] }>('/api/v1/projects');
    return response.data.items;
  },

  async createProject(payload: { name?: string | null; description?: string | null }): Promise<ProjectSummary> {
    const response = await apiClient.post<ProjectSummary>('/api/v1/projects', payload);
    return response.data;
  },

  async getProject(projectId: number): Promise<ProjectSummary> {
    const response = await apiClient.get<ProjectSummary>(`/api/v1/projects/${projectId}`);
    return response.data;
  },

  async listProjectJobs(projectId: number): Promise<CompareJobSummary[]> {
    const response = await apiClient.get<{ items: CompareJobSummary[] }>(`/api/v1/projects/${projectId}/jobs`);
    return response.data.items;
  },

  async runProjectComparisons(projectId: number): Promise<CompareJobSummary> {
    const response = await apiClient.post<CompareJobSummary>(
      `/api/v1/projects/${projectId}/run-comparisons`
    );
    return response.data;
  },

  async listPairs(jobId: number): Promise<ComparePairSummary[]> {
    const response = await apiClient.get<{ items: ComparePairSummary[] }>(`/api/v1/compare-jobs/${jobId}/pairs`);
    return response.data.items;
  },

  async getPairReport(pairId: number): Promise<PairReport> {
    const response = await apiClient.get<PairReport>(`/api/v1/compare-jobs/pairs/${pairId}`);
    return response.data;
  },
};

apiClient.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.data);
      throw new Error(error.response.data.detail || 'API request failed');
    }
    if (error.request) {
      console.error('Network Error:', error.request);
      throw new Error('Network error - please check if the server is running');
    }
    console.error('Request Error:', error.message);
    throw new Error(error.message);
  }
);

export default apiClient;
