import axios from 'axios';

// API Base URL - configurable via Vite env (VITE_API_BASE_URL)
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || '';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL || '/',
  timeout: 30000, // 30 seconds for plagiarism detection
  headers: {
    'Content-Type': 'application/json',
  },
});

// 简化：只保留双文档对比相关的类型
export interface SimilarityMatch {
  query_text: string;
  matched_text: string;
  similarity_score: number;
  document_id: string; // 匹配到的文档ID（目标文档）
  query_document_id: string; // 查询来源文档ID（必填项）
  position: number;
  query_index?: number;
  match_index?: number;
}

export type Granularity = 'paragraph' | 'sentence';

export interface ComparisonRequest {
  document1: File;
  document2: File;
  granularity: Granularity;
  threshold?: number; // 可选；不提供则后端按粒度默认
  max_total_matches?: number; // 全局上限
}

export interface DocumentInfo {
  filename: string;
  extension: string;
  size: number;
  text_length: number;
  content?: string; // 文档解析后的纯文本内容
}

export interface ComparisonResult {
  task_id: string;
  status: string;
  granularity: Granularity;
  document1_info: DocumentInfo;
  document2_info: DocumentInfo;
  document1_spans: Array<{ index: number; start: number; end: number }>;
  document2_spans: Array<{ index: number; start: number; end: number }>;
  matches: SimilarityMatch[];
  processing_time: number;
  created_at: string;
}

// API methods - 简化：只保留双文档对比
export const plagiarismApi = {
  /**
   * Check API health
   */
  async checkHealth(): Promise<{ status: string }> {
    const response = await apiClient.get('/api/v1/health');
    return response.data;
  },

  /**
   * Compare two documents for plagiarism
   */
  async compareDocuments(request: ComparisonRequest): Promise<ComparisonResult> {
    console.log('Comparing documents with request:', {
      granularity: request.granularity,
      threshold: request.threshold,
      max_total_matches: request.max_total_matches
    });

    const formData = new FormData();
    formData.append('document1', request.document1);
    formData.append('document2', request.document2);
    formData.append('granularity', request.granularity);
    if (request.threshold) {
      formData.append('threshold', request.threshold.toString());
    }
    if (request.max_total_matches) {
      formData.append('max_total_matches', request.max_total_matches.toString());
    }

    const response = await apiClient.post<ComparisonResult>(
      '/api/v1/comparison/upload-and-compare',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds for dual document processing
      }
    );

    console.log('Server response granularity:', response.data.granularity);
    console.log('Number of matches:', response.data.matches.length);

    return response.data;
  },
};

// Request interceptor to add authentication if needed in the future
apiClient.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.data);
      throw new Error(error.response.data.detail || 'API request failed');
    } else if (error.request) {
      console.error('Network Error:', error.request);
      throw new Error('Network error - please check if the server is running');
    } else {
      console.error('Request Error:', error.message);
      throw new Error(error.message);
    }
  }
);

export default apiClient;
