/**
 * Configuration utilities and constants for the frontend application
 */

/**
 * Get the full API URL by combining base URL with path
 * Handles both development proxy and production environments
 */
export const getApiUrl = (path: string): string => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;

  // In development, the proxy handles /api routes, so we don't need the base URL
  if (!baseUrl && path.startsWith('api/')) {
    return `/${cleanPath}`;
  }

  return baseUrl ? `${baseUrl}/${cleanPath}` : `/${cleanPath}`;
};

/**
 * Application configuration
 */
export const config = {
  api: {
    baseUrl: import.meta.env.VITE_API_BASE_URL || '',
    timeout: 30000,
  },
  llm: {
    // Default LLM model for report generation
    defaultModel: import.meta.env.VITE_LLM_MODEL || 'google/gemini-2.5-flash-lite',
    // Maximum tokens for LLM responses
    maxTokens: 4096,
    // Temperature for LLM generation (0-1, higher = more creative)
    temperature: 0.7,
    // Streaming response chunk size
    streamChunkSize: 1024,
  },
  reports: {
    // Maximum number of matches to include in detailed report
    maxDetailedMatches: 20,
    // Minimum similarity score to include in report
    minSimilarityThreshold: 0.6,
    // Report generation timeout (ms)
    generationTimeout: 60000,
  },
};