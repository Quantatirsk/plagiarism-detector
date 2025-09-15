import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { plagiarismApi, type ComparisonRequest, type ComparisonResult, type SimilarityMatch, type Granularity } from '../api/plagiarismApi';

export interface DualDocumentComparisonState {
  isLoading: boolean;
  result: ComparisonResult | null;
  error: string | null;
  progress: {
    stage: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
    message: string;
  };
}

// 简化的高亮项
export interface HighlightItem {
  id: string;
  text: string;
  startIndex: number;
  endIndex: number;
  level: 'very-high' | 'high' | 'medium' | 'low';
  similarity_score: number;
  documentNumber: 1 | 2;
  partnerHighlightId: string;
}

export interface DualDocumentComparisonHook {
  state: DualDocumentComparisonState;
  documentAHighlights: HighlightItem[];
  documentBHighlights: HighlightItem[];
  compareDocuments: (
    document1: File,
    document2: File,
    granularity: Granularity,
    threshold?: number,
    topKPerQuery?: number,
    maxTotalMatches?: number,
  ) => Promise<void>;
  reset: () => void;
  getSimilarityLevel: (score: number) => 'very-high' | 'high' | 'medium' | 'low';
  jumpToPartner: (highlightId: string) => void;
  registerHighlightElement: (highlightId: string, element: HTMLElement) => void;
}

/**
 * Hook for managing dual-document comparison workflow - 简化版
 */
export const useDualDocumentComparison = (): DualDocumentComparisonHook => {
  const [state, setState] = useState<DualDocumentComparisonState>({
    isLoading: false,
    result: null,
    error: null,
    progress: {
      stage: 'idle',
      message: '准备对比',
    },
  });

  /**
   * 相似度等级判断 - 简化版
   */
  const getSimilarityLevel = useCallback((score: number): 'very-high' | 'high' | 'medium' | 'low' => {
    // 高度相似: 0.90 - 1.00
    if (score >= 0.9) return 'very-high';
    // 中高相似: 0.85 - 0.90
    if (score >= 0.85) return 'high';
    // 中度相似: 0.80 - 0.85
    if (score >= 0.8) return 'medium';
    // 低度相似: 0.70 - 0.80
    if (score >= 0.7) return 'low';
    return 'low';
  }, []);

  // 索引驱动定位：从服务返回的 chunk spans 直接计算高亮位置
  const getSpanByIndex = (
    spans: Array<{ index: number; start: number; end: number }> | undefined,
    idx: number | undefined | null
  ): { start: number; end: number } => {
    if (!spans || idx === undefined || idx === null) return { start: 0, end: 0 };
    const span = spans.find(s => s.index === idx);
    if (!span) return { start: 0, end: 0 };
    return { start: span.start, end: span.end };
  };

  /**
   * 简化的匹配处理 - 直接转换为高亮项（一对一匹配）
   */
  const processMatches = useCallback((matches: SimilarityMatch[], document1Content: string, document2Content: string, spans1: Array<{ index: number; start: number; end: number }>, spans2: Array<{ index: number; start: number; end: number }>): {
    documentAHighlights: HighlightItem[];
    documentBHighlights: HighlightItem[];
  } => {
    const documentAHighlights: HighlightItem[] = [];
    const documentBHighlights: HighlightItem[] = [];

    // 由于后端现在保证一对一匹配，直接处理每个匹配对
    matches.forEach((match, index) => {
      const level = getSimilarityLevel(match.similarity_score);
      const highlightA_Id = `highlight-a-${index}`;
      const highlightB_Id = `highlight-b-${index}`;

      const posA = getSpanByIndex(spans1, match.query_index);
      const posB = getSpanByIndex(spans2, match.match_index);

      const highlightA: HighlightItem = {
        id: highlightA_Id,
        text: match.query_text,
        startIndex: posA.start,
        endIndex: posA.end,
        level,
        similarity_score: match.similarity_score,
        documentNumber: 1,
        partnerHighlightId: highlightB_Id,
      };

      const highlightB: HighlightItem = {
        id: highlightB_Id,
        text: match.matched_text,
        startIndex: posB.start,
        endIndex: posB.end,
        level,
        similarity_score: match.similarity_score,
        documentNumber: 2,
        partnerHighlightId: highlightA_Id,
      };

      documentAHighlights.push(highlightA);
      documentBHighlights.push(highlightB);
    });

    // console.debug(`生成高亮: 文档A(${documentAHighlights.length}) 文档B(${documentBHighlights.length})`);
    return { documentAHighlights, documentBHighlights };
  }, [getSimilarityLevel]);

  /**
   * 文件验证
   */
  const validateFiles = (document1: File, document2: File): string | null => {
    const supportedFormats = ['pdf', 'docx', 'doc', 'md', 'txt'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (document1.size > maxSize || document2.size > maxSize) {
      return '文件大小不能超过 10MB';
    }

    const getFileExtension = (filename: string) => {
      return filename.toLowerCase().split('.').pop() || '';
    };

    const ext1 = getFileExtension(document1.name);
    const ext2 = getFileExtension(document2.name);

    if (!supportedFormats.includes(ext1) || !supportedFormats.includes(ext2)) {
      return `仅支持以下格式: ${supportedFormats.join(', ')}`;
    }

    return null;
  };

  /**
   * 文档对比主函数
   */
  const compareDocuments = useCallback(async (
    document1: File,
    document2: File,
    granularity: Granularity = 'paragraph',
    threshold?: number,
    topKPerQuery?: number,
    maxTotalMatches?: number,
  ) => {
    const validationError = validateFiles(document1, document2);
    if (validationError) {
      setState(prev => ({
        ...prev,
        error: validationError,
        progress: { stage: 'error', message: validationError }
      }));
      return;
    }

    setState(prev => ({
      ...prev,
      isLoading: true,
      error: null,
      result: null,
      progress: { stage: 'uploading', message: '正在上传文档...' }
    }));

    try {
      const comparisonRequest: ComparisonRequest = {
        document1,
        document2,
        granularity,
        threshold,
        top_k_per_query: topKPerQuery,
        max_total_matches: maxTotalMatches,
      };

      setState(prev => ({
        ...prev,
        progress: { stage: 'processing', message: '正在分析文档相似度...' }
      }));

      const result = await plagiarismApi.compareDocuments(comparisonRequest);

      console.log('Comparison result:', {
        granularity: result.granularity,
        doc1_spans: result.document1_spans?.length,
        doc2_spans: result.document2_spans?.length,
        matches: result.matches.length
      });

      setState(prev => ({
        ...prev,
        isLoading: false,
        result,
        progress: { stage: 'completed', message: `对比完成，发现 ${result.matches.length} 个相似匹配` }
      }));

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '对比失败，请重试';

      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
        progress: { stage: 'error', message: errorMessage }
      }));
    }
  }, []);

  /**
   * 重置状态
   */
  const reset = useCallback(() => {
    setState({
      isLoading: false,
      result: null,
      error: null,
      progress: {
        stage: 'idle',
        message: '准备对比',
      },
    });
  }, []);

  // 跳转功能
  const highlightElementsRef = useRef<Map<string, HTMLElement>>(new Map());

  const registerHighlightElement = useCallback((highlightId: string, element: HTMLElement) => {
    highlightElementsRef.current.set(highlightId, element);
  }, []);

  const highlightDataRef = useRef<{
    documentAHighlights: HighlightItem[];
    documentBHighlights: HighlightItem[];
  }>({ documentAHighlights: [], documentBHighlights: [] });

  // Keep track of currently hovered element
  const currentHoveredRef = useRef<HTMLElement | null>(null);

  const jumpToPartner = useCallback((highlightId: string) => {
    const { documentAHighlights, documentBHighlights } = highlightDataRef.current;
    const allHighlights = [...documentAHighlights, ...documentBHighlights];
    const currentHighlight = allHighlights.find(h => h.id === highlightId);

    if (!currentHighlight?.partnerHighlightId) {
      return;
    }

    const targetElement = highlightElementsRef.current.get(currentHighlight.partnerHighlightId);

    // Remove hover state from previously hovered element
    if (currentHoveredRef.current && currentHoveredRef.current !== targetElement) {
      currentHoveredRef.current.classList.remove('partner-hover');
    }

    if (targetElement) {
      targetElement.scrollIntoView({
        behavior: 'instant' as ScrollBehavior,
        block: 'center',
        inline: 'nearest'
      });

      // Add hover state class
      targetElement.classList.add('partner-hover');
      currentHoveredRef.current = targetElement;

      // Also add flash effect for better visibility
      targetElement.classList.add('flash-highlight');
      setTimeout(() => {
        targetElement.classList.remove('flash-highlight');
      }, 1000);
    }
  }, []);

  // 处理匹配数据
  const highlightData = useMemo(() => {
    if (!state.result) {
      return { documentAHighlights: [], documentBHighlights: [] };
    }
    return processMatches(
      state.result.matches || [],
      state.result.document1_info?.content || '',
      state.result.document2_info?.content || '',
      state.result.document1_spans || [],
      state.result.document2_spans || []
    );
  }, [state.result, processMatches]);

  const { documentAHighlights, documentBHighlights } = highlightData;

  useEffect(() => {
    highlightDataRef.current = { documentAHighlights, documentBHighlights };
  }, [documentAHighlights, documentBHighlights]);

  // 监控元素注册状态
  useEffect(() => {
    // 可选调试：检查注册状态
  }, [documentAHighlights.length, documentBHighlights.length]);

  return {
    state,
    documentAHighlights,
    documentBHighlights,
    compareDocuments,
    reset,
    getSimilarityLevel,
    jumpToPartner,
    registerHighlightElement,
  };
};
