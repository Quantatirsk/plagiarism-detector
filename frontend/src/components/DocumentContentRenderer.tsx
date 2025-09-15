import React, { useMemo, useCallback, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import type { HighlightItem } from '../hooks/useDualDocumentComparison';

interface HighlightSegment {
  text: string;
  isMatch: boolean;
  highlightId?: string;
  similarity_score?: number;
  level?: 'very-high' | 'high' | 'medium' | 'low';
  startIndex: number;
  endIndex: number;
}

interface DocumentContentRendererProps {
  content: string;
  highlights: HighlightItem[];
  documentNumber: 1 | 2;
  onHighlightClick: (highlightId: string) => void;
  registerHighlightElement: (highlightId: string, element: HTMLElement) => void;
  className?: string;
}

interface ContentAnalysis {
  isMarkdown: boolean;
  confidence: number;
  reasons: string[];
}

const DEBUG = false;

export const DocumentContentRenderer: React.FC<DocumentContentRendererProps> = ({
  content,
  highlights,
  documentNumber,
  onHighlightClick,
  registerHighlightElement,
  className = '',
}) => {
  // 容器引用（用于事件委托与元素注册）
  const markdownContainerRef = useRef<HTMLDivElement | null>(null);
  /**
   * 智能检测内容格式
   */
  const analyzeContent = useCallback((text: string): ContentAnalysis => {
    if (!text) {
      return { isMarkdown: false, confidence: 0, reasons: ['Empty content'] };
    }

    const reasons: string[] = [];
    let markdownScore = 0;

    // 检测 Markdown 特征
    const markdownPatterns = [
      { pattern: /^#{1,6}\s+.+$/m, weight: 3, name: 'Headers' },
      { pattern: /\*\*[^*]+\*\*/g, weight: 2, name: 'Bold text' },
      { pattern: /\*[^*]+\*/g, weight: 1.5, name: 'Italic text' },
      { pattern: /```[\s\S]*?```/g, weight: 3, name: 'Code blocks' },
      { pattern: /`[^`]+`/g, weight: 1, name: 'Inline code' },
      { pattern: /^\s*[-*+]\s+/m, weight: 2, name: 'Unordered lists' },
      { pattern: /^\s*\d+\.\s+/m, weight: 2, name: 'Ordered lists' },
      { pattern: /\[([^\]]+)\]\(([^)]+)\)/g, weight: 2, name: 'Links' },
      { pattern: /!\[([^\]]*)\]\(([^)]+)\)/g, weight: 2, name: 'Images' },
      { pattern: /^>\s+/m, weight: 2, name: 'Blockquotes' },
      { pattern: /^\|.+\|$/m, weight: 2, name: 'Tables' },
      { pattern: /---+/g, weight: 1.5, name: 'Horizontal rules' },
    ];

    markdownPatterns.forEach(({ pattern, weight, name }) => {
      const matches = text.match(pattern);
      if (matches) {
        markdownScore += weight * Math.min(matches.length, 5); // Cap influence per pattern
        reasons.push(`${name}: ${matches.length} occurrences`);
      }
    });

    // 文本长度因子
    const textLength = text.length;
    const normalizedScore = markdownScore / Math.max(textLength / 1000, 1);

    // 计算置信度
    const confidence = Math.min(normalizedScore * 10, 100) / 100;
    const isMarkdown = confidence > 0.3;

    if (!isMarkdown) {
      reasons.push('Plain text format detected');
    }

    return { isMarkdown, confidence, reasons };
  }, []);


  /**
   * 处理HighlightItem数据以进行高亮显示
   */
  const processHighlightSegments = useCallback((text: string, documentNumber: 1 | 2): HighlightSegment[] => {
    if (!text || !highlights || highlights.length === 0) {
      return [{ text, isMatch: false, startIndex: 0, endIndex: text.length }];
    }

    // 过滤对应文档的高亮项
    const documentHighlights = highlights.filter(highlight => highlight.documentNumber === documentNumber);
    if (DEBUG) {
      console.log(`🔍 [DEBUG] 文档${documentNumber} 高亮处理: ${documentHighlights.length} 个高亮`);
    }

    // 按起始位置排序
    const sortedHighlights = [...documentHighlights].sort((a, b) => a.startIndex - b.startIndex);

    // 优化重叠检测：只跳过真正严重重叠的高亮（重叠度>80%）
    const nonOverlappingHighlights: HighlightItem[] = [];
    const skippedHighlights: HighlightItem[] = [];

    sortedHighlights.forEach((highlight) => {
      let shouldSkip = false;
      let overlapReason = '';

      for (const existing of nonOverlappingHighlights) {
        // 计算重叠区域
        const overlapStart = Math.max(highlight.startIndex, existing.startIndex);
        const overlapEnd = Math.min(highlight.endIndex, existing.endIndex);

        if (overlapStart < overlapEnd) {
          const overlapLength = overlapEnd - overlapStart;
          const highlightLength = highlight.endIndex - highlight.startIndex;
          const existingLength = existing.endIndex - existing.startIndex;

          // 计算重叠百分比
          const overlapPercent = Math.max(
            overlapLength / highlightLength,
            overlapLength / existingLength
          );

          // 只跳过重叠度超过80%的高亮
          if (overlapPercent > 0.8) {
            shouldSkip = true;
            overlapReason = `与${existing.id}重叠${(overlapPercent*100).toFixed(1)}%`;
            break;
          }
        }
      }

      if (!shouldSkip) {
        nonOverlappingHighlights.push(highlight);
      } else {
        skippedHighlights.push(highlight);
        if (DEBUG) {
          console.warn(`⚠️ [DEBUG] 跳过高重叠高亮: ${highlight.id} (文档${documentNumber})`);
          console.warn(`  原因: ${overlapReason}`);
          console.warn(`  位置: [${highlight.startIndex}-${highlight.endIndex}]`);
          console.warn(`  相似度: ${(highlight.similarity_score*100).toFixed(2)}%`);
        }
      }
    });

    if (DEBUG && skippedHighlights.length > 0) {
      console.warn(`⚠️ [DEBUG] 文档${documentNumber} 跳过 ${skippedHighlights.length} 个高重叠高亮，这些将无法跳转!`);
    }

    const segments: HighlightSegment[] = [];
    let lastIndex = 0;

    nonOverlappingHighlights.forEach((highlight) => {
      // 添加高亮前的文本
      if (highlight.startIndex > lastIndex) {
        segments.push({
          text: text.slice(lastIndex, highlight.startIndex),
          isMatch: false,
          startIndex: lastIndex,
          endIndex: highlight.startIndex,
        });
      }

      // 添加高亮匹配 - 使用文档中的实际文本
      const actualText = text.slice(highlight.startIndex, highlight.endIndex) || highlight.text;
      segments.push({
        text: actualText,
        isMatch: true,
        highlightId: highlight.id,
        similarity_score: highlight.similarity_score,
        level: highlight.level,
        startIndex: highlight.startIndex,
        endIndex: highlight.endIndex,
      });

      lastIndex = highlight.endIndex;
    });

    // 添加剩余文本
    if (lastIndex < text.length) {
      segments.push({
        text: text.slice(lastIndex),
        isMatch: false,
        startIndex: lastIndex,
        endIndex: text.length,
      });
    }

    return segments;
  }, [highlights]);

  /**
   * 获取相似度等级的 CSS 类名
   */
  const getSimilarityClasses = useCallback((level: string) => {
    const baseClasses = 'similarity-highlight transition-all duration-200 cursor-pointer';
    switch (level) {
      case 'very-high':
        return `${baseClasses} similarity-very-high`;
      case 'high':
        return `${baseClasses} similarity-high`;
      case 'medium':
        return `${baseClasses} similarity-medium`;
      case 'low':
        return `${baseClasses} similarity-low`;
      default:
        return baseClasses;
    }
  }, []);

  /**
   * 获取高亮上下文信息
   */
  const getHighlightContext = useCallback((similarity_score: number, level: string, documentNumber: 1 | 2) => {
    const score = `${Math.round(similarity_score * 100)}%`;
    // 来源应该是另一侧文档，而不是当前文档
    const source = `文档 ${documentNumber === 1 ? 'B' : 'A'}`;
    // 中文等级名称
    const levelLabel = level === 'very-high'
      ? '高度相似'
      : level === 'high'
        ? '中高相似'
        : level === 'medium'
          ? '中度相似'
          : '低度相似';
    return { score, level: levelLabel, source };
  }, []);


  /**
   * 内容分析结果
   */
  const contentAnalysis = useMemo(() => analyzeContent(content), [content, analyzeContent]);

  /**
   * 处理后的文本片段
   */
  const textSegments = useMemo(() => {
    return processHighlightSegments(content, documentNumber);
  }, [content, documentNumber, processHighlightSegments]);

  // 安全转义HTML内容
  const escapeHtml = useCallback((s: string) => (
    s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
  ), []);

  // rehype-sanitize 白名单，允许我们注入的少量安全属性
  const sanitizeSchema = useMemo(() => ({
    ...defaultSchema,
    attributes: {
      ...(defaultSchema.attributes || {}),
      span: [
        ...(defaultSchema.attributes?.span || []),
        'className',
        [ 'data-highlight-id' ],
        'role',
        'tabIndex',
      ],
      div: [
        ...(defaultSchema.attributes?.div || []),
        'className'
      ],
    },
    clobberPrefix: 'md-',
  }), []);

  /**
   * 渲染 Markdown 内容（带高亮）
   */
  const renderMarkdownContent = useCallback(() => {
    // 对于 Markdown 内容，我们需要将高亮片段嵌入到 markdown 中
    let highlightedMarkdown = content;

    // 从后往前替换，避免位置偏移问题
    const sortedSegments = [...textSegments]
      .filter(seg => seg.isMatch && seg.highlightId)
      .sort((a, b) => b.startIndex - a.startIndex);

    sortedSegments.forEach((segment) => {
      if (segment.isMatch && segment.highlightId) {
        const context = getHighlightContext(
          segment.similarity_score || 0,
          segment.level || 'low',
          documentNumber
        );
        const highlightClass = getSimilarityClasses(segment.level || 'low');

        // Escape user-provided text to avoid HTML injection
        const actualText = escapeHtml(segment.text);
        const highlightedText = `<span
          class="${highlightClass}"
          data-highlight-id="${segment.highlightId}"
          tabindex="0"
          role="button"
        >
          ${actualText}
          <div class="similarity-tooltip">
            <div class="text-xs space-y-1">
              <div class="font-semibold">${context.level}</div>
              <div>相似度: ${context.score}</div>
              <div>来源: ${context.source}</div>
              <div class="text-gray-300 text-xs">点击跳转到对应句子</div>
            </div>
          </div>
        </span>`;

        highlightedMarkdown = highlightedMarkdown.slice(0, segment.startIndex) +
          highlightedText +
          highlightedMarkdown.slice(segment.endIndex);
      }
    });

    return (
      <div className="markdown-content" ref={markdownContainerRef}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight, rehypeRaw, [rehypeSanitize, sanitizeSchema]]}
        >
          {highlightedMarkdown}
        </ReactMarkdown>
      </div>
    );
  }, [content, textSegments, getHighlightContext, getSimilarityClasses, documentNumber, sanitizeSchema, escapeHtml]);

  /**
   * 渲染纯文本内容（带高亮）
   */
  const renderPlainTextContent = useCallback(() => {
    // Debug: Track how many segments are being rendered
    const highlightSegments = textSegments.filter(seg => seg.isMatch && seg.highlightId);
    console.log(`🎨 [DEBUG] 渲染文档${documentNumber}: ${textSegments.length}个片段, 其中${highlightSegments.length}个高亮`);


    return (
      <div className="plain-text-content">
        {textSegments.map((segment, index) => {
          if (!segment.isMatch || !segment.highlightId) {
            return (
              <span key={`segment-${index}`}>
                {segment.text}
              </span>
            );
          }

          const context = getHighlightContext(
            segment.similarity_score || 0,
            segment.level || 'low',
            documentNumber
          );

          return (
            <span
              key={segment.highlightId}
              data-highlight-id={segment.highlightId}
              className={getSimilarityClasses(segment.level || 'low')}
              onMouseEnter={() => onHighlightClick(segment.highlightId!)}
              onMouseLeave={() => {
                // Clear partner hover state when mouse leaves
                const allElements = document.querySelectorAll('.partner-hover');
                allElements.forEach(el => el.classList.remove('partner-hover'));
              }}
              onClick={() => onHighlightClick(segment.highlightId!)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onHighlightClick(segment.highlightId!);
                }
              }}
              ref={(element) => {
                if (element && segment.highlightId) {
                  if (DEBUG) {
                    console.log(`📝 [DEBUG] 注册元素: ${segment.highlightId} (文档${documentNumber})`);
                  }
                  registerHighlightElement(segment.highlightId, element);
                } else {
                  if (DEBUG) console.warn(`⚠️ [DEBUG] 元素注册失败: element=${!!element}, highlightId=${segment.highlightId}`);
                }
              }}
            >
              {segment.text}

              {/* Tooltip */}
              <div className="similarity-tooltip">
                <div className="text-xs space-y-1">
                  <div className="font-semibold">{context.level}</div>
                  <div>相似度: {context.score}</div>
                  <div>来源: {context.source}</div>
                  <div className="text-gray-300 text-xs">
                    悬停定位对应内容
                  </div>
                </div>
              </div>
            </span>
          );
        })}
      </div>
    );
  }, [textSegments, getHighlightContext, getSimilarityClasses, onHighlightClick, registerHighlightElement, documentNumber]);

  // Debug: Track final rendering completion（默认关闭）
  React.useEffect(() => {
    if (!DEBUG) return;
    const highlightSegments = textSegments.filter(seg => seg.isMatch && seg.highlightId);
    console.log(`🏁 [DEBUG] 文档${documentNumber}渲染完成: ${highlightSegments.length}个高亮片段`);
  }, [textSegments, documentNumber]);

  /**
   * 设置全局高亮点击和悬停处理器（用于 Markdown 中的 HTML）
   */
  // 事件委托：捕获悬停和点击事件
  useEffect(() => {
    const container = markdownContainerRef.current;
    if (!container) return;

    const handleMouseEnter = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const el = target.closest('[data-highlight-id]') as HTMLElement | null;
      const id = el?.getAttribute('data-highlight-id');
      if (id) {
        onHighlightClick(id);
      }
    };

    const handleMouseLeave = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const el = target.closest('[data-highlight-id]') as HTMLElement | null;
      if (el) {
        // Clear partner hover state when mouse leaves
        const allElements = document.querySelectorAll('.partner-hover');
        allElements.forEach(el => el.classList.remove('partner-hover'));
      }
    };

    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const el = target.closest('[data-highlight-id]') as HTMLElement | null;
      const id = el?.getAttribute('data-highlight-id');
      if (id) {
        onHighlightClick(id);
      }
    };

    container.addEventListener('mouseover', handleMouseEnter);
    container.addEventListener('mouseout', handleMouseLeave);
    container.addEventListener('click', handleClick);
    return () => {
      container.removeEventListener('mouseover', handleMouseEnter);
      container.removeEventListener('mouseout', handleMouseLeave);
      container.removeEventListener('click', handleClick);
    };
  }, [onHighlightClick]);

  // 注册渲染后的高亮元素，便于跳转定位
  useEffect(() => {
    const container = markdownContainerRef.current;
    if (!container) return;
    const nodes = container.querySelectorAll('[data-highlight-id]');
    nodes.forEach((node) => {
      const el = node as HTMLElement;
      const id = el.getAttribute('data-highlight-id');
      if (id) registerHighlightElement(id, el);
    });
  }, [textSegments, registerHighlightElement]);

  // 如果没有内容，显示占位符
  if (!content) {
    return (
      <div className={`text-center py-8 text-gray-500 ${className}`}>
        <p>文档内容为空</p>
      </div>
    );
  }

  return (
    <div className={`document-content-renderer ${className}`}>
      {/* 开发调试信息已禁用，如需启用请取消注释 */}
      {false && process.env.NODE_ENV === 'development' && (
        <div className="text-xs text-gray-400 mb-2 font-mono">
          格式: {contentAnalysis.isMarkdown ? 'Markdown' : 'Plain Text'}
          (置信度: {Math.round(contentAnalysis.confidence * 100)}%)
        </div>
      )}

      {/* 根据内容类型渲染 */}
      {contentAnalysis.isMarkdown ? renderMarkdownContent() : renderPlainTextContent()}
    </div>
  );
};

export default DocumentContentRenderer;
