import React, { useMemo, useCallback } from 'react';

// 独立的最小匹配类型，避免依赖不存在的模块
export interface ProcessedMatch {
  id: string;
  query_text: string;
  matched_text: string;
  similarity_score: number;
  document_id: string;
  position: number;
  level: 'very-high' | 'high' | 'medium' | 'low';
}

interface SimilarityHighlighterProps {
  sourceText: string;
  matches: ProcessedMatch[];
  onMatchClick?: (matchId: string) => void;
  className?: string;
}

interface HighlightSegment {
  text: string;
  isMatch: boolean;
  match?: ProcessedMatch;
  startIndex: number;
  endIndex: number;
}

export const SimilarityHighlighter: React.FC<SimilarityHighlighterProps> = ({
  sourceText,
  matches,
  onMatchClick,
  className = '',
}) => {
  /**
   * Process text into segments for highlighting
   */
  const highlightSegments = useMemo(() => {
    if (!sourceText || matches.length === 0) {
      return [{ text: sourceText, isMatch: false, startIndex: 0, endIndex: sourceText.length }];
    }

    // Sort matches by start index to process them in order
    const sortedMatches = [...matches].sort((a, b) => a.startIndex - b.startIndex);

    // Remove overlapping matches - keep the first one in case of overlap
    const nonOverlappingMatches: ProcessedMatch[] = [];
    sortedMatches.forEach((match) => {
      const hasOverlap = nonOverlappingMatches.some((existing) =>
        (match.startIndex < existing.endIndex && match.endIndex > existing.startIndex)
      );
      if (!hasOverlap) {
        nonOverlappingMatches.push(match);
      }
    });

    const segments: HighlightSegment[] = [];
    let lastIndex = 0;

    nonOverlappingMatches.forEach((match) => {
      // Add text before the match
      if (match.startIndex > lastIndex) {
        segments.push({
          text: sourceText.slice(lastIndex, match.startIndex),
          isMatch: false,
          startIndex: lastIndex,
          endIndex: match.startIndex,
        });
      }

      // Add the highlighted match
      segments.push({
        text: match.query_text,
        isMatch: true,
        match,
        startIndex: match.startIndex,
        endIndex: match.endIndex,
      });

      lastIndex = match.endIndex;
    });

    // Add remaining text after the last match
    if (lastIndex < sourceText.length) {
      segments.push({
        text: sourceText.slice(lastIndex),
        isMatch: false,
        startIndex: lastIndex,
        endIndex: sourceText.length,
      });
    }

    return segments;
  }, [sourceText, matches]);

  /**
   * Handle match click
   */
  const handleMatchClick = useCallback((match: ProcessedMatch) => {
    onMatchClick?.(match.id);
  }, [onMatchClick]);

  /**
   * Get CSS classes for similarity level
   */
  const getSimilarityClasses = useCallback((level: ProcessedMatch['level']) => {
    const baseClasses = 'similarity-highlight';

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
   * Format similarity score for display
   */
  const formatScore = useCallback((score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  }, []);

  /**
   * Get match context information
   */
  const getMatchContext = useCallback((match: ProcessedMatch) => {
    const levelLabel = match.level === 'very-high'
      ? '高度相似'
      : match.level === 'high'
        ? '中高相似'
        : match.level === 'medium'
          ? '中度相似'
          : '低度相似';
    return {
      level: levelLabel,
      score: formatScore(match.similarity_score),
      source: `文档 ${match.document_id}`,
      position: `位置 ${match.position}`,
    };
  }, [formatScore]);

  if (!sourceText) {
    return (
      <div className={`text-gray-500 text-center py-8 ${className}`}>
        <p>暂无内容可显示</p>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Summary Header */}
      {matches.length > 0 && (
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <h4 className="font-medium text-gray-900 mb-2">相似度分析摘要</h4>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-600 border border-red-600 rounded"></div>
                <span>高度相似 ({matches.filter(m => m.level === 'very-high').length})</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-fuchsia-600 border border-fuchsia-600 rounded"></div>
                <span>中高相似 ({matches.filter(m => m.level === 'high').length})</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-orange-500 border border-orange-500 rounded"></div>
                <span>中度相似 ({matches.filter(m => m.level === 'medium').length})</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-yellow-200 border border-yellow-200 rounded"></div>
                <span>低度相似 ({matches.filter(m => m.level === 'low').length})</span>
              </div>
            </div>
          </div>
      )}

      {/* Highlighted Text */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="text-sm leading-relaxed font-mono whitespace-pre-wrap">
          {highlightSegments.map((segment, index) => {
            if (!segment.isMatch || !segment.match) {
              return (
                <span key={`segment-${index}`} className="text-gray-800">
                  {segment.text}
                </span>
              );
            }

            const context = getMatchContext(segment.match);

            return (
              <span
                key={`match-${segment.match.id}`}
                data-match-id={segment.match.id}
                className={getSimilarityClasses(segment.match.level)}
                onClick={() => handleMatchClick(segment.match!)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleMatchClick(segment.match!);
                  }
                }}
              >
                {segment.text}

                {/* Tooltip */}
                <div className="similarity-tooltip">
                  <div className="text-xs space-y-1">
                    <div className="font-semibold">
                      {context.level === 'high' ? '高相似度' :
                       context.level === 'medium' ? '中相似度' : '低相似度'}
                    </div>
                    <div>相似度: {context.score}</div>
                    <div>来源: {context.source}</div>
                    <div className="text-gray-300 text-xs">
                      点击定位到对应位置
                    </div>
                  </div>
                </div>
              </span>
            );
          })}
        </div>
      </div>

      {/* Match Details List */}
      {matches.length > 0 && (
        <div className="space-y-3">
          <h4 className="font-medium text-gray-900">详细匹配信息</h4>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {matches.map((match) => {
              const context = getMatchContext(match);

              return (
                <div
                  key={match.id}
                  className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 cursor-pointer transition-colors"
                  onClick={() => handleMatchClick(match)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        match.level === 'high' ? 'bg-red-500' :
                        match.level === 'medium' ? 'bg-orange-500' : 'bg-yellow-500'
                      }`}></div>
                      <span className="text-sm font-medium">
                        相似度 {context.score}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">{context.source}</span>
                  </div>

                  <div className="text-sm space-y-1">
                    <div>
                      <span className="text-gray-600">原文:</span>
                      <span className="ml-2 text-gray-800 font-mono">
                        "{match.query_text.slice(0, 100)}..."
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">匹配:</span>
                      <span className="ml-2 text-gray-800 font-mono">
                        "{match.matched_text.slice(0, 100)}..."
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
