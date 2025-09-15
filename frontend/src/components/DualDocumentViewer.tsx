import React, { useMemo, useCallback, useEffect, useRef } from 'react';
import type { HighlightItem } from '../hooks/useDualDocumentComparison';
import type { DocumentInfo } from '../api/plagiarismApi';
import { DocumentContentRenderer } from './DocumentContentRenderer';

interface DualDocumentViewerProps {
  document1Info: DocumentInfo;
  document2Info: DocumentInfo;
  document1Content: string;
  document2Content: string;
  documentAHighlights: HighlightItem[];
  documentBHighlights: HighlightItem[];
  onHighlightClick: (highlightId: string) => void;
  registerHighlightElement: (highlightId: string, element: HTMLElement) => void;
  className?: string;
}


export const DualDocumentViewer: React.FC<DualDocumentViewerProps> = ({
  document1Info,
  document2Info,
  document1Content,
  document2Content,
  documentAHighlights,
  documentBHighlights,
  onHighlightClick,
  registerHighlightElement,
  className = '',
}) => {
  // 简单的ref管理
  const leftPanelRef = useRef<HTMLDivElement>(null);
  const rightPanelRef = useRef<HTMLDivElement>(null);

  /**
   * Handle highlight click
   */
  const handleHighlightClick = useCallback((highlightId: string) => {
    onHighlightClick(highlightId);
  }, [onHighlightClick]);


  // Debug: Track highlights being passed to renderers
  React.useEffect(() => {
    console.log(`🎯 [DEBUG] DualDocumentViewer render: 文档A有${documentAHighlights.length}个高亮, 文档B有${documentBHighlights.length}个高亮`);

    const targetA = documentAHighlights.find(h => h.id.includes('117'));
    const targetB = documentBHighlights.find(h => h.id.includes('117'));

    if (targetA) console.log(`🎯 [DEBUG] 文档A找到117号: ${targetA.id}`);
    if (targetB) console.log(`🎯 [DEBUG] 文档B找到117号: ${targetB.id}`);

    if (!targetB && documentBHighlights.length > 100) {
      console.warn(`⚠️ [DEBUG] 文档B没有117号高亮，但有${documentBHighlights.length}个高亮`);
      console.log('前5个B高亮:', documentBHighlights.slice(0, 5).map(h => h.id));
      console.log('后5个B高亮:', documentBHighlights.slice(-5).map(h => h.id));
    }
  }, [documentAHighlights, documentBHighlights]);

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Summary Header */}
      {(documentAHighlights.length > 0 || documentBHighlights.length > 0) && (
        <div className="flex-none bg-gray-50 border-b border-gray-200 p-4">
          <h4 className="font-medium text-gray-900 mb-2">相似度分析摘要</h4>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-600 border border-red-600 rounded"></div>
              <span>高度相似 ({[...documentAHighlights, ...documentBHighlights].filter(h => h.level === 'very-high').length / 2})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-fuchsia-600 border border-fuchsia-600 rounded"></div>
              <span>中高相似 ({[...documentAHighlights, ...documentBHighlights].filter(h => h.level === 'high').length / 2})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-orange-500 border border-orange-500 rounded"></div>
              <span>中度相似 ({[...documentAHighlights, ...documentBHighlights].filter(h => h.level === 'medium').length / 2})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-yellow-200 border border-yellow-200 rounded"></div>
              <span>低度相似 ({[...documentAHighlights, ...documentBHighlights].filter(h => h.level === 'low').length / 2})</span>
            </div>
          </div>
        </div>
      )}

      {/* Dual Document Display */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Document A */}
        <div className="flex-1 flex flex-col border-r border-gray-200">
          <div className="flex-none bg-gray-50 px-4 py-2 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900">
                文档 A - {document1Info?.filename || '未选择文件'}
              </h3>
              <div className="text-sm text-gray-600">
                {((document1Info?.extension || '').replace(/^\./, '') || '未知格式').toUpperCase()} • {document1Info?.text_length || 0} 字符
              </div>
            </div>
          </div>

          <div
            ref={leftPanelRef}
            className="flex-1 overflow-auto p-4 relative"
          >
            <DocumentContentRenderer
              content={document1Content}
              highlights={documentAHighlights}
              documentNumber={1}
              onHighlightClick={handleHighlightClick}
              registerHighlightElement={registerHighlightElement}
            />
          </div>
        </div>

        {/* Right Panel - Document B */}
        <div className="flex-1 flex flex-col">
          <div className="flex-none bg-gray-50 px-4 py-2 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900">
                文档 B - {document2Info?.filename || '未选择文件'}
              </h3>
              <div className="text-sm text-gray-600">
                {((document2Info?.extension || '').replace(/^\./, '') || '未知格式').toUpperCase()} • {document2Info?.text_length || 0} 字符
              </div>
            </div>
          </div>

          <div
            ref={rightPanelRef}
            className="flex-1 overflow-auto p-4 relative"
          >
            <DocumentContentRenderer
              content={document2Content}
              highlights={documentBHighlights}
              documentNumber={2}
              onHighlightClick={handleHighlightClick}
              registerHighlightElement={registerHighlightElement}
            />
          </div>
        </div>
      </div>

    </div>
  );
};
