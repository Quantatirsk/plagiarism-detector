import React, { useState, useCallback, useMemo } from 'react';
import { FileText, Upload, Search, AlertTriangle, CheckCircle, Loader2, X } from 'lucide-react';
import { useDualDocumentComparison } from '../hooks/useDualDocumentComparison';
import { DualDocumentViewer } from './DualDocumentViewer';
// UI components will be used when implementing the interface
import type { Granularity } from '../api/plagiarismApi';

interface DualDocumentComparerProps {
  className?: string;
}

interface DocumentData {
  file: File | null;
  content: string;
  name: string;
}

export const DualDocumentComparer: React.FC<DualDocumentComparerProps> = ({ className = '' }) => {
  const [document1, setDocument1] = useState<DocumentData>({ file: null, content: '', name: '' });
  const [document2, setDocument2] = useState<DocumentData>({ file: null, content: '', name: '' });
  const [granularity, setGranularity] = useState<Granularity>('sentence');
  const [threshold, setThreshold] = useState(0.75);
  const [maxTotalMatches, setMaxTotalMatches] = useState(2000);

  // Dual document comparison hook - 简化版
  const {
    state: comparisonState,
    documentAHighlights,
    documentBHighlights,
    compareDocuments,
    reset: resetComparison,
    getSimilarityLevel,
    jumpToPartner,
    registerHighlightElement,
  } = useDualDocumentComparison();

  /**
   * Handle file upload
   */
  const handleFileUpload = useCallback((
    event: React.ChangeEvent<HTMLInputElement>,
    documentNumber: 1 | 2
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const supportedTypes = ['application/pdf', 'application/msword',
                          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                          'text/plain', 'text/markdown'];

    const supportedExtensions = ['.pdf', '.doc', '.docx', '.txt', '.md'];
    const fileExtension = file.name.toLowerCase().split('.').pop();

    if (!supportedTypes.includes(file.type) && !supportedExtensions.includes(`.${fileExtension}`)) {
      alert('仅支持 PDF、DOC、DOCX、TXT、MD 格式的文件');
      return;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      alert('文件大小不能超过 10MB');
      return;
    }

    const documentData: DocumentData = {
      file,
      content: '', // Will be extracted by backend
      name: file.name,
    };

    if (documentNumber === 1) {
      setDocument1(documentData);
    } else {
      setDocument2(documentData);
    }

    // Clear the input value to allow re-selecting the same file
    event.target.value = '';
  }, []);

  /**
   * Handle file removal
   */
  const handleRemoveFile = useCallback((documentNumber: 1 | 2) => {
    if (documentNumber === 1) {
      setDocument1({ file: null, content: '', name: '' });
    } else {
      setDocument2({ file: null, content: '', name: '' });
    }
  }, []);

  /**
   * Handle document comparison
   */
  const handleCompareDocuments = useCallback(async () => {
    if (!document1.file || !document2.file) {
      alert('请选择两个文档进行对比');
      return;
    }

    console.log('Starting comparison with granularity:', granularity);

    await compareDocuments(
      document1.file,
      document2.file,
      granularity,
      threshold,
      undefined, // topKPerQuery no longer used in one-to-one matching
      maxTotalMatches
    );
  }, [document1.file, document2.file, granularity, threshold, maxTotalMatches, compareDocuments]);


  /**
   * Handle highlight click
   */
  const handleHighlightClick = useCallback((highlightId: string) => {
    console.log('Highlight clicked:', highlightId);
    jumpToPartner(highlightId);
  }, [jumpToPartner]);

  /**
   * Generate statistics
   */
  const statistics = useMemo(() => {
    if (!comparisonState.result) return null;

    const { matches } = comparisonState.result;
    const allMatches = matches;

    const veryHighMatches = allMatches.filter(m => getSimilarityLevel(m.similarity_score) === 'very-high');
    const highMatches = allMatches.filter(m => getSimilarityLevel(m.similarity_score) === 'high');
    const mediumMatches = allMatches.filter(m => getSimilarityLevel(m.similarity_score) === 'medium');
    const lowMatches = allMatches.filter(m => getSimilarityLevel(m.similarity_score) === 'low');

    return {
      total: allMatches.length,
      veryHigh: veryHighMatches.length,
      high: highMatches.length,
      medium: mediumMatches.length,
      low: lowMatches.length,
      avgScore: allMatches.length > 0
        ? (allMatches.reduce((sum, m) => sum + m.similarity_score, 0) / allMatches.length).toFixed(3)
        : '0',
    };
  }, [comparisonState.result, getSimilarityLevel]);

  /**
   * File upload area component
   */
  const FileUploadArea: React.FC<{
    documentNumber: 1 | 2;
    document: DocumentData;
    title: string
  }> = ({ documentNumber, document, title }) => (
    <div className="flex-1 border border-gray-300 border-dashed rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
      <div className="space-y-4">
        <div className="flex items-center justify-center">
          <FileText className="w-12 h-12 text-gray-400" />
        </div>

        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>

          {document.file ? (
            <div className="space-y-3">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <FileText className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-900">
                      {document.name}
                    </span>
                  </div>
                  <button
                    onClick={() => handleRemoveFile(documentNumber)}
                    className="text-gray-400 hover:text-red-500"
                    title="移除文件"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="mt-1 text-xs text-blue-600">
                  {(document.file.size / 1024).toFixed(1)} KB
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-gray-600">
                支持 PDF、DOC、DOCX、TXT、MD 格式
              </p>
              <p className="text-xs text-gray-500">
                文件大小限制：10MB
              </p>
            </div>
          )}

          <label className="mt-4 cursor-pointer inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
            <Upload className="w-4 h-4 mr-2" />
            {document.file ? '更换文件' : '选择文件'}
            <input
              type="file"
              accept=".pdf,.doc,.docx,.txt,.md"
              onChange={(e) => handleFileUpload(e, documentNumber)}
              className="hidden"
            />
          </label>
        </div>
      </div>
    </div>
  );

  return (
    <div className={`h-full flex flex-col ${className}`}>
      {/* Header */}
      <div className="flex-none bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Search className="w-5 h-5 text-blue-600" />
            <h1 className="text-xl font-semibold text-gray-900">文档相似度对比</h1>
          </div>

          {/* Detection Controls */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">检测粒度:</label>
              <select
                value={granularity}
                onChange={(e) => setGranularity(e.target.value as Granularity)}
                className="text-sm border border-gray-300 rounded-md px-2 py-1"
                disabled={comparisonState.isLoading}
              >
                <option value="paragraph">段落</option>
                <option value="sentence">句子</option>
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">阈值:</label>
              <input
                type="range"
                min="0.6"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-24"
                disabled={comparisonState.isLoading}
              />
              <span className="text-sm text-gray-600 min-w-[3rem]">
                {(threshold * 100).toFixed(0)}%
              </span>
            </div>


            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">最大匹配数:</label>
              <input
                type="number"
                min={50}
                max={5000}
                value={maxTotalMatches}
                onChange={(e) => setMaxTotalMatches(Math.max(50, Math.min(5000, Number(e.target.value) || 50)))}
                className="w-24 border border-gray-300 rounded-md px-2 py-1"
                disabled={comparisonState.isLoading}
              />
            </div>

            <button
              onClick={handleCompareDocuments}
              disabled={comparisonState.isLoading || !document1.file || !document2.file}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {comparisonState.isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>对比中...</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>开始对比</span>
                </>
              )}
            </button>

            {comparisonState.result && (
              <button
                onClick={resetComparison}
                className="px-3 py-2 text-gray-600 hover:text-gray-800"
              >
                重置
              </button>
            )}
          </div>
        </div>

        {/* Progress and Statistics */}
        {comparisonState.progress.stage !== 'idle' && (
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {comparisonState.progress.stage === 'error' ? (
                <AlertTriangle className="w-4 h-4 text-red-500" />
              ) : comparisonState.progress.stage === 'completed' ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
              )}
              <span className="text-sm text-gray-700">
                {comparisonState.progress.message}
              </span>
            </div>

            {statistics && (
              <div className="flex items-center space-x-4 text-sm">
                <span className="text-gray-600">
                  总匹配: <span className="font-medium">{statistics.total}</span>
                </span>
                <span className="text-red-700">
                  高度相似: <span className="font-medium">{statistics.veryHigh}</span>
                </span>
                <span className="text-red-600">
                  中高相似: <span className="font-medium">{statistics.high}</span>
                </span>
                <span className="text-orange-600">
                  中度相似: <span className="font-medium">{statistics.medium}</span>
                </span>
                <span className="text-yellow-600">
                  低度相似: <span className="font-medium">{statistics.low}</span>
                </span>
                <span className="text-gray-600">
                  平均分: <span className="font-medium">{statistics.avgScore}</span>
                </span>
              </div>
            )}
          </div>
        )}

        {comparisonState.error && (
          <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-red-500" />
              <span className="text-sm text-red-700">{comparisonState.error}</span>
            </div>
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* File Upload Area */}
        {!comparisonState.result && (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="w-full max-w-4xl space-y-6">
              <div className="text-center space-y-2 mb-8">
                <h2 className="text-2xl font-semibold text-gray-900">选择要对比的文档</h2>
                <p className="text-gray-600">上传两个文档，系统将分析它们之间的相似度</p>
              </div>

              <div className="flex space-x-6">
                <FileUploadArea
                  documentNumber={1}
                  document={document1}
                  title="文档 A"
                />
                <FileUploadArea
                  documentNumber={2}
                  document={document2}
                  title="文档 B"
                />
              </div>
            </div>
          </div>
        )}

        {/* Comparison Results */}
        {comparisonState.result && (
          <DualDocumentViewer
            document1Info={comparisonState.result.document1_info}
            document2Info={comparisonState.result.document2_info}
            document1Content={comparisonState.result.document1_info?.content || ''}
            document2Content={comparisonState.result.document2_info?.content || ''}
            documentAHighlights={documentAHighlights}
            documentBHighlights={documentBHighlights}
            onHighlightClick={handleHighlightClick}
            registerHighlightElement={registerHighlightElement}
            className="flex-1"
          />
        )}
      </div>
    </div>
  );
};
