/**
 * ComparePage - 全屏对比页面
 * 直接全屏渲染 PlanComparePage，避免嵌套滚动容器
 */

import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Spin } from 'antd';
import PlanComparePage from './PlanComparePage';
import { plagiarismApi, type PairReport, type DocumentDetail, type ComparePairSummary, type DocumentSummary } from '@/api/plagiarismApi';
import { designSystem } from '@/styles/DesignSystem';

export default function ComparePage() {
  const { pairId } = useParams<{ pairId: string }>();
  const navigate = useNavigate();

  const [report, setReport] = useState<PairReport | null>(null);
  const [leftDocument, setLeftDocument] = useState<DocumentDetail | null>(null);
  const [rightDocument, setRightDocument] = useState<DocumentDetail | null>(null);
  const [pairs, setPairs] = useState<ComparePairSummary[]>([]);
  const [documentLookup, setDocumentLookup] = useState<Record<number, DocumentSummary>>({});

  const [loading, setLoading] = useState(false);
  const [pairsLoading, setPairsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ==================== 数据加载 ====================

  // 加载 pair report 和 documents
  useEffect(() => {
    if (!pairId) return;

    const numericPairId = Number(pairId);
    if (isNaN(numericPairId)) {
      setError('无效的对比ID');
      return;
    }

    let mounted = true;
    setLoading(true);
    setError(null);

    const loadPairData = async () => {
      try {
        // 1. 获取 pair report
        const pairReport = await plagiarismApi.getPairReport(numericPairId);
        if (!mounted) return;

        // 2. 并行获取左右文档
        const [leftDoc, rightDoc] = await Promise.all([
          plagiarismApi.getDocument(pairReport.left_document_id),
          plagiarismApi.getDocument(pairReport.right_document_id),
        ]);
        if (!mounted) return;

        setReport(pairReport);
        setLeftDocument(leftDoc);
        setRightDocument(rightDoc);

        // 3. 加载同一任务的其他 pairs（用于切换）
        const jobId = pairReport.pair.job_id;
        if (jobId) {
          loadJobPairs(jobId);
        }
      } catch (err) {
        if (!mounted) return;
        console.error('Failed to load pair data:', err);
        setError((err as Error).message || '加载对比数据失败');
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    loadPairData();

    return () => {
      mounted = false;
    };
  }, [pairId]);

  // 加载同一任务的所有 pairs
  const loadJobPairs = async (jobId: number) => {
    setPairsLoading(true);
    try {
      const pairsList = await plagiarismApi.listPairs(jobId);
      setPairs(pairsList);

      // 批量获取文档信息
      const docIds = new Set<number>();
      for (const pair of pairsList) {
        docIds.add(pair.left_document_id);
        docIds.add(pair.right_document_id);
      }

      const lookup: Record<number, DocumentSummary> = {};
      await Promise.all(
        Array.from(docIds).map(async (docId) => {
          try {
            const doc = await plagiarismApi.getDocument(docId);
            lookup[docId] = doc;
          } catch (err) {
            console.error(`Failed to load document ${docId}:`, err);
          }
        })
      );

      setDocumentLookup(lookup);
    } catch (err) {
      console.error('Failed to load pairs:', err);
    } finally {
      setPairsLoading(false);
    }
  };

  // ==================== 交互函数 ====================

  const handleBack = () => {
    // 返回比对任务页
    navigate('/comparison/tasks');
  };

  const handleSwitchPair = (newPairId: number) => {
    navigate(`/comparison/results/${newPairId}`);
  };

  // ==================== 渲染 ====================

  if (loading || !report) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          width: '100vw',
          backgroundColor: designSystem.semantic.surface.base,
        }}
      >
        <Spin tip="正在加载对比数据..." size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          width: '100vw',
          gap: designSystem.spacing[2],
          backgroundColor: designSystem.semantic.surface.base,
        }}
      >
        <div style={{ color: designSystem.colors.error, fontSize: designSystem.typography.fontSize.lg }}>
          加载失败
        </div>
        <div style={{ color: designSystem.semantic.text.secondary }}>
          {error}
        </div>
      </div>
    );
  }

  // 提供全屏容器，确保 PageLayout 有明确的高度参考点
  // 这样内部的 [data-document-pane] 容器才能正确计算 overflow: auto
  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
      }}
    >
      <PlanComparePage
        report={report}
        leftDocument={leftDocument}
        rightDocument={rightDocument}
        pairs={pairs}
        pairsLoading={pairsLoading}
        pairsError={null}
        documentLookup={documentLookup}
        onSwitchPair={handleSwitchPair}
        onBack={handleBack}
        isTransitioning={loading}
      />
    </div>
  );
}
