/**
 * CompareTab - 对比结果预览
 * 显示当前任务的所有 pairs，点击进入全屏对比页
 */

import { useEffect, useState } from 'react';
import { Card, Row, Col, Spin, Empty, Progress, Tag } from 'antd';
import { EyeOutlined, FileTextOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { plagiarismApi, type ComparePairSummary, type DocumentSummary } from '@/api/plagiarismApi';
import { designSystem } from '@/styles/DesignSystem';

export default function CompareTab() {
  const navigate = useNavigate();
  const { selectedTaskId } = useWorkspaceStore();

  const [pairs, setPairs] = useState<ComparePairSummary[]>([]);
  const [documentLookup, setDocumentLookup] = useState<Record<number, DocumentSummary>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ==================== 数据加载 ====================
  useEffect(() => {
    if (!selectedTaskId) {
      setPairs([]);
      setDocumentLookup({});
      return;
    }

    let mounted = true;
    setLoading(true);
    setError(null);

    const loadPairs = async () => {
      try {
        const pairsList = await plagiarismApi.listPairs(selectedTaskId);
        if (!mounted) return;

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

        if (!mounted) return;
        setDocumentLookup(lookup);
      } catch (err) {
        if (!mounted) return;
        console.error('Failed to load pairs:', err);
        setError((err as Error).message || '加载对比列表失败');
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    loadPairs();

    return () => {
      mounted = false;
    };
  }, [selectedTaskId]);

  // ==================== 交互函数 ====================
  const handleViewPair = (pairId: number) => {
    navigate(`/compare/${pairId}`);
  };

  const getDocumentName = (docId: number) => {
    const doc = documentLookup[docId];
    return doc?.title || doc?.filename || `文档 #${docId}`;
  };

  const getMatchScore = (pair: ComparePairSummary) => {
    return pair.metrics?.similarity_score ?? 0;
  };

  const getStatusColor = (status: string) => {
    const statusColors: Record<string, string> = {
      completed: 'success',
      running: 'processing',
      failed: 'error',
      pending: 'default',
    };
    return statusColors[status] || 'default';
  };

  // ==================== 渲染 ====================

  if (!selectedTaskId) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: designSystem.semantic.text.tertiary,
        }}
      >
        请在任务列表中选择一个任务以查看对比结果
      </div>
    );
  }

  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
        }}
      >
        <Spin tip="正在加载对比结果..." size="large" />
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
          height: '100%',
          gap: designSystem.spacing[2],
        }}
      >
        <div style={{ color: designSystem.colors.error, fontSize: designSystem.typography.fontSize.base }}>
          加载失败
        </div>
        <div style={{ color: designSystem.semantic.text.secondary, fontSize: designSystem.typography.fontSize.sm }}>
          {error}
        </div>
      </div>
    );
  }

  if (pairs.length === 0) {
    return (
      <div style={{ padding: designSystem.spacing[6] }}>
        <Empty
          description="暂无对比结果"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      </div>
    );
  }

  return (
    <div
      style={{
        padding: designSystem.spacing[3],
        height: '100%',
        overflow: 'auto',
      }}
    >
      <div style={{ marginBottom: designSystem.spacing[3] }}>
        <span style={{ fontSize: designSystem.typography.fontSize.base, fontWeight: designSystem.typography.fontWeight.semibold }}>
          对比结果 ({pairs.length})
        </span>
      </div>

      <Row gutter={[16, 16]}>
        {pairs.map((pair) => {
          const score = getMatchScore(pair);
          const leftDoc = getDocumentName(pair.left_document_id);
          const rightDoc = getDocumentName(pair.right_document_id);

          return (
            <Col xs={24} sm={12} lg={8} key={pair.id}>
              <Card
                hoverable
                onClick={() => handleViewPair(pair.id)}
                style={{
                  borderRadius: designSystem.borderRadius.lg,
                  height: '100%',
                }}
                bodyStyle={{
                  padding: designSystem.spacing[3],
                }}
              >
                {/* 状态标签 */}
                <div style={{ marginBottom: designSystem.spacing[2] }}>
                  <Tag color={getStatusColor(pair.status)}>
                    {pair.status}
                  </Tag>
                </div>

                {/* 文档对 */}
                <div style={{ marginBottom: designSystem.spacing[3] }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: designSystem.spacing[1],
                      marginBottom: designSystem.spacing[1],
                      fontSize: designSystem.typography.fontSize.sm,
                    }}
                  >
                    <FileTextOutlined style={{ color: designSystem.colors.primary[500] }} />
                    <span
                      style={{
                        fontWeight: designSystem.typography.fontWeight.medium,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={leftDoc}
                    >
                      {leftDoc}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: designSystem.typography.fontSize.xs,
                      color: designSystem.semantic.text.secondary,
                      textAlign: 'center',
                      margin: `${designSystem.spacing[1]} 0`,
                    }}
                  >
                    ⟷
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: designSystem.spacing[1],
                      fontSize: designSystem.typography.fontSize.sm,
                    }}
                  >
                    <FileTextOutlined style={{ color: designSystem.colors.primary[500] }} />
                    <span
                      style={{
                        fontWeight: designSystem.typography.fontWeight.medium,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={rightDoc}
                    >
                      {rightDoc}
                    </span>
                  </div>
                </div>

                {/* 相似度进度条 */}
                <div style={{ marginBottom: designSystem.spacing[2] }}>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: designSystem.spacing[1],
                      fontSize: designSystem.typography.fontSize.xs,
                      color: designSystem.semantic.text.secondary,
                    }}
                  >
                    <span>相似度</span>
                    <span style={{ fontWeight: designSystem.typography.fontWeight.semibold, color: designSystem.semantic.text.primary }}>
                      {(score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    percent={score * 100}
                    strokeColor={
                      score > 0.7
                        ? designSystem.colors.error
                        : score > 0.4
                        ? designSystem.colors.warning
                        : designSystem.colors.success
                    }
                    showInfo={false}
                    size="small"
                  />
                </div>

                {/* 查看按钮 */}
                <div
                  style={{
                    textAlign: 'center',
                    paddingTop: designSystem.spacing[2],
                    borderTop: `1px solid ${designSystem.semantic.border.light}`,
                  }}
                >
                  <EyeOutlined style={{ marginRight: designSystem.spacing[1] }} />
                  <span style={{ fontSize: designSystem.typography.fontSize.sm }}>查看详情</span>
                </div>
              </Card>
            </Col>
          );
        })}
      </Row>
    </div>
  );
}
