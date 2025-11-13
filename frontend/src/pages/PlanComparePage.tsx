/**
 * PlanComparePage - 文档对比页
 *
 * 功能：
 * - 并排查看两个文档
 * - 高亮显示匹配片段
 * - 交互式匹配选择和跳转
 * - 导出对比报告
 */

import { useCallback, useEffect, useMemo, useState, memo } from 'react';
import { Card, Button, Select, Space, Typography, Spin, Row, Col } from 'antd';
import {
  ArrowLeftOutlined,
  DownloadOutlined,
  LeftOutlined,
  RightOutlined,
} from '@ant-design/icons';
import type {
  DocumentDetail,
  DocumentSummary,
  MatchDetailModel,
  MatchGroupModel,
  PairReport,
  ComparePairSummary,
} from '@/api/plagiarismApi';
import PageLayout from '@/layout/PageLayout';
import { MatchInfoTooltip, type MatchData } from '@/components/ui/match-info-popover';
import { buildSegmentsWithOverlap, type HighlightInterval as ImportedHighlightInterval } from '@/utils/highlightUtilsSimple';
import { saveAs } from 'file-saver';
import { designSystem } from '@/styles/DesignSystem';

const { Text } = Typography;

type Side = 'left' | 'right';
type DocumentLookup = Record<number, DocumentSummary>;

interface PairComparePageProps {
  report: PairReport | null;
  leftDocument: DocumentDetail | null;
  rightDocument: DocumentDetail | null;
  pairs: ComparePairSummary[];
  pairsLoading: boolean;
  pairsError: string | null;
  documentLookup: DocumentLookup;
  onSwitchPair: (pairId: number) => void;
  onBack: () => void;
  isTransitioning?: boolean;
}

type NormalisedMatch = {
  key: string;
  group: MatchGroupModel;
  details: MatchDetailModel[];
};

type HighlightMode = 'block' | 'fragment';
type HighlightInterval = ImportedHighlightInterval;

type Segment = {
  text: string;
  matchKey?: string;
  ordinal?: number;
  mode?: HighlightMode;
  allMatches?: Array<{
    matchKey: string;
    ordinal: number;
    mode: HighlightMode;
  }>;
};

export default function PlanComparePage({
  report,
  leftDocument,
  rightDocument,
  pairs,
  pairsLoading,
  pairsError,
  documentLookup,
  onSwitchPair,
  onBack,
  isTransitioning = false,
}: PairComparePageProps) {
  // ==================== 状态管理 ====================
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(true);

  // ==================== 数据处理 ====================
  const matches = useMemo<NormalisedMatch[]>(() => {
    if (!report) return [];
    const detailByGroup = report.details.reduce<Record<number, MatchDetailModel[]>>((acc, detail) => {
      acc[detail.group_id] = acc[detail.group_id] || [];
      acc[detail.group_id].push(detail);
      return acc;
    }, {});
    return report.groups.map((group) => ({
      key: `group-${group.id}`,
      group,
      details: detailByGroup[group.id] || [],
    }));
  }, [report]);

  useEffect(() => {
    if (matches.length > 0 && !activeKey) {
      setActiveKey(matches[0].key);
    }
  }, [matches, activeKey]);

  const currentPairId = report?.pair.id;
  const pairOptions = useMemo(
    () => (pairs.length ? pairs : report ? [report.pair] : []),
    [pairs, report],
  );

  const currentPair = useMemo(
    () => pairOptions.find((pair) => pair.id === currentPairId) ?? report?.pair,
    [pairOptions, currentPairId, report],
  );

  const leftIntervals = useMemo(
    () => prepareIntervals(leftDocument?.processed_text ?? '', matches, 'left'),
    [leftDocument?.processed_text, matches]
  );
  const rightIntervals = useMemo(
    () => prepareIntervals(rightDocument?.processed_text ?? '', matches, 'right'),
    [rightDocument?.processed_text, matches]
  );

  const leftSegments = useMemo(
    () => buildSegmentsWithOverlap(leftDocument?.processed_text ?? '', leftIntervals),
    [leftDocument?.processed_text, leftIntervals]
  );
  const rightSegments = useMemo(
    () => buildSegmentsWithOverlap(rightDocument?.processed_text ?? '', rightIntervals),
    [rightDocument?.processed_text, rightIntervals]
  );

  // ==================== 交互函数 ====================
  const jumpToMatch = useCallback((target: Side, matchKey: string, shouldFocus: boolean = false) => {
    if (!matchKey) return;
    const targetNode = document.querySelector<HTMLElement>(`mark[data-side="${target}"][data-match-keys*="${matchKey}"]`);
    if (targetNode) {
      const scrollContainer = targetNode.closest('[data-document-pane]');
      if (scrollContainer) {
        const containerRect = scrollContainer.getBoundingClientRect();
        const targetRect = targetNode.getBoundingClientRect();
        const relativeTop = targetRect.top - containerRect.top + scrollContainer.scrollTop;
        const scrollPosition = relativeTop - (scrollContainer.clientHeight / 2) + (targetRect.height / 2);
        scrollContainer.scrollTo({
          top: Math.max(0, scrollPosition),
          behavior: 'smooth'
        });
      }
      if (shouldFocus) {
        targetNode.focus({ preventScroll: true });
      }
    }
  }, []);

  const handleSelectMatch = useCallback(
    (matchKey: string, source: Side) => {
      if (!matchKey) return;

      document.querySelectorAll('mark.active-match').forEach(el => {
        el.classList.remove('active-match');
      });

      document.querySelectorAll(`mark[data-match-key="${matchKey}"][data-mode="fragment"]`).forEach(el => {
        el.classList.add('active-match');
      });

      if (document.querySelectorAll(`mark[data-match-key="${matchKey}"][data-mode="fragment"]`).length === 0) {
        document.querySelectorAll(`mark[data-match-key="${matchKey}"][data-mode="block"]`).forEach(el => {
          el.classList.add('active-match');
        });
      }

      setActiveKey(matchKey);
      jumpToMatch(source, matchKey, false);
      const opposite: Side = source === 'left' ? 'right' : 'left';
      jumpToMatch(opposite, matchKey, false);

      const matchButton = document.querySelector(`button[data-match-key="${matchKey}"]`);
      if (matchButton) {
        const matchListContainer = matchButton.closest('[data-match-list]');
        if (matchListContainer) {
          const containerRect = matchListContainer.getBoundingClientRect();
          const buttonRect = matchButton.getBoundingClientRect();
          const relativeTop = buttonRect.top - containerRect.top + matchListContainer.scrollTop;
          const scrollPosition = relativeTop - (matchListContainer.clientHeight / 2) + (buttonRect.height / 2);
          matchListContainer.scrollTo({
            top: Math.max(0, scrollPosition),
            behavior: 'smooth'
          });
        }
      }
    },
    [jumpToMatch],
  );

  const handleDownloadComparison = useCallback(() => {
    if (!report) return;
    const markdown = generateComparisonMarkdown({
      report,
      matches,
      pair: currentPair,
      documentLookup,
      leftDocument,
      rightDocument,
    });
    const filename = `comparison-${report.pair.id}.md`;
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
    saveAs(blob, filename);
  }, [report, matches, currentPair, documentLookup, leftDocument, rightDocument]);

  // ==================== 加载状态 ====================
  if (!report) {
    return (
      <div style={{
        display: 'flex',
        height: '100vh',
        alignItems: 'center',
        justifyContent: 'center',
        background: designSystem.semantic.surface.base,
      }}>
        <Spin tip="正在加载配对报告..." />
      </div>
    );
  }

  // ==================== 布局区域 ====================

  // topBar 工具栏
  const topBar = (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: designSystem.spacing[1],
        padding: designSystem.spacing[1],
        width: '100%',
      }}
    >
      <Space>
        <Button icon={<ArrowLeftOutlined />} onClick={onBack}>
          返回任务
        </Button>
        <div style={{
          borderLeft: `1px solid ${designSystem.colors.neutral[200]}`,
          paddingLeft: designSystem.spacing[3],
          fontSize: designSystem.typography.fontSize.sm,
          fontWeight: designSystem.typography.fontWeight.semibold,
        }}>
          {formatDocumentLabel(report.pair.left_document_id, documentLookup, leftDocument)} ↔{' '}
          {formatDocumentLabel(report.pair.right_document_id, documentLookup, rightDocument)}
        </div>
      </Space>
      <Space>
        <Button
          icon={<DownloadOutlined />}
          onClick={handleDownloadComparison}
          disabled={!report}
        >
          下载对比结果
        </Button>
        {currentPair && pairOptions.length > 0 && (
          <PairSwitcher
            pairs={pairOptions}
            currentPair={currentPair}
            loading={pairsLoading}
            error={pairsError}
            documentLookup={documentLookup}
            currentLeftDocument={leftDocument}
            currentRightDocument={rightDocument}
            onSwitchPair={onSwitchPair}
          />
        )}
      </Space>
    </div>
  );

  // 左侧匹配列表
  const leftSidebar = (
    <div
      data-match-list
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <div style={{
        padding: `${designSystem.spacing[2]} ${designSystem.spacing[3]}`,
        borderBottom: `1px solid ${designSystem.colors.neutral[200]}`,
        background: designSystem.semantic.surface.background,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <Text strong style={{ fontSize: designSystem.typography.fontSize.xs, textTransform: 'uppercase' }}>
          匹配列表
        </Text>
        <span style={{
          fontSize: designSystem.typography.fontSize.xs,
          fontFamily: 'monospace',
          color: designSystem.semantic.text.secondary,
          background: designSystem.semantic.surface.base,
          padding: `${designSystem.spacing[0]} ${designSystem.spacing[1]}`,
          borderRadius: designSystem.borderRadius.sm,
        }}>
          {matches.length}
        </span>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {matches.length === 0 ? (
          <div style={{ padding: designSystem.spacing[3], fontSize: designSystem.typography.fontSize.xs, color: designSystem.semantic.text.secondary }}>
            未发现匹配结果。
          </div>
        ) : (
          matches.map((match, index) => {
            const { group } = match;
            const isActive = match.key === activeKey;
            const finalScore = group.final_score || 0;
            return (
              <button
                key={match.key}
                onClick={() => handleSelectMatch(match.key, 'left')}
                data-match-key={match.key}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: designSystem.spacing[1],
                  padding: `${designSystem.spacing[2]} ${designSystem.spacing[3]}`,
                  width: '100%',
                  textAlign: 'left',
                  fontSize: designSystem.typography.fontSize.sm,
                  border: 'none',
                  borderBottom: `1px solid ${designSystem.colors.neutral[100]}`,
                  borderLeft: isActive ? `2px solid ${designSystem.colors.primary[500]}` : '2px solid transparent',
                  background: isActive ? designSystem.colors.neutral[50] : designSystem.semantic.surface.base,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.background = designSystem.semantic.surface.background;
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.background = designSystem.semantic.surface.base;
                  }
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{
                    fontSize: '10px',
                    fontWeight: designSystem.typography.fontWeight.medium,
                    color: designSystem.semantic.text.secondary,
                  }}>
                    #{index + 1}
                  </span>
                  <span style={{
                    fontSize: designSystem.typography.fontSize.xs,
                    fontFamily: 'monospace',
                    fontWeight: designSystem.typography.fontWeight.semibold,
                    color: getScoreColor(group.final_score),
                  }}>
                    {formatScore(group.final_score)}
                  </span>
                </div>
                <div style={{
                  width: '100%',
                  height: '4px',
                  background: designSystem.semantic.surface.background,
                  borderRadius: designSystem.borderRadius.full,
                  overflow: 'hidden',
                }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${finalScore * 100}%`,
                      background: getScoreBackground(finalScore),
                      transition: 'all 0.3s',
                    }}
                  />
                </div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: designSystem.spacing[2],
                  fontSize: '10px',
                  color: designSystem.semantic.text.secondary,
                }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                    <span style={{
                      width: '4px',
                      height: '4px',
                      borderRadius: '50%',
                      background: designSystem.colors.primary[500],
                    }} />
                    {formatScore(group.semantic_score)}
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                    <span style={{
                      width: '4px',
                      height: '4px',
                      borderRadius: '50%',
                      background: '#a855f7',
                    }} />
                    {formatScore(group.cross_score)}
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );

  // 右侧信息面板（可选）
  const rightSidebar = (
    <Card
      size="small"
      title="匹配信息"
      style={{ borderRadius: designSystem.borderRadius.lg }}
    >
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[2],
        fontSize: designSystem.typography.fontSize.sm,
      }}>
        <div>
          <Text type="secondary">匹配数量:</Text>
          <div style={{ marginTop: designSystem.spacing[1] }}>
            <Text strong style={{ fontSize: designSystem.typography.fontSize.lg }}>
              {matches.length}
            </Text>
          </div>
        </div>
        {activeKey && (() => {
          const match = matches.find(m => m.key === activeKey);
          if (!match) return null;
          return (
            <>
              <div>
                <Text type="secondary">当前匹配:</Text>
                <div style={{ marginTop: designSystem.spacing[1] }}>
                  <Text>#{matches.findIndex(m => m.key === activeKey) + 1}</Text>
                </div>
              </div>
              <div>
                <Text type="secondary">相似度:</Text>
                <div style={{ marginTop: designSystem.spacing[1] }}>
                  <Text strong style={{ fontSize: designSystem.typography.fontSize.lg, color: getScoreColor(match.group.final_score) }}>
                    {formatScore(match.group.final_score)}
                  </Text>
                </div>
              </div>
            </>
          );
        })()}
      </div>
    </Card>
  );

  // 底部状态栏
  const bottomBar = (
    <>
      <span>匹配数量: {matches.length}</span>
      <span>文档对 ID: {report.pair.id}</span>
      {activeKey && <span>当前选中: #{matches.findIndex(m => m.key === activeKey) + 1}</span>}
    </>
  );

  // ==================== 渲染 ====================
  return (
    <PageLayout
      topBar={topBar}
      leftSidebar={leftSidebar}
      leftSidebarWidth="280px"
      leftDefaultCollapsed={leftCollapsed}
      onLeftCollapsedChange={setLeftCollapsed}
      rightSidebar={rightSidebar}
      rightDefaultCollapsed={rightCollapsed}
      onRightCollapsedChange={setRightCollapsed}
      bottomBar={bottomBar}
    >
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        background: designSystem.semantic.surface.base,
        padding: designSystem.spacing[1],
        position: 'relative',
      }}>
        {isTransitioning && (
          <div style={{
            position: 'absolute',
            inset: 0,
            zIndex: 50,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(4px)',
          }}>
            <Spin tip="正在切换文档..." />
          </div>
        )}
        <Row gutter={[parseInt(designSystem.spacing[1]), parseInt(designSystem.spacing[1])]} style={{ flex: 1, minHeight: 0, display: 'flex' }}>
          <Col xs={24} xl={12} style={{ display: 'flex', flexDirection: 'column', minHeight: 0, flex: 1, height: '100%' }}>
            <DocumentPane
              title="左侧文档"
              segments={leftSegments}
              activeKey={activeKey}
              side="left"
              matches={matches}
              onSelectMatch={(key) => handleSelectMatch(key, 'left')}
            />
          </Col>
          <Col xs={24} xl={12} style={{ display: 'flex', flexDirection: 'column', minHeight: 0, flex: 1, height: '100%' }}>
            <DocumentPane
              title="右侧文档"
              segments={rightSegments}
              activeKey={activeKey}
              side="right"
              matches={matches}
              onSelectMatch={(key) => handleSelectMatch(key, 'right')}
            />
          </Col>
        </Row>
      </div>
    </PageLayout>
  );
}

// ==================== 子组件 ====================

interface DocumentPaneProps {
  title: string;
  segments: Segment[];
  activeKey: string | null;
  side: Side;
  matches: NormalisedMatch[];
  onSelectMatch: (key: string) => void;
}

function DocumentPane({ title, segments, activeKey, side, matches, onSelectMatch }: DocumentPaneProps) {
  return (
    <Card
      title={title}
      size="small"
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        borderRadius: designSystem.borderRadius.lg,
      }}
      bodyStyle={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        overflow: 'hidden',
        padding: 0,
      }}
    >
      <div
        data-document-pane
        style={{
          flex: 1,
          overflow: 'auto',
          background: designSystem.semantic.surface.base,
          padding: designSystem.spacing[3],
          fontSize: '13px',
          lineHeight: 1.8,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          textAlign: 'justify',
        }}
      >
        <RenderSegments
          segments={segments}
          activeKey={activeKey}
          side={side}
          matches={matches}
          onSelectMatch={onSelectMatch}
        />
      </div>
    </Card>
  );
}

interface PairSwitcherProps {
  pairs: ComparePairSummary[];
  currentPair: ComparePairSummary | undefined;
  loading: boolean;
  error: string | null;
  documentLookup: DocumentLookup;
  currentLeftDocument: DocumentDetail | null;
  currentRightDocument: DocumentDetail | null;
  onSwitchPair: (pairId: number) => void;
}

function PairSwitcher({
  pairs,
  currentPair,
  loading,
  error,
  documentLookup,
  currentLeftDocument,
  currentRightDocument,
  onSwitchPair,
}: PairSwitcherProps) {
  // 使用 currentPair 的实际值，如果不存在则使用第一个 pair 的值
  const [leftSelection, setLeftSelection] = useState<number>(() =>
    currentPair?.left_document_id ?? pairs[0]?.left_document_id ?? 0
  );
  const [rightSelection, setRightSelection] = useState<number>(() =>
    currentPair?.id ?? pairs[0]?.id ?? 0
  );

  const uniqueLefts = useMemo(() => Array.from(new Set(pairs.map((pair) => pair.left_document_id))), [pairs]);

  useEffect(() => {
    if (currentPair) {
      setLeftSelection(currentPair.left_document_id);
      setRightSelection(currentPair.id);
    }
  }, [currentPair]);

  useEffect(() => {
    if (uniqueLefts.length === 0 || !currentPair) return;
    if (!uniqueLefts.includes(leftSelection)) {
      setLeftSelection(currentPair.left_document_id);
    }
  }, [leftSelection, uniqueLefts, currentPair]);

  const pairsForLeft = useMemo(() => {
    const subset = pairs.filter((pair) => pair.left_document_id === leftSelection);
    const base = subset.length ? subset : (currentPair ? [currentPair] : []);
    return base
      .slice()
      .sort((a, b) =>
        a.right_document_id !== b.right_document_id ? a.right_document_id - b.right_document_id : a.id - b.id,
      );
  }, [pairs, leftSelection, currentPair]);

  const currentIndex = pairsForLeft.findIndex((pair) => pair.id === currentPair?.id);
  const canPrev = currentIndex > 0;
  const canNext = currentIndex >= 0 && currentIndex < pairsForLeft.length - 1;
  const disableLeftSelect = uniqueLefts.length <= 1;
  const disableRightSelect = pairsForLeft.length <= 1;

  const handleLeftChange = (value: string) => {
    const nextLeft = Number(value);
    if (Number.isNaN(nextLeft) || !currentPair) return;
    setLeftSelection(nextLeft);
    if (currentPair.left_document_id !== nextLeft) {
      const nextPair = pairs
        .filter((pair) => pair.left_document_id === nextLeft)
        .sort((a, b) =>
          a.right_document_id !== b.right_document_id ? a.right_document_id - b.right_document_id : a.id - b.id,
        )[0];
      if (nextPair) {
        setRightSelection(nextPair.id);
        if (nextPair.id !== currentPair.id) {
          onSwitchPair(nextPair.id);
        }
      }
    }
  };

  const handleRightChange = (value: string) => {
    const nextId = Number(value);
    if (!Number.isNaN(nextId) && currentPair) {
      setRightSelection(nextId);
      if (nextId !== currentPair.id) {
        onSwitchPair(nextId);
      }
    }
  };

  const handlePrev = () => {
    if (!canPrev) return;
    const prevPair = pairsForLeft[currentIndex - 1];
    if (prevPair) {
      onSwitchPair(prevPair.id);
    }
  };

  const handleNext = () => {
    if (!canNext) return;
    const nextPair = pairsForLeft[currentIndex + 1];
    if (nextPair) {
      onSwitchPair(nextPair.id);
    }
  };

  return (
    <Space size="middle">
      {error && (
        <Text type="danger" style={{
          fontSize: designSystem.typography.fontSize.sm,
          background: 'rgba(255, 77, 79, 0.1)',
          padding: `${designSystem.spacing[1]} ${designSystem.spacing[2]}`,
          borderRadius: designSystem.borderRadius.sm,
        }}>
          {error}
        </Text>
      )}
      <Text style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>
        左侧:
      </Text>
      <Select
        value={String(leftSelection)}
        onChange={handleLeftChange}
        disabled={disableLeftSelect || loading}
        style={{ width: 160 }}
        options={uniqueLefts.map((left) => ({
          value: String(left),
          label: formatDocumentLabel(
            left,
            documentLookup,
            left === currentPair?.left_document_id ? currentLeftDocument : null,
          ),
        }))}
      />
      <Text style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>
        右侧:
      </Text>
      <Select
        value={String(rightSelection)}
        onChange={handleRightChange}
        disabled={disableRightSelect || loading}
        style={{ width: 160 }}
        options={pairsForLeft.map((pair) => ({
          value: String(pair.id),
          label: formatDocumentLabel(
            pair.right_document_id,
            documentLookup,
            pair.id === currentPair?.id ? currentRightDocument : null,
          ),
        }))}
      />
      <div style={{ height: '32px', width: '1px', background: designSystem.semantic.border.light }} />
      <Button
        type="default"
        onClick={handlePrev}
        disabled={!canPrev || loading}
        title="上一个配对"
        icon={<LeftOutlined />}
      />
      <Button
        type="default"
        onClick={handleNext}
        disabled={!canNext || loading}
        title="下一个配对"
        icon={<RightOutlined />}
      />
    </Space>
  );
}

interface RenderSegmentsProps {
  segments: Segment[];
  activeKey: string | null;
  side: Side;
  matches: NormalisedMatch[];
  onSelectMatch: (key: string) => void;
}

const RenderSegments = memo(({ segments, activeKey, side, matches, onSelectMatch }: RenderSegmentsProps) => {
  if (!segments.length) return null;

  return (
    <>
      {segments.map((segment, index) => {
        if (!segment.matchKey) {
          return <span key={`plain-${index}`}>{segment.text}</span>;
        }

        const match = matches.find(m => m.key === segment.matchKey) || null;
        const isActive = activeKey === segment.matchKey;
        const ordinal = segment.ordinal ?? 0;

        const hasOverlaps = segment.allMatches && segment.allMatches.length > 1;
        const overlappingMatches: MatchData[] = hasOverlaps
          ? segment.allMatches!
              .map(am => matches.find(m => m.key === am.matchKey))
              .filter((m): m is NormalisedMatch => m !== undefined && m !== null)
              .map(m => ({ group: m.group, details: m.details }))
          : (match ? [{ group: match.group, details: match.details }] : []);

        const baseStyle: React.CSSProperties = {
          borderRadius: designSystem.borderRadius.sm,
          transition: 'all 0.2s',
          cursor: 'pointer',
          wordBreak: 'break-word',
        };

        const colorStyle = getScoreStyle(match?.group?.final_score);
        const backgroundStyle: React.CSSProperties =
          segment.mode === 'block'
            ? { ...colorStyle, color: designSystem.semantic.text.primary }
            : {
                ...colorStyle,
                color: designSystem.semantic.text.primary,
                textDecoration: 'underline',
                textDecorationColor: '#000',
                textDecorationThickness: '2px',
                textUnderlineOffset: '2px',
              };

        const activeStyle: React.CSSProperties = isActive
          ? {
              outline: `2px solid ${designSystem.colors.primary[500]}`,
              outlineOffset: '1px',
            }
          : {};

        const overlappingStyle: React.CSSProperties = hasOverlaps
          ? {
              boxShadow: `0 0 0 2px ${designSystem.colors.primary[300]}`,
            }
          : {};

        const finalStyle = { ...baseStyle, ...backgroundStyle, ...activeStyle, ...overlappingStyle };

        if (!match) {
          return (
            <mark
              key={`highlight-${index}`}
              id={makeHighlightId(side, segment.matchKey || '', ordinal)}
              className="match-highlight"
              style={finalStyle}
              tabIndex={0}
              data-match-keys={segment.matchKey}
              data-match-key={segment.matchKey}
              data-side={side}
              data-mode={segment.mode}
              data-ordinal={ordinal}
              onClick={() => onSelectMatch(segment.matchKey!)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onSelectMatch(segment.matchKey!);
                }
              }}
            >
              {segment.text}
            </mark>
          );
        }

        return (
          <MatchInfoTooltip key={`highlight-${index}`} match={match} allMatches={overlappingMatches}>
            <mark
              id={makeHighlightId(side, segment.matchKey || '', ordinal)}
              className="match-highlight"
              style={finalStyle}
              tabIndex={0}
              data-match-keys={segment.matchKey}
              data-match-key={segment.matchKey}
              data-side={side}
              data-mode={segment.mode}
              data-ordinal={ordinal}
              onClick={() => onSelectMatch(segment.matchKey!)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onSelectMatch(segment.matchKey!);
                }
              }}
            >
              {segment.text}
            </mark>
          </MatchInfoTooltip>
        );
      })}
    </>
  );
});

// ==================== 辅助函数 ====================

function prepareIntervals(text: string, matches: NormalisedMatch[], side: Side): HighlightInterval[] {
  if (!text) return [];
  const raw: HighlightInterval[] = [];
  const ordinalMap = new Map<string, number>();
  const seen = new Set<string>();
  matches.forEach((match) => {
    const spans = match.group.document_spans ?? match.group.paragraph_spans ?? [];
    const sideSpans = spans
      .map((span) => ({
        start: side === 'left' ? span.left_start : span.right_start,
        end: side === 'left' ? span.left_end : span.right_end,
      }))
      .filter((item) => {
        return Number.isFinite(item.start) && Number.isFinite(item.end) &&
               item.start >= 0 && item.end >= 0 && item.end > item.start;
      });

    if (!sideSpans.length) return;

    const coverage = match.group.alignment_ratio ?? 0;
    const coverageThreshold = 0.75;

    const blockOrdinal = ordinalMap.get(match.key) ?? 0;
    const minStart = Math.min(...sideSpans.map((item) => item.start));
    const maxEnd = Math.max(...sideSpans.map((item) => item.end));

    const blockKey = `${match.key}:${Math.floor(minStart)}:${Math.floor(maxEnd)}:block`;
    if (!seen.has(blockKey)) {
      seen.add(blockKey);
      raw.push({
        start: minStart,
        end: maxEnd,
        matchKey: match.key,
        ordinal: blockOrdinal,
        side,
        mode: 'block',
      });
    }

    let nextOrdinal = blockOrdinal + 1;

    if (coverage < coverageThreshold) {
      sideSpans.forEach(({ start, end }) => {
        const key = `${match.key}:${Math.floor(start)}:${Math.floor(end)}:fragment`;
        if (seen.has(key)) return;
        seen.add(key);
        raw.push({
          start: start,
          end: end,
          matchKey: match.key,
          ordinal: nextOrdinal,
          side,
          mode: 'fragment',
        });
        nextOrdinal += 1;
      });
    }

    ordinalMap.set(match.key, nextOrdinal);
  });
  return raw;
}

function makeHighlightId(side: Side, matchKey: string, ordinal: number) {
  return `${side}-match-${matchKey}-${ordinal}`;
}

function formatScore(value: number | null | undefined) {
  if (value == null) return '—';
  return value.toFixed(3);
}

function getScoreColor(score: number | null | undefined): string {
  const finalScore = score || 0;
  if (finalScore > 0.9) return '#b91c1c';
  if (finalScore >= 0.85) return '#c2410c';
  if (finalScore >= 0.8) return '#a16207';
  if (finalScore >= 0.7) return '#15803d';
  return '#6b7280';
}

function getScoreBackground(score: number | null | undefined): string {
  const finalScore = score || 0;
  if (finalScore > 0.9) return 'rgba(239, 68, 68, 0.75)';
  if (finalScore >= 0.85) return 'rgba(249, 115, 22, 0.7)';
  if (finalScore >= 0.8) return 'rgba(250, 204, 21, 0.65)';
  if (finalScore >= 0.7) return 'rgba(34, 197, 94, 0.65)';
  return 'rgba(156, 163, 175, 0.6)';
}

function getScoreStyle(score: number | null | undefined): React.CSSProperties {
  const finalScore = score || 0;
  if (finalScore > 0.9) {
    return { background: 'rgba(239, 68, 68, 0.75)' };
  } else if (finalScore >= 0.85) {
    return { background: 'rgba(249, 115, 22, 0.7)' };
  } else if (finalScore >= 0.8) {
    return { background: 'rgba(250, 204, 21, 0.65)' };
  } else if (finalScore >= 0.7) {
    return { background: 'rgba(34, 197, 94, 0.65)' };
  } else {
    return { background: 'rgba(156, 163, 175, 0.6)' };
  }
}

function formatDocumentLabel(
  documentId: number,
  lookup: DocumentLookup,
  fallback?: DocumentDetail | DocumentSummary | null,
) {
  const info = lookup[documentId] ?? fallback ?? null;
  if (!info) return `文档 ${documentId}`;
  const name = info.title || info.filename || `文档 ${info.id ?? documentId}`;
  return name;
}

// ==================== Markdown 导出函数 ====================

function formatMetricTitle(key: string): string {
  return key
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function formatMetricValue(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toFixed(3);
  }
  return String(value);
}

function formatAverageScore(values: Array<number | null | undefined>): string {
  const numeric = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  if (!numeric.length) return '—';
  const average = numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
  return average.toFixed(3);
}

interface ComparisonMarkdownContext {
  report: PairReport;
  matches: NormalisedMatch[];
  pair: ComparePairSummary | null | undefined;
  documentLookup: DocumentLookup;
  leftDocument: DocumentDetail | null;
  rightDocument: DocumentDetail | null;
}

function generateComparisonMarkdown({
  report,
  matches,
  pair,
  documentLookup,
  leftDocument,
  rightDocument,
}: ComparisonMarkdownContext): string {
  const leftLabel = formatDocumentLabel(report.pair.left_document_id, documentLookup, leftDocument);
  const rightLabel = formatDocumentLabel(report.pair.right_document_id, documentLookup, rightDocument);
  const matchGroupCount = matches.length;
  const detailCount = report.details.length;
  const metricsEntries = Object.entries(pair?.metrics ?? {}).filter(([, value]) => value !== null && value !== undefined);
  const sortedMatches = [...matches].sort((a, b) => (b.group.final_score ?? 0) - (a.group.final_score ?? 0));
  const now = new Date().toLocaleString('zh-CN');

  const warningLines = detailCount > 1000
    ? ['> ⚠️ 匹配详情数量超过 1000 条，为避免影响分析，仅列出部分关键片段。', '']
    : [];

  const metricSection = metricsEntries.length
    ? [
        '## 指标概览',
        '| 指标 | 数值 |',
        '| --- | --- |',
        ...metricsEntries.map(([key, value]) => `| ${formatMetricTitle(key)} | ${formatMetricValue(value)} |`),
        '',
      ]
    : [];

  const distributionSection = [
    '## 匹配分布速览',
    `- 平均 Final 分数：${formatAverageScore(sortedMatches.map((item) => item.group.final_score))}`,
    `- 最高 Final 分数：${formatScore(sortedMatches[0]?.group.final_score)}`,
    `- 最低 Final 分数：${formatScore(sortedMatches[sortedMatches.length - 1]?.group.final_score)}`,
    '',
  ];

  const detailedTableSection = buildHighSimilarityMarkdownTable({
    matches: sortedMatches,
    leftDocument,
    rightDocument,
  });

  const lines = [
    '# 文档对比分析报告',
    '',
    '## 基本信息',
    `- 导出时间：${now}`,
    `- 对比配对 ID：${report.pair.id}`,
    `- 左侧文档：${leftLabel} (ID: ${report.pair.left_document_id})`,
    `- 右侧文档：${rightLabel} (ID: ${report.pair.right_document_id})`,
    `- 匹配组数量：${matchGroupCount}`,
    `- 匹配详情数量：${detailCount}`,
    '',
    ...warningLines,
    ...metricSection,
    ...distributionSection,
    ...detailedTableSection,
    '---',
    '',
    '（本报告为系统自动生成的结构化对比摘要，可用于归档或人工复核。）',
    '',
  ];

  return lines.join('\n');
}

interface HighSimilarityTableOptions {
  matches: NormalisedMatch[];
  leftDocument: DocumentDetail | null;
  rightDocument: DocumentDetail | null;
  maxRows?: number;
}

function buildHighSimilarityMarkdownTable({
  matches,
  leftDocument,
  rightDocument,
  maxRows = 50,
}: HighSimilarityTableOptions): string[] {
  if (!matches.length) return [];

  const leftText = leftDocument?.processed_text ?? null;
  const rightText = rightDocument?.processed_text ?? null;

  const rows: string[] = [];
  const limited = matches.slice(0, maxRows);

  limited.forEach((match, index) => {
    const { group, details } = match;
    const primaryDetail = selectPrimaryDetail(details);
    const leftExcerpt = resolveExcerpt({
      detail: primaryDetail,
      side: 'left',
      fullText: leftText,
    });
    const rightExcerpt = resolveExcerpt({
      detail: primaryDetail,
      side: 'right',
      fullText: rightText,
    });

    const score = group.final_score ?? 0;
    const rowStyle = scoreStyleForMarkdown(score);

    rows.push(
      `<tr style="${rowStyle}">
        <td style="padding:8px 12px;border:1px solid #d7dce3;vertical-align:top;">${index + 1}</td>
        <td style="padding:8px 12px;border:1px solid #d7dce3;vertical-align:top;">${formatScore(group.final_score)}</td>
        <td style="padding:8px 12px;border:1px solid #d7dce3;vertical-align:top;">${escapeHtml(leftExcerpt)}</td>
        <td style="padding:8px 12px;border:1px solid #d7dce3;vertical-align:top;">${escapeHtml(rightExcerpt)}</td>
      </tr>`
    );
  });

  const table = [
    '## 高相似片段详情',
    '<table style="width:100%;border-collapse:collapse;margin-top:12px;">',
    '<thead>',
    '<tr style="background:#f4f6fb;color:#1f2937;font-weight:600;">',
    '<th style="padding:8px 12px;border:1px solid #d7dce3;">序号</th>',
    '<th style="padding:8px 12px;border:1px solid #d7dce3;">相似度</th>',
    '<th style="padding:8px 12px;border:1px solid #d7dce3;">左侧片段</th>',
    '<th style="padding:8px 12px;border:1px solid #d7dce3;">右侧片段</th>',
    '</tr>',
    '</thead>',
    '<tbody>',
    ...rows,
    '</tbody>',
    '</table>',
    '',
  ];

  return table;
}

function selectPrimaryDetail(details: MatchDetailModel[]): MatchDetailModel | undefined {
  if (!details?.length) return undefined;
  return [...details].sort((a, b) => (b.final_score ?? 0) - (a.final_score ?? 0))[0];
}

function extractExcerpt(fullText: string, start: number, end: number, radius = 80): string {
  if (!fullText) return '—';

  const safeStart = Number.isFinite(start) ? Math.max(0, Math.min(start, fullText.length)) : 0;
  const safeEnd = Number.isFinite(end) ? Math.max(safeStart, Math.min(end, fullText.length)) : safeStart;

  const windowStart = Math.max(0, safeStart - radius);
  const windowEnd = Math.min(fullText.length, safeEnd + radius);
  let snippet = fullText.slice(windowStart, windowEnd).replace(/\s+/g, ' ').trim();

  if (!snippet) return '—';

  if (windowStart > 0) snippet = `…${snippet}`;
  if (windowEnd < fullText.length) snippet = `${snippet}…`;
  return snippet;
}

function scoreStyleForMarkdown(score: number): string {
  if (!Number.isFinite(score)) return 'background:#f8fafc;color:#1f2937;';
  if (score >= 0.9) return 'background:rgba(239,68,68,0.18);color:#7f1d1d;';
  if (score >= 0.85) return 'background:rgba(249,115,22,0.16);color:#7c2d12;';
  if (score >= 0.8) return 'background:rgba(250,204,21,0.14);color:#713f12;';
  return 'background:#f8fafc;color:#1f2937;';
}

interface ResolveExcerptOptions {
  detail: MatchDetailModel | undefined;
  side: 'left' | 'right';
  fullText: string | null;
  radius?: number;
}

function resolveExcerpt({ detail, side, fullText, radius = 80 }: ResolveExcerptOptions): string {
  if (!detail) return '—';

  const direct = side === 'left' ? detail.left_excerpt : detail.right_excerpt;
  if (direct && direct.trim()) return direct.trim();

  const spans = detail.spans;
  if (!Array.isArray(spans) || spans.length === 0) return '—';
  const first = spans[0];
  const start = side === 'left' ? first.left_start : first.right_start;
  const end = side === 'left' ? first.left_end : first.right_end;

  if (typeof start !== 'number' || typeof end !== 'number' || end <= start) return '—';
  if (!fullText) return '—';

  return extractExcerpt(fullText, start, end, radius);
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/\n/g, '<br/>');
}
