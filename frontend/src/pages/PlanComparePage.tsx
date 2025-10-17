import React, { useCallback, useEffect, useMemo, useState, memo } from 'react';
import type {
  DocumentDetail,
  DocumentSummary,
  MatchDetailModel,
  MatchGroupModel,
  PairReport,
  ComparePairSummary,
} from '@/api/plagiarismApi';
import { Button } from '@/components/ui/button';
import { PageShell, PageHeader } from '@/components/layout/Page';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { MatchInfoTooltip } from '@/components/ui/match-info-popover';
import { cn } from '@/lib/utils';
import { buildSegmentsWithOverlap, type HighlightInterval as ImportedHighlightInterval } from '@/utils/highlightUtilsSimple';
import { ChevronLeft, ChevronRight, Download } from 'lucide-react';
import { saveAs } from 'file-saver';

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

// Use the imported type and extend it for compatibility
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

export function PlanComparePage({
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
  const matches = useMemo<NormalisedMatch[]>(() => {
    if (!report) return [];
    const detailByGroup = report.details.reduce<Record<number, MatchDetailModel[]>>((acc, detail) => {
      acc[detail.group_id] = acc[detail.group_id] || [];
      acc[detail.group_id].push(detail);
      return acc;
    }, {});
    // Use group.id as the primary key for consistent matching across sides
    return report.groups.map((group) => ({
      key: `group-${group.id}`,
      group,
      details: detailByGroup[group.id] || [],
    }));
  }, [report]);

  const currentPairId = report?.pair.id;
  const pairOptions = useMemo(
    () => (pairs.length ? pairs : report ? [report.pair] : []),
    [pairs, report],
  );

  const currentPair = useMemo(
    () => pairOptions.find((pair) => pair.id === currentPairId) ?? report?.pair,
    [pairOptions, currentPairId, report],
  );

  const [activeKey, setActiveKey] = useState<string | null>(matches[0]?.key ?? null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const leftIntervals = useMemo(() => prepareIntervals(leftDocument?.processed_text ?? '', matches, 'left'), [leftDocument?.processed_text, matches]);
  const rightIntervals = useMemo(() => prepareIntervals(rightDocument?.processed_text ?? '', matches, 'right'), [rightDocument?.processed_text, matches]);

  const leftSegments = useMemo(() => buildSegmentsWithOverlap(leftDocument?.processed_text ?? '', leftIntervals), [leftDocument?.processed_text, leftIntervals]);
  const rightSegments = useMemo(() => buildSegmentsWithOverlap(rightDocument?.processed_text ?? '', rightIntervals), [rightDocument?.processed_text, rightIntervals]);

  const jumpToMatch = useCallback((target: Side, matchKey: string, shouldFocus: boolean = false) => {
    if (!matchKey) {
      return;
    }
    // 修复选择器以支持新的 data-match-keys 属性
    const targetNode = document.querySelector<HTMLElement>(`mark[data-side="${target}"][data-match-keys*="${matchKey}"]`);
    if (targetNode) {
      // Find the scrollable container for the document pane
      const scrollContainer = targetNode.closest('.overflow-auto');
      if (scrollContainer) {
        // Calculate the position relative to the scroll container
        const containerRect = scrollContainer.getBoundingClientRect();
        const targetRect = targetNode.getBoundingClientRect();
        const relativeTop = targetRect.top - containerRect.top + scrollContainer.scrollTop;

        // Center the element in the container
        const scrollPosition = relativeTop - (scrollContainer.clientHeight / 2) + (targetRect.height / 2);

        // Scroll only within the container, not the entire page
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
      if (!matchKey) {
        return;
      }

      // Remove active class from all elements
      document.querySelectorAll('mark.active-match').forEach(el => {
        el.classList.remove('active-match');
      });

      // Add active class to matching elements on both sides
      // Only highlight fragments, not the whole block
      document.querySelectorAll(`mark[data-match-key="${matchKey}"][data-mode="fragment"]`).forEach(el => {
        el.classList.add('active-match');
      });

      // If no fragments, highlight the block
      if (document.querySelectorAll(`mark[data-match-key="${matchKey}"][data-mode="fragment"]`).length === 0) {
        document.querySelectorAll(`mark[data-match-key="${matchKey}"][data-mode="block"]`).forEach(el => {
          el.classList.add('active-match');
        });
      }

      setActiveKey(matchKey);

      // Jump to both sides immediately
      jumpToMatch(source, matchKey, false);
      const opposite: Side = source === 'left' ? 'right' : 'left';
      jumpToMatch(opposite, matchKey, false);

      const matchButton = document.querySelector(`button[data-match-key="${matchKey}"]`);
      if (matchButton) {
        // Find the scrollable container for the match list
        const matchListContainer = matchButton.closest('.overflow-auto');
        if (matchListContainer) {
          const containerRect = matchListContainer.getBoundingClientRect();
          const buttonRect = matchButton.getBoundingClientRect();
          const relativeTop = buttonRect.top - containerRect.top + matchListContainer.scrollTop;

          // Center the button in the container
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
    if (!report) {
      return;
    }

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

  if (!report) {
    return (
      <PageShell>
        <div className="flex h-full items-center justify-center bg-muted/40 text-sm text-muted-foreground">
          <div className="animate-pulse">正在加载配对报告...</div>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell>
      <PageHeader
        title={
          <>
            <Button
              variant="default"
              size="sm"
              className="text-xs"
              onClick={onBack}
            >
              ← 返回任务
            </Button>
            <div className="border-l border-border pl-3">
              <div className="text-sm font-medium">
                {formatDocumentLabel(report.pair.left_document_id, documentLookup, leftDocument)} ↔{' '}
                {formatDocumentLabel(report.pair.right_document_id, documentLookup, rightDocument)}
              </div>
              <div className="text-xs text-muted-foreground">匹配数量：{matches.length}</div>
            </div>
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="text-xs"
              onClick={handleDownloadComparison}
              disabled={!report}
            >
              <Download className="mr-2 h-4 w-4" /> 下载对比结果
            </Button>
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
          </div>
        }
      />

      <div className="relative flex flex-1 min-h-0 divide-x divide-border bg-background">
        {isTransitioning && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
            <div className="animate-pulse text-sm text-muted-foreground">正在切换文档...</div>
          </div>
        )}
        <main className="flex-1 min-h-0 overflow-hidden">
          <div className="relative flex h-full min-h-0 flex-col gap-5 p-6">
            <div className="grid flex-1 min-h-0 grid-cols-1 gap-5 xl:grid-cols-2">
              <DocumentPane
                title="左侧文档"
                segments={leftSegments}
                activeKey={activeKey}
                side="left"
                matches={matches}
                onSelectMatch={(key) => handleSelectMatch(key, 'left')}
              />
              <DocumentPane
                title="右侧文档"
                segments={rightSegments}
                activeKey={activeKey}
                side="right"
                matches={matches}
                onSelectMatch={(key) => handleSelectMatch(key, 'right')}
              />
            </div>

          </div>
        </main>
        <aside className={cn(
          "relative flex flex-col border-l border-border bg-card shadow-sm transition-all duration-300",
          sidebarCollapsed ? "w-0 max-w-0" : "w-72 min-w-[18rem]"
        )}>
          <button
            type="button"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className={cn(
              'absolute left-0 top-1/2 z-20 inline-flex h-12 w-5 -translate-y-1/2 -translate-x-full items-center justify-center rounded-md border border-border bg-background shadow-lg transition hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary/40'
            )}
            title={sidebarCollapsed ? '展开匹配列表' : '折叠匹配列表'}
          >
            {sidebarCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
            <span className="sr-only">{sidebarCollapsed ? '展开' : '折叠'}匹配列表</span>
          </button>
          {!sidebarCollapsed && (
            <>
              <div className="flex items-center justify-between border-b border-border px-4 py-3 flex-shrink-0">
                <h2 className="text-sm font-medium text-muted-foreground">匹配列表</h2>
                <span className="text-xs text-muted-foreground">{matches.length}</span>
              </div>
              <div className="flex-1 min-h-0 overflow-auto">
                {matches.length === 0 ? (
                  <p className="px-4 py-4 text-sm text-muted-foreground">未发现匹配结果。</p>
                ) : (
                  <ul className="divide-y divide-border/60">
                    {matches.map((match, index) => {
                      const { group } = match;
                      const isActive = match.key === activeKey;
                      return (
                        <li key={match.key}>
                          <button
                            onClick={() => handleSelectMatch(match.key, 'left')}
                            className={cn(
                              'flex w-full flex-col gap-1 px-4 py-3 text-left text-sm transition',
                              isActive ? 'font-medium ring-1 ring-primary/30 bg-accent' : 'hover:bg-accent/50',
                              'hover:bg-accent'
                            )}
                            data-match-key={match.key}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-medium text-muted-foreground">#{index + 1}</span>
                              <span className={cn('text-xs font-mono', getScoreColorClasses(group.final_score))}>{formatScore(group.final_score)}</span>
                            </div>
                            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                              <span>语义 {formatScore(group.semantic_score)}</span>
                              <span>交叉编码 {formatScore(group.cross_score)}</span>
                            </div>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            </>
          )}
        </aside>
      </div>
    </PageShell>
  );
}

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
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border border-border bg-card shadow-sm">
      <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-2 text-sm font-medium text-muted-foreground">
        <span>{title}</span>
      </div>
      <div className="relative flex-1 min-h-0 min-w-0 overflow-auto bg-background">
        <article className="min-w-0 whitespace-pre-wrap break-words px-6 py-4 text-sm leading-relaxed text-justify">
          <RenderSegments segments={segments} activeKey={activeKey} side={side} matches={matches} onSelectMatch={onSelectMatch} />
        </article>
      </div>
    </div>
  );
}

interface PairSwitcherProps {
  pairs: ComparePairSummary[];
  currentPair: ComparePairSummary;
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
  const [leftSelection, setLeftSelection] = useState<number>(currentPair.left_document_id);
  const [rightSelection, setRightSelection] = useState<number>(currentPair.id);

  const uniqueLefts = useMemo(() => Array.from(new Set(pairs.map((pair) => pair.left_document_id))), [pairs]);

  useEffect(() => {
    setLeftSelection(currentPair.left_document_id);
    setRightSelection(currentPair.id);
  }, [currentPair.id, currentPair.left_document_id]);

  useEffect(() => {
    if (uniqueLefts.length === 0) {
      return;
    }
    if (!uniqueLefts.includes(leftSelection)) {
      setLeftSelection(currentPair.left_document_id);
    }
  }, [leftSelection, uniqueLefts, currentPair.left_document_id]);

  const pairsForLeft = useMemo(() => {
    const subset = pairs.filter((pair) => pair.left_document_id === leftSelection);
    const base = subset.length ? subset : [currentPair];
    return base
      .slice()
      .sort((a, b) =>
        a.right_document_id !== b.right_document_id ? a.right_document_id - b.right_document_id : a.id - b.id,
      );
  }, [pairs, leftSelection, currentPair]);

  const currentIndex = pairsForLeft.findIndex((pair) => pair.id === currentPair.id);
  const canPrev = currentIndex > 0;
  const canNext = currentIndex >= 0 && currentIndex < pairsForLeft.length - 1;
  const disableLeftSelect = uniqueLefts.length <= 1;
  const disableRightSelect = pairsForLeft.length <= 1;

  const handleLeftChange = (value: string) => {
    const nextLeft = Number(value);
    if (Number.isNaN(nextLeft)) {
      return;
    }
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
    if (!Number.isNaN(nextId)) {
      setRightSelection(nextId);
      if (nextId !== currentPair.id) {
        onSwitchPair(nextId);
      }
    }
  };

  const handlePrev = () => {
    if (!canPrev) {
      return;
    }
    onSwitchPair(pairsForLeft[currentIndex - 1].id);
  };

  const handleNext = () => {
    if (!canNext) {
      return;
    }
    onSwitchPair(pairsForLeft[currentIndex + 1].id);
  };

  return (
    <div className="flex items-center gap-3">
      {error && (
        <span className="text-sm text-destructive bg-destructive/10 px-2 py-1 rounded">{error}</span>
      )}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-sm text-muted-foreground">左侧:</span>
          <Select
            value={String(leftSelection)}
            onValueChange={handleLeftChange}
            disabled={disableLeftSelect || loading}
          >
            <SelectTrigger className="h-9 w-[140px] px-3 text-sm hover:border-primary focus:border-primary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent align="start" className="max-h-64">
              {uniqueLefts.map((left) => (
                <SelectItem key={left} value={String(left)} className="text-sm">
                  <span className="block truncate">
                    {formatDocumentLabel(
                      left,
                      documentLookup,
                      left === currentPair.left_document_id ? currentLeftDocument : null,
                    )}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-sm text-muted-foreground">右侧:</span>
          <Select
            value={String(rightSelection)}
            onValueChange={handleRightChange}
            disabled={disableRightSelect || loading}
          >
            <SelectTrigger className="h-9 w-[140px] px-3 text-sm hover:border-primary focus:border-primary">
              <SelectValue />
            </SelectTrigger>
            <SelectContent align="start" className="max-h-64">
              {pairsForLeft.map((pair) => (
                <SelectItem key={pair.id} value={String(pair.id)} className="text-sm">
                  <span className="block truncate">
                    {formatDocumentLabel(
                      pair.right_document_id,
                      documentLookup,
                      pair.id === currentPair.id ? currentRightDocument : null,
                    )}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="h-9 w-px bg-border" />
      <div className="flex items-center gap-1">
        <Button
          variant="default"
          size="sm"
          onClick={handlePrev}
          disabled={!canPrev || loading}
          title="上一个配对"
        >
          <ChevronLeft className="h-4 w-4" />
          <span className="sr-only">上一个</span>
        </Button>
        <Button
          variant="default"
          size="sm"
          onClick={handleNext}
          disabled={!canNext || loading}
          title="下一个配对"
        >
          <ChevronRight className="h-4 w-4" />
          <span className="sr-only">下一个</span>
        </Button>
      </div>
    </div>
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

  if (!segments.length) {
    return null;
  }


  return <>{segments.map((segment, index) => {
    if (!segment.matchKey) {
      return <span key={`plain-${index}`}>{segment.text}</span>;
    }

    const match = matches.find(m => m.key === segment.matchKey) || null;
    const isActive = activeKey === segment.matchKey;
    const ordinal = segment.ordinal ?? 0;

    // Check if this segment has overlapping matches
    const hasOverlaps = segment.allMatches && segment.allMatches.length > 1;
    const overlappingMatches = hasOverlaps
      ? segment.allMatches!.map(am => matches.find(m => m.key === am.matchKey)).filter(Boolean)
      : [match].filter(Boolean);
    const baseClasses =
      'rounded-sm transition cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-primary/60 break-words match-highlight';

    // Determine background color based on final_score
    const colorClasses = getScoreColorClasses(match?.group?.final_score, true);
    const backgroundClasses =
      segment.mode === 'block'
        ? `${colorClasses} text-foreground`
        : `${colorClasses} text-foreground underline decoration-black decoration-2 underline-offset-2`;
    const activeClasses = isActive ? 'active-match' : '';

    // Add visual indicator for multiple overlapping matches
    const overlappingClasses = hasOverlaps
      ? 'ring-2 ring-primary/40 ring-offset-1'
      : '';

    if (!match) {
      // No match found - still render for visual consistency
      return (
        <mark
          key={`highlight-${index}`}
          id={makeHighlightId(side, segment.matchKey || '', ordinal)}
          className={cn(baseClasses, backgroundClasses, activeClasses, overlappingClasses)}
          tabIndex={0}
          data-match-keys={segment.matchKey}
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
          className={cn(baseClasses, backgroundClasses, activeClasses, overlappingClasses)}
          tabIndex={0}
          data-match-keys={segment.matchKey}
          data-side={side}
          data-mode={segment.mode}
          data-ordinal={ordinal}
          onClick={() => {
            onSelectMatch(segment.matchKey!);
          }}
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
  })}</>;
});

function prepareIntervals(text: string, matches: NormalisedMatch[], side: Side): HighlightInterval[] {
  if (!text) {
    return [];
  }
  const raw: HighlightInterval[] = [];
  const ordinalMap = new Map<string, number>();
  const seen = new Set<string>();
  matches.forEach((match) => {
    // Try document_spans first, fallback to paragraph_spans
    const spans = match.group.document_spans ?? match.group.paragraph_spans ?? [];


    const sideSpans = spans
      .map((span) => ({
        start: side === 'left' ? span.left_start : span.right_start,
        end: side === 'left' ? span.left_end : span.right_end,
      }))
      .filter((item) => {
        // More robust validation - allow zero positions
        return Number.isFinite(item.start) && Number.isFinite(item.end) &&
               item.start >= 0 && item.end >= 0 && item.end > item.start;
      });

    if (!sideSpans.length) {
      return;
    }

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
        if (seen.has(key)) {
          return;
        }
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
  return raw;  // 不再需要 mergeIntervals，因为 buildSegmentsWithOverlap 会处理重叠
}

function makeHighlightId(side: Side, matchKey: string, ordinal: number) {
  return `${side}-match-${matchKey}-${ordinal}`;
}

function formatMetricTitle(key: string): string {
  return key
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function formatMetricValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toFixed(3);
  }
  return String(value);
}

function formatAverageScore(values: Array<number | null | undefined>): string {
  const numeric = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  if (!numeric.length) {
    return '—';
  }
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
  if (!matches.length) {
    return [];
  }

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
  if (!details?.length) {
    return undefined;
  }
  return [...details].sort((a, b) => (b.final_score ?? 0) - (a.final_score ?? 0))[0];
}

function extractExcerpt(fullText: string, start: number, end: number, radius = 80): string {
  if (!fullText) {
    return '—';
  }

  const safeStart = Number.isFinite(start) ? Math.max(0, Math.min(start, fullText.length)) : 0;
  const safeEnd = Number.isFinite(end) ? Math.max(safeStart, Math.min(end, fullText.length)) : safeStart;

  const windowStart = Math.max(0, safeStart - radius);
  const windowEnd = Math.min(fullText.length, safeEnd + radius);
  let snippet = fullText.slice(windowStart, windowEnd).replace(/\s+/g, ' ').trim();

  if (!snippet) {
    return '—';
  }

  if (windowStart > 0) {
    snippet = `…${snippet}`;
  }
  if (windowEnd < fullText.length) {
    snippet = `${snippet}…`;
  }
  return snippet;
}

function scoreStyleForMarkdown(score: number): string {
  if (!Number.isFinite(score)) {
    return 'background:#f8fafc;color:#1f2937;';
  }
  if (score >= 0.9) {
    return 'background:rgba(239,68,68,0.18);color:#7f1d1d;';
  }
  if (score >= 0.85) {
    return 'background:rgba(249,115,22,0.16);color:#7c2d12;';
  }
  if (score >= 0.8) {
    return 'background:rgba(250,204,21,0.14);color:#713f12;';
  }
  return 'background:#f8fafc;color:#1f2937;';
}

interface ResolveExcerptOptions {
  detail: MatchDetailModel | undefined;
  side: 'left' | 'right';
  fullText: string | null;
  radius?: number;
}

function resolveExcerpt({ detail, side, fullText, radius = 80 }: ResolveExcerptOptions): string {
  if (!detail) {
    return '—';
  }

  const direct = side === 'left' ? detail.left_excerpt : detail.right_excerpt;
  if (direct && direct.trim()) {
    return direct.trim();
  }

  const spans = detail.spans;
  if (!Array.isArray(spans) || spans.length === 0) {
    return '—';
  }
  const first = spans[0];
  const start = side === 'left' ? first.left_start : first.right_start;
  const end = side === 'left' ? first.left_end : first.right_end;

  if (typeof start !== 'number' || typeof end !== 'number' || end <= start) {
    return '—';
  }

  if (!fullText) {
    return '—';
  }

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

function formatDocumentLabel(
  documentId: number,
  lookup: DocumentLookup,
  fallback?: DocumentDetail | DocumentSummary | null,
) {
  const info = lookup[documentId] ?? fallback ?? null;
  if (!info) {
    return `文档 ${documentId}`;
  }
  const name = info.title || info.filename || `文档 ${info.id ?? documentId}`;
  return name;
}


function formatScore(value: number | null | undefined) {
  if (value == null) {
    return '—';
  }
  return value.toFixed(3);
}

function getScoreColorClasses(score: number | null | undefined, isBackground = false): string {
  const finalScore = score || 0;
  if (finalScore > 0.9) {
    return isBackground ? 'bg-red-400/50 hover:bg-red-400/65' : 'text-red-600';
  } else if (finalScore >= 0.85) {
    return isBackground ? 'bg-orange-300/50 hover:bg-orange-300/65' : 'text-orange-600';
  } else if (finalScore >= 0.8) {
    return isBackground ? 'bg-yellow-300/50 hover:bg-yellow-300/65' : 'text-yellow-600';
  } else if (finalScore >= 0.7) {
    return isBackground ? 'bg-green-300/50 hover:bg-green-300/65' : 'text-green-600';
  } else {
    return isBackground ? 'bg-gray-200/50 hover:bg-gray-200/65' : 'text-gray-500';
  }
}

// 删除了 clamp 函数 - 不再需要

export default PlanComparePage;
