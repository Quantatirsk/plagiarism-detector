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
  onReloadPairs: () => void;
  onBack: () => void;
  isTransitioning?: boolean;
}

type NormalisedMatch = {
  key: string;
  group: MatchGroupModel;
  details: MatchDetailModel[];
};

type HighlightMode = 'block' | 'fragment';

type HighlightInterval = {
  start: number;
  end: number;
  matchKey: string;
  ordinal: number;
  side: Side;
  mode: HighlightMode;
};

type Segment = {
  text: string;
  matchKey?: string;
  ordinal?: number;
  mode?: HighlightMode;
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
  onReloadPairs,
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
    return report.groups.map((group) => ({
      key: `${group.id}-${group.left_chunk_id}-${group.right_chunk_id}`,
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

  const leftIntervals = useMemo(() => prepareIntervals(leftDocument?.processed_text ?? '', matches, 'left'), [leftDocument?.processed_text, matches]);
  const rightIntervals = useMemo(() => prepareIntervals(rightDocument?.processed_text ?? '', matches, 'right'), [rightDocument?.processed_text, matches]);

  const leftSegments = useMemo(() => buildSegments(leftDocument?.processed_text ?? '', leftIntervals), [leftDocument?.processed_text, leftIntervals]);
  const rightSegments = useMemo(() => buildSegments(rightDocument?.processed_text ?? '', rightIntervals), [rightDocument?.processed_text, rightIntervals]);

  const jumpToMatch = useCallback((target: Side, matchKey: string, shouldFocus: boolean = false) => {
    if (!matchKey) {
      return;
    }
    const targetNode = document.querySelector<HTMLElement>(`mark[data-side="${target}"][data-match-key="${matchKey}"]`);
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
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="text-xs text-muted-foreground hover:text-foreground"
              onClick={onBack}
            >
              ← 返回任务
            </Button>
            <span>
              {formatDocumentLabel(report.pair.left_document_id, documentLookup, leftDocument)} ↔{' '}
              {formatDocumentLabel(report.pair.right_document_id, documentLookup, rightDocument)}
            </span>
          </div>
        }
        subtitle={`匹配数量：${matches.length}`}
      />

      <div className="border-b border-border bg-background px-6 py-3">
        <PairSwitcher
          pairs={pairOptions}
          currentPair={currentPair}
          loading={pairsLoading}
          documentLookup={documentLookup}
          currentLeftDocument={leftDocument}
          currentRightDocument={rightDocument}
          onSwitchPair={onSwitchPair}
          onReloadPairs={onReloadPairs}
        />
        {pairsError && (
          <div className="mt-2 text-xs text-destructive">刷新配对列表失败：{pairsError}</div>
        )}
      </div>

      <div className="relative flex flex-1 min-h-0 divide-x divide-border bg-background">
        {isTransitioning && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
            <div className="animate-pulse text-sm text-muted-foreground">正在切换文档...</div>
          </div>
        )}
          <aside className="flex flex-col w-72 min-w-[18rem] border-r border-border bg-card shadow-sm">
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
                          <span className={cn("text-xs font-mono", getScoreColorClasses(group.final_score))}>{formatScore(group.final_score)}</span>
                        </div>
                        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                          <span>语义 {formatScore(group.semantic_score)}</span>
                          <span>词汇 {formatScore(group.lexical_overlap)}</span>
                          <span>片段 {group.match_count}</span>
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </aside>

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
  documentLookup: DocumentLookup;
  currentLeftDocument: DocumentDetail | null;
  currentRightDocument: DocumentDetail | null;
  onSwitchPair: (pairId: number) => void;
  onReloadPairs: () => void;
}

function PairSwitcher({
  pairs,
  currentPair,
  loading,
  documentLookup,
  currentLeftDocument,
  currentRightDocument,
  onSwitchPair,
  onReloadPairs,
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
    <div className="flex items-center gap-4 text-xs text-muted-foreground sm:text-sm">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">左侧文档</span>
          <Select
            value={String(leftSelection)}
            onValueChange={handleLeftChange}
            disabled={disableLeftSelect || loading}
          >
            <SelectTrigger className="h-8 w-[200px] px-3 text-sm">
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
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">右侧文档</span>
          <Select
            value={String(rightSelection)}
            onValueChange={handleRightChange}
            disabled={disableRightSelect || loading}
          >
            <SelectTrigger className="h-8 w-[200px] px-3 text-sm">
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
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" className="h-8" onClick={handlePrev} disabled={!canPrev || loading}>
          上一个
        </Button>
        <Button variant="outline" size="sm" className="h-8" onClick={handleNext} disabled={!canNext || loading}>
          下一个
        </Button>
        <Button variant="outline" size="sm" className="h-8" onClick={onReloadPairs} disabled={loading}>
          刷新
        </Button>
        {loading && <span className="text-xs">刷新中…</span>}
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

  // Debug: Log segment and match information
  React.useEffect(() => {
    const segmentsWithKeys = segments.filter(s => s.matchKey);
    const uniqueSegmentKeys = new Set(segmentsWithKeys.map(s => s.matchKey));
    const availableMatchKeys = new Set(matches.map(m => m.key));
    const missingKeys = Array.from(uniqueSegmentKeys).filter(key => !availableMatchKeys.has(key!));

    console.log(`[${side}] RenderSegments Debug:`, {
      totalSegments: segments.length,
      segmentsWithKeys: segmentsWithKeys.length,
      uniqueKeys: uniqueSegmentKeys.size,
      totalMatches: matches.length,
      sampleSegmentKeys: Array.from(uniqueSegmentKeys).slice(0, 3),
      sampleMatchKeys: matches.slice(0, 3).map(m => m.key),
      missingKeysCount: missingKeys.length
    });

    if (missingKeys.length > 0) {
      console.warn(`[${side}] Segments with matchKeys not found in matches:`, {
        missingKeys,
        sampleMissingSegment: segments.find(s => s.matchKey && missingKeys.includes(s.matchKey))
      });
    }
  }, [segments, matches, side]);

  return <>{segments.map((segment, index) => {
    if (!segment.matchKey) {
      return <span key={`plain-${index}`}>{segment.text}</span>;
    }

    // Find all matches for this segment (handle 1-to-many cases)
    const matchesForSegment = matches.filter(m => m.key === segment.matchKey);
    const match = matchesForSegment[0] || null;
    const isActive = activeKey === segment.matchKey;
    const ordinal = segment.ordinal ?? 0;
    const baseClasses =
      'rounded-sm transition cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-primary/60 break-words match-highlight';

    // Determine background color based on final_score
    const colorClasses = getScoreColorClasses(match?.group?.final_score, true);
    const backgroundClasses =
      segment.mode === 'block'
        ? `${colorClasses} text-foreground`
        : `${colorClasses} text-foreground underline decoration-black decoration-2 underline-offset-2`;
    const activeClasses = isActive ? 'active-match' : '';

    if (!match) {
      // No match found - just render the mark without popover
      console.warn('No match found for segment:', {
        matchKey: segment.matchKey,
        text: segment.text.substring(0, 50) + '...',
        availableMatches: matches.map(m => m.key)
      });

      return (
        <mark
          key={`highlight-${index}`}
          id={makeHighlightId(side, segment.matchKey, ordinal)}
          className={cn(baseClasses, backgroundClasses, activeClasses)}
          tabIndex={0}
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
      <MatchInfoTooltip key={`highlight-${index}`} match={match} allMatches={matchesForSegment}>
        <mark
          id={makeHighlightId(side, segment.matchKey, ordinal)}
          className={cn(baseClasses, backgroundClasses, activeClasses)}
          tabIndex={0}
          data-match-key={segment.matchKey}
          data-side={side}
          data-mode={segment.mode}
          data-ordinal={ordinal}
          onClick={() => {
            console.log('Mark clicked:', {
              matchKey: segment.matchKey,
              hasMatch: !!match,
              matchDetails: match?.details?.length || 0
            });
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

    // Debug logging
    if (!match.group.document_spans?.length) {
      console.log(`[${side}] Match ${match.key} using paragraph_spans (${match.group.paragraph_spans?.length || 0} spans)`);
    }

    const sideSpans = spans
      .map((span) => ({
        start: side === 'left' ? span.left_start : span.right_start,
        end: side === 'left' ? span.left_end : span.right_end,
      }))
      .filter((item) => Number.isFinite(item.start) && Number.isFinite(item.end) && item.end > item.start);

    if (!sideSpans.length) {
      console.warn(`[${side}] Match ${match.key} has no valid spans after filtering`, {
        documentSpans: match.group.document_spans?.length || 0,
        paragraphSpans: match.group.paragraph_spans?.length || 0,
        group: match.group
      });
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
        start: clampNumber(minStart),
        end: clampNumber(maxEnd),
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
          start: clampNumber(start),
          end: clampNumber(end),
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
  return mergeIntervals(raw);
}

function mergeIntervals(intervals: HighlightInterval[]): HighlightInterval[] {
  if (intervals.length <= 1) {
    return intervals.slice();
  }
  const sorted = intervals
    .map((interval) => ({ ...interval }))
    .sort((a, b) => (a.start !== b.start ? a.start - b.start : a.end - b.end));

  const merged: HighlightInterval[] = [];
  const MERGE_DISTANCE = 16;

  sorted.forEach((interval) => {
    const last = merged[merged.length - 1];
    if (
      last &&
      last.matchKey === interval.matchKey &&
      last.mode === interval.mode &&
      interval.start <= last.end + MERGE_DISTANCE
    ) {
      last.end = Math.max(last.end, interval.end);
      last.ordinal = Math.min(last.ordinal, interval.ordinal);
    } else {
      merged.push(interval);
    }
  });
  return merged;
}

function buildSegments(text: string, intervals: HighlightInterval[]): Segment[] {
  if (!intervals.length) {
    return text ? [{ text }] : [];
  }
  const sorted = intervals
    .map((interval) => ({ ...interval }))
    .filter((interval) => interval.end > interval.start)
    .sort((a, b) => (a.start !== b.start ? a.start - b.start : a.end - b.end));

  const segments: Segment[] = [];
  let cursor = 0;
  sorted.forEach((interval) => {
    const start = clamp(interval.start, text.length);
    const end = clamp(interval.end, text.length);
    if (end <= cursor) {
      return;
    }

    const clippedStart = Math.max(start, cursor);
    if (clippedStart > cursor) {
      segments.push({ text: text.slice(cursor, clippedStart) });
    }

    if (end > clippedStart) {
      segments.push({
        text: text.slice(clippedStart, end),
        matchKey: interval.matchKey,
        ordinal: interval.ordinal,
        mode: interval.mode,
      });
      cursor = end;
    } else {
      cursor = clippedStart;
    }
  });
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor) });
  }
  return segments;
}

function makeHighlightId(side: Side, matchKey: string, ordinal: number) {
  return `${side}-match-${matchKey}-${ordinal}`;
}

function clampNumber(value: number) {
  return Number.isFinite(value) ? value : 0;
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

function clamp(value: number, length: number) {
  if (Number.isNaN(value) || !Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(Math.floor(value), length));
}

export default PlanComparePage;
