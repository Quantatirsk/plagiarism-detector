/**
 * Simplified utilities for handling overlapping highlights
 */

export type HighlightInterval = {
  start: number;
  end: number;
  matchKey: string;
  ordinal: number;
  side: 'left' | 'right';
  mode: 'block' | 'fragment';
};

export type Segment = {
  text: string;
  matchKey?: string;
  ordinal?: number;
  mode?: 'block' | 'fragment';
  // 用于存储该段落包含的所有匹配信息（处理重叠）
  allMatches?: Array<{
    matchKey: string;
    ordinal: number;
    mode: 'block' | 'fragment';
  }>;
};

/**
 * Build segments from text and intervals, handling overlaps properly
 */
export function buildSegmentsWithOverlap(text: string, intervals: HighlightInterval[]): Segment[] {
  if (!text || !intervals.length) {
    return text ? [{ text }] : [];
  }

  // Sort intervals by start position
  const sorted = [...intervals]
    .filter(interval => interval.end > interval.start)
    .sort((a, b) => a.start !== b.start ? a.start - b.start : a.end - b.end);

  // Create events for interval boundaries
  const events: Array<{pos: number; type: 'start' | 'end'; interval: HighlightInterval}> = [];
  sorted.forEach(interval => {
    events.push({pos: interval.start, type: 'start', interval});
    events.push({pos: interval.end, type: 'end', interval});
  });

  // Sort events by position
  events.sort((a, b) => {
    if (a.pos !== b.pos) return a.pos - b.pos;
    // Process 'end' events before 'start' events at the same position
    return a.type === 'end' ? -1 : 1;
  });

  // Build segments
  const segments: Segment[] = [];
  const activeIntervals = new Set<HighlightInterval>();
  let cursor = 0;

  events.forEach((event, i) => {
    const pos = Math.min(event.pos, text.length);

    // Add plain text before this position
    if (pos > cursor) {
      if (activeIntervals.size === 0) {
        // No active intervals - plain text
        segments.push({ text: text.slice(cursor, pos) });
      } else {
        // Active intervals - highlighted text
        const allMatches = Array.from(activeIntervals).map(interval => ({
          matchKey: interval.matchKey,
          ordinal: interval.ordinal,
          mode: interval.mode
        }));

        // Use the first match as primary (for compatibility)
        const primary = allMatches[0];
        segments.push({
          text: text.slice(cursor, pos),
          matchKey: primary.matchKey,
          ordinal: primary.ordinal,
          mode: primary.mode,
          allMatches: allMatches
        });
      }
      cursor = pos;
    }

    // Update active intervals
    if (event.type === 'start') {
      activeIntervals.add(event.interval);
    } else {
      activeIntervals.delete(event.interval);
    }
  });

  // Add remaining text
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor) });
  }

  return segments;
}