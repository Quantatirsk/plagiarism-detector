import { useRef, useCallback, useEffect, useState } from 'react';

// 独立的最小匹配类型，避免依赖不存在的模块
export interface ProcessedMatch {
  id: string;
  similarity_score: number;
  startIndex: number;
}

export interface ScrollSyncOptions {
  enabled: boolean;
  debounceMs?: number;
  smoothScroll?: boolean;
}

export interface ScrollPosition {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

export interface SyncScrollingHook {
  leftPanelRef: React.RefObject<HTMLElement | null>;
  rightPanelRef: React.RefObject<HTMLElement | null>;
  isScrollSyncing: boolean;
  scrollToMatch: (matchId: string) => void;
  scrollToPosition: (panel: 'left' | 'right', position: number) => void;
  syncScrollPosition: (fromPanel: 'left' | 'right') => void;
}

/**
 * Hook for synchronized scrolling between two document panels
 */
export const useSyncScrolling = (
  matches: ProcessedMatch[],
  options: ScrollSyncOptions = { enabled: true, debounceMs: 50, smoothScroll: true }
): SyncScrollingHook => {
  const leftPanelRef = useRef<HTMLElement>(null);
  const rightPanelRef = useRef<HTMLElement>(null);
  const [isScrollSyncing, setIsScrollSyncing] = useState(false);

  // Debounce timer for scroll events
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastScrollSource = useRef<'left' | 'right' | null>(null);

  /**
   * Get scroll position information
   */
  const getScrollInfo = useCallback((element: HTMLElement): ScrollPosition => {
    return {
      scrollTop: element.scrollTop,
      scrollHeight: element.scrollHeight,
      clientHeight: element.clientHeight,
    };
  }, []);

  /**
   * Calculate relative scroll position (0-1)
   */
  const getRelativeScrollPosition = useCallback((element: HTMLElement): number => {
    const { scrollTop, scrollHeight, clientHeight } = getScrollInfo(element);
    const maxScroll = scrollHeight - clientHeight;
    return maxScroll > 0 ? scrollTop / maxScroll : 0;
  }, [getScrollInfo]);

  /**
   * Scroll to a specific match by ID
   */
  const scrollToMatch = useCallback((matchId: string) => {
    const match = matches.find(m => m.id === matchId);
    if (!match || !leftPanelRef.current) return;

    // Find element with the match ID or calculate position
    const matchElement = leftPanelRef.current.querySelector(`[data-match-id="${matchId}"]`);

    if (matchElement) {
      matchElement.scrollIntoView({
        behavior: options.smoothScroll ? 'smooth' : 'auto',
        block: 'center',
      });
    } else {
      // Fallback: estimate position based on match.startIndex
      const textContent = leftPanelRef.current.textContent || '';
      const relativePosition = match.startIndex / textContent.length;
      const targetScrollTop = relativePosition * (leftPanelRef.current.scrollHeight - leftPanelRef.current.clientHeight);

      leftPanelRef.current.scrollTo({
        top: targetScrollTop,
        behavior: options.smoothScroll ? 'smooth' : 'auto',
      });
    }
  }, [matches, options.smoothScroll]);

  /**
   * Scroll to a specific position in a panel
   */
  const scrollToPosition = useCallback((panel: 'left' | 'right', position: number) => {
    const panelRef = panel === 'left' ? leftPanelRef : rightPanelRef;
    if (!panelRef.current) return;

    const element = panelRef.current;
    const maxScroll = element.scrollHeight - element.clientHeight;
    const targetScrollTop = Math.max(0, Math.min(maxScroll, position));

    element.scrollTo({
      top: targetScrollTop,
      behavior: options.smoothScroll ? 'smooth' : 'auto',
    });
  }, [options.smoothScroll]);

  /**
   * Synchronize scroll position between panels
   */
  const syncScrollPosition = useCallback((fromPanel: 'left' | 'right') => {
    if (!options.enabled || isScrollSyncing) return;

    const sourceRef = fromPanel === 'left' ? leftPanelRef : rightPanelRef;
    const targetRef = fromPanel === 'left' ? rightPanelRef : leftPanelRef;

    if (!sourceRef.current || !targetRef.current) return;

    setIsScrollSyncing(true);
    lastScrollSource.current = fromPanel;

    // Calculate relative position and sync
    const relativePosition = getRelativeScrollPosition(sourceRef.current);
    const targetMaxScroll = targetRef.current.scrollHeight - targetRef.current.clientHeight;
    const targetScrollTop = relativePosition * targetMaxScroll;

    targetRef.current.scrollTo({
      top: Math.max(0, targetScrollTop),
      behavior: 'auto', // Use instant scroll for sync to avoid lag
    });

    // Reset syncing flag after a short delay
    setTimeout(() => {
      setIsScrollSyncing(false);
      lastScrollSource.current = null;
    }, 100);
  }, [options.enabled, isScrollSyncing, getRelativeScrollPosition]);

  /**
   * Handle scroll events with debouncing
   */
  const handleScroll = useCallback((panel: 'left' | 'right') => {
    if (!options.enabled || isScrollSyncing || lastScrollSource.current === panel) return;

    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }

    scrollTimeoutRef.current = setTimeout(() => {
      syncScrollPosition(panel);
    }, options.debounceMs || 50);
  }, [options.enabled, options.debounceMs, isScrollSyncing, syncScrollPosition]);

  /**
   * Set up scroll event listeners
   */
  useEffect(() => {
    if (!options.enabled) return;

    const leftElement = leftPanelRef.current;
    const rightElement = rightPanelRef.current;

    if (!leftElement || !rightElement) return;

    const handleLeftScroll = () => handleScroll('left');
    const handleRightScroll = () => handleScroll('right');

    leftElement.addEventListener('scroll', handleLeftScroll, { passive: true });
    rightElement.addEventListener('scroll', handleRightScroll, { passive: true });

    return () => {
      leftElement.removeEventListener('scroll', handleLeftScroll);
      rightElement.removeEventListener('scroll', handleRightScroll);

      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, [options.enabled, handleScroll]);

  return {
    leftPanelRef,
    rightPanelRef,
    isScrollSyncing,
    scrollToMatch,
    scrollToPosition,
    syncScrollPosition,
  };
};
