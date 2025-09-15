import { useRef, useState, useCallback, useEffect } from 'react';

// 独立的最小匹配类型，避免跨模块耦合
export interface ProcessedMatch {
  id: string;
  similarity_score: number;
  startIndex: number;
  sourceDocument: 1 | 2;
  query_text: string;
  matched_text: string;
}

interface SyncScrollingOptions {
  enabled?: boolean;
  debounceMs?: number;
  smoothScroll?: boolean;
  syncThreshold?: number;
}

interface PositionMapping {
  document1: { [key: string]: number };
  document2: { [key: string]: number };
}

export interface EnhancedSyncScrollingHook {
  leftPanelRef: React.RefObject<HTMLDivElement>;
  rightPanelRef: React.RefObject<HTMLDivElement>;
  isScrollSyncing: boolean;
  scrollToMatch: (matchId: string) => void;
  scrollToPosition: (position: number, sourcePanel?: 'left' | 'right') => void;
  setSyncEnabled: (enabled: boolean) => void;
  positionMapping: PositionMapping;
}

/**
 * Enhanced synchronized scrolling hook for dual document viewing
 */
export const useEnhancedSyncScrolling = (
  matches: ProcessedMatch[],
  options: SyncScrollingOptions = {}
): EnhancedSyncScrollingHook => {
  const {
    enabled = true,
    debounceMs = 100,
    smoothScroll = true,
    syncThreshold = 10,
  } = options;

  const leftPanelRef = useRef<HTMLDivElement>(null);
  const rightPanelRef = useRef<HTMLDivElement>(null);
  const [isScrollSyncing, setIsScrollSyncing] = useState(false);
  const [syncEnabled, setSyncEnabled] = useState(enabled);
  const [positionMapping, setPositionMapping] = useState<PositionMapping>({
    document1: {},
    document2: {},
  });

  // Debounce timer
  const debounceTimer = useRef<NodeJS.Timeout | undefined>(undefined);
  const isInternalSync = useRef(false);

  /**
   * Calculate position mapping based on matches
   */
  const calculatePositionMapping = useCallback(() => {
    const doc1Positions: { [key: string]: number } = {};
    const doc2Positions: { [key: string]: number } = {};

    matches.forEach(match => {
      if (match.sourceDocument === 1) {
        doc1Positions[match.id] = match.startIndex;
      } else {
        doc2Positions[match.id] = match.startIndex;
      }
    });

    setPositionMapping({
      document1: doc1Positions,
      document2: doc2Positions,
    });
  }, [matches]);

  /**
   * Get element position within its container
   */
  const getElementPosition = useCallback((element: Element, container: HTMLDivElement): number => {
    const containerRect = container.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();
    return elementRect.top - containerRect.top + container.scrollTop;
  }, []);

  /**
   * Find corresponding match in the other document
   */
  const findCorrespondingMatch = useCallback((matchId: string): ProcessedMatch | null => {
    const currentMatch = matches.find(m => m.id === matchId);
    if (!currentMatch) return null;

    // Find match with same similarity score and similar content in the other document
    const correspondingMatch = matches.find(m =>
      m.id !== matchId &&
      m.sourceDocument !== currentMatch.sourceDocument &&
      Math.abs(m.similarity_score - currentMatch.similarity_score) < 0.01 &&
      (m.query_text === currentMatch.matched_text || m.matched_text === currentMatch.query_text)
    );

    return correspondingMatch || null;
  }, [matches]);

  /**
   * Sync scroll position between panels
   */
  const syncScrollPosition = useCallback((
    sourcePanel: HTMLDivElement,
    targetPanel: HTMLDivElement,
    smoothSync: boolean = true
  ) => {
    if (isInternalSync.current) return;

    const sourceScrollRatio = sourcePanel.scrollTop / (sourcePanel.scrollHeight - sourcePanel.clientHeight);
    const targetScrollTop = sourceScrollRatio * (targetPanel.scrollHeight - targetPanel.clientHeight);

    isInternalSync.current = true;
    setIsScrollSyncing(true);

    if (smoothSync && smoothScroll) {
      targetPanel.scrollTo({
        top: targetScrollTop,
        behavior: 'smooth',
      });
    } else {
      targetPanel.scrollTop = targetScrollTop;
    }

    // Reset sync flag after a short delay
    setTimeout(() => {
      isInternalSync.current = false;
      setIsScrollSyncing(false);
    }, debounceMs + 50);
  }, [smoothScroll, debounceMs]);

  /**
   * Handle scroll event with debouncing
   */
  const handleScroll = useCallback((
    event: Event,
    sourcePanel: 'left' | 'right'
  ) => {
    if (!syncEnabled || isInternalSync.current) return;

    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(() => {
      const leftPanel = leftPanelRef.current;
      const rightPanel = rightPanelRef.current;

      if (!leftPanel || !rightPanel) return;

      const source = sourcePanel === 'left' ? leftPanel : rightPanel;
      const target = sourcePanel === 'left' ? rightPanel : leftPanel;

      syncScrollPosition(source, target);
    }, debounceMs);
  }, [syncEnabled, debounceMs, syncScrollPosition]);

  /**
   * Scroll to specific match with highlighting
   */
  const scrollToMatch = useCallback((matchId: string) => {
    const leftPanel = leftPanelRef.current;
    const rightPanel = rightPanelRef.current;

    if (!leftPanel || !rightPanel) return;

    // Find the match element
    const matchElement = document.querySelector(`[data-match-id="${matchId}"]`) as HTMLElement;
    if (!matchElement) return;

    // Determine which panel contains this match
    const isInLeftPanel = leftPanel.contains(matchElement);
    const targetPanel = isInLeftPanel ? leftPanel : rightPanel;
    const otherPanel = isInLeftPanel ? rightPanel : leftPanel;

    // Calculate position and scroll to match
    const elementPosition = getElementPosition(matchElement, targetPanel);
    const scrollPosition = Math.max(0, elementPosition - targetPanel.clientHeight / 2);

    // Scroll to match with smooth animation
    targetPanel.scrollTo({
      top: scrollPosition,
      behavior: 'smooth',
    });

    // Highlight the match temporarily
    matchElement.classList.add('ring-4', 'ring-blue-400', 'ring-opacity-75');
    setTimeout(() => {
      matchElement.classList.remove('ring-4', 'ring-blue-400', 'ring-opacity-75');
    }, 2000);

    // Try to scroll to corresponding match in the other panel
    const correspondingMatch = findCorrespondingMatch(matchId);
    if (correspondingMatch) {
      const correspondingElement = document.querySelector(`[data-match-id="${correspondingMatch.id}"]`) as HTMLElement;
      if (correspondingElement) {
        const correspondingPosition = getElementPosition(correspondingElement, otherPanel);
        const correspondingScrollPosition = Math.max(0, correspondingPosition - otherPanel.clientHeight / 2);

        setTimeout(() => {
          otherPanel.scrollTo({
            top: correspondingScrollPosition,
            behavior: 'smooth',
          });

          // Highlight corresponding match
          correspondingElement.classList.add('ring-4', 'ring-green-400', 'ring-opacity-75');
          setTimeout(() => {
            correspondingElement.classList.remove('ring-4', 'ring-green-400', 'ring-opacity-75');
          }, 2000);
        }, 200);
      }
    }
  }, [getElementPosition, findCorrespondingMatch]);

  /**
   * Scroll to specific position in document
   */
  const scrollToPosition = useCallback((
    position: number,
    sourcePanel: 'left' | 'right' = 'left'
  ) => {
    const leftPanel = leftPanelRef.current;
    const rightPanel = rightPanelRef.current;

    if (!leftPanel || !rightPanel) return;

    const targetPanel = sourcePanel === 'left' ? leftPanel : rightPanel;
    const maxScrollTop = targetPanel.scrollHeight - targetPanel.clientHeight;
    const scrollPosition = Math.min(position, maxScrollTop);

    targetPanel.scrollTo({
      top: scrollPosition,
      behavior: 'smooth',
    });

    // Sync the other panel
    setTimeout(() => {
      const otherPanel = sourcePanel === 'left' ? rightPanel : leftPanel;
      syncScrollPosition(targetPanel, otherPanel);
    }, 100);
  }, [syncScrollPosition]);

  /**
   * Set up scroll event listeners
   */
  useEffect(() => {
    const leftPanel = leftPanelRef.current;
    const rightPanel = rightPanelRef.current;

    if (!leftPanel || !rightPanel) return;

    const leftScrollHandler = (e: Event) => handleScroll(e, 'left');
    const rightScrollHandler = (e: Event) => handleScroll(e, 'right');

    leftPanel.addEventListener('scroll', leftScrollHandler, { passive: true });
    rightPanel.addEventListener('scroll', rightScrollHandler, { passive: true });

    return () => {
      leftPanel.removeEventListener('scroll', leftScrollHandler);
      rightPanel.removeEventListener('scroll', rightScrollHandler);
    };
  }, [handleScroll]);

  /**
   * Update position mapping when matches change
   */
  useEffect(() => {
    calculatePositionMapping();
  }, [calculatePositionMapping]);

  /**
   * Clean up debounce timer
   */
  useEffect(() => {
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, []);

  return {
    leftPanelRef,
    rightPanelRef,
    isScrollSyncing,
    scrollToMatch,
    scrollToPosition,
    setSyncEnabled,
    positionMapping,
  };
};
