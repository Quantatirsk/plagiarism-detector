import React from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './tooltip';

interface MatchInfoProps {
  children: React.ReactNode;
  match: any;
  allMatches?: any[];
}

export function MatchInfoTooltip({ children, match, allMatches }: MatchInfoProps) {
  if (!match) {
    return <>{children}</>;
  }

  const hasMultipleMatches = allMatches && allMatches.length > 1;
  const hasDetails = match.details && match.details.length > 0;

  const formatScore = (value: number | null | undefined) => {
    if (value == null) return '—';
    return (value * 100).toFixed(0);
  };

  return (
    <TooltipProvider>
      <Tooltip delayDuration={0}>
        <TooltipTrigger asChild>
          {children}
        </TooltipTrigger>
        <TooltipContent className="max-w-md p-0 bg-background text-foreground border" side="top" align="start">
          {!hasDetails ? (
            <div className="p-3 space-y-1.5 text-xs">
              <div className="font-medium mb-1">段落级匹配</div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-muted-foreground">
                <div>综合: {formatScore(match.group.final_score)}%</div>
                <div>语义: {formatScore(match.group.semantic_score)}%</div>
                <div>词汇: {formatScore(match.group.lexical_overlap)}%</div>
                <div>覆盖: {formatScore(match.group.alignment_ratio)}%</div>
              </div>
              {hasMultipleMatches && (
                <div className="text-muted-foreground mt-1 pt-1 border-t">共{allMatches.length}个匹配</div>
              )}
            </div>
          ) : (
            <div className="max-h-72 overflow-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-muted">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">片段</th>
                    <th className="px-3 py-2 text-right font-medium">综合</th>
                    <th className="px-3 py-2 text-right font-medium">语义</th>
                    <th className="px-3 py-2 text-right font-medium">词汇</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {match.details.slice(0, 10).map((detail: any, index: number) => (
                    <tr key={index} className="hover:bg-muted/30">
                      <td className="px-3 py-2 font-mono text-xs">
                        L{detail.left_chunk_id}-R{detail.right_chunk_id}
                      </td>
                      <td className="px-3 py-2 text-right">{formatScore(detail.final_score)}%</td>
                      <td className="px-3 py-2 text-right">{formatScore(detail.semantic_score)}%</td>
                      <td className="px-3 py-2 text-right">{formatScore(detail.lexical_overlap)}%</td>
                    </tr>
                  ))}
                  {match.details.length > 10 && (
                    <tr>
                      <td colSpan={4} className="px-3 py-2 text-center text-muted-foreground">
                        还有 {match.details.length - 10} 条...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
              {hasMultipleMatches && (
                <div className="px-3 py-2 text-xs text-muted-foreground border-t bg-background">
                  共{allMatches.length}个匹配组
                </div>
              )}
            </div>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}