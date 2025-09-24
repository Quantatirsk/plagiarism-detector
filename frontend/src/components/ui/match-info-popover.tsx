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
              <div className="font-medium mb-1">
                段落级匹配
                {hasMultipleMatches && (
                  <span className="ml-2 px-1.5 py-0.5 text-xs bg-primary/20 text-primary rounded">
                    {allMatches.length}个重叠
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-muted-foreground">
                <div>综合: {formatScore(match.group.final_score)}%</div>
                <div>语义: {formatScore(match.group.semantic_score)}%</div>
                <div>交叉: {formatScore(match.group.cross_score)}%</div>
                <div>覆盖: {formatScore(match.group.alignment_ratio)}%</div>
              </div>
              {hasMultipleMatches && (
                <div className="text-muted-foreground mt-1 pt-1 border-t">
                  <div className="font-medium mb-1">所有重叠匹配：</div>
                  {allMatches.map((m, i) => (
                    <div key={i} className="text-xs">
                      匹配{i + 1}: 综合{formatScore(m.group.final_score)}%
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="max-h-72 overflow-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-muted">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">段落范围</th>
                    <th className="px-3 py-2 text-right font-medium">综合</th>
                    <th className="px-3 py-2 text-right font-medium">语义</th>
                    <th className="px-3 py-2 text-right font-medium">交叉</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {match.details.slice(0, 10).map((detail: any, index: number) => (
                    <tr key={index} className="hover:bg-muted/30">
                      <td className="px-3 py-2 font-mono text-xs">
                        {(() => {
                          // Try to get span information from detail or match
                          const spans = detail.spans ||
                                       match.group.document_spans ||
                                       match.group.paragraph_spans;

                          if (spans && spans.length > 0) {
                            const firstSpan = spans[0];
                            return `L[${firstSpan.left_start}-${firstSpan.left_end}] ↔ R[${firstSpan.right_start}-${firstSpan.right_end}]`;
                          }
                          // Fallback to chunk IDs if no span data
                          return `L${detail.left_chunk_id}-R${detail.right_chunk_id}`;
                        })()}
                      </td>
                      <td className="px-3 py-2 text-right">{formatScore(detail.final_score)}%</td>
                      <td className="px-3 py-2 text-right">{formatScore(detail.semantic_score)}%</td>
                      <td className="px-3 py-2 text-right">{formatScore(detail.cross_score)}%</td>
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