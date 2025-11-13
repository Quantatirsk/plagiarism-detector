/**
 * MatchInfoTooltip - 匹配信息提示组件
 * 使用 Ant Design Tooltip 显示匹配详情
 */
import { Tooltip } from 'antd';
import type { ReactNode } from 'react';
import type { MatchGroupModel, MatchDetailModel } from '@/api/plagiarismApi';

export interface MatchData {
  group: MatchGroupModel;
  details: MatchDetailModel[];
}

interface MatchInfoTooltipProps {
  match: { group: MatchGroupModel; details: MatchDetailModel[] };
  allMatches?: MatchData[];
  children: ReactNode;
}

export function MatchInfoTooltip({ match, allMatches, children }: MatchInfoTooltipProps) {
  const displayMatches = allMatches && allMatches.length > 0 ? allMatches : [match];

  const tooltipContent = (
    <div style={{ maxWidth: 300 }}>
      {displayMatches.map((m, idx) => (
        <div key={idx} style={{ marginBottom: idx < displayMatches.length - 1 ? 8 : 0 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            匹配 #{idx + 1}
          </div>
          <div style={{ fontSize: 12 }}>
            <div>最终得分: {(m.group.final_score || 0).toFixed(3)}</div>
            <div>语义得分: {(m.group.semantic_score || 0).toFixed(3)}</div>
            <div>交叉得分: {(m.group.cross_score || 0).toFixed(3)}</div>
            <div>匹配片段: {m.details.length} 个</div>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <Tooltip title={tooltipContent} mouseEnterDelay={0.3}>
      {children}
    </Tooltip>
  );
}
