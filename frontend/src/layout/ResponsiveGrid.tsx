/**
 * ResponsiveGrid - 响应式网格布局
 * 基于 Ant Design Grid 系统，自动计算列数
 */

import { Row, Col } from 'antd';
import type { ReactNode } from 'react';
import { designSystem } from '@/styles/DesignSystem';

interface ResponsiveGridProps {
  children: ReactNode[];
  columns?: {
    xs?: number;  // < 576px
    sm?: number;  // ≥ 576px
    md?: number;  // ≥ 768px
    lg?: number;  // ≥ 992px
    xl?: number;  // ≥ 1200px
  };
  gutter?: number | [number, number];
}

export function ResponsiveGrid({
  children,
  columns = { xs: 1, sm: 2, md: 3, lg: 4 },
  gutter = [parseInt(designSystem.spacing[1]), parseInt(designSystem.spacing[1])],
}: ResponsiveGridProps) {
  const { xs = 1, sm = 2, md = 3, lg = 4, xl = lg } = columns;

  return (
    <Row gutter={gutter}>
      {children.map((child, index) => (
        <Col
          key={index}
          xs={24 / xs}
          sm={24 / sm}
          md={24 / md}
          lg={24 / lg}
          xl={24 / xl}
        >
          {child}
        </Col>
      ))}
    </Row>
  );
}
