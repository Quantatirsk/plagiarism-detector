import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const statusBadgeVariants = cva(
  'inline-flex h-6 items-center rounded-full px-2 text-xs font-medium ring-1 ring-inset',
  {
    variants: {
      tone: {
        neutral: 'bg-slate-100 text-slate-700 ring-slate-200',
        info: 'bg-sky-100 text-sky-800 ring-sky-200',
        success: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
        warning: 'bg-amber-100 text-amber-800 ring-amber-200',
        error: 'bg-destructive/10 text-destructive ring-destructive/30',
      },
    },
    defaultVariants: {
      tone: 'neutral',
    },
  },
);

export type StatusBadgeTone = NonNullable<VariantProps<typeof statusBadgeVariants>['tone']>;

export interface StatusBadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof statusBadgeVariants> {}

export const StatusBadge = React.forwardRef<HTMLSpanElement, StatusBadgeProps>(
  ({ className, tone, ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(statusBadgeVariants({ tone }), className)}
        {...props}
      />
    );
  },
);

StatusBadge.displayName = 'StatusBadge';
