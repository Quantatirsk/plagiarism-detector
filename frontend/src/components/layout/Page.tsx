import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

function isPrimitiveNode(value: ReactNode): value is string | number {
  return typeof value === 'string' || typeof value === 'number';
}

interface PageShellProps {
  children: ReactNode;
  className?: string;
}

export function PageShell({ children, className }: PageShellProps) {
  return (
    <div className={cn('flex h-full min-h-0 flex-col overflow-hidden bg-muted/40 text-foreground', className)}>
      {children}
    </div>
  );
}

interface PageHeaderProps {
  title: ReactNode;
  subtitle?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  meta?: ReactNode;
  children?: ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  subtitle,
  description,
  actions,
  meta,
  children,
  className,
}: PageHeaderProps) {
  let renderedTitle: ReactNode;
  if (isPrimitiveNode(title)) {
    renderedTitle = <h1 className="truncate text-lg font-semibold tracking-tight">{title}</h1>;
  } else {
    renderedTitle = (
      <div className="flex min-w-0 items-center gap-3 truncate text-lg font-semibold tracking-tight">{title}</div>
    );
  }

  let renderedSubtitle: ReactNode = null;
  if (subtitle != null && subtitle !== false) {
    if (isPrimitiveNode(subtitle)) {
      renderedSubtitle = <p className="truncate text-sm text-muted-foreground">{subtitle}</p>;
    } else {
      renderedSubtitle = <div className="truncate text-sm text-muted-foreground">{subtitle}</div>;
    }
  }

  return (
    <header
      className={cn(
        'border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80',
        className,
      )}
    >
      <div className="flex flex-col gap-3 px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-4">
            <div className="flex min-w-0 flex-wrap items-center gap-4">
              {renderedTitle}
              {renderedSubtitle}
            </div>
            {children ? <div className="flex items-center gap-3 text-xs text-muted-foreground sm:text-sm">{children}</div> : null}
          </div>
          {(meta || actions) && (
            <div className="flex flex-wrap items-center justify-end gap-3">
              {meta ? <div className="text-xs text-muted-foreground sm:text-sm">{meta}</div> : null}
              {actions}
            </div>
          )}
        </div>
        {description != null && description !== false
          ? isPrimitiveNode(description)
            ? <p className="text-sm text-muted-foreground">{description}</p>
            : <div className="text-sm text-muted-foreground">{description}</div>
          : null}
      </div>
    </header>
  );
}

interface PageContentProps {
  children: ReactNode;
  className?: string;
  containerClassName?: string;
}

export function PageContent({ children, className, containerClassName }: PageContentProps) {
  return (
    <main className={cn('flex-1 overflow-auto min-h-0', className)}>
      <div className={cn('flex flex-col gap-6 px-6 py-6', containerClassName)}>{children}</div>
    </main>
  );
}

interface SectionCardProps {
  children: ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md';
}

const paddingMap: Record<NonNullable<SectionCardProps['padding']>, string> = {
  none: '',
  sm: 'p-4',
  md: 'p-5',
};

export function SectionCard({ children, className, padding = 'md' }: SectionCardProps) {
  return (
    <section className={cn('rounded-lg border border-border bg-card shadow-sm', paddingMap[padding], className)}>
      {children}
    </section>
  );
}
