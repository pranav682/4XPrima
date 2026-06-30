import * as React from "react";
import { Inbox, TriangleAlert, RefreshCw } from "lucide-react";
import { Skeleton } from "@/components/ui/misc";
import { Card, CardContent } from "@/components/ui/card";
import type { AsyncState } from "@/hooks/useApi";

export function EmptyState({
  title,
  message,
  icon,
}: {
  title: string;
  message: string;
  icon?: React.ReactNode;
}) {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
        <div className="rounded-full border border-border bg-elevated p-3 text-muted-foreground">
          {icon ?? <Inbox className="h-5 w-5" aria-hidden />}
        </div>
        <div className="flex flex-col gap-1">
          <p className="text-sm font-medium text-foreground">{title}</p>
          <p className="mx-auto max-w-md text-xs text-muted-foreground">{message}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
        <div className="rounded-full border border-killed/40 bg-killed/10 p-3 text-killed">
          <TriangleAlert className="h-5 w-5" aria-hidden />
        </div>
        <div className="flex flex-col gap-1">
          <p className="text-sm font-medium text-foreground">Couldn’t load this view</p>
          <p className="mx-auto max-w-md text-xs text-muted-foreground">{error.message}</p>
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-1 inline-flex items-center gap-1.5 rounded-md border border-border bg-elevated px-3 py-1.5 text-xs font-medium text-foreground hover:bg-muted"
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden /> Retry
          </button>
        )}
      </CardContent>
    </Card>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 pt-5" aria-busy="true" aria-label="Loading">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex items-center gap-4">
            <Skeleton className="h-4 w-1/4" />
            <Skeleton className="h-4 w-1/5" />
            <Skeleton className="h-4 w-1/6" />
            <Skeleton className="h-4 w-1/5" />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

/** Render the right state for an async resource: loading skeleton, error, empty,
 *  or the data view. */
export function AsyncBoundary<T>({
  state,
  empty,
  children,
  skeleton,
  emptyView,
}: {
  state: AsyncState<T>;
  empty: (data: T) => boolean;
  children: (data: T) => React.ReactNode;
  skeleton?: React.ReactNode;
  emptyView?: React.ReactNode;
}) {
  if (state.loading) return <>{skeleton ?? <TableSkeleton />}</>;
  if (state.error) return <ErrorState error={state.error} onRetry={state.reload} />;
  if (state.data == null || empty(state.data)) {
    return (
      <>
        {emptyView ?? (
          <EmptyState
            title="Nothing here yet"
            message="No data has been produced for this view yet."
          />
        )}
      </>
    );
  }
  return <>{children(state.data)}</>;
}
