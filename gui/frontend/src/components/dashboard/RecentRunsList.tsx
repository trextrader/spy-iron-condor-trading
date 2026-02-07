/**
 * Recent Runs List
 * Phase 6.2 - Dashboard
 *
 * Displays a compact list of recent backtest runs with quick status.
 */

import { clsx } from 'clsx';
import { formatDistanceToNow } from 'date-fns';
import type { BacktestStatus } from '@shared/types';

interface RecentRun {
  run_id: string;
  status: BacktestStatus;
  started_at: string;
  config_hash: string;
  net_pnl?: number;
  trade_count?: number;
}

interface RecentRunsListProps {
  runs: RecentRun[];
  onSelectRun?: (runId: string) => void;
  maxItems?: number;
}

const statusConfig: Record<
  BacktestStatus,
  { color: string; bgColor: string; label: string; icon: string }
> = {
  pending: {
    color: 'text-surface-400',
    bgColor: 'bg-surface-500/20',
    label: 'Pending',
    icon: '⏳',
  },
  running: {
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/20',
    label: 'Running',
    icon: '🔄',
  },
  completed: {
    color: 'text-green-400',
    bgColor: 'bg-green-500/20',
    label: 'Complete',
    icon: '✅',
  },
  failed: {
    color: 'text-red-400',
    bgColor: 'bg-red-500/20',
    label: 'Failed',
    icon: '❌',
  },
  cancelled: {
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/20',
    label: 'Cancelled',
    icon: '⛔',
  },
};

export default function RecentRunsList({
  runs,
  onSelectRun,
  maxItems = 5,
}: RecentRunsListProps) {
  const displayRuns = runs.slice(0, maxItems);

  if (!displayRuns.length) {
    return (
      <div className="flex h-32 items-center justify-center rounded-lg border border-dashed border-surface-700">
        <div className="text-center">
          <span className="text-2xl">📊</span>
          <p className="mt-2 text-sm text-surface-500">No recent runs</p>
          <p className="text-xs text-surface-600">Start a backtest to see results here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="divide-y divide-surface-800">
      {displayRuns.map((run) => {
        const status = statusConfig[run.status];
        const timeAgo = formatDistanceToNow(new Date(run.started_at), { addSuffix: true });

        return (
          <div
            key={run.run_id}
            className={clsx(
              'flex items-center justify-between py-3 transition-colors',
              onSelectRun && 'cursor-pointer hover:bg-surface-800/50'
            )}
            onClick={() => onSelectRun?.(run.run_id)}
          >
            {/* Left: Status + Run ID */}
            <div className="flex items-center gap-3">
              <div
                className={clsx(
                  'flex h-8 w-8 items-center justify-center rounded-full text-sm',
                  status.bgColor
                )}
              >
                {status.icon}
              </div>
              <div>
                <p className="font-mono text-sm text-surface-100">
                  {run.run_id.slice(0, 20)}...
                </p>
                <p className="text-xs text-surface-500">{timeAgo}</p>
              </div>
            </div>

            {/* Right: Results */}
            <div className="text-right">
              {run.status === 'completed' && run.net_pnl !== undefined ? (
                <>
                  <p
                    className={clsx(
                      'font-mono text-sm font-medium',
                      run.net_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    )}
                  >
                    {run.net_pnl >= 0 ? '+' : ''}${run.net_pnl.toLocaleString()}
                  </p>
                  {run.trade_count !== undefined && (
                    <p className="text-xs text-surface-500">{run.trade_count} trades</p>
                  )}
                </>
              ) : (
                <span className={clsx('text-xs', status.color)}>{status.label}</span>
              )}
            </div>
          </div>
        );
      })}

      {runs.length > maxItems && (
        <div className="pt-3 text-center">
          <button className="text-sm text-brand-400 hover:text-brand-300">
            View all {runs.length} runs →
          </button>
        </div>
      )}
    </div>
  );
}
