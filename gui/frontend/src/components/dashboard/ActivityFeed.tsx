/**
 * Activity Feed Component
 * Phase 6.2 - Dashboard
 *
 * Real-time activity stream showing recent events.
 */

import { useMemo } from 'react';
import { clsx } from 'clsx';
import { formatDistanceToNow } from 'date-fns';

type ActivityType = 'backtest' | 'optimization' | 'config' | 'model' | 'trade' | 'system';

interface ActivityItem {
  id: string;
  type: ActivityType;
  title: string;
  description?: string;
  timestamp: string;
  status?: 'success' | 'error' | 'warning' | 'info' | 'pending';
  metadata?: Record<string, unknown>;
}

interface ActivityFeedProps {
  items: ActivityItem[];
  maxItems?: number;
  showTimestamp?: boolean;
}

const typeIcons: Record<ActivityType, string> = {
  backtest: '⏱️',
  optimization: '🎯',
  config: '⚙️',
  model: '🧠',
  trade: '💹',
  system: '🔧',
};

const statusColors: Record<string, string> = {
  success: 'text-green-400 bg-green-500/10',
  error: 'text-red-400 bg-red-500/10',
  warning: 'text-amber-400 bg-amber-500/10',
  info: 'text-blue-400 bg-blue-500/10',
  pending: 'text-surface-400 bg-surface-500/10',
};

export default function ActivityFeed({
  items,
  maxItems = 10,
  showTimestamp = true,
}: ActivityFeedProps) {
  const displayItems = useMemo(() => {
    return items.slice(0, maxItems);
  }, [items, maxItems]);

  if (!displayItems.length) {
    return (
      <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-surface-700">
        <div className="text-center">
          <span className="text-2xl">📭</span>
          <p className="mt-2 text-sm text-surface-500">No recent activity</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {displayItems.map((item, index) => (
        <ActivityRow
          key={item.id}
          item={item}
          showTimestamp={showTimestamp}
          isLast={index === displayItems.length - 1}
        />
      ))}

      {items.length > maxItems && (
        <div className="pt-2 text-center">
          <button className="text-xs text-brand-400 hover:text-brand-300">
            View {items.length - maxItems} more
          </button>
        </div>
      )}
    </div>
  );
}

function ActivityRow({
  item,
  showTimestamp,
  isLast,
}: {
  item: ActivityItem;
  showTimestamp: boolean;
  isLast: boolean;
}) {
  const timeAgo = useMemo(() => {
    try {
      return formatDistanceToNow(new Date(item.timestamp), { addSuffix: true });
    } catch {
      return item.timestamp;
    }
  }, [item.timestamp]);

  return (
    <div
      className={clsx(
        'flex items-start gap-3 rounded-lg px-3 py-2 transition-colors hover:bg-surface-800/50',
        !isLast && 'border-b border-surface-800/50'
      )}
    >
      {/* Icon */}
      <div
        className={clsx(
          'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full',
          item.status ? statusColors[item.status] : 'bg-surface-800'
        )}
      >
        <span className="text-sm">{typeIcons[item.type]}</span>
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm font-medium text-surface-100 truncate">{item.title}</p>
          {showTimestamp && (
            <span className="flex-shrink-0 text-xs text-surface-500">{timeAgo}</span>
          )}
        </div>
        {item.description && (
          <p className="mt-0.5 text-xs text-surface-400 line-clamp-2">{item.description}</p>
        )}
      </div>
    </div>
  );
}

// Helper to generate mock activity for development
export function generateMockActivity(): ActivityItem[] {
  const now = new Date();
  return [
    {
      id: '1',
      type: 'backtest',
      title: 'Backtest completed',
      description: 'run_20260206_143522 finished with 127 trades',
      timestamp: new Date(now.getTime() - 5 * 60000).toISOString(),
      status: 'success',
    },
    {
      id: '2',
      type: 'model',
      title: 'Training checkpoint saved',
      description: 'Epoch 2 - val_loss: -379.65',
      timestamp: new Date(now.getTime() - 15 * 60000).toISOString(),
      status: 'success',
    },
    {
      id: '3',
      type: 'config',
      title: 'Configuration updated',
      description: 'Enabled CondorNet ETD-1 module',
      timestamp: new Date(now.getTime() - 30 * 60000).toISOString(),
      status: 'info',
    },
    {
      id: '4',
      type: 'optimization',
      title: 'Ablation study running',
      description: '45% complete - evaluating MoE impact',
      timestamp: new Date(now.getTime() - 45 * 60000).toISOString(),
      status: 'pending',
    },
    {
      id: '5',
      type: 'trade',
      title: 'Paper trade executed',
      description: 'Iron Condor on SPY - $1,234 credit',
      timestamp: new Date(now.getTime() - 60 * 60000).toISOString(),
      status: 'success',
    },
  ];
}
