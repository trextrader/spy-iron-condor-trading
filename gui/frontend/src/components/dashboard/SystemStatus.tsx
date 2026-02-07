/**
 * System Status Component
 * Phase 6.2 - Dashboard
 *
 * Shows health status of all system components.
 */

import { clsx } from 'clsx';

type StatusLevel = 'healthy' | 'warning' | 'error' | 'unknown';

interface SystemComponent {
  name: string;
  status: StatusLevel;
  message?: string;
  lastCheck?: string;
}

interface SystemStatusProps {
  components: SystemComponent[];
  compact?: boolean;
}

const statusConfig: Record<StatusLevel, { color: string; bgColor: string; icon: string }> = {
  healthy: { color: 'text-green-400', bgColor: 'bg-green-500', icon: '✓' },
  warning: { color: 'text-amber-400', bgColor: 'bg-amber-500', icon: '!' },
  error: { color: 'text-red-400', bgColor: 'bg-red-500', icon: '×' },
  unknown: { color: 'text-surface-400', bgColor: 'bg-surface-500', icon: '?' },
};

export default function SystemStatus({ components, compact = false }: SystemStatusProps) {
  // Calculate overall status
  const overallStatus: StatusLevel = components.some((c) => c.status === 'error')
    ? 'error'
    : components.some((c) => c.status === 'warning')
      ? 'warning'
      : components.every((c) => c.status === 'healthy')
        ? 'healthy'
        : 'unknown';

  const healthyCount = components.filter((c) => c.status === 'healthy').length;

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div
          className={clsx(
            'h-2 w-2 rounded-full',
            statusConfig[overallStatus].bgColor,
            overallStatus === 'healthy' && 'animate-pulse'
          )}
        />
        <span className={clsx('text-sm', statusConfig[overallStatus].color)}>
          {healthyCount}/{components.length} Systems Online
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Overall status header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={clsx(
              'h-3 w-3 rounded-full',
              statusConfig[overallStatus].bgColor,
              overallStatus === 'healthy' && 'animate-pulse'
            )}
          />
          <span className={clsx('text-sm font-medium', statusConfig[overallStatus].color)}>
            {overallStatus === 'healthy'
              ? 'All Systems Operational'
              : overallStatus === 'warning'
                ? 'Partial Degradation'
                : overallStatus === 'error'
                  ? 'System Issues Detected'
                  : 'Status Unknown'}
          </span>
        </div>
        <span className="text-xs text-surface-500">
          {healthyCount}/{components.length} healthy
        </span>
      </div>

      {/* Component list */}
      <div className="grid gap-2">
        {components.map((component) => {
          const config = statusConfig[component.status];
          return (
            <div
              key={component.name}
              className="flex items-center justify-between rounded-lg bg-surface-800/50 px-3 py-2"
            >
              <div className="flex items-center gap-2">
                <div
                  className={clsx(
                    'flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold text-surface-950',
                    config.bgColor
                  )}
                >
                  {config.icon}
                </div>
                <span className="text-sm text-surface-200">{component.name}</span>
              </div>
              <div className="text-right">
                <span className={clsx('text-xs', config.color)}>
                  {component.status.charAt(0).toUpperCase() + component.status.slice(1)}
                </span>
                {component.message && (
                  <p className="text-xs text-surface-500">{component.message}</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Default system components
export function getDefaultSystemComponents(): SystemComponent[] {
  return [
    { name: 'Backend API', status: 'healthy', message: 'Response time: 12ms' },
    { name: 'WebSocket', status: 'healthy', message: 'Connected' },
    { name: 'CondorNet Model', status: 'healthy', message: 'Loaded' },
    { name: 'Data Pipeline', status: 'healthy', message: 'Synced' },
    { name: 'Execution Reality', status: 'healthy', message: 'Enabled' },
  ];
}
