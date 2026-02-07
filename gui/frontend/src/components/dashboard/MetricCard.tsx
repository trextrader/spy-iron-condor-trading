/**
 * Metric Card Component
 * Phase 6.2 - Dashboard
 *
 * Displays a single KPI with value, change indicator, and optional sparkline.
 */

import { useMemo } from 'react';
import { clsx } from 'clsx';

interface MetricCardProps {
  label: string;
  value: number | string;
  format?: 'number' | 'currency' | 'percent' | 'ratio';
  decimals?: number;
  change?: number;
  changeLabel?: string;
  icon?: string;
  variant?: 'default' | 'highlight' | 'warning' | 'danger';
  sparklineData?: number[];
  isLoading?: boolean;
}

export default function MetricCard({
  label,
  value,
  format = 'number',
  decimals = 2,
  change,
  changeLabel,
  icon,
  variant = 'default',
  sparklineData,
  isLoading = false,
}: MetricCardProps) {
  // Format value
  const formattedValue = useMemo(() => {
    if (typeof value === 'string') return value;
    if (isNaN(value)) return '--';

    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        }).format(value);
      case 'percent':
        return `${(value * 100).toFixed(decimals)}%`;
      case 'ratio':
        return value.toFixed(decimals);
      default:
        return value.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        });
    }
  }, [value, format, decimals]);

  // Variant styles
  const variantStyles = {
    default: 'border-surface-800 bg-surface-900',
    highlight: 'border-brand-500/30 bg-brand-500/5',
    warning: 'border-warning-500/30 bg-warning-500/5',
    danger: 'border-danger-500/30 bg-danger-500/5',
  };

  // Change indicator
  const changeColor = change === undefined ? '' : change >= 0 ? 'text-green-400' : 'text-red-400';
  const changeIcon = change === undefined ? '' : change >= 0 ? '↑' : '↓';

  return (
    <div
      className={clsx(
        'rounded-lg border p-4 transition-all hover:shadow-lg',
        variantStyles[variant],
        isLoading && 'animate-pulse'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-surface-400">{label}</span>
        {icon && <span className="text-lg">{icon}</span>}
      </div>

      {/* Value */}
      <div className="mt-2">
        {isLoading ? (
          <div className="h-8 w-24 rounded bg-surface-800" />
        ) : (
          <span
            className={clsx(
              'font-mono text-2xl font-bold tabular-nums',
              variant === 'highlight' && 'text-brand-400',
              variant === 'warning' && 'text-warning-500',
              variant === 'danger' && 'text-danger-500',
              variant === 'default' && 'text-surface-100'
            )}
          >
            {formattedValue}
          </span>
        )}
      </div>

      {/* Change indicator */}
      {change !== undefined && (
        <div className={clsx('mt-2 flex items-center gap-1 text-sm', changeColor)}>
          <span>{changeIcon}</span>
          <span className="font-mono">
            {change >= 0 ? '+' : ''}
            {(change * 100).toFixed(1)}%
          </span>
          {changeLabel && <span className="text-surface-500">({changeLabel})</span>}
        </div>
      )}

      {/* Mini sparkline */}
      {sparklineData && sparklineData.length > 1 && (
        <div className="mt-3">
          <MiniSparkline data={sparklineData} />
        </div>
      )}
    </div>
  );
}

// Simple SVG sparkline
function MiniSparkline({ data }: { data: number[] }) {
  const width = 100;
  const height = 24;
  const padding = 2;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((value, i) => {
    const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((value - min) / range) * (height - 2 * padding);
    return `${x},${y}`;
  });

  const isPositive = data[data.length - 1]! >= data[0]!;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke={isPositive ? '#22c55e' : '#ef4444'}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* End point dot */}
      <circle
        cx={points[points.length - 1]?.split(',')[0]}
        cy={points[points.length - 1]?.split(',')[1]}
        r={2}
        fill={isPositive ? '#22c55e' : '#ef4444'}
      />
    </svg>
  );
}
