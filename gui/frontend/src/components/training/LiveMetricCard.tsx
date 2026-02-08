/**
 * Live Metric Card
 * Phase 6.8 - Real-time Training Visualization
 *
 * Single metric display with real-time sparkline.
 */

import { useMemo } from 'react';
import { clsx } from 'clsx';

interface LiveMetricCardProps {
  label: string;
  value: number | undefined;
  history: number[];
  format?: (v: number) => string;
  goodWhen?: 'low' | 'high' | 'zero';
  threshold?: number;
  colorHex: string;
}

export default function LiveMetricCard({
  label,
  value,
  history,
  format,
  goodWhen,
  threshold,
  colorHex,
}: LiveMetricCardProps) {
  const formatted = value !== undefined ? (format ? format(value) : value.toFixed(4)) : '--';

  // Determine color based on threshold
  let valueColor = 'text-surface-100';
  if (value !== undefined && goodWhen && threshold !== undefined) {
    let isGood = false;
    if (goodWhen === 'low') isGood = value < threshold;
    else if (goodWhen === 'high') isGood = value > threshold;
    else if (goodWhen === 'zero') isGood = Math.abs(value) < threshold;
    valueColor = isGood ? 'text-green-400' : 'text-amber-400';
  }

  // Generate sparkline path
  const sparklinePath = useMemo(() => {
    if (!history.length) return '';

    const data = history.slice(-50); // Last 50 points
    if (data.length < 2) return '';

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const width = 100;
    const height = 24;
    const padding = 2;

    const points = data.map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - padding - ((v - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    });

    return `M ${points.join(' L ')}`;
  }, [history]);

  // Trend direction
  const trend = useMemo(() => {
    if (history.length < 2) return 0;
    const recent = history.slice(-10);
    const first = recent[0];
    const last = recent[recent.length - 1];
    if (first === undefined || last === undefined) return 0;
    return last - first;
  }, [history]);

  return (
    <div className="rounded-lg border border-surface-700 bg-surface-800/50 p-3 transition-all hover:border-surface-600">
      {/* Label */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-surface-400">{label}</span>
        <div
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: colorHex }}
        />
      </div>

      {/* Value */}
      <div className="mt-1 flex items-baseline gap-2">
        <span className={clsx('font-mono text-lg font-semibold tabular-nums', valueColor)}>
          {formatted}
        </span>
        {trend !== 0 && (
          <span
            className={clsx(
              'text-xs',
              trend < 0 ? 'text-green-500' : 'text-red-500'
            )}
          >
            {trend < 0 ? '↓' : '↑'}
          </span>
        )}
      </div>

      {/* Sparkline */}
      <div className="mt-2">
        <svg width="100%" height="24" viewBox="0 0 100 24" preserveAspectRatio="none">
          {sparklinePath && (
            <path
              d={sparklinePath}
              fill="none"
              stroke={colorHex}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.8"
            />
          )}
          {/* End dot */}
          {history.length > 0 && (
            <circle
              cx="100"
              cy={
                24 -
                2 -
                (((history[history.length - 1] ?? 0) - Math.min(...history.slice(-50))) /
                  (Math.max(...history.slice(-50)) - Math.min(...history.slice(-50)) || 1)) *
                  20
              }
              r="2"
              fill={colorHex}
            />
          )}
        </svg>
      </div>
    </div>
  );
}
