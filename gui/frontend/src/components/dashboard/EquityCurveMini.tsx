/**
 * Mini Equity Curve Chart
 * Phase 6.2 - Dashboard
 *
 * Compact equity curve visualization for dashboard overview.
 */

import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown?: number;
}

interface EquityCurveMiniProps {
  data: EquityPoint[];
  height?: number;
  showDrawdown?: boolean;
}

export default function EquityCurveMini({
  data,
  height = 200,
  showDrawdown = false,
}: EquityCurveMiniProps) {
  // Calculate stats
  const stats = useMemo(() => {
    if (!data.length) return null;

    const equities = data.map((d) => d.equity);
    const start = equities[0] ?? 0;
    const end = equities[equities.length - 1] ?? 0;
    const max = Math.max(...equities);
    const min = Math.min(...equities);
    const returns = ((end - start) / start) * 100;

    // Calculate max drawdown
    let maxDD = 0;
    let peak = equities[0] ?? 0;
    for (const eq of equities) {
      if (eq > peak) peak = eq;
      const dd = (peak - eq) / peak;
      if (dd > maxDD) maxDD = dd;
    }

    return { start, end, max, min, returns, maxDD };
  }, [data]);

  if (!data.length || !stats) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-surface-800 bg-surface-900/50"
        style={{ height }}
      >
        <span className="text-sm text-surface-500">No equity data</span>
      </div>
    );
  }

  const isPositive = stats.returns >= 0;

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="0%"
                stopColor={isPositive ? '#22c55e' : '#ef4444'}
                stopOpacity={0.3}
              />
              <stop
                offset="100%"
                stopColor={isPositive ? '#22c55e' : '#ef4444'}
                stopOpacity={0}
              />
            </linearGradient>
            {showDrawdown && (
              <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            )}
          </defs>

          <XAxis dataKey="timestamp" hide />
          <YAxis domain={['auto', 'auto']} hide />

          {/* Starting equity reference line */}
          <ReferenceLine
            y={stats.start}
            stroke="#71717a"
            strokeDasharray="3 3"
            strokeWidth={1}
          />

          {/* Drawdown area (underwater) */}
          {showDrawdown && (
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke="none"
              fill="url(#ddGradient)"
            />
          )}

          {/* Equity curve */}
          <Area
            type="monotone"
            dataKey="equity"
            stroke={isPositive ? '#22c55e' : '#ef4444'}
            strokeWidth={2}
            fill="url(#equityGradient)"
          />

          <Tooltip
            contentStyle={{
              backgroundColor: '#18181b',
              border: '1px solid #27272a',
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number, name: string) => [
              `$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`,
              name === 'equity' ? 'Equity' : 'Drawdown',
            ]}
            labelFormatter={(label) => {
              const date = new Date(label);
              return date.toLocaleDateString();
            }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Stats bar */}
      <div className="flex items-center justify-between text-xs">
        <div className="text-surface-500">
          Start:{' '}
          <span className="text-surface-300">${stats.start.toLocaleString()}</span>
        </div>
        <div className={isPositive ? 'text-green-400' : 'text-red-400'}>
          {isPositive ? '+' : ''}
          {stats.returns.toFixed(2)}%
        </div>
        <div className="text-surface-500">
          Max DD:{' '}
          <span className="text-red-400">{(stats.maxDD * 100).toFixed(1)}%</span>
        </div>
        <div className="text-surface-500">
          End:{' '}
          <span className="text-surface-300">${stats.end.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}
