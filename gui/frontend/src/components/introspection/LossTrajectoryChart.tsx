/**
 * Loss Trajectory Chart
 * Phase 6.7 - Model Introspection
 *
 * Multi-line chart showing all 12 loss components over training steps.
 * The "heartbeat" of CondorNet.
 */

import { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from 'recharts';

// CondorNet neon palette
const COMPONENT_COLORS: Record<string, string> = {
  loss: '#ef4444',        // Red - total loss
  mse: '#f97316',         // Orange - strike geometry
  npdd: '#22c55e',        // Green - profit distribution
  sharpe: '#3b82f6',      // Blue - risk-adjusted
  dd: '#a855f7',          // Purple - drawdown
  turnover: '#ec4899',    // Pink - flip penalty
  fuzzy: '#14b8a6',       // Teal - fuzzy variance
  pattern_ent: '#eab308', // Yellow - gate diversity
  group_inv: '#6366f1',   // Indigo - group invariance
  rho: '#64748b',         // Slate - correlation
  energy: '#06b6d4',      // Cyan - energy bound
  growth: '#84cc16',      // Lime - growth penalty
};

const COMPONENT_LABELS: Record<string, string> = {
  loss: 'Total Loss',
  mse: 'MSE (Strike)',
  npdd: 'NPDD',
  sharpe: 'Sharpe',
  dd: 'Drawdown',
  turnover: 'Turnover',
  fuzzy: 'Fuzzy Var',
  pattern_ent: 'Pattern Ent',
  group_inv: 'Group Inv',
  rho: 'Rho',
  energy: 'Energy',
  growth: 'Growth',
};

interface LossTrajectoryChartProps {
  steps: number[];
  components: Record<string, number[]>;
  height?: number;
}

export default function LossTrajectoryChart({
  steps,
  components,
  height = 500,
}: LossTrajectoryChartProps) {
  const [visibleComponents, setVisibleComponents] = useState<Set<string>>(
    new Set(['loss', 'mse', 'npdd', 'sharpe', 'turnover', 'pattern_ent'])
  );

  // Transform data for Recharts
  const data = useMemo(() => {
    if (!steps.length) return [];

    return steps.map((step, i) => {
      const point: Record<string, number> = { step };
      Object.entries(components).forEach(([key, values]) => {
        point[key] = values[i] ?? 0;
      });
      return point;
    });
  }, [steps, components]);

  const toggleComponent = (key: string) => {
    setVisibleComponents((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const availableComponents = Object.keys(components);

  if (!data.length) {
    return (
      <div className="flex h-64 items-center justify-center text-surface-500">
        No trajectory data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Component toggles */}
      <div className="flex flex-wrap gap-2">
        {availableComponents.map((key) => (
          <button
            key={key}
            onClick={() => toggleComponent(key)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-all ${
              visibleComponents.has(key)
                ? 'ring-2 ring-offset-2 ring-offset-surface-900'
                : 'opacity-40 hover:opacity-70'
            }`}
            style={{
              backgroundColor: visibleComponents.has(key)
                ? COMPONENT_COLORS[key]
                : 'transparent',
              color: visibleComponents.has(key) ? 'white' : COMPONENT_COLORS[key],
              borderColor: COMPONENT_COLORS[key],
              borderWidth: 1,
            }}
          >
            {COMPONENT_LABELS[key] || key}
          </button>
        ))}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="step"
            stroke="#71717a"
            tick={{ fill: '#a1a1aa', fontSize: 11 }}
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
          />
          <YAxis
            stroke="#71717a"
            tick={{ fill: '#a1a1aa', fontSize: 11 }}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#18181b',
              border: '1px solid #27272a',
              borderRadius: 8,
              fontSize: 12,
            }}
            labelFormatter={(v) => `Step ${v}`}
            formatter={(value: number, name: string) => [
              value.toFixed(4),
              COMPONENT_LABELS[name] || name,
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: 11 }}
            formatter={(value) => COMPONENT_LABELS[value] || value}
          />
          <Brush
            dataKey="step"
            height={30}
            stroke="#3b82f6"
            fill="#18181b"
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
          />

          {availableComponents.map(
            (key) =>
              visibleComponents.has(key) && (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={COMPONENT_COLORS[key]}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 2 }}
                />
              )
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
