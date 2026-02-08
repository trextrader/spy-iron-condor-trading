/**
 * Streaming Loss Chart
 * Phase 6.8 - Real-time Training Visualization
 *
 * Real-time multi-line chart showing loss components as they stream in.
 */

import { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { TrainingStep } from '@/hooks/useTrainingTelemetry';

// CondorNet neon palette
const COMPONENT_COLORS: Record<string, string> = {
  loss: '#ef4444',
  mse: '#f97316',
  npdd: '#22c55e',
  sharpe: '#3b82f6',
  dd: '#a855f7',
  turnover: '#ec4899',
  fuzzy: '#14b8a6',
  pattern_ent: '#eab308',
  group_inv: '#6366f1',
  rho: '#64748b',
  energy: '#06b6d4',
  growth: '#84cc16',
};

const COMPONENT_LABELS: Record<string, string> = {
  loss: 'Total Loss',
  mse: 'MSE',
  npdd: 'NPDD',
  sharpe: 'Sharpe',
  dd: 'Drawdown',
  turnover: 'Turnover',
  fuzzy: 'Fuzzy',
  pattern_ent: 'Pattern Ent',
  group_inv: 'Group Inv',
  rho: 'Rho',
  energy: 'Energy',
  growth: 'Growth',
};

const ALL_COMPONENTS = Object.keys(COMPONENT_COLORS);

interface StreamingLossChartProps {
  stepHistory: TrainingStep[];
  visibleWindow?: number;
  height?: number;
}

export default function StreamingLossChart({
  stepHistory,
  visibleWindow = 500,
  height = 400,
}: StreamingLossChartProps) {
  const [visibleComponents, setVisibleComponents] = useState<Set<string>>(
    new Set(['loss', 'mse', 'npdd', 'sharpe'])
  );

  // Transform data for Recharts (only show last visibleWindow steps)
  const data = useMemo(() => {
    const slice = stepHistory.slice(-visibleWindow);
    return slice.map((step) => ({
      step: step.step,
      ...Object.fromEntries(
        ALL_COMPONENTS.map((key) => [key, step[key as keyof TrainingStep] ?? 0])
      ),
    }));
  }, [stepHistory, visibleWindow]);

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

  if (!data.length) {
    return (
      <div className="flex items-center justify-center text-surface-500" style={{ height }}>
        <div className="text-center">
          <p className="text-lg">Waiting for loss data...</p>
          <p className="mt-1 text-sm">Chart will populate as training progresses</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Component toggles */}
      <div className="flex flex-wrap gap-2">
        {ALL_COMPONENTS.map((key) => (
          <button
            key={key}
            onClick={() => toggleComponent(key)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-all ${
              visibleComponents.has(key)
                ? 'ring-2 ring-offset-2 ring-offset-surface-900'
                : 'opacity-40 hover:opacity-70'
            }`}
            style={{
              backgroundColor: visibleComponents.has(key) ? COMPONENT_COLORS[key] : 'transparent',
              color: visibleComponents.has(key) ? 'white' : COMPONENT_COLORS[key],
              borderColor: COMPONENT_COLORS[key],
              borderWidth: 1,
            }}
          >
            {COMPONENT_LABELS[key]}
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
            tickFormatter={(v) => `${(v / 1000).toFixed(1)}k`}
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

          {ALL_COMPONENTS.map(
            (key) =>
              visibleComponents.has(key) && (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={COMPONENT_COLORS[key]}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              )
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Stats bar */}
      <div className="flex items-center justify-between text-xs text-surface-500">
        <span>Showing last {Math.min(visibleWindow, stepHistory.length)} steps</span>
        <span>
          Step range: {data[0]?.step.toLocaleString()} - {data[data.length - 1]?.step.toLocaleString()}
        </span>
      </div>
    </div>
  );
}
