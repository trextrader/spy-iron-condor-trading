/**
 * Metric Grid
 * Phase 6.8 - Real-time Training Visualization
 *
 * Grid of all 12 loss components with live sparklines.
 */

import { useMemo } from 'react';
import LiveMetricCard from './LiveMetricCard';
import { TrainingStep } from '@/hooks/useTrainingTelemetry';

// CondorNet neon palette (from LossTrajectoryChart)
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

const METRIC_CONFIG: Record<
  string,
  { goodWhen?: 'low' | 'high' | 'zero'; threshold?: number; format?: (v: number) => string }
> = {
  loss: { goodWhen: 'low', threshold: 1.0, format: (v) => v.toFixed(3) },
  mse: { goodWhen: 'low', threshold: 0.5, format: (v) => v.toFixed(4) },
  npdd: { goodWhen: 'zero', threshold: 0.1, format: (v) => v.toFixed(4) },
  sharpe: { goodWhen: 'high', threshold: 0, format: (v) => v.toFixed(3) },
  dd: { goodWhen: 'low', threshold: 0.1, format: (v) => v.toFixed(4) },
  turnover: { goodWhen: 'low', threshold: 0.05, format: (v) => v.toFixed(4) },
  fuzzy: { goodWhen: 'low', threshold: 0.1, format: (v) => v.toFixed(4) },
  pattern_ent: { goodWhen: 'low', threshold: -2, format: (v) => v.toFixed(2) },
  group_inv: { goodWhen: 'low', threshold: 0.1, format: (v) => v.toFixed(4) },
  rho: { goodWhen: 'zero', threshold: 0.1, format: (v) => v.toFixed(4) },
  energy: { goodWhen: 'low', threshold: 0.1, format: (v) => v.toFixed(4) },
  growth: { goodWhen: 'low', threshold: 0.1, format: (v) => v.toFixed(4) },
};

const METRIC_KEYS = [
  'loss',
  'mse',
  'npdd',
  'sharpe',
  'dd',
  'turnover',
  'fuzzy',
  'pattern_ent',
  'group_inv',
  'rho',
  'energy',
  'growth',
] as const;

interface MetricGridProps {
  latestStep: TrainingStep | null;
  stepHistory: TrainingStep[];
}

export default function MetricGrid({ latestStep, stepHistory }: MetricGridProps) {
  // Extract history arrays for each metric
  const historyArrays = useMemo(() => {
    const result: Record<string, number[]> = {};
    for (const key of METRIC_KEYS) {
      result[key] = stepHistory.map((s) => s[key] ?? 0);
    }
    return result;
  }, [stepHistory]);

  if (!latestStep && !stepHistory.length) {
    return (
      <div className="flex h-48 items-center justify-center text-surface-500">
        <div className="text-center">
          <p className="text-lg">Waiting for training data...</p>
          <p className="mt-1 text-sm">Metrics will appear once training starts</p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
      {METRIC_KEYS.map((key) => (
        <LiveMetricCard
          key={key}
          label={COMPONENT_LABELS[key]}
          value={latestStep?.[key]}
          history={historyArrays[key]}
          colorHex={COMPONENT_COLORS[key]}
          {...METRIC_CONFIG[key]}
        />
      ))}
    </div>
  );
}
