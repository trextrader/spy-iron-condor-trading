/**
 * Epoch Summary Panel
 * Phase 6.7 - Model Introspection
 *
 * Shows per-epoch aggregated metrics in a compact, scannable format.
 * Highlights key training milestones:
 * - MSE compression
 * - Pattern entropy diversification
 * - Sharpe stabilization
 * - Scaler recovery
 */

import { useMemo } from 'react';

interface EpochData {
  epoch: number;
  loss: number;
  mse: number;
  npdd: number;
  sharpe: number;
  dd: number;
  turnover: number;
  fuzzy: number;
  pattern_ent: number;
  energy: number;
  growth: number;
  [key: string]: number;
}

interface EpochSummaryPanelProps {
  epochs: EpochData[];
}

// Format number with color based on direction
function MetricCell({
  value,
  prevValue,
  format = 'default',
  goodDirection = 'down',
}: {
  value: number;
  prevValue?: number;
  format?: 'default' | 'percent' | 'scientific';
  goodDirection?: 'up' | 'down' | 'zero';
}) {
  const formatted = useMemo(() => {
    if (format === 'percent') return `${(value * 100).toFixed(1)}%`;
    if (format === 'scientific') return value.toExponential(2);
    if (Math.abs(value) < 0.01) return value.toExponential(2);
    return value.toFixed(4);
  }, [value, format]);

  // Determine change direction
  let changeColor = 'text-surface-300';
  if (prevValue !== undefined) {
    const delta = value - prevValue;
    const isGood =
      (goodDirection === 'down' && delta < 0) ||
      (goodDirection === 'up' && delta > 0) ||
      (goodDirection === 'zero' && Math.abs(value) < Math.abs(prevValue));

    changeColor = isGood ? 'text-green-400' : 'text-red-400';
  }

  return <span className={`font-mono text-sm ${changeColor}`}>{formatted}</span>;
}

export default function EpochSummaryPanel({ epochs }: EpochSummaryPanelProps) {
  if (!epochs.length) {
    return (
      <div className="flex h-48 items-center justify-center text-surface-500">
        No epoch summary data available
      </div>
    );
  }

  // Key metrics to display
  const metrics = [
    { key: 'loss', label: 'Loss', direction: 'down' as const },
    { key: 'mse', label: 'MSE', direction: 'down' as const },
    { key: 'npdd', label: 'NPDD', direction: 'zero' as const },
    { key: 'sharpe', label: 'Sharpe', direction: 'zero' as const },
    { key: 'dd', label: 'DD', direction: 'down' as const },
    { key: 'turnover', label: 'Turnover', direction: 'down' as const },
    { key: 'pattern_ent', label: 'Pat.Ent', direction: 'down' as const },
    { key: 'energy', label: 'Energy', direction: 'down' as const },
    { key: 'growth', label: 'Growth', direction: 'down' as const },
  ];

  // Calculate deltas from first to last epoch
  const firstEpoch = epochs[0]!;
  const lastEpoch = epochs[epochs.length - 1]!;

  const deltas = useMemo(() => {
    return metrics.map(({ key, direction }) => {
      const start = firstEpoch[key] ?? 0;
      const end = lastEpoch[key] ?? 0;
      const delta = end - start;
      const pctChange = start !== 0 ? (delta / Math.abs(start)) * 100 : 0;

      const isGood =
        (direction === 'down' && delta < 0) ||
        (direction === 'up' && delta > 0) ||
        (direction === 'zero' && Math.abs(end) < Math.abs(start));

      return { key, delta, pctChange, isGood };
    });
  }, [epochs, firstEpoch, lastEpoch]);

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {deltas.slice(0, 5).map(({ key, delta, pctChange, isGood }) => (
          <div
            key={key}
            className="rounded-lg border border-surface-800 bg-surface-900/50 p-3"
          >
            <div className="text-xs uppercase text-surface-500">
              {metrics.find((m) => m.key === key)?.label}
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="font-mono text-lg text-surface-100">
                {lastEpoch[key]?.toFixed(3)}
              </span>
              <span
                className={`text-xs ${isGood ? 'text-green-400' : 'text-red-400'}`}
              >
                {pctChange >= 0 ? '+' : ''}
                {pctChange.toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Epoch table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-surface-800">
              <th className="px-3 py-2 text-left text-xs font-medium uppercase text-surface-500">
                Epoch
              </th>
              {metrics.map(({ key, label }) => (
                <th
                  key={key}
                  className="px-3 py-2 text-right text-xs font-medium uppercase text-surface-500"
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {epochs.map((epoch, i) => {
              const prevEpoch = epochs[i - 1];
              return (
                <tr
                  key={epoch.epoch}
                  className="border-b border-surface-800/50 hover:bg-surface-800/30"
                >
                  <td className="px-3 py-2 font-medium text-surface-100">
                    {epoch.epoch}
                  </td>
                  {metrics.map(({ key, direction }) => (
                    <td key={key} className="px-3 py-2 text-right">
                      <MetricCell
                        value={epoch[key] ?? 0}
                        prevValue={prevEpoch?.[key]}
                        goodDirection={direction}
                      />
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Key observations */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <ObservationCard
          title="MSE Compression"
          value={`${((1 - lastEpoch.mse / firstEpoch.mse) * 100).toFixed(0)}%`}
          description="Strike geometry learned"
          isPositive={(lastEpoch.mse ?? 0) < (firstEpoch.mse ?? 1)}
        />
        <ObservationCard
          title="Pattern Diversity"
          value={lastEpoch.pattern_ent?.toFixed(2) ?? '--'}
          description="Gate activation entropy"
          isPositive={(lastEpoch.pattern_ent ?? 0) < -1}
        />
        <ObservationCard
          title="Sharpe Direction"
          value={lastEpoch.sharpe?.toFixed(2) ?? '--'}
          description="Risk-adjusted signal"
          isPositive={(lastEpoch.sharpe ?? -100) > (firstEpoch.sharpe ?? 0)}
        />
        <ObservationCard
          title="Energy Bound"
          value={lastEpoch.energy?.toFixed(3) ?? '--'}
          description="Stability constraint"
          isPositive={(lastEpoch.energy ?? 1) < 0.1}
        />
      </div>
    </div>
  );
}

function ObservationCard({
  title,
  value,
  description,
  isPositive,
}: {
  title: string;
  value: string;
  description: string;
  isPositive: boolean;
}) {
  return (
    <div
      className={`rounded-lg border p-3 ${
        isPositive
          ? 'border-green-500/30 bg-green-500/5'
          : 'border-surface-800 bg-surface-900/50'
      }`}
    >
      <div className="text-xs text-surface-500">{title}</div>
      <div
        className={`mt-1 font-mono text-xl ${
          isPositive ? 'text-green-400' : 'text-surface-300'
        }`}
      >
        {value}
      </div>
      <div className="mt-1 text-xs text-surface-500">{description}</div>
    </div>
  );
}
