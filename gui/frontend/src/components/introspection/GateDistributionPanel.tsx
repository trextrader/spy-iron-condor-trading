/**
 * Gate Distribution Panel
 * Phase 6.7 - Model Introspection
 *
 * Shows histogram distributions of fuzzy gate activations for a specific epoch.
 * Reveals whether gates are:
 * - Binary (crisp logic)
 * - Soft (graded logic)
 * - Dead (always ~0)
 * - Chaotic (unstable)
 */

import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface GateStats {
  counts: number[];
  bin_edges: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
}

interface GateDistributionPanelProps {
  epoch: number;
  gates: Record<string, GateStats>;
}

// Classify gate behavior
function classifyGate(stats: GateStats): {
  type: 'binary' | 'soft' | 'dead' | 'chaotic' | 'saturated';
  color: string;
  label: string;
} {
  const { mean, std, min, max } = stats;

  if (max < 0.1) {
    return { type: 'dead', color: '#64748b', label: 'Dead' };
  }
  if (min > 0.9) {
    return { type: 'saturated', color: '#f59e0b', label: 'Saturated' };
  }
  if (std < 0.1 && mean > 0.4 && mean < 0.6) {
    return { type: 'soft', color: '#22c55e', label: 'Soft' };
  }
  if (std > 0.3) {
    return { type: 'binary', color: '#3b82f6', label: 'Binary' };
  }
  if (std > 0.2 && (mean < 0.2 || mean > 0.8)) {
    return { type: 'chaotic', color: '#ef4444', label: 'Chaotic' };
  }

  return { type: 'soft', color: '#22c55e', label: 'Soft' };
}

function GateHistogram({ gateId, stats }: { gateId: string; stats: GateStats }) {
  const classification = classifyGate(stats);

  // Transform histogram data for Recharts
  const data = useMemo(() => {
    const { counts, bin_edges } = stats;
    return counts.map((count, i) => ({
      bin: ((bin_edges[i] ?? 0) + (bin_edges[i + 1] ?? 1)) / 2,
      count,
      label: `${(bin_edges[i] ?? 0).toFixed(2)}-${(bin_edges[i + 1] ?? 1).toFixed(2)}`,
    }));
  }, [stats]);

  return (
    <div className="rounded-lg border border-surface-800 bg-surface-900/50 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-sm font-medium text-surface-100">{gateId}</span>
        <span
          className="rounded-full px-2 py-0.5 text-xs font-medium"
          style={{ backgroundColor: `${classification.color}20`, color: classification.color }}
        >
          {classification.label}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={100}>
        <BarChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="bin"
            tick={{ fill: '#71717a', fontSize: 9 }}
            tickFormatter={(v) => v.toFixed(1)}
            domain={[0, 1]}
          />
          <YAxis hide />
          <Tooltip
            contentStyle={{
              backgroundColor: '#18181b',
              border: '1px solid #27272a',
              borderRadius: 6,
              fontSize: 11,
            }}
            formatter={(value: number) => [value, 'Count']}
            labelFormatter={(v) => `Activation: ${v.toFixed(2)}`}
          />
          <Bar dataKey="count" fill={classification.color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div className="text-surface-500">
          Mean: <span className="text-surface-300">{stats.mean.toFixed(3)}</span>
        </div>
        <div className="text-surface-500">
          Std: <span className="text-surface-300">{stats.std.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

export default function GateDistributionPanel({ epoch, gates }: GateDistributionPanelProps) {
  const gateIds = Object.keys(gates).sort((a, b) => {
    const numA = parseInt(a.replace(/\D/g, ''));
    const numB = parseInt(b.replace(/\D/g, ''));
    return numA - numB;
  });

  // Summary statistics
  const summary = useMemo(() => {
    const classifications = gateIds.map((id) => classifyGate(gates[id]!));
    return {
      binary: classifications.filter((c) => c.type === 'binary').length,
      soft: classifications.filter((c) => c.type === 'soft').length,
      dead: classifications.filter((c) => c.type === 'dead').length,
      saturated: classifications.filter((c) => c.type === 'saturated').length,
      chaotic: classifications.filter((c) => c.type === 'chaotic').length,
    };
  }, [gates, gateIds]);

  if (!gateIds.length) {
    return (
      <div className="flex h-48 items-center justify-center text-surface-500">
        No gate distribution data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Epoch indicator */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-surface-100">Epoch {epoch} Gate Distributions</h3>
        <div className="flex gap-3 text-xs">
          <span className="text-blue-400">Binary: {summary.binary}</span>
          <span className="text-green-400">Soft: {summary.soft}</span>
          <span className="text-slate-400">Dead: {summary.dead}</span>
          <span className="text-amber-400">Saturated: {summary.saturated}</span>
          <span className="text-red-400">Chaotic: {summary.chaotic}</span>
        </div>
      </div>

      {/* Gate grid */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {gateIds.map((gateId) => (
          <GateHistogram key={gateId} gateId={gateId} stats={gates[gateId]!} />
        ))}
      </div>
    </div>
  );
}
