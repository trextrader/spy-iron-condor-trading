/**
 * Diagnostics Panel
 * Phase 6.8 - Real-time Training Visualization
 *
 * Mini charts for learning rate, gradient norm, and scaler scale.
 */

import { useMemo } from 'react';
import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';
import { TrainingStep } from '@/hooks/useTrainingTelemetry';

interface DiagnosticsPanelProps {
  stepHistory: TrainingStep[];
}

interface MiniChartProps {
  title: string;
  data: { step: number; value: number }[];
  color: string;
  format?: (v: number) => string;
}

function MiniChart({ title, data, color, format }: MiniChartProps) {
  const latestValue = data[data.length - 1]?.value;
  const formatted = latestValue !== undefined ? (format ? format(latestValue) : latestValue.toFixed(4)) : '--';

  return (
    <div className="rounded-lg border border-surface-700 bg-surface-800/50 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-surface-400">{title}</span>
        <span className="font-mono text-sm text-surface-100">{formatted}</span>
      </div>
      <div className="mt-2 h-16">
        {data.length > 1 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id={`gradient-${title}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid #27272a',
                  borderRadius: 6,
                  fontSize: 11,
                }}
                formatter={(value: number) => [format ? format(value) : value.toFixed(4), title]}
                labelFormatter={(v) => `Step ${v}`}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke={color}
                fill={`url(#gradient-${title})`}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-surface-600">
            Waiting for data...
          </div>
        )}
      </div>
    </div>
  );
}

export default function DiagnosticsPanel({ stepHistory }: DiagnosticsPanelProps) {
  // Extract data for each diagnostic metric (last 100 steps for performance)
  const lrData = useMemo(() => {
    return stepHistory.slice(-100).map((s) => ({
      step: s.step,
      value: s.lr ?? 0,
    }));
  }, [stepHistory]);

  const gradData = useMemo(() => {
    return stepHistory.slice(-100).map((s) => ({
      step: s.step,
      value: s.grad_norm ?? 0,
    }));
  }, [stepHistory]);

  const scalerData = useMemo(() => {
    return stepHistory.slice(-100).map((s) => ({
      step: s.step,
      value: s.scaler_scale ?? 0,
    }));
  }, [stepHistory]);

  return (
    <div className="space-y-3">
      <MiniChart
        title="Learning Rate"
        data={lrData}
        color="#3b82f6"
        format={(v) => v.toExponential(2)}
      />
      <MiniChart
        title="Gradient Norm"
        data={gradData}
        color="#f97316"
        format={(v) => v.toFixed(2)}
      />
      <MiniChart
        title="Scaler Scale"
        data={scalerData}
        color="#22c55e"
        format={(v) => v.toFixed(0)}
      />
    </div>
  );
}
