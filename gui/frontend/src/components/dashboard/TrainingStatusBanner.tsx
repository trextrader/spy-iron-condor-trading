/**
 * Training Status Banner
 * Phase 6.2 - Dashboard
 *
 * Real-time training status display with live metrics.
 */

import { useNavigate } from 'react-router-dom';
import { useTrainingTelemetry } from '@/hooks/useTrainingTelemetry';
import { clsx } from 'clsx';

interface TrainingStatusBannerProps {
  className?: string;
}

export default function TrainingStatusBanner({ className }: TrainingStatusBannerProps) {
  const navigate = useNavigate();
  const { status, latestStep, isConnected } = useTrainingTelemetry();

  // Don't show if not training
  if (!status.isTraining && !latestStep) {
    return null;
  }

  // Format ETA
  const formatEta = (seconds?: number): string => {
    if (!seconds) return '--';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  // Determine banner state
  const isLive = isConnected && status.isTraining;
  const isComplete = !status.isTraining && latestStep;

  return (
    <div
      className={clsx(
        'rounded-lg border p-4 transition-all',
        isLive && 'border-brand-500/30 bg-brand-500/5',
        isComplete && 'border-green-500/30 bg-green-500/5',
        !isLive && !isComplete && 'border-surface-700 bg-surface-800/50',
        className
      )}
    >
      <div className="flex items-center justify-between">
        {/* Left: Status icon and title */}
        <div className="flex items-center gap-4">
          <div
            className={clsx(
              'flex h-10 w-10 items-center justify-center rounded-full',
              isLive && 'bg-brand-500/20',
              isComplete && 'bg-green-500/20',
              !isLive && !isComplete && 'bg-surface-700'
            )}
          >
            <span className={clsx('text-xl', isLive && 'animate-pulse')}>
              {isLive ? '🧠' : isComplete ? '✅' : '⏸️'}
            </span>
          </div>
          <div>
            <h3
              className={clsx(
                'font-medium',
                isLive && 'text-brand-400',
                isComplete && 'text-green-400',
                !isLive && !isComplete && 'text-surface-400'
              )}
            >
              {isLive
                ? 'Training in Progress'
                : isComplete
                  ? 'Training Complete'
                  : 'Training Paused'}
            </h3>
            <p className="text-sm text-surface-400">
              {latestStep ? (
                <>
                  Epoch {latestStep.epoch} • Step {latestStep.step.toLocaleString()}
                </>
              ) : (
                'Waiting for data...'
              )}
            </p>
          </div>
        </div>

        {/* Center: Live metrics */}
        {latestStep && (
          <div className="hidden gap-6 md:flex">
            <MetricPill
              label="MSE"
              value={latestStep.mse}
              format={(v) => v.toFixed(3)}
              goodWhen="low"
              threshold={1.0}
            />
            <MetricPill
              label="Pattern Ent"
              value={latestStep.pattern_ent}
              format={(v) => v.toFixed(2)}
              goodWhen="low"
              threshold={-2}
            />
            <MetricPill
              label="Energy"
              value={latestStep.energy}
              format={(v) => v.toFixed(4)}
              goodWhen="low"
              threshold={0.1}
            />
            <MetricPill
              label="Scaler"
              value={latestStep.scaler_scale ?? 0}
              format={(v) => v.toFixed(0)}
            />
          </div>
        )}

        {/* Right: Progress and actions */}
        <div className="text-right">
          <div className="flex items-center gap-3">
            {/* Progress bar */}
            {status.progressPct > 0 && (
              <div className="hidden w-32 sm:block">
                <div className="h-2 overflow-hidden rounded-full bg-surface-700">
                  <div
                    className={clsx(
                      'h-full rounded-full transition-all',
                      isLive ? 'bg-brand-500' : 'bg-green-500'
                    )}
                    style={{ width: `${status.progressPct}%` }}
                  />
                </div>
                <p className="mt-1 text-xs text-surface-500">
                  {status.progressPct.toFixed(1)}%
                </p>
              </div>
            )}

            {/* ETA */}
            <div>
              <p className="font-mono text-sm text-surface-300">
                {isComplete ? 'Done' : `~${formatEta(status.etaSeconds)}`}
              </p>
              <button
                className="mt-1 text-xs text-brand-400 hover:text-brand-300"
                onClick={() => navigate('/model')}
              >
                View Diagnostics →
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Connection indicator */}
      <div className="mt-3 flex items-center gap-2 border-t border-surface-800/50 pt-3">
        <div
          className={clsx(
            'h-2 w-2 rounded-full',
            isConnected ? 'bg-green-500' : 'bg-red-500'
          )}
        />
        <span className="text-xs text-surface-500">
          {isConnected ? 'Live connection' : 'Reconnecting...'}
        </span>
        {latestStep && (
          <span className="ml-auto text-xs text-surface-600">
            Last update: {new Date().toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  );
}

// Mini metric display
function MetricPill({
  label,
  value,
  format,
  goodWhen,
  threshold,
}: {
  label: string;
  value: number;
  format?: (v: number) => string;
  goodWhen?: 'low' | 'high';
  threshold?: number;
}) {
  const formatted = format ? format(value) : value.toFixed(2);

  let color = 'text-surface-300';
  if (goodWhen && threshold !== undefined) {
    const isGood =
      (goodWhen === 'low' && value < threshold) ||
      (goodWhen === 'high' && value > threshold);
    color = isGood ? 'text-green-400' : 'text-amber-400';
  }

  return (
    <div className="text-center">
      <p className="text-xs text-surface-500">{label}</p>
      <p className={clsx('font-mono text-sm font-medium', color)}>{formatted}</p>
    </div>
  );
}
