/**
 * Training Header
 * Phase 6.8 - Real-time Training Visualization
 *
 * Large status header showing current training progress prominently.
 */

import { clsx } from 'clsx';
import { TrainingStatus } from '@/hooks/useTrainingTelemetry';

interface TrainingHeaderProps {
  status: TrainingStatus;
  isConnected: boolean;
}

export default function TrainingHeader({ status, isConnected }: TrainingHeaderProps) {
  const isLive = isConnected && status.isTraining;
  const isComplete = !status.isTraining && status.progressPct >= 100;
  const isIdle = !status.isTraining && status.progressPct < 100;

  // Format ETA
  const formatEta = (seconds?: number): string => {
    if (!seconds) return '--:--:--';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div
      className={clsx(
        'rounded-lg border p-6 transition-all',
        isLive && 'border-brand-500/50 bg-gradient-to-r from-brand-500/10 to-transparent',
        isComplete && 'border-green-500/50 bg-gradient-to-r from-green-500/10 to-transparent',
        isIdle && 'border-surface-700 bg-surface-800/50'
      )}
    >
      <div className="flex items-center justify-between">
        {/* Left: Status icon and title */}
        <div className="flex items-center gap-5">
          <div
            className={clsx(
              'flex h-16 w-16 items-center justify-center rounded-full',
              isLive && 'bg-brand-500/20 ring-2 ring-brand-500/30',
              isComplete && 'bg-green-500/20 ring-2 ring-green-500/30',
              isIdle && 'bg-surface-700'
            )}
          >
            <span className={clsx('text-3xl', isLive && 'animate-pulse')}>
              {isLive ? '🧠' : isComplete ? '✅' : '⏸️'}
            </span>
          </div>
          <div>
            <h2
              className={clsx(
                'text-xl font-semibold',
                isLive && 'text-brand-400',
                isComplete && 'text-green-400',
                isIdle && 'text-surface-400'
              )}
            >
              {isLive
                ? 'Training in Progress'
                : isComplete
                  ? 'Training Complete'
                  : 'Awaiting Training'}
            </h2>
            <p className="text-surface-400">
              {isLive || isComplete
                ? 'CondorNet neural network optimization'
                : 'Start a training run to see live metrics'}
            </p>
          </div>
        </div>

        {/* Center: Epoch and Step counters */}
        <div className="hidden items-center gap-8 lg:flex">
          <div className="text-center">
            <p className="text-sm text-surface-500">Epoch</p>
            <p className="font-mono text-3xl font-bold text-surface-100">
              {status.currentEpoch || '--'}
            </p>
          </div>
          <div className="h-12 w-px bg-surface-700" />
          <div className="text-center">
            <p className="text-sm text-surface-500">Step</p>
            <p className="font-mono text-3xl font-bold text-surface-100">
              {status.currentStep ? status.currentStep.toLocaleString() : '--'}
            </p>
          </div>
          <div className="h-12 w-px bg-surface-700" />
          <div className="text-center">
            <p className="text-sm text-surface-500">Total</p>
            <p className="font-mono text-lg text-surface-400">
              {status.totalSteps ? status.totalSteps.toLocaleString() : '--'}
            </p>
          </div>
        </div>

        {/* Right: ETA and circular progress */}
        <div className="flex items-center gap-6">
          {/* Circular progress */}
          <div className="relative h-20 w-20">
            <svg className="h-20 w-20 -rotate-90 transform">
              <circle
                cx="40"
                cy="40"
                r="36"
                fill="none"
                stroke="currentColor"
                strokeWidth="6"
                className="text-surface-700"
              />
              <circle
                cx="40"
                cy="40"
                r="36"
                fill="none"
                stroke="currentColor"
                strokeWidth="6"
                strokeDasharray={`${2 * Math.PI * 36}`}
                strokeDashoffset={`${2 * Math.PI * 36 * (1 - status.progressPct / 100)}`}
                strokeLinecap="round"
                className={clsx(
                  'transition-all duration-500',
                  isLive && 'text-brand-500',
                  isComplete && 'text-green-500',
                  isIdle && 'text-surface-600'
                )}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="font-mono text-lg font-bold text-surface-100">
                {status.progressPct.toFixed(0)}%
              </span>
            </div>
          </div>

          {/* ETA */}
          <div className="text-right">
            <p className="text-sm text-surface-500">ETA</p>
            <p className="font-mono text-2xl font-medium text-surface-200">
              {isComplete ? 'Done!' : formatEta(status.etaSeconds)}
            </p>
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-6">
        <div className="h-2 overflow-hidden rounded-full bg-surface-700">
          <div
            className={clsx(
              'h-full rounded-full transition-all duration-500',
              isLive && 'bg-gradient-to-r from-brand-600 to-brand-400',
              isComplete && 'bg-gradient-to-r from-green-600 to-green-400',
              isIdle && 'bg-surface-600'
            )}
            style={{ width: `${status.progressPct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
