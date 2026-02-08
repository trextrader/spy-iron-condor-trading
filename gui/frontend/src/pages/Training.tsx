/**
 * Training Page
 * Phase 6.8 - Real-time Training Visualization
 *
 * Live monitoring dashboard for CondorNet training sessions.
 * Shows all metrics, loss trajectories, and fuzzy gate activations in real-time.
 */

import { useTrainingTelemetry } from '@/hooks/useTrainingTelemetry';
import {
  TrainingHeader,
  MetricGrid,
  StreamingLossChart,
  StreamingHeatmap,
  DiagnosticsPanel,
  TrainingControls,
} from '@/components/training';

export default function Training() {
  const {
    status,
    latestStep,
    stepHistory,
    fuzzyHistory,
    isConnected,
    clearHistory,
  } = useTrainingTelemetry();

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-surface-100">Training Monitor</h1>
          <p className="mt-1 text-sm text-surface-400">
            Real-time neural telemetry - watch CondorNet train
          </p>
        </div>
        <TrainingControls
          isConnected={isConnected}
          isTraining={status.isTraining}
          onClearHistory={clearHistory}
        />
      </div>

      {/* Training Status Header */}
      <TrainingHeader status={status} isConnected={isConnected} />

      {/* Live Metrics Grid - All 12 Components */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-surface-100">Live Metrics</h2>
          <p className="text-sm text-surface-400">All 12 loss components with sparklines</p>
        </div>
        <div className="p-4">
          <MetricGrid latestStep={latestStep} stepHistory={stepHistory} />
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Streaming Loss Chart - 2 columns */}
        <div className="card lg:col-span-2">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">Loss Trajectory</h2>
            <p className="text-sm text-surface-400">Real-time multi-component loss</p>
          </div>
          <div className="p-4">
            <StreamingLossChart stepHistory={stepHistory} height={350} />
          </div>
        </div>

        {/* Diagnostics Panel - 1 column */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">Diagnostics</h2>
            <p className="text-sm text-surface-400">LR, Gradients, Scaler</p>
          </div>
          <div className="p-4">
            <DiagnosticsPanel stepHistory={stepHistory} />
          </div>
        </div>
      </div>

      {/* Fuzzy Heatmap */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-surface-100">Fuzzy Gate Activations</h2>
          <p className="text-sm text-surface-400">
            Watch which gates wake up, saturate, or oscillate
          </p>
        </div>
        <div className="p-4">
          <StreamingHeatmap fuzzyHistory={fuzzyHistory} height={250} />
        </div>
      </div>

      {/* Connection Status Footer */}
      <div className="flex items-center justify-between rounded-lg border border-surface-800 bg-surface-900/50 px-4 py-3 text-xs text-surface-500">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span>{isConnected ? 'Live connection' : 'Disconnected'}</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Steps: {stepHistory.length.toLocaleString()}</span>
          <span>Fuzzy: {fuzzyHistory.length.toLocaleString()}</span>
          <span>
            Last update:{' '}
            {latestStep ? new Date().toLocaleTimeString() : '--'}
          </span>
        </div>
      </div>
    </div>
  );
}
