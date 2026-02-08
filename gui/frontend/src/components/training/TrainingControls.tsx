/**
 * Training Controls
 * Phase 6.8 - Real-time Training Visualization
 *
 * Start/Stop simulation buttons for testing/demo.
 */

import { useState } from 'react';
import { clsx } from 'clsx';

interface TrainingControlsProps {
  isConnected: boolean;
  isTraining: boolean;
  onClearHistory?: () => void;
}

export default function TrainingControls({
  isConnected,
  isTraining,
  onClearHistory,
}: TrainingControlsProps) {
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  const handleStartSimulation = async () => {
    setIsStarting(true);
    try {
      await fetch('/api/training/simulate/start?epochs=5&steps_per_epoch=100&delay_ms=100', {
        method: 'POST',
      });
    } catch (e) {
      console.error('Failed to start simulation:', e);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopSimulation = async () => {
    setIsStopping(true);
    try {
      await fetch('/api/training/simulate/stop', {
        method: 'POST',
      });
    } catch (e) {
      console.error('Failed to stop simulation:', e);
    } finally {
      setIsStopping(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      {/* Connection indicator */}
      <div className="flex items-center gap-2 rounded-full bg-surface-800 px-3 py-1.5">
        <div
          className={clsx(
            'h-2 w-2 rounded-full',
            isConnected ? 'bg-green-500' : 'bg-red-500'
          )}
        />
        <span className="text-xs text-surface-400">
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {/* Start simulation button */}
      <button
        onClick={handleStartSimulation}
        disabled={isTraining || isStarting || !isConnected}
        className={clsx(
          'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all',
          isTraining || isStarting || !isConnected
            ? 'cursor-not-allowed bg-surface-700 text-surface-500'
            : 'bg-brand-600 text-white hover:bg-brand-500'
        )}
      >
        {isStarting ? (
          <>
            <span className="animate-spin">⏳</span> Starting...
          </>
        ) : (
          <>
            <span>▶️</span> Start Simulation
          </>
        )}
      </button>

      {/* Stop button */}
      <button
        onClick={handleStopSimulation}
        disabled={!isTraining || isStopping}
        className={clsx(
          'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all',
          !isTraining || isStopping
            ? 'cursor-not-allowed bg-surface-700 text-surface-500'
            : 'bg-red-600 text-white hover:bg-red-500'
        )}
      >
        {isStopping ? (
          <>
            <span className="animate-spin">⏳</span> Stopping...
          </>
        ) : (
          <>
            <span>⏹️</span> Stop
          </>
        )}
      </button>

      {/* Clear history button */}
      <button
        onClick={onClearHistory}
        className="flex items-center gap-2 rounded-lg bg-surface-800 px-4 py-2 text-sm font-medium text-surface-300 transition-all hover:bg-surface-700"
      >
        <span>🗑️</span> Clear
      </button>
    </div>
  );
}
