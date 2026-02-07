import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  BacktestStatus,
  BacktestProgress,
  BacktestResult,
  BacktestRequest,
} from '@shared/types';

interface BacktestRun {
  runId: string;
  status: BacktestStatus;
  progress: BacktestProgress | null;
  result: BacktestResult | null;
  startedAt: Date;
  request: BacktestRequest;
}

interface BacktestState {
  // Active runs
  runs: Map<string, BacktestRun>;
  activeRunId: string | null;

  // UI state
  isStarting: boolean;
  error: string | null;

  // Actions
  startRun: (runId: string, request: BacktestRequest) => void;
  updateProgress: (runId: string, progress: BacktestProgress) => void;
  completeRun: (runId: string, result: BacktestResult) => void;
  failRun: (runId: string, error: string) => void;
  cancelRun: (runId: string) => void;
  setActiveRun: (runId: string | null) => void;
  setStarting: (starting: boolean) => void;
  setError: (error: string | null) => void;
  clearRun: (runId: string) => void;
}

export const useBacktestStore = create<BacktestState>()(
  devtools(
    (set, get) => ({
      runs: new Map(),
      activeRunId: null,
      isStarting: false,
      error: null,

      startRun: (runId, request) => {
        const { runs } = get();
        const newRuns = new Map(runs);
        newRuns.set(runId, {
          runId,
          status: 'running',
          progress: null,
          result: null,
          startedAt: new Date(),
          request,
        });
        set({ runs: newRuns, activeRunId: runId, isStarting: false }, false, 'startRun');
      },

      updateProgress: (runId, progress) => {
        const { runs } = get();
        const run = runs.get(runId);
        if (!run) return;

        const newRuns = new Map(runs);
        newRuns.set(runId, {
          ...run,
          status: progress.status,
          progress,
        });
        set({ runs: newRuns }, false, 'updateProgress');
      },

      completeRun: (runId, result) => {
        const { runs } = get();
        const run = runs.get(runId);
        if (!run) return;

        const newRuns = new Map(runs);
        newRuns.set(runId, {
          ...run,
          status: 'completed',
          result,
        });
        set({ runs: newRuns }, false, 'completeRun');
      },

      failRun: (runId, error) => {
        const { runs } = get();
        const run = runs.get(runId);
        if (!run) return;

        const newRuns = new Map(runs);
        newRuns.set(runId, {
          ...run,
          status: 'failed',
        });
        set({ runs: newRuns, error }, false, 'failRun');
      },

      cancelRun: (runId) => {
        const { runs } = get();
        const run = runs.get(runId);
        if (!run) return;

        const newRuns = new Map(runs);
        newRuns.set(runId, {
          ...run,
          status: 'cancelled',
        });
        set({ runs: newRuns }, false, 'cancelRun');
      },

      setActiveRun: (runId) => {
        set({ activeRunId: runId }, false, 'setActiveRun');
      },

      setStarting: (isStarting) => {
        set({ isStarting }, false, 'setStarting');
      },

      setError: (error) => {
        set({ error }, false, 'setError');
      },

      clearRun: (runId) => {
        const { runs, activeRunId } = get();
        const newRuns = new Map(runs);
        newRuns.delete(runId);
        set(
          {
            runs: newRuns,
            activeRunId: activeRunId === runId ? null : activeRunId,
          },
          false,
          'clearRun'
        );
      },
    }),
    { name: 'backtest-store' }
  )
);

// Selector helpers
export const useActiveRun = () => {
  const { runs, activeRunId } = useBacktestStore();
  return activeRunId ? runs.get(activeRunId) ?? null : null;
};

export const useRunsList = () => {
  const { runs } = useBacktestStore();
  return Array.from(runs.values()).sort(
    (a, b) => b.startedAt.getTime() - a.startedAt.getTime()
  );
};
