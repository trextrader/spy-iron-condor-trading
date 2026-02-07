import { get, post, del } from './client';
import type {
  BacktestRequest,
  BacktestStartResponse,
  BacktestResult,
  BacktestStatus,
} from '@shared/types';

interface BacktestStatusResponse {
  run_id: string;
  status: BacktestStatus;
  progress: {
    progress_pct: number;
    bars_processed: number;
    total_bars: number;
    current_pnl: number;
    eta_seconds?: number;
  } | null;
  result: BacktestResult | null;
}

interface BacktestListResponse {
  runs: Array<{
    run_id: string;
    status: BacktestStatus;
    started_at: string;
    config_hash: string;
  }>;
  total: number;
}

// Start a new backtest
export async function startBacktest(request: BacktestRequest): Promise<BacktestStartResponse> {
  return post<BacktestStartResponse>('/backtest/run', request);
}

// Get backtest status
export async function getBacktestStatus(runId: string): Promise<BacktestStatusResponse> {
  return get<BacktestStatusResponse>(`/backtest/status/${runId}`);
}

// Get backtest results
export async function getBacktestResults(runId: string): Promise<BacktestResult> {
  return get<BacktestResult>(`/backtest/results/${runId}`);
}

// List recent backtests
export async function listBacktests(
  limit = 50,
  statusFilter?: BacktestStatus
): Promise<BacktestListResponse> {
  const params: Record<string, unknown> = { limit };
  if (statusFilter) {
    params.status_filter = statusFilter;
  }
  return get<BacktestListResponse>('/backtest/list', params);
}

// Cancel a running backtest
export async function cancelBacktest(runId: string): Promise<{ message: string }> {
  return del<{ message: string }>(`/backtest/${runId}`);
}

// Replay a previous backtest
export async function replayBacktest(
  runId?: string,
  configHash?: string
): Promise<{
  original_run_id: string;
  replay_run_id: string;
  determinism_verified: boolean;
  diff_fingerprint: string;
}> {
  return post('/backtest/replay', { run_id: runId, config_hash: configHash });
}
