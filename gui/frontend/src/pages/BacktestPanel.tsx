/**
 * Backtest Control Panel
 * Phase 6.5 - Run backtests with kaggle backtester, track progress, view results
 */

import { useState, useEffect, useCallback } from 'react';
import { useWebSocketContext } from '@/providers/WebSocketProvider';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

// API base URL
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Tape {
  id: string;
  name: string;
  path: string;
  size_mb: number;
  modified: string;
}

interface Model {
  id: string;
  name: string;
  path: string;
  size_mb: number;
  modified: string;
}

interface BacktestProgress {
  run_id: string;
  status: string;
  progress_pct: number;
  bars_processed: number;
  total_bars: number;
  current_equity: number;
  current_pnl: number;
  trades_so_far: number;
  eta_seconds?: number;
}

interface BacktestMetrics {
  total_trades: number;
  win_count: number;
  loss_count: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  max_pnl: number;
  min_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  npdd: number;
}

interface TradeSummary {
  trade_id: string;
  entry_dt: string;
  exit_dt?: string;
  pnl: number;
  pnl_pct: number;
  reason?: string;
}

interface BacktestResult {
  run_id: string;
  status: string;
  metrics?: BacktestMetrics;
  equity_curve?: Array<{ equity: number }>;
  trades?: TradeSummary[];
  error_message?: string;
}

export default function BacktestPanel() {
  // State
  const [tapes, setTapes] = useState<Tape[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedTape, setSelectedTape] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [seed, setSeed] = useState<number>(42);
  const [device, setDevice] = useState<string>('cpu');
  const [limit, setLimit] = useState<number | null>(10000);
  const [useExecutionReality, setUseExecutionReality] = useState<boolean>(true);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState<BacktestProgress | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [equityHistory, setEquityHistory] = useState<Array<{ idx: number; equity: number }>>([]);

  const { subscribe, isConnected } = useWebSocketContext();

  // Fetch available tapes
  useEffect(() => {
    fetch(`${API_BASE}/api/backtest/tapes`)
      .then((res) => res.json())
      .then((data) => {
        setTapes(data.tapes || []);
        if (data.tapes?.length > 0) {
          setSelectedTape(data.tapes[0].id);
        }
      })
      .catch((err) => console.error('Failed to fetch tapes:', err));
  }, []);

  // Fetch available models
  useEffect(() => {
    fetch(`${API_BASE}/api/backtest/models`)
      .then((res) => res.json())
      .then((data) => {
        setModels(data.models || []);
        if (data.models?.length > 0) {
          setSelectedModel(data.models[0].id);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));
  }, []);

  // Subscribe to backtest progress via WebSocket
  useEffect(() => {
    if (!currentRunId || !isConnected) return;

    const channel = `backtest:${currentRunId}`;
    const unsubscribe = subscribe(channel, (data: unknown) => {
      const progressData = data as BacktestProgress;
      setProgress(progressData);

      // Add to equity history
      setEquityHistory((prev) => [
        ...prev.slice(-500), // Keep last 500 points
        { idx: prev.length, equity: progressData.current_equity },
      ]);

      // Check if completed
      if (progressData.status === 'completed' || progressData.status === 'failed') {
        setIsRunning(false);
        fetchResult(currentRunId);
      }
    });

    return () => unsubscribe();
  }, [currentRunId, isConnected, subscribe]);

  // Fetch final result
  const fetchResult = async (runId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/backtest/results/${runId}`);
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      }
    } catch (err) {
      console.error('Failed to fetch result:', err);
    }
  };

  // Start backtest
  const startBacktest = useCallback(async () => {
    if (!selectedTape || !selectedModel) {
      setError('Please select a tape and model');
      return;
    }

    setError(null);
    setIsRunning(true);
    setProgress(null);
    setResult(null);
    setEquityHistory([]);

    try {
      const res = await fetch(`${API_BASE}/api/backtest/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tape_id: selectedTape,
          model_id: selectedModel,
          seed,
          device,
          limit,
          use_execution_reality: useExecutionReality,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      setCurrentRunId(data.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start backtest');
      setIsRunning(false);
    }
  }, [selectedTape, selectedModel, seed, device, limit, useExecutionReality]);

  // Cancel backtest
  const cancelBacktest = async () => {
    if (!currentRunId) return;

    try {
      await fetch(`${API_BASE}/api/backtest/${currentRunId}`, {
        method: 'DELETE',
      });
      setIsRunning(false);
    } catch (err) {
      console.error('Failed to cancel:', err);
    }
  };

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Format percent
  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-surface-100">Backtest Control</h1>
        <p className="mt-1 text-sm text-surface-400">
          Run backtests with CondorBrain V2.2 and view detailed results
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Configuration Form */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">Run Configuration</h2>
          </div>
          <div className="space-y-4">
            {/* Tape Selection */}
            <div>
              <label className="input-label">Data Tape</label>
              <select
                className="input"
                value={selectedTape}
                onChange={(e) => setSelectedTape(e.target.value)}
                disabled={isRunning}
              >
                <option value="">Select tape...</option>
                {tapes.map((tape) => (
                  <option key={tape.id} value={tape.id}>
                    {tape.name} ({tape.size_mb.toFixed(1)} MB)
                  </option>
                ))}
              </select>
            </div>

            {/* Model Selection */}
            <div>
              <label className="input-label">Model Checkpoint</label>
              <select
                className="input"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isRunning}
              >
                <option value="">Select model...</option>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.size_mb.toFixed(1)} MB)
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Seed */}
              <div>
                <label className="input-label">Random Seed</label>
                <input
                  type="number"
                  className="input"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
                  disabled={isRunning}
                />
              </div>

              {/* Limit */}
              <div>
                <label className="input-label">Bar Limit</label>
                <input
                  type="number"
                  className="input"
                  value={limit || ''}
                  onChange={(e) => setLimit(parseInt(e.target.value) || null)}
                  placeholder="All bars"
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Device */}
            <div>
              <label className="input-label">Device</label>
              <select
                className="input"
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                disabled={isRunning}
              >
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA (GPU)</option>
                <option value="mps">MPS (Apple Silicon)</option>
              </select>
            </div>

            {/* Execution Reality Toggle */}
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="execReality"
                checked={useExecutionReality}
                onChange={(e) => setUseExecutionReality(e.target.checked)}
                disabled={isRunning}
                className="h-4 w-4 rounded border-surface-600 bg-surface-800 text-brand-500"
              />
              <label htmlFor="execReality" className="text-sm text-surface-300">
                Use Execution Reality Engine (Phase 5.5)
              </label>
            </div>

            {/* Error Message */}
            {error && (
              <div className="rounded-md bg-red-900/30 p-3 text-sm text-red-400">{error}</div>
            )}

            {/* Run/Cancel Buttons */}
            <div className="flex gap-2">
              <button
                className="btn-primary flex-1"
                onClick={startBacktest}
                disabled={isRunning || !selectedTape || !selectedModel}
              >
                {isRunning ? 'Running...' : 'Start Backtest'}
              </button>
              {isRunning && (
                <button className="btn-secondary" onClick={cancelBacktest}>
                  Cancel
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Progress Panel */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">Progress</h2>
            {isConnected && (
              <span className="text-xs text-green-500">WebSocket Connected</span>
            )}
          </div>

          {isRunning && progress ? (
            <div className="space-y-4">
              {/* Progress Bar */}
              <div>
                <div className="mb-1 flex justify-between text-sm">
                  <span className="text-surface-400">
                    {progress.bars_processed.toLocaleString()} / {progress.total_bars.toLocaleString()} bars
                  </span>
                  <span className="text-surface-300">{progress.progress_pct.toFixed(1)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-surface-700">
                  <div
                    className="h-full bg-brand-500 transition-all duration-300"
                    style={{ width: `${progress.progress_pct}%` }}
                  />
                </div>
              </div>

              {/* Live Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg bg-surface-800 p-3">
                  <div className="text-xs text-surface-500">Current Equity</div>
                  <div className="text-xl font-bold text-surface-100">
                    {formatCurrency(progress.current_equity)}
                  </div>
                </div>
                <div className="rounded-lg bg-surface-800 p-3">
                  <div className="text-xs text-surface-500">P&L</div>
                  <div
                    className={`text-xl font-bold ${progress.current_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  >
                    {formatCurrency(progress.current_pnl)}
                  </div>
                </div>
                <div className="rounded-lg bg-surface-800 p-3">
                  <div className="text-xs text-surface-500">Trades</div>
                  <div className="text-xl font-bold text-surface-100">
                    {progress.trades_so_far}
                  </div>
                </div>
                <div className="rounded-lg bg-surface-800 p-3">
                  <div className="text-xs text-surface-500">Status</div>
                  <div className="text-xl font-bold text-brand-400 capitalize">
                    {progress.status}
                  </div>
                </div>
              </div>

              {/* Mini Equity Chart */}
              {equityHistory.length > 1 && (
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={equityHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="idx" hide />
                      <YAxis domain={['auto', 'auto']} hide />
                      <Line
                        type="monotone"
                        dataKey="equity"
                        stroke="#10b981"
                        dot={false}
                        strokeWidth={2}
                      />
                      <ReferenceLine y={100000} stroke="#666" strokeDasharray="3 3" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          ) : result ? (
            <div className="text-center text-surface-400">
              <span className="text-green-500">Completed</span> - See results below
            </div>
          ) : (
            <div className="flex h-48 items-center justify-center text-surface-500">
              No active backtest
            </div>
          )}
        </div>
      </div>

      {/* Results Section */}
      {result && result.status === 'completed' && result.metrics && (
        <div className="space-y-6">
          {/* Metrics Grid */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-surface-100">Performance Metrics</h2>
            </div>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">Total P&L</div>
                <div
                  className={`text-2xl font-bold ${result.metrics.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}
                >
                  {formatCurrency(result.metrics.total_pnl)}
                </div>
              </div>
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">Win Rate</div>
                <div className="text-2xl font-bold text-surface-100">
                  {result.metrics.win_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-surface-500">
                  {result.metrics.win_count}W / {result.metrics.loss_count}L
                </div>
              </div>
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">Sharpe Ratio</div>
                <div className="text-2xl font-bold text-surface-100">
                  {result.metrics.sharpe_ratio.toFixed(2)}
                </div>
              </div>
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">Max Drawdown</div>
                <div className="text-2xl font-bold text-red-400">
                  {formatPercent(-result.metrics.max_drawdown_pct)}
                </div>
              </div>
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">Total Trades</div>
                <div className="text-2xl font-bold text-surface-100">
                  {result.metrics.total_trades}
                </div>
              </div>
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">Avg P&L per Trade</div>
                <div
                  className={`text-2xl font-bold ${result.metrics.avg_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}
                >
                  {formatCurrency(result.metrics.avg_pnl)}
                </div>
              </div>
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">NPDD Ratio</div>
                <div className="text-2xl font-bold text-surface-100">
                  {result.metrics.npdd.toFixed(2)}
                </div>
              </div>
              <div className="rounded-lg bg-surface-800 p-4">
                <div className="text-xs text-surface-500">Best / Worst Trade</div>
                <div className="flex gap-2 text-lg font-bold">
                  <span className="text-green-400">{formatCurrency(result.metrics.max_pnl)}</span>
                  <span className="text-surface-500">/</span>
                  <span className="text-red-400">{formatCurrency(result.metrics.min_pnl)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Equity Curve */}
          {result.equity_curve && result.equity_curve.length > 0 && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-surface-100">Equity Curve</h2>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={result.equity_curve.map((p, i) => ({ idx: i, equity: p.equity }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="idx" stroke="#888" fontSize={10} />
                    <YAxis
                      domain={['auto', 'auto']}
                      stroke="#888"
                      fontSize={10}
                      tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1a1a2e', border: 'none' }}
                      labelStyle={{ color: '#888' }}
                      formatter={(value: number) => [formatCurrency(value), 'Equity']}
                    />
                    <ReferenceLine y={100000} stroke="#666" strokeDasharray="3 3" />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      stroke="#10b981"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Trades Table */}
          {result.trades && result.trades.length > 0 && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-surface-100">
                  Trade History ({result.trades.length} trades)
                </h2>
              </div>
              <div className="max-h-96 overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface-900">
                    <tr className="text-left text-surface-400">
                      <th className="p-2">ID</th>
                      <th className="p-2">Entry</th>
                      <th className="p-2">Exit</th>
                      <th className="p-2 text-right">P&L</th>
                      <th className="p-2 text-right">P&L %</th>
                      <th className="p-2">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades.slice(0, 100).map((trade, i) => (
                      <tr key={trade.trade_id || i} className="border-t border-surface-800">
                        <td className="p-2 text-surface-300">{trade.trade_id}</td>
                        <td className="p-2 text-surface-400">
                          {new Date(trade.entry_dt).toLocaleString()}
                        </td>
                        <td className="p-2 text-surface-400">
                          {trade.exit_dt ? new Date(trade.exit_dt).toLocaleString() : '-'}
                        </td>
                        <td
                          className={`p-2 text-right font-medium ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}
                        >
                          {formatCurrency(trade.pnl)}
                        </td>
                        <td
                          className={`p-2 text-right ${trade.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}
                        >
                          {formatPercent(trade.pnl_pct)}
                        </td>
                        <td className="p-2 text-surface-500">{trade.reason || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.trades.length > 100 && (
                  <div className="p-2 text-center text-sm text-surface-500">
                    Showing first 100 of {result.trades.length} trades
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error Result */}
      {result && result.status === 'failed' && (
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-red-400">Backtest Failed</h2>
          </div>
          <div className="rounded-lg bg-red-900/20 p-4">
            <pre className="whitespace-pre-wrap text-sm text-red-300">
              {result.error_message || 'Unknown error'}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
