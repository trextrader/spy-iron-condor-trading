/**
 * Dashboard Page
 * Phase 6.2 - Main overview with metrics, activity feed, and quick actions
 *
 * The mission control center for CondorBrain.
 */

import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import MetricCard from '@/components/dashboard/MetricCard';
import EquityCurveMini from '@/components/dashboard/EquityCurveMini';
import ActivityFeed, { generateMockActivity } from '@/components/dashboard/ActivityFeed';
import RecentRunsList from '@/components/dashboard/RecentRunsList';
import SystemStatus from '@/components/dashboard/SystemStatus';
import TrainingStatusBanner from '@/components/dashboard/TrainingStatusBanner';
import { useUIStore } from '@/stores/uiStore';
import { useTrainingTelemetry } from '@/hooks/useTrainingTelemetry';
import { listBacktests } from '@/api/backtest';
import { getCurrentConfig } from '@/api/config';

export default function Dashboard() {
  const navigate = useNavigate();
  const [currentTime, setCurrentTime] = useState(new Date());
  const { wsConnected } = useUIStore();
  const { status: trainingStatus, isConnected: trainingConnected } = useTrainingTelemetry();

  // Update clock every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch config
  const { data: configData, isLoading: configLoading, isError: configError } = useQuery({
    queryKey: ['config'],
    queryFn: getCurrentConfig,
  });

  // Fetch recent backtests
  const { data: backtestsData } = useQuery({
    queryKey: ['recent-backtests'],
    queryFn: () => listBacktests(10),
  });

  // Mock data for demo (replace with real data later)
  const mockMetrics = {
    netPnl: 12847.32,
    winRate: 0.634,
    sharpe: 1.87,
    maxDrawdown: 0.0823,
    totalTrades: 247,
    avgTradeDuration: 45.2,
  };

  const mockEquityData = generateMockEquityData();
  const mockActivity = generateMockActivity();

  // Dynamic system components with real WebSocket status
  type StatusLevel = 'healthy' | 'warning' | 'error' | 'unknown';
  const systemComponents = useMemo(
    () => [
      {
        name: 'Backend API',
        status: (configError ? 'error' : configLoading ? 'unknown' : 'healthy') as StatusLevel,
        message: configError ? 'Connection failed' : configLoading ? 'Connecting...' : 'Response OK',
      },
      {
        name: 'WebSocket',
        status: (wsConnected ? 'healthy' : 'warning') as StatusLevel,
        message: wsConnected ? 'Live connection' : 'Reconnecting...',
      },
      {
        name: 'CondorNet Model',
        status: (trainingStatus.isTraining ? 'healthy' : 'unknown') as StatusLevel,
        message: trainingStatus.isTraining
          ? `Training Epoch ${trainingStatus.currentEpoch}`
          : 'Idle',
      },
      {
        name: 'Training Telemetry',
        status: (trainingConnected ? 'healthy' : 'warning') as StatusLevel,
        message: trainingConnected ? 'Streaming' : 'Waiting...',
      },
      {
        name: 'Data Pipeline',
        status: 'healthy' as StatusLevel,
        message: 'Synced',
      },
    ],
    [wsConnected, trainingStatus, trainingConnected, configLoading, configError]
  );

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-surface-100">Dashboard</h1>
          <p className="mt-1 text-sm text-surface-400">
            CondorBrain Mission Control - Real-time system overview
          </p>
        </div>
        <div className="text-right">
          <div className="font-mono text-lg tabular-nums text-surface-100">
            {currentTime.toLocaleTimeString()}
          </div>
          <div className="text-sm text-surface-500">
            {currentTime.toLocaleDateString(undefined, {
              weekday: 'long',
              year: 'numeric',
              month: 'long',
              day: 'numeric',
            })}
          </div>
        </div>
      </div>

      {/* Top Metrics Row */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Net P&L"
          value={mockMetrics.netPnl}
          format="currency"
          change={0.127}
          changeLabel="7d"
          icon="💰"
          variant="highlight"
          sparklineData={mockEquityData.map((d) => d.equity)}
        />
        <MetricCard
          label="Win Rate"
          value={mockMetrics.winRate}
          format="percent"
          decimals={1}
          change={0.023}
          changeLabel="7d"
          icon="🎯"
        />
        <MetricCard
          label="Sharpe Ratio"
          value={mockMetrics.sharpe}
          format="ratio"
          change={0.15}
          changeLabel="7d"
          icon="📈"
        />
        <MetricCard
          label="Max Drawdown"
          value={mockMetrics.maxDrawdown}
          format="percent"
          decimals={2}
          icon="📉"
          variant={mockMetrics.maxDrawdown > 0.1 ? 'warning' : 'default'}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Equity Curve - Takes 2 columns */}
        <div className="card lg:col-span-2">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-lg font-semibold text-surface-100">Equity Curve</h2>
            <div className="flex gap-2">
              <button className="btn-ghost text-xs">7D</button>
              <button className="btn-ghost text-xs">30D</button>
              <button className="btn-ghost text-xs">90D</button>
              <button className="btn-secondary text-xs">All</button>
            </div>
          </div>
          <EquityCurveMini data={mockEquityData} height={250} showDrawdown />
        </div>

        {/* System Status */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">System Status</h2>
          </div>
          <SystemStatus components={systemComponents} />
        </div>
      </div>

      {/* Secondary Row */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Recent Runs */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-lg font-semibold text-surface-100">Recent Runs</h2>
            <button
              className="text-xs text-brand-400 hover:text-brand-300"
              onClick={() => navigate('/backtest')}
            >
              View All →
            </button>
          </div>
          <RecentRunsList
            runs={
              backtestsData?.runs.map((r) => ({
                ...r,
                net_pnl: Math.random() * 10000 - 3000, // Mock for demo
                trade_count: Math.floor(Math.random() * 200) + 50,
              })) ?? []
            }
            onSelectRun={(runId) => navigate(`/backtest?run=${runId}`)}
          />
        </div>

        {/* Activity Feed */}
        <div className="card lg:col-span-2">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-lg font-semibold text-surface-100">Activity Feed</h2>
            <button className="text-xs text-surface-400 hover:text-surface-300">
              Clear All
            </button>
          </div>
          <ActivityFeed items={mockActivity} maxItems={6} />
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-surface-100">Quick Actions</h2>
        </div>
        <div className="flex flex-wrap gap-3">
          <button className="btn-primary" onClick={() => navigate('/backtest')}>
            <span className="mr-2">⏱️</span>
            New Backtest
          </button>
          <button className="btn-secondary" onClick={() => navigate('/model')}>
            <span className="mr-2">🧠</span>
            Model Introspection
          </button>
          <button className="btn-secondary" onClick={() => navigate('/intelligence')}>
            <span className="mr-2">🔧</span>
            Configure Intelligence
          </button>
          <button className="btn-secondary" onClick={() => navigate('/optimization')}>
            <span className="mr-2">🎯</span>
            Run Optimization
          </button>
          <button className="btn-ghost">
            <span className="mr-2">📥</span>
            Export Config
          </button>
        </div>
      </div>

      {/* Training Status Banner - Live WebSocket Updates */}
      <TrainingStatusBanner />

      {/* Config Hash Footer */}
      <div className="flex items-center justify-between text-xs text-surface-500">
        <div>
          Config Hash:{' '}
          <code className="rounded bg-surface-800 px-2 py-0.5 font-mono text-brand-400">
            {configData?.config_hash?.slice(0, 12) ?? 'loading...'}
          </code>
        </div>
        <div>Last updated: {new Date().toLocaleTimeString()}</div>
      </div>
    </div>
  );
}

// Generate mock equity data for demo
function generateMockEquityData() {
  const data = [];
  let equity = 100000;
  const now = new Date();

  for (let i = 90; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);

    // Random walk with slight upward bias
    const change = (Math.random() - 0.45) * 2000;
    equity = Math.max(90000, equity + change);

    // Calculate drawdown from peak
    const peak = Math.max(...data.map((d) => d.equity), equity);
    const drawdown = (peak - equity) / peak;

    data.push({
      timestamp: date.toISOString(),
      equity: Math.round(equity * 100) / 100,
      drawdown: Math.round(drawdown * 10000) / 10000,
    });
  }

  return data;
}
