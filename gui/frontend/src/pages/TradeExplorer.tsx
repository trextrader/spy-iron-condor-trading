/**
 * Trade Explorer Page
 * Phase 6.8 - Analyze individual trades from backtests
 */

export default function TradeExplorer() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-surface-100">Trade Explorer</h1>
        <p className="mt-1 text-sm text-surface-400">
          Analyze individual trades and P&L breakdown
        </p>
      </div>

      {/* Run Selection and Filters */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex-1">
            <label className="input-label">Backtest Run</label>
            <select className="input">
              <option>Select run...</option>
            </select>
          </div>
          <div>
            <label className="input-label">Outcome</label>
            <select className="input">
              <option value="">All</option>
              <option value="win">Winners</option>
              <option value="loss">Losers</option>
            </select>
          </div>
          <div className="pt-6">
            <button className="btn-secondary">Export CSV</button>
          </div>
        </div>
      </div>

      {/* Trade Table */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-surface-100">Trades</h2>
            <span className="text-sm text-surface-500">0 trades</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Entry Time</th>
                <th>Exit Time</th>
                <th>Duration</th>
                <th>P&L</th>
                <th>P&L %</th>
                <th>Type</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={8} className="text-center text-surface-500">
                  Select a backtest run to view trades
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Analytics */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">Win/Loss Distribution</h2>
          </div>
          <div className="flex h-48 items-center justify-center text-surface-500">
            Distribution chart will be rendered here
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">Duration Histogram</h2>
          </div>
          <div className="flex h-48 items-center justify-center text-surface-500">
            Histogram will be rendered here
          </div>
        </div>
      </div>
    </div>
  );
}
