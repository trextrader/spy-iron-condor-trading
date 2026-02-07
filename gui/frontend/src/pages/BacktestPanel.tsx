/**
 * Backtest Control Panel
 * Phase 6.5 - Run backtests, track progress, view results
 */

export default function BacktestPanel() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-surface-100">Backtest Control</h1>
        <p className="mt-1 text-sm text-surface-400">
          Configure and run backtests with full reproducibility
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
              <label className="input-label">Tape</label>
              <select className="input">
                <option>Select tape...</option>
              </select>
            </div>

            {/* Model Selection */}
            <div>
              <label className="input-label">Model</label>
              <select className="input">
                <option>Select model...</option>
              </select>
            </div>

            {/* Seed */}
            <div>
              <label className="input-label">Random Seed</label>
              <input type="number" className="input" placeholder="42" defaultValue={42} />
            </div>

            {/* Device */}
            <div>
              <label className="input-label">Device</label>
              <select className="input">
                <option value="cuda">CUDA</option>
                <option value="cpu">CPU</option>
                <option value="mps">MPS</option>
              </select>
            </div>

            {/* Run Button */}
            <button className="btn-primary w-full">Start Backtest</button>
          </div>
        </div>

        {/* Progress / Status */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-surface-100">Progress</h2>
          </div>
          <div className="flex h-48 items-center justify-center text-surface-500">
            No active backtest
          </div>
        </div>
      </div>

      {/* Results Section */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-surface-100">Results</h2>
        </div>
        <div className="flex h-64 items-center justify-center text-surface-500">
          Run a backtest to see results here
        </div>
      </div>
    </div>
  );
}
