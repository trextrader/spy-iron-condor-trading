/**
 * Intelligence Control Matrix Panel
 * Phase 6.3 - Toggle and configure intelligence components
 */

export default function IntelligencePanel() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-surface-100">Intelligence Control Matrix</h1>
        <p className="mt-1 text-sm text-surface-400">
          Configure CondorNet, Fuzzy Logic, and indicator components
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* CondorNet Section */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-surface-100">CondorNet</h2>
              <span className="badge-success">Enabled</span>
            </div>
          </div>
          <div className="space-y-3">
            {['ETD-1 Integrator', 'TFT Controller', 'Neural CDE', 'MoE Decoder'].map(
              (component) => (
                <div
                  key={component}
                  className="flex items-center justify-between rounded-lg bg-surface-800/50 p-3"
                >
                  <span className="text-sm text-surface-300">{component}</span>
                  <button className="btn-ghost text-xs">Toggle</button>
                </div>
              )
            )}
          </div>
        </div>

        {/* Fuzzy Logic Section */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-surface-100">Fuzzy Logic</h2>
              <span className="badge-success">Enabled</span>
            </div>
          </div>
          <div className="space-y-3">
            {['MTF Consensus', 'IV Rank', 'VIX Regime', 'RSI', 'Stochastic'].map((factor) => (
              <div
                key={factor}
                className="flex items-center justify-between rounded-lg bg-surface-800/50 p-3"
              >
                <span className="text-sm text-surface-300">{factor}</span>
                <button className="btn-ghost text-xs">Toggle</button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Config Actions */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-surface-400">Current Configuration Hash</p>
            <code className="font-mono text-sm text-brand-400">loading...</code>
          </div>
          <div className="flex gap-3">
            <button className="btn-secondary">Save Preset</button>
            <button className="btn-secondary">Load Preset</button>
            <button className="btn-primary">Apply Changes</button>
          </div>
        </div>
      </div>
    </div>
  );
}
