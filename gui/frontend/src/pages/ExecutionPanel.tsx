/**
 * Execution Reality Engine Panel
 * Phase 6.4 - Configure the 8 execution reality components
 */

const executionComponents = [
  { name: 'Slippage Model', description: 'Base slippage and volatility scaling', enabled: true },
  { name: 'Spread Dynamics', description: 'Bid-ask spread modeling', enabled: true },
  { name: 'Fill Probability', description: 'Order fill likelihood based on size', enabled: true },
  { name: 'Partial Fills', description: 'Simulate incomplete order fills', enabled: true },
  { name: 'Latency Model', description: 'Network and processing delays', enabled: true },
  { name: 'Queue Position', description: 'Order book queue simulation', enabled: false },
  { name: 'Rejections', description: 'Order rejection probability', enabled: false },
  { name: 'Market Impact', description: 'Price impact from order flow', enabled: true },
];

export default function ExecutionPanel() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-surface-100">Execution Reality Engine</h1>
        <p className="mt-1 text-sm text-surface-400">
          Configure realistic market execution simulation
        </p>
      </div>

      {/* Master Toggle */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-surface-100">Execution Reality</h2>
            <p className="text-sm text-surface-400">Enable realistic execution simulation</p>
          </div>
          <button className="btn-primary">Enabled</button>
        </div>
      </div>

      {/* Component Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {executionComponents.map((component) => (
          <div key={component.name} className="card">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium text-surface-100">{component.name}</h3>
                <p className="mt-1 text-xs text-surface-500">{component.description}</p>
              </div>
              <span
                className={component.enabled ? 'status-dot-success' : 'status-dot-pending'}
                style={{ width: 8, height: 8, borderRadius: '50%' }}
              />
            </div>
            <div className="mt-4">
              <button className="btn-ghost w-full text-xs">Configure</button>
            </div>
          </div>
        ))}
      </div>

      {/* Monte Carlo Preview */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-surface-100">Expected Impact Distribution</h2>
        </div>
        <div className="flex h-48 items-center justify-center text-surface-500">
          Monte Carlo visualization will be rendered here
        </div>
      </div>

      {/* Profile Management */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-surface-400">Active Profile</p>
            <p className="font-medium text-surface-100">Realistic</p>
          </div>
          <div className="flex gap-3">
            <button className="btn-secondary">Ideal</button>
            <button className="btn-secondary">Conservative</button>
            <button className="btn-primary">Realistic</button>
          </div>
        </div>
      </div>
    </div>
  );
}
