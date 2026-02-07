/**
 * Tape Viewer Page
 * Phase 6.6 - Browse and filter historical tape data
 */

export default function TapeViewer() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-surface-100">Tape Viewer</h1>
        <p className="mt-1 text-sm text-surface-400">
          Browse historical market data with annotations
        </p>
      </div>

      {/* Tape Selection and Filters */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex-1">
            <label className="input-label">Tape</label>
            <select className="input">
              <option>Select tape...</option>
            </select>
          </div>
          <div className="flex-1">
            <label className="input-label">Date Range</label>
            <input type="text" className="input" placeholder="2024-01-01 to 2024-12-31" />
          </div>
          <div className="flex-1">
            <label className="input-label">Search</label>
            <input type="text" className="input" placeholder="Search..." />
          </div>
          <div className="pt-6">
            <button className="btn-primary">Apply Filters</button>
          </div>
        </div>
      </div>

      {/* Data Grid */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-surface-100">Data</h2>
            <span className="text-sm text-surface-500">0 records</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Open</th>
                <th>High</th>
                <th>Low</th>
                <th>Close</th>
                <th>Volume</th>
                <th>Annotations</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={7} className="text-center text-surface-500">
                  Select a tape to view data
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
