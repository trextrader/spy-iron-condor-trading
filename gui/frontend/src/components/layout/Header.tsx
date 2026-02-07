import { useConfigStore } from '@/stores/configStore';

export default function Header() {
  const { configHash, isLoading } = useConfigStore();

  return (
    <header className="flex h-16 items-center justify-between border-b border-surface-800 bg-surface-900 px-6">
      {/* Left side - breadcrumb / page title */}
      <div className="flex items-center space-x-4">
        <h1 className="text-lg font-semibold text-surface-100">CondorBrain Control Center</h1>
      </div>

      {/* Right side - status indicators */}
      <div className="flex items-center space-x-6">
        {/* Config hash indicator */}
        <div className="flex items-center space-x-2">
          <span className="text-xs text-surface-500">Config:</span>
          <code className="rounded bg-surface-800 px-2 py-1 font-mono text-xs text-brand-400">
            {isLoading ? '...' : configHash?.slice(0, 8) || 'default'}
          </code>
        </div>

        {/* Connection status */}
        <div className="flex items-center space-x-2">
          <span className="status-dot status-dot-success" />
          <span className="text-xs text-surface-400">Connected</span>
        </div>

        {/* Time */}
        <div className="text-sm tabular-nums text-surface-400">
          {new Date().toLocaleTimeString()}
        </div>
      </div>
    </header>
  );
}
