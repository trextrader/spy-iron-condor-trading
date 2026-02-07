import { NavLink } from 'react-router-dom';
import { clsx } from 'clsx';

interface NavItem {
  path: string;
  label: string;
  icon: string;
  category: string;
}

const navItems: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: '📊', category: 'Overview' },
  { path: '/backtest', label: 'Backtest', icon: '⏱️', category: 'Control' },
  { path: '/intelligence', label: 'Intelligence', icon: '🧠', category: 'Control' },
  { path: '/execution', label: 'Execution', icon: '⚡', category: 'Control' },
  { path: '/tape', label: 'Tape Viewer', icon: '📜', category: 'Data' },
  { path: '/model', label: 'Model', icon: '🔬', category: 'Data' },
  { path: '/trades', label: 'Trades', icon: '💹', category: 'Data' },
  { path: '/optimization', label: 'Optimizer', icon: '🎯', category: 'Advanced' },
];

// Group items by category
const groupedNav = navItems.reduce(
  (acc, item) => {
    if (!acc[item.category]) {
      acc[item.category] = [];
    }
    acc[item.category].push(item);
    return acc;
  },
  {} as Record<string, NavItem[]>
);

export default function Sidebar() {
  return (
    <aside className="flex w-64 flex-col border-r border-surface-800 bg-surface-900">
      {/* Logo */}
      <div className="flex h-16 items-center border-b border-surface-800 px-6">
        <span className="text-2xl">🦅</span>
        <span className="ml-3 text-lg font-semibold text-surface-100">CondorBrain</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-4">
        {Object.entries(groupedNav).map(([category, items]) => (
          <div key={category} className="mb-6">
            <h3 className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-surface-500">
              {category}
            </h3>
            <ul className="space-y-1">
              {items.map((item) => (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    className={({ isActive }) =>
                      clsx(
                        'flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                        isActive
                          ? 'bg-brand-600/20 text-brand-400'
                          : 'text-surface-400 hover:bg-surface-800 hover:text-surface-100'
                      )
                    }
                  >
                    <span className="mr-3 text-lg">{item.icon}</span>
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-surface-800 p-4">
        <div className="rounded-lg bg-surface-800/50 p-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-surface-500">System Status</span>
            <span className="status-dot status-dot-success" />
          </div>
          <p className="mt-1 text-sm font-medium text-surface-300">All Systems Operational</p>
        </div>
      </div>
    </aside>
  );
}
