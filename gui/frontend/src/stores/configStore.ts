import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { FullConfig, IntelligenceConfig, ExecutionRealityConfig } from '@shared/types';

interface ConfigState {
  // Data
  config: FullConfig | null;
  configHash: string | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  setConfig: (config: FullConfig, hash: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  toggleComponent: (path: string, enabled: boolean) => void;
  updateIntelligence: (updates: Partial<IntelligenceConfig>) => void;
  updateExecutionReality: (updates: Partial<ExecutionRealityConfig>) => void;
  reset: () => void;
}

const initialState = {
  config: null,
  configHash: null,
  isLoading: false,
  error: null,
};

export const useConfigStore = create<ConfigState>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setConfig: (config, hash) => {
        set({ config, configHash: hash, error: null }, false, 'setConfig');
      },

      setLoading: (isLoading) => {
        set({ isLoading }, false, 'setLoading');
      },

      setError: (error) => {
        set({ error, isLoading: false }, false, 'setError');
      },

      toggleComponent: (path, enabled) => {
        const { config } = get();
        if (!config) return;

        // Deep clone and update the config based on path
        const newConfig = JSON.parse(JSON.stringify(config)) as FullConfig;

        // Parse path like "intelligence.condornet_enabled"
        const parts = path.split('.');
        let current: Record<string, unknown> = newConfig;

        for (let i = 0; i < parts.length - 1; i++) {
          current = current[parts[i]!] as Record<string, unknown>;
        }

        const lastKey = parts[parts.length - 1]!;
        current[lastKey] = enabled;

        set({ config: newConfig }, false, 'toggleComponent');
      },

      updateIntelligence: (updates) => {
        const { config } = get();
        if (!config) return;

        const newConfig = {
          ...config,
          intelligence: { ...config.intelligence, ...updates },
        };

        set({ config: newConfig }, false, 'updateIntelligence');
      },

      updateExecutionReality: (updates) => {
        const { config } = get();
        if (!config) return;

        const newConfig = {
          ...config,
          execution_reality: { ...config.execution_reality, ...updates },
        };

        set({ config: newConfig }, false, 'updateExecutionReality');
      },

      reset: () => {
        set(initialState, false, 'reset');
      },
    }),
    { name: 'config-store' }
  )
);
