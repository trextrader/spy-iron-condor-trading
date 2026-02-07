import { get, post } from './client';
import type { FullConfig } from '@shared/types';

interface ConfigResponse {
  config: FullConfig;
  config_hash: string;
}

interface ToggleResponse {
  success: boolean;
  new_config_hash: string;
  component_path: string;
  new_value: boolean;
}

// Get current configuration
export async function getCurrentConfig(): Promise<ConfigResponse> {
  return get<ConfigResponse>('/config/current');
}

// Update full configuration
export async function updateConfig(config: FullConfig): Promise<ConfigResponse> {
  return post<ConfigResponse>('/config/update', config);
}

// Toggle a single component
export async function toggleComponent(
  componentPath: string,
  enabled: boolean
): Promise<ToggleResponse> {
  return post<ToggleResponse>('/config/toggle', {
    component_path: componentPath,
    enabled,
  });
}

// Get configuration by hash
export async function getConfigByHash(hash: string): Promise<ConfigResponse> {
  return get<ConfigResponse>(`/config/hash/${hash}`);
}

// Get configuration history
export async function getConfigHistory(limit = 50): Promise<{
  entries: Array<{ hash: string; timestamp: string; changes: string[] }>;
  total: number;
}> {
  return get('/config/history', { limit });
}
