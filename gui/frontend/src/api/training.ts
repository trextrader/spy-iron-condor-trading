/**
 * Training Diagnostics API
 * Phase 6.7 - Model Introspection
 */

import { get } from './client';

export interface DiagnosticsFile {
  filename: string;
  path: string;
  start_time: string;
  end_time: string;
  total_steps: number;
}

export interface LossTrajectory {
  steps: number[];
  components: Record<string, number[]>;
}

export interface FuzzyHeatmap {
  steps: number[];
  gates: string[];
  values: number[][];
}

export interface EpochSummary {
  epoch: number;
  loss: number;
  mse: number;
  npdd: number;
  sharpe: number;
  dd: number;
  turnover: number;
  fuzzy: number;
  pattern_ent: number;
  energy: number;
  growth: number;
}

export interface GateDistribution {
  counts: number[];
  bin_edges: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
}

// List available diagnostic files
export async function listDiagnostics(): Promise<{ files: DiagnosticsFile[]; total: number }> {
  return get('/training/diagnostics/list');
}

// Get loss trajectory data
export async function getLossTrajectory(
  filename: string,
  window = 200,
  components?: string[]
): Promise<LossTrajectory> {
  const params: Record<string, unknown> = { window };
  if (components) {
    params.components = components.join(',');
  }
  return get(`/training/diagnostics/${filename}/loss-trajectory`, params);
}

// Get fuzzy heatmap data
export async function getFuzzyHeatmap(filename: string): Promise<FuzzyHeatmap> {
  return get(`/training/diagnostics/${filename}/fuzzy-heatmap`);
}

// Get epoch summary
export async function getEpochSummary(filename: string): Promise<{ epochs: EpochSummary[] }> {
  return get(`/training/diagnostics/${filename}/epoch-summary`);
}

// Get gate distributions for a specific epoch
export async function getGateDistribution(
  filename: string,
  epoch: number,
  bins = 30
): Promise<{ epoch: number; gates: Record<string, GateDistribution> }> {
  return get(`/training/diagnostics/${filename}/gate-distribution`, { epoch, bins });
}
