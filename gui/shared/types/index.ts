/**
 * CondorBrain GUI - Shared Type Definitions
 * Phase 6.0 - Project Scaffolding
 *
 * These types are shared between frontend and can be used for API contracts.
 */

// =============================================================================
// ENUMS
// =============================================================================

export type DeviceType = 'cpu' | 'cuda' | 'mps';

export type BacktestStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type OptimizationMode = 'ablation' | 'bayesian' | 'evolutionary' | 'certification';

export type OptimizationObjective = 'npdd' | 'sharpe' | 'dd' | 'composite';

export type OptimizationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

// =============================================================================
// CONFIG TYPES
// =============================================================================

export interface IntelligenceConfig {
  condornet_enabled: boolean;
  fuzzy_enabled: boolean;
  indicators_enabled: boolean;
  diffusion_enabled: boolean;

  // CondorNet sub-components
  etd1_enabled: boolean;
  tft_enabled: boolean;
  cde_enabled: boolean;
  moe_enabled: boolean;

  // Model settings
  model_id: string;
  checkpoint_path: string;
}

export interface ExecutionRealityConfig {
  enabled: boolean;

  // 8 Components
  slippage: SlippageConfig;
  spread_dynamics: SpreadDynamicsConfig;
  fill_probability: FillProbabilityConfig;
  partial_fills: PartialFillsConfig;
  latency: LatencyConfig;
  queue_position: QueuePositionConfig;
  rejections: RejectionsConfig;
  market_impact: MarketImpactConfig;
}

export interface SlippageConfig {
  enabled: boolean;
  base_bps: number;
  volatility_multiplier: number;
}

export interface SpreadDynamicsConfig {
  enabled: boolean;
  base_spread_pct: number;
  volatility_scaling: number;
}

export interface FillProbabilityConfig {
  enabled: boolean;
  base_fill_rate: number;
  size_decay: number;
}

export interface PartialFillsConfig {
  enabled: boolean;
  min_fill_pct: number;
  expected_fills: number;
}

export interface LatencyConfig {
  enabled: boolean;
  mean_ms: number;
  std_ms: number;
}

export interface QueuePositionConfig {
  enabled: boolean;
  position_factor: number;
}

export interface RejectionsConfig {
  enabled: boolean;
  base_rejection_rate: number;
}

export interface MarketImpactConfig {
  enabled: boolean;
  impact_coefficient: number;
  decay_rate: number;
}

export interface FullConfig {
  tape_id: string;
  model_id: string;
  seed: number;
  device: DeviceType;
  batch_size: number;
  intelligence: IntelligenceConfig;
  execution_reality: ExecutionRealityConfig;
}

// =============================================================================
// BACKTEST TYPES
// =============================================================================

export interface BacktestRequest {
  tape_id: string;
  model_id: string;
  seed?: number;
  device?: DeviceType;
  batch_size?: number;
  limit?: number;
  use_execution_reality?: boolean;
  intelligence_config?: Partial<IntelligenceConfig>;
  execution_reality_config?: Partial<ExecutionRealityConfig>;
}

export interface BacktestStartResponse {
  run_id: string;
  config_hash: string;
  status: BacktestStatus;
  message: string;
}

export interface BacktestProgress {
  run_id: string;
  status: BacktestStatus;
  progress_pct: number;
  bars_processed: number;
  total_bars: number;
  current_pnl: number;
  eta_seconds?: number;
}

export interface EquityCurvePoint {
  timestamp: string;
  equity: number;
  drawdown: number;
  position_count: number;
}

export interface TradeSummary {
  trade_id: string;
  entry_time: string;
  exit_time: string;
  pnl: number;
  pnl_pct: number;
  duration_minutes: number;
  trade_type: string;
}

export interface BacktestMetrics {
  net_pnl: number;
  total_trades: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown: number;
  npdd_ratio: number;
  avg_trade_pnl: number;
  avg_trade_duration: number;
  profit_factor: number;
}

export interface BacktestResult {
  run_id: string;
  config_hash: string;
  status: BacktestStatus;
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  metrics: BacktestMetrics;
  equity_curve: EquityCurvePoint[];
  trades: TradeSummary[];
  logs?: string[];
}

// =============================================================================
// OPTIMIZATION TYPES
// =============================================================================

export interface OptimizationRequest {
  mode: OptimizationMode;
  objective: OptimizationObjective;
  max_evaluations?: number;
  base_config_hash?: string;
  search_space?: {
    components_to_search: string[];
    param_ranges: Record<string, [number, number]>;
  };
  objective_weights?: Record<string, number>;
  tape_id: string;
  model_id: string;
  seed?: number;
}

export interface ConfigCandidate {
  config_hash: string;
  score: number;
  metrics: Record<string, number>;
  components_active: string[];
  params: Record<string, unknown>;
  evaluation_number: number;
  is_best: boolean;
}

export interface ComponentImportance {
  component_path: string;
  importance: number;
  direction: 'positive' | 'negative' | 'neutral';
  confidence: number;
}

export interface ComponentSynergy {
  component_a: string;
  component_b: string;
  synergy_score: number;
  confidence: number;
}

export interface OptimizationProgress {
  optimization_id: string;
  status: OptimizationStatus;
  progress_pct: number;
  evaluations_completed: number;
  total_evaluations: number;
  best_score: number;
  best_config_hash: string;
  current_evaluation?: ConfigCandidate;
  eta_seconds?: number;
}

export interface OptimizationResult {
  optimization_id: string;
  status: OptimizationStatus;
  mode: OptimizationMode;
  objective: OptimizationObjective;
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  best_config_hash: string;
  best_score: number;
  best_metrics: Record<string, number>;
  all_candidates: ConfigCandidate[];
  component_importance?: ComponentImportance[];
  synergies?: ComponentSynergy[];
  error_message?: string;
}

// =============================================================================
// WEBSOCKET TYPES
// =============================================================================

export type WebSocketChannel =
  | 'backtest.progress'
  | 'optimization.progress'
  | 'config.changed'
  | 'logs';

export interface WebSocketMessage<T = unknown> {
  channel: WebSocketChannel;
  event: string;
  data: T;
  timestamp: string;
}

// =============================================================================
// API RESPONSE TYPES
// =============================================================================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}
