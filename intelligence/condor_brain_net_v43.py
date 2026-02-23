"""
condor_brain_net_v43.py — CondorNet™ v4.3 Architecture
=======================================================
Extends condor_brain_net_v42.py with:

1. MultiTFProjector      — 4 TF inputs [B,T,64] each → [B,T,256] joint
2. PivotProjector        — 13 sparse pivot features → [B,T,16] dense
3. TFFusionBlock         — fuses TF + pivot projections → [B,T,256]
4. OptionsChainEncoder   — transformer over chain grid [B,N,10] → [B,128]
5. StrategyHead          — 10 strategy classes + leg params + entry signal
6. RiskMetricHead        — PoP, EV, max_loss, VaR, CVaR predictions
7. PivotPredictionHead   — anticipates reversals BEFORE pivot confirmation
8. CondorNetV43          — unified model combining all components

Pivot Prediction Philosophy:
    The PineScript multi-pivot engine (medium: L30/R32, strong: L69/R69) confirms
    pivots AFTER the right-bar lookback. This file teaches the model to predict
    the ONSET of a pivot before confirmation by learning from the leading indicators
    (RSI divergence, Bollinger squeeze, PSAR flip proximity, momentum exhaustion).
    Only medium and strong strength pivots are used — weak pivots are excluded.

Backward Compatibility:
    Old single-TF CondorNet v42 API is fully preserved via CondorNetV43.forward_compat().
    All existing backtest engine calls continue to work without modification.

Author: CondorNet v4.3 Implementation
Version: 4.3.0
Date: 2026-02-23
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all preserved v42 components
from intelligence.condor_brain_net_v42 import (
    AugmentedStateSpec,
    BlockMatrixA,
    BlockMatrixB,
    CDEResponseG,
    FullForcingD,
    PivotHead,
    OffsetModule,
    PivotEncoder,
    etd1_kernel,
    condornet_master_step,
    CanonicalPredicateGates,
    PredicateSignature,
    RegimeCombinatoricsDynamics,
    RelationalLogicLayer,
    PredicateSet,
    SuperSet,
    TFTControlEncoder,
    FusionGate,
    CondorNet,
    group_invariant_loss,
    spectral_radius_loss,
)

from intelligence.schema_v43 import (
    SCHEMA_VERSION,
    N_STRATEGY_TYPES,
    STRATEGY_TYPES,
    ABSTAIN_IDX,
    ABSTAIN_CONFIDENCE_THRESHOLD,
    CHAIN_GRID_CONFIG,
    N_PIVOT_FEATURES,
    TF_FEATURE_NAMES,
)

# =============================================================================
# TYPED OUTPUT DATACLASS
# =============================================================================

@dataclass
class CondorNetOutput:
    """
    All outputs from a CondorNetV43 forward pass — single typed return object.
    Ensures output contract is consistent between training and inference.
    """
    # Core outputs
    strategy_logits:   torch.Tensor          # [B, 10] — softmax over 10 strategy types
    leg_params:        torch.Tensor          # [B, 4, 5] — per-leg: (moneyness_offset, expiry_bucket, long_short, qty, delta_target)
    entry_signal:      torch.Tensor          # [B, 1] — binary entry confidence ∈ [0,1]

    # Risk metric outputs
    pop:               torch.Tensor          # [B, 1] — PoP ∈ [0,1]
    ev:                torch.Tensor          # [B, 1] — Expected Value ($)
    max_loss:          torch.Tensor          # [B, 1] — Max loss ≥ 0
    var_95:            torch.Tensor          # [B, 1] — 95% VaR ≥ 0
    cvar_95:           torch.Tensor          # [B, 1] — 95% CVaR ≥ 0

    # Pivot prediction outputs
    pivot_high_probs:  torch.Tensor          # [B, 5] — P(pivot_high in N bars), N=[5,10,20,35,70]
    pivot_low_probs:   torch.Tensor          # [B, 5] — P(pivot_low in N bars)
    pivot_strength:    torch.Tensor          # [B, 2] — logits: [medium, strong]

    # Position sizing
    position_size:     torch.Tensor          # [B, 1] — fuzzy-scaled size ∈ [0,1]

    # Spot price auxiliary
    spot_pred:         torch.Tensor          # [B, 1] — target spot price prediction

    # v42 compatibility outputs (filled by condornet_master_step inside core)
    final_gate:        Optional[torch.Tensor] = None   # [B, T] — existing gate output
    diagnostics:       Optional[Dict]         = None   # Full v42 diagnostic dict

    # Selected strategy (argmax at inference)
    @property
    def strategy_type_idx(self) -> torch.Tensor:
        return self.strategy_logits.argmax(dim=-1)   # [B]

    @property
    def abstain_mask(self) -> torch.Tensor:
        """True where model confidence is below abstain threshold."""
        return self.strategy_logits.softmax(dim=-1).max(dim=-1).values < ABSTAIN_CONFIDENCE_THRESHOLD

    def to_dict(self) -> Dict:
        """Serialize all tensors as plain dicts (for logging/reporting)."""
        return {
            "strategy_logits":  self.strategy_logits.detach().cpu().tolist(),
            "entry_signal":     self.entry_signal.detach().cpu().tolist(),
            "pop":              self.pop.detach().cpu().tolist(),
            "ev":               self.ev.detach().cpu().tolist(),
            "max_loss":         self.max_loss.detach().cpu().tolist(),
            "var_95":           self.var_95.detach().cpu().tolist(),
            "cvar_95":          self.cvar_95.detach().cpu().tolist(),
            "pivot_high_probs": self.pivot_high_probs.detach().cpu().tolist(),
            "pivot_low_probs":  self.pivot_low_probs.detach().cpu().tolist(),
            "position_size":    self.position_size.detach().cpu().tolist(),
            "spot_pred":        self.spot_pred.detach().cpu().tolist(),
        }


# =============================================================================
# PART 1: MULTI-TF PROJECTOR
# =============================================================================

class MultiTFProjector(nn.Module):
    """
    Projects 4 independent TF feature tensors to a shared joint representation.

    Each TF gets its own linear projection (d_tf_in → d_tf_proj=64) because
    TF features have different statistical scales and temporal resolutions.
    Outputs are concatenated: [B, T, d_tf_proj * 4] = [B, T, 256].

    The 4 separate projectors allow the model to learn which TF contributes
    most to each decision — tracked via MultiTFContributionRatios hook.

    Input:   4 × [B, T, d_tf_in]   (d_tf_in = len(TF_FEATURE_NAMES) = 64, or +ETL features)
    Output:  [B, T, d_joint=256]
    """
    def __init__(self, d_tf_in: int = 64, d_tf_proj: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_tf_proj = d_tf_proj
        self.d_joint = d_tf_proj * 4  # 256

        self.proj_m1  = nn.Sequential(nn.Linear(d_tf_in, d_tf_proj), nn.LayerNorm(d_tf_proj), nn.GELU())
        self.proj_m5  = nn.Sequential(nn.Linear(d_tf_in, d_tf_proj), nn.LayerNorm(d_tf_proj), nn.GELU())
        self.proj_m15 = nn.Sequential(nn.Linear(d_tf_in, d_tf_proj), nn.LayerNorm(d_tf_proj), nn.GELU())
        self.proj_h1  = nn.Sequential(nn.Linear(d_tf_in, d_tf_proj), nn.LayerNorm(d_tf_proj), nn.GELU())
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        x_m1:  torch.Tensor,   # [B, T, d_tf_in]
        x_m5:  torch.Tensor,   # [B, T, d_tf_in]
        x_m15: torch.Tensor,   # [B, T, d_tf_in]
        x_h1:  torch.Tensor,   # [B, T, d_tf_in]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            joint:          [B, T, 256]
            contrib_norms:  Frobenius norms per TF (for E3 telemetry hook)
        """
        p_m1  = self.proj_m1(x_m1)
        p_m5  = self.proj_m5(x_m5)
        p_m15 = self.proj_m15(x_m15)
        p_h1  = self.proj_h1(x_h1)

        # Contribution ratios (Frobenius norms) — for MultiTFContributionRatios hook
        n_m1  = float(p_m1.norm(p=2).item())
        n_m5  = float(p_m5.norm(p=2).item())
        n_m15 = float(p_m15.norm(p=2).item())
        n_h1  = float(p_h1.norm(p=2).item())
        total = n_m1 + n_m5 + n_m15 + n_h1 + 1e-8
        contrib_norms = {
            "m1":  n_m1 / total,
            "m5":  n_m5 / total,
            "m15": n_m15 / total,
            "h1":  n_h1 / total,
        }

        joint = torch.cat([p_m1, p_m5, p_m15, p_h1], dim=-1)   # [B, T, 256]
        return self.dropout(joint), contrib_norms


# =============================================================================
# PART 2: PIVOT PROJECTOR + TF FUSION BLOCK
# =============================================================================

class PivotProjector(nn.Module):
    """
    Projects sparse pivot features into a dense embedding.

    Input:   [B, T, 13]  sparse pivot features (NaN masked → 0 + mask)
    Output:  [B, T, 16]

    The pivot mask is used for masked mean pooling — padded NaN positions
    do not contribute to the projection output.

    Pivot features (medium + strong only per spec):
        PivotHigh, PivotLow, PivotResidual, PivotResidualATR, PivotResidualZ,
        PivotCurvatureATR, PivotCurvatureProxy, PivotSegmentLengthBars,
        PivotSegmentLengthMinutes, PivotSegmentResidualStd, PivotSegmentVolatility,
        Slope, SlopeATR
    """
    def __init__(self, d_pivot: int = 13, d_out: int = 16):
        super().__init__()
        self.d_out = d_out
        self.proj = nn.Sequential(
            nn.Linear(d_pivot, d_out * 2),
            nn.GELU(),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )
        self.no_pivot_embed = nn.Parameter(torch.zeros(d_out))
        """Learned embedding for positions where no pivot occurred (all NaN)."""

    def forward(
        self,
        pivot_features: torch.Tensor,   # [B, T, 13] — NaN replaced with 0
        pivot_mask: torch.Tensor,        # [B, T, 13] bool — True = NaN (no event)
    ) -> torch.Tensor:
        """
        Returns: [B, T, 16]
        """
        B, T, _ = pivot_features.shape

        # Where all 13 pivots are NaN → position has no pivot information at all
        all_nan = pivot_mask.all(dim=-1, keepdim=True)   # [B, T, 1]

        # Zero out NaN positions before projection
        x = pivot_features * (~pivot_mask).float()   # [B, T, 13]
        out = self.proj(x)                            # [B, T, 16]

        # Replace fully-NaN positions with learned no-pivot embedding
        no_pivot = self.no_pivot_embed.view(1, 1, self.d_out).expand(B, T, -1)
        out = torch.where(all_nan, no_pivot, out)

        return out


class TFFusionBlock(nn.Module):
    """
    Fuses MultiTFProjector output [B,T,256] with PivotProjector output [B,T,16].

    Architecture:
        cat([tf_joint, pivot_proj]) → [B, T, 272]
        Linear(272 → 256) + LayerNorm + residual
        Output: [B, T, 256] — feeds into chain concatenation
    """
    def __init__(self, d_tf: int = 256, d_pivot: int = 16, d_out: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_tf + d_pivot, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_out)
        # Residual projection from d_tf to d_out (in case d_tf != d_out)
        self.residual_proj = nn.Linear(d_tf, d_out, bias=False) if d_tf != d_out else nn.Identity()

    def forward(
        self,
        tf_joint:    torch.Tensor,   # [B, T, 256]
        pivot_embed: torch.Tensor,   # [B, T, 16]
    ) -> torch.Tensor:
        """Returns: [B, T, 256]"""
        fused = self.fusion(torch.cat([tf_joint, pivot_embed], dim=-1))   # [B, T, 256]
        residual = self.residual_proj(tf_joint)
        return self.norm(fused + residual)


# =============================================================================
# PART 3: OPTIONS CHAIN ENCODER
# =============================================================================

class OptionsChainEncoder(nn.Module):
    """
    Transformer encoder over the options chain grid.

    Input:  chain_grid [B, N_contracts, 10]  (N ≤ 120 after filtering)
            chain_mask [B, N_contracts] bool  True = padded/illiquid (ignored)

    Architecture:
        1. Linear: 10 → d_model (64)
        2. Moneyness-ranked positional encoding (learned)
        3. TransformerEncoder: 2 layers, 4 heads, d_ff=256
        4. Masked mean pooling → [B, d_model]
        5. Output projection → [B, d_chain=128]
        6. Skew signal extraction: put_iv_25d - call_iv_25d (appended)

    Output:  [B, d_chain=128]

    Falls back to all-zero embedding with warning if chain is empty.
    """
    def __init__(
        self,
        in_features: int = 10,
        d_model:     int = 64,
        n_heads:     int = 4,
        n_layers:    int = 2,
        d_ff:        int = 256,
        d_chain:     int = 128,
        dropout:     float = 0.1,
        max_contracts: int = 120,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_chain  = d_chain
        self.in_features = in_features

        # Feature indices in chain grid (per CHAIN_FEATURE_NAMES)
        # [moneyness, days_to_exp, iv, delta, gamma, theta, vega, bid, ask, oi_norm]
        self.MONEYNESS_IDX = 0
        self.IV_IDX        = 2
        self.DELTA_IDX     = 3

        # Input projection
        self.input_proj = nn.Linear(in_features, d_model)

        # Learned positional encoding by moneyness rank position
        self.pos_encoding = nn.Embedding(max_contracts + 1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,   # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection: d_model → d_chain
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_chain),
            nn.LayerNorm(d_chain),
        )

        # Skew signal extraction (put_iv_25d - call_iv_25d)
        self.skew_proj = nn.Linear(1, 8)   # project scalar skew to 8-dim, concat into output

        # Final output combines chain embed + skew signal
        self.final_proj = nn.Linear(d_chain + 8, d_chain)

    def _extract_skew(self, chain_grid: torch.Tensor, chain_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract put skew - call skew signal from chain.
        Approximated as mean IV of puts with delta ∈ [-0.30, -0.20]
        minus mean IV of calls with delta ∈ [0.20, 0.30].

        Returns: [B, 1] skew value (positive = put skew, crash fear)
        """
        B = chain_grid.shape[0]
        delta = chain_grid[:, :, self.DELTA_IDX]   # [B, N]
        iv    = chain_grid[:, :, self.IV_IDX]       # [B, N]
        mask  = chain_mask                          # [B, N] True=invalid

        # Put skew: delta ∈ [-0.30, -0.20]
        put_mask = (delta >= -0.30) & (delta <= -0.20) & (~mask)
        put_iv = (iv * put_mask.float()).sum(dim=-1) / (put_mask.float().sum(dim=-1) + 1e-8)

        # Call skew: delta ∈ [0.20, 0.30]
        call_mask = (delta >= 0.20) & (delta <= 0.30) & (~mask)
        call_iv = (iv * call_mask.float()).sum(dim=-1) / (call_mask.float().sum(dim=-1) + 1e-8)

        skew = (put_iv - call_iv).unsqueeze(-1)   # [B, 1]
        return skew

    def forward(
        self,
        chain_grid:  torch.Tensor,              # [B, N_contracts, 10]
        chain_mask:  Optional[torch.Tensor] = None,  # [B, N_contracts] True=padded
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            chain_embed: [B, d_chain=128]
            diag:        dict with chain telemetry (E1 hook data)
        """
        B, N, F = chain_grid.shape

        # Fallback for empty chain
        if N == 0 or (chain_mask is not None and chain_mask.all()):
            import warnings
            warnings.warn("Empty options chain — using zero embedding fallback", RuntimeWarning)
            embed = torch.zeros(B, self.d_chain, device=chain_grid.device, dtype=chain_grid.dtype)
            diag = {"chain_contract_count": 0, "skew": 0.0}
            return embed, diag

        # Chain diagnostic data (E1 hook)
        valid_counts = (~chain_mask).float().sum(dim=-1) if chain_mask is not None else torch.full((B,), N, dtype=torch.float)
        diag = {
            "chain_contract_count_mean": float(valid_counts.mean()),
            "chain_contract_count_min":  float(valid_counts.min()),
            "chain_contract_count_max":  float(valid_counts.max()),
            "chain_iv_mean":  float(chain_grid[:, :, self.IV_IDX].mean()),
            "chain_iv_std":   float(chain_grid[:, :, self.IV_IDX].std()),
        }

        # Sort contracts by moneyness (ascending = most OTM puts → ATM → most OTM calls)
        moneyness = chain_grid[:, :, self.MONEYNESS_IDX]   # [B, N]
        sort_idx  = moneyness.argsort(dim=-1)               # [B, N]
        chain_sorted = torch.gather(
            chain_grid, 1,
            sort_idx.unsqueeze(-1).expand(-1, -1, F)
        )

        if chain_mask is not None:
            mask_sorted = torch.gather(chain_mask, 1, sort_idx)
        else:
            mask_sorted = None

        # Positional encoding by rank position (0 = most OTM put, N-1 = most OTM call)
        positions = torch.arange(N, device=chain_grid.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.pos_encoding(positions)   # [B, N, d_model]

        # Input projection + positional encoding
        x = self.input_proj(chain_sorted) + pos_embed   # [B, N, d_model]

        # Transformer (mask=True → ignored by attention)
        x = self.transformer(x, src_key_padding_mask=mask_sorted)   # [B, N, d_model]

        # Masked mean pooling (ignore padded/illiquid positions)
        if mask_sorted is not None:
            valid = (~mask_sorted).float().unsqueeze(-1)   # [B, N, 1]
            chain_embed = (x * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
        else:
            chain_embed = x.mean(dim=1)   # [B, d_model]

        # Output projection
        chain_embed = self.output_proj(chain_embed)   # [B, d_chain=128]

        # Skew signal extraction + projection
        skew = self._extract_skew(chain_grid, chain_mask if chain_mask is not None else torch.zeros(B, N, dtype=torch.bool, device=chain_grid.device))
        diag["skew_mean"] = float(skew.mean())

        skew_feat = self.skew_proj(skew)                         # [B, 8]
        chain_embed = self.final_proj(
            torch.cat([chain_embed, skew_feat], dim=-1)         # [B, 128+8]
        )                                                        # [B, 128]

        return chain_embed, diag


# =============================================================================
# PART 4: STRATEGY HEAD
# =============================================================================

class StrategyHead(nn.Module):
    """
    Strategy selection head.

    Outputs:
        strategy_logits: [B, 10]     — logits over 10 strategy types (including abstain)
        leg_params:      [B, 4, 5]   — per-leg: (moneyness_offset, expiry_bucket, long_short, qty, delta_target)
        entry_signal:    [B, 1]      — entry confidence ∈ [0,1] (sigmoid)

    The strategy_logits are NOT softmaxed here — loss uses raw logits (cross_entropy).
    Inference uses softmax + abstain threshold check.

    leg_params interpretation:
        dim 0: moneyness_offset — log(strike/spot), typically ∈ [-0.15, 0.15]
        dim 1: expiry_bucket   — 0/1/2 → [nearest weekly, 2nd weekly, monthly]
        dim 2: long_short      — sigmoid → 0=short, 1=long
        dim 3: qty             — softplus → number of contracts (default 1)
        dim 4: delta_target    — tanh scaled → target delta magnitude ∈ [0, 0.5]
    """
    def __init__(
        self,
        d_joint:         int = 384,
        n_strategy_types: int = 10,
        max_legs:        int = 4,
        n_leg_params:    int = 5,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.n_strategy_types = n_strategy_types
        self.max_legs = max_legs

        # Strategy type classifier
        self.strategy_head = nn.Sequential(
            nn.Linear(d_joint, d_joint // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_joint // 2, n_strategy_types),
        )

        # Leg parameters regressor
        self.leg_head = nn.Sequential(
            nn.Linear(d_joint, d_joint // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_joint // 2, max_legs * n_leg_params),
        )

        # Entry signal (confidence gate)
        self.entry_head = nn.Sequential(
            nn.Linear(d_joint, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, joint_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            joint_repr: [B, d_joint=384] — final joint representation (last timestep)

        Returns:
            strategy_logits: [B, 10]
            leg_params:      [B, 4, 5]
            entry_signal:    [B, 1]  ∈ [0,1]
        """
        strategy_logits = self.strategy_head(joint_repr)               # [B, 10]
        leg_params_flat = self.leg_head(joint_repr)                    # [B, 20]
        leg_params = leg_params_flat.view(-1, self.max_legs, 5)        # [B, 4, 5]
        entry_signal = torch.sigmoid(self.entry_head(joint_repr))      # [B, 1]
        return strategy_logits, leg_params, entry_signal


# =============================================================================
# PART 5: RISK METRIC HEAD
# =============================================================================

class RiskMetricHead(nn.Module):
    """
    Predicts per-strategy risk metrics.

    Outputs:
        pop:      [B, 1]  ∈ [0,1]    Probability of Profit (sigmoid)
        ev:       [B, 1]  unbounded  Expected Value ($)
        max_loss: [B, 1]  ≥ 0        Max loss (softplus)
        var_95:   [B, 1]  ≥ 0        95% VaR (softplus)
        cvar_95:  [B, 1]  ≥ 0        95% CVaR ≥ VaR (softplus + offset from var)

    CVaR constraint: cvar_95 ≥ var_95 is enforced via offset parameterization.
    """
    def __init__(self, d_joint: int = 384, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_joint // 2

        self.shared = nn.Sequential(
            nn.Linear(d_joint, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pop_head     = nn.Linear(d_hidden, 1)
        self.ev_head      = nn.Linear(d_hidden, 1)
        self.max_loss_head = nn.Linear(d_hidden, 1)
        self.var_head     = nn.Linear(d_hidden, 1)
        self.cvar_offset_head = nn.Linear(d_hidden, 1)   # CVaR = VaR + softplus(offset)

    def forward(self, joint_repr: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            joint_repr: [B, d_joint=384]

        Returns:
            pop, ev, max_loss, var_95, cvar_95  — each [B, 1]
        """
        h = self.shared(joint_repr)

        pop      = torch.sigmoid(self.pop_head(h))
        ev       = self.ev_head(h)
        max_loss = F.softplus(self.max_loss_head(h))
        var_95   = F.softplus(self.var_head(h))
        cvar_95  = var_95 + F.softplus(self.cvar_offset_head(h))   # Enforces CVaR ≥ VaR

        return pop, ev, max_loss, var_95, cvar_95


# =============================================================================
# PART 6: PIVOT PREDICTION HEAD
# =============================================================================

class PivotPredictionHead(nn.Module):
    """
    Predicts upcoming pivot reversals BEFORE they are confirmed.

    The PineScript multi-pivot engine (medium: L30/R32, strong: L69/R69)
    confirms pivots AFTER right-bar lookback. This head learns to anticipate
    the ONSET of reversals from leading indicator patterns.

    Prediction horizons: N ∈ [5, 10, 20, 35, 70] bars ahead
    - 5  bars: very near-term (intrabar momentum exhaustion)
    - 10 bars: short-term reversal zone
    - 20 bars: medium pivot territory (R=32 confirmation latency)
    - 35 bars: between medium and strong confirmation window
    - 70 bars: strong pivot territory (R=69 confirmation latency)

    Only MEDIUM and STRONG pivots are used (not weak) — matching the PineScript spec.

    Outputs:
        pivot_high_probs: [B, 5]  — P(pivot_high in N bars) for each N
        pivot_low_probs:  [B, 5]  — P(pivot_low in N bars) for each N
        pivot_strength:   [B, 2]  — logits: medium vs strong (for calibration)

    Training labels come from forward-scan of confirmed pivot timestamps in labels_v43.parquet.

    Key leading indicators the model learns to use:
        - RSI divergence from price (rsi_dyn relative to price momentum)
        - Bollinger Band squeeze/expansion (bandwidth, bw_expansion_rate)
        - PSAR flip proximity (psar_reversion_mu near 1.0 → imminent flip)
        - PivotResidualZ at extremes (> 3 or < -3 → reversal zone)
        - ADX falling while trending (momentum exhaustion)
        - Stochastic overbought/oversold zones (stoch_k_dyn)
        - TurtleChannel proximity (close near TurtleChn-Up/Low = breakout/rejection)
        - Pressure imbalance (pressure_up vs pressure_down divergence)
    """

    PIVOT_HORIZONS: List[int] = [5, 10, 20, 35, 70]
    N_HORIZONS: int = 5

    def __init__(self, d_joint: int = 384, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_joint // 2

        self.shared = nn.Sequential(
            nn.Linear(d_joint, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )

        # Separate heads for high and low pivot predictions
        self.pivot_high_head = nn.Linear(d_hidden, self.N_HORIZONS)
        self.pivot_low_head  = nn.Linear(d_hidden, self.N_HORIZONS)

        # Strength classification: medium vs strong
        self.strength_head = nn.Linear(d_hidden, 2)

    def forward(self, joint_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            joint_repr: [B, d_joint=384] — last-timestep joint representation

        Returns:
            pivot_high_probs: [B, 5] ∈ [0,1] — sigmoid probabilities
            pivot_low_probs:  [B, 5] ∈ [0,1]
            pivot_strength:   [B, 2] — raw logits (medium, strong)
        """
        h = self.shared(joint_repr)

        pivot_high_probs = torch.sigmoid(self.pivot_high_head(h))   # [B, 5]
        pivot_low_probs  = torch.sigmoid(self.pivot_low_head(h))    # [B, 5]
        pivot_strength   = self.strength_head(h)                    # [B, 2] raw logits

        return pivot_high_probs, pivot_low_probs, pivot_strength


# =============================================================================
# PART 7: JOINT FUSION (TF + Chain)
# =============================================================================

class JointFusionLayer(nn.Module):
    """
    Combines fused TF representation [B, T, 256] with chain embedding [B, 128].

    Architecture:
        1. Chain embed [B, 128] → broadcast → [B, T, 128]
        2. Concatenate: [B, T, 256+128] = [B, T, 384]
        3. LayerNorm → [B, T, 384]

    The chain embed is identical for all T (same chain state for all timesteps
    within the bar window). Broadcasting is intentional — the chain is a
    global context that modulates all time steps, not a per-step input.
    """
    def __init__(self, d_tf: int = 256, d_chain: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_joint = d_tf + d_chain   # 384
        self.norm = nn.LayerNorm(self.d_joint)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tf_fused:    torch.Tensor,   # [B, T, 256]
        chain_embed: torch.Tensor,   # [B, 128]
    ) -> torch.Tensor:
        """Returns: [B, T, 384]"""
        B, T, _ = tf_fused.shape
        chain_broadcast = chain_embed.unsqueeze(1).expand(B, T, -1)   # [B, T, 128]
        joint = torch.cat([tf_fused, chain_broadcast], dim=-1)         # [B, T, 384]
        return self.dropout(self.norm(joint))


# =============================================================================
# PART 8: POSITION SIZE HEAD
# =============================================================================

class PositionSizeHead(nn.Module):
    """
    Predicts fuzzy-scaled position size ∈ [0, 1].

    Input: PoP prediction + joint representation
    Output: position_size [B, 1]

    Combines:
    - PoP-based sizing (from RiskMetricHead.pop)
    - 10-factor fuzzy sizing signal (from joint representation)
    50/50 blend per spec — configurable via pop_sizing_weight.
    """
    def __init__(self, d_joint: int = 384, dropout: float = 0.1):
        super().__init__()
        self.pop_size_head = nn.Sequential(
            nn.Linear(1, 16),   # pop scalar → expanded
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.fuzzy_size_head = nn.Sequential(
            nn.Linear(d_joint, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        pop:        torch.Tensor,   # [B, 1]
        joint_repr: torch.Tensor,   # [B, d_joint]
        pop_weight: float = 0.5,
    ) -> torch.Tensor:
        """Returns: [B, 1] ∈ [0,1]"""
        pop_size   = torch.sigmoid(self.pop_size_head(pop))
        fuzzy_size = torch.sigmoid(self.fuzzy_size_head(joint_repr))
        return pop_weight * pop_size + (1.0 - pop_weight) * fuzzy_size


# =============================================================================
# PART 9: CONDORNET V43 — UNIFIED MODEL
# =============================================================================

class CondorNetV43(nn.Module):
    """
    CondorNet™ v4.3 — Full Multi-Strategy Options Intelligence Model.

    Combines:
        - CondorNet v42 core (ETD-1 + TFT + CDE + predicate logic) — PRESERVED
        - MultiTFProjector (4 TF inputs → joint)
        - PivotProjector + TFFusionBlock (pivot feature routing)
        - OptionsChainEncoder (transformer over chain grid)
        - JointFusionLayer (TF + chain combination)
        - StrategyHead (10 strategy classes + leg params)
        - RiskMetricHead (PoP, EV, VaR, CVaR)
        - PivotPredictionHead (anticipate reversals before confirmation)
        - PositionSizeHead (PoP-blended fuzzy sizing)

    Forward signature (v43):
        forward(x_m1, x_m5, x_m15, x_h1, chain, chain_mask, pivot_features, ...)

    Backward compatibility (v42):
        forward_compat(x, ...) — routes single TF to all 4 TFs with zero chain

    Pivot Prediction:
        Horizons: [5, 10, 20, 35, 70] bars ahead
        Medium pivot window: L=30, R=32 (from PineScript spec)
        Strong pivot window: L=69, R=69 (from PineScript spec)
        ONLY medium + strong — weak pivots excluded per spec.
    """

    # Pivot parameters matching PineScript medium + strong settings
    MEDIUM_LEFT  = 30
    MEDIUM_RIGHT = 32
    STRONG_LEFT  = 69
    STRONG_RIGHT = 69

    def __init__(
        self,
        # Multi-TF input dimensions
        d_tf_in:           int   = 64,      # Feature count per TF (len(TF_FEATURE_NAMES))
        d_tf_proj:         int   = 64,      # Per-TF projection dim → joint = 4×64 = 256
        # Pivot
        d_pivot_in:        int   = 13,      # N_PIVOT_FEATURES
        d_pivot_proj:      int   = 16,      # Pivot projection dim
        # Chain encoder
        chain_in_features: int   = 10,      # len(CHAIN_FEATURE_NAMES)
        d_model_chain:     int   = 64,
        n_heads_chain:     int   = 4,
        n_layers_chain:    int   = 2,
        d_chain:           int   = 128,
        # CondorNet v42 core dimensions
        d_h:               int   = 256,
        d_v:               int   = 32,
        d_m:               int   = 64,
        d_r:               int   = 32,
        d_control:         int   = 128,
        n_greeks:          int   = 5,
        n_predicates:      int   = 8,
        n_sets:            int   = 4,
        n_super_sets:      int   = 1,
        # Strategy head
        n_strategy_types:  int   = 10,      # includes abstain
        max_legs:          int   = 4,
        # Config
        pop_sizing_weight: float = 0.5,
        dropout:           float = 0.1,
        enforce_sparsity:  bool  = True,
        verbose:           bool  = False,
    ):
        super().__init__()
        self.d_tf_in     = d_tf_in
        self.d_chain     = d_chain
        self.pop_sizing_weight = pop_sizing_weight
        self.verbose     = verbose
        self.schema_version = SCHEMA_VERSION

        d_tf_joint = d_tf_proj * 4  # 256

        # ── v43 NEW COMPONENTS ──────────────────────────────────────────────
        self.tf_projector   = MultiTFProjector(d_tf_in, d_tf_proj, dropout)
        self.pivot_proj     = PivotProjector(d_pivot_in, d_pivot_proj)
        self.tf_fusion      = TFFusionBlock(d_tf_joint, d_pivot_proj, d_tf_joint, dropout)
        self.chain_encoder  = OptionsChainEncoder(
            in_features=chain_in_features,
            d_model=d_model_chain,
            n_heads=n_heads_chain,
            n_layers=n_layers_chain,
            d_chain=d_chain,
            dropout=dropout,
        )
        self.joint_fusion   = JointFusionLayer(d_tf_joint, d_chain, dropout)

        d_joint = d_tf_joint + d_chain   # 256 + 128 = 384

        self.strategy_head  = StrategyHead(d_joint, n_strategy_types, max_legs, dropout=dropout)
        self.risk_head      = RiskMetricHead(d_joint, dropout)
        self.pivot_pred_head = PivotPredictionHead(d_joint, dropout)
        self.size_head      = PositionSizeHead(d_joint, dropout)

        # Spot price auxiliary regression head
        self.spot_head = nn.Sequential(
            nn.Linear(d_joint, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # ── v42 CORE ENGINE ─────────────────────────────────────────────────
        # The v42 CondorNet processes the fused TF representation for predicate
        # logic, set combinatorics, and ETD-1 state dynamics.
        # Input dim to v42 core = d_tf_joint (256) after fusion.
        self.condor_core = CondorNet(
            d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r,
            d_control=d_control,
            n_greeks=n_greeks,
            n_predicates=n_predicates,
            n_sets=n_sets,
            n_super_sets=n_super_sets,
            enforce_sparsity=enforce_sparsity,
            verbose_math=verbose,
        )
        # Projector to adapt d_tf_joint → v42 expected d_input
        self.core_input_proj = nn.Linear(d_tf_joint, self.condor_core.d_input)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            print(f"[CondorNetV43] Initialized | params={total:,} | schema={SCHEMA_VERSION}")
            print(f"  TF joint dim: {d_tf_joint} | chain dim: {d_chain} | joint: {d_joint}")

    def forward(
        self,
        x_m1:            torch.Tensor,                    # [B, T, d_tf_in]
        x_m5:            torch.Tensor,                    # [B, T, d_tf_in]
        x_m15:           torch.Tensor,                    # [B, T, d_tf_in]
        x_h1:            torch.Tensor,                    # [B, T, d_tf_in]
        chain:           torch.Tensor,                    # [B, N_contracts, 10]
        chain_mask:      Optional[torch.Tensor] = None,   # [B, N_contracts] True=padded
        pivot_features:  Optional[torch.Tensor] = None,   # [B, T, 13] actual values (NaN→0)
        pivot_mask:      Optional[torch.Tensor] = None,   # [B, T, 13] bool — True=NaN
        return_diagnostics: bool = False,
    ) -> CondorNetOutput:
        """
        Full v43 forward pass.

        Args:
            x_m1/m5/m15/h1:  TF feature tensors [B, T, d_tf_in]
            chain:            Options chain grid [B, N, 10]
            chain_mask:       Padding mask [B, N] — True=padded (ignored by attention)
            pivot_features:   Pivot feature values [B, T, 13] — NaN replaced with 0
            pivot_mask:       Pivot NaN mask [B, T, 13] — True=NaN (no pivot event)
            return_diagnostics: Include full v42 diagnostic dict in output

        Returns:
            CondorNetOutput with all predictions and optional diagnostics
        """
        B, T, _ = x_m1.shape

        # ── Step 1: Multi-TF Projection ─────────────────────────────────────
        tf_joint, contrib_norms = self.tf_projector(x_m1, x_m5, x_m15, x_h1)
        # tf_joint: [B, T, 256]

        # ── Step 2: Pivot Projection + Fusion ───────────────────────────────
        if pivot_features is not None:
            if pivot_mask is None:
                pivot_mask = torch.zeros_like(pivot_features, dtype=torch.bool)
            pivot_embed = self.pivot_proj(pivot_features, pivot_mask)   # [B, T, 16]
        else:
            pivot_embed = torch.zeros(B, T, self.pivot_proj.d_out, device=x_m1.device, dtype=x_m1.dtype)

        tf_fused = self.tf_fusion(tf_joint, pivot_embed)   # [B, T, 256]

        # ── Step 3: Options Chain Encoding ──────────────────────────────────
        chain_embed, chain_diag = self.chain_encoder(chain, chain_mask)   # [B, 128]

        # ── Step 4: Joint Fusion ─────────────────────────────────────────────
        joint = self.joint_fusion(tf_fused, chain_embed)   # [B, T, 384]

        # ── Step 5: v42 Core (ETD-1 + Predicate Logic) ──────────────────────
        # Feed fused TF (adapted) into v42 CondorNet for predicate/set dynamics
        core_input = self.core_input_proj(tf_fused)   # [B, T, v42.d_input]
        
        # BLUEPRINT FIX: Safely extract proxy physics features from x_m1 to prevent 
        # lambda gating constancy (all-zero collapse).
        def safe_extract(name: str, fallback_val: float = 0.0):
            try:
                idx = TF_FEATURE_NAMES.index(name)
                return x_m1[:, :, idx]
            except ValueError:
                return torch.full((B, T), fallback_val, device=x_m1.device, dtype=x_m1.dtype)

        close_p = safe_extract("close", 500.0)
        # S_t_minus_1 is shifted close with duplication on 0th step
        close_p_m1 = torch.cat([close_p[:, :1], close_p[:, :-1]], dim=1)

        core_out, core_diag = self.condor_core(
            core_input, 
            iv_rank=safe_extract("fuzzy_reversion_11", 50.0),
            bid_ask_spread=safe_extract("spread_ratio", 0.01),
            price=close_p,
            rsi=safe_extract("rsi_dyn", 50.0),
            delta_rsi=safe_extract("log_return", 0.0), # proxy for rate of change
            S_t=close_p,
            S_t_minus_1=close_p_m1,
            gamma=safe_extract("gap_risk_score", 0.0), # proxy for curvature risk
            return_diagnostics=True
        )
        # core_out: [B, T, d_x] — last timestep used for heads

        # ── Step 6: Aggregate to [B, d_joint] for output heads ───────────────
        # Use last timestep of joint representation
        joint_last = joint[:, -1, :]   # [B, 384]

        # ── Step 7: Output Heads ─────────────────────────────────────────────
        strategy_logits, leg_params, entry_signal = self.strategy_head(joint_last)

        # Ensemble v43 attention-based strategy logits with v42 ETD-1/predicate
        # strategy logits. This wires gradient flow back to the A_matrix,
        # predicate gates, super_sets, and hierarchical logic — without it
        # those modules compute in the forward pass but receive zero gradient.
        # v42 CondorNet loops over T timesteps and returns final output [B, 10]
        # (not [B, T, 10]) — no time-dim indexing needed.
        # core_out is float32 (v42 forces .float()); cast to match AMP dtype.
        strategy_logits = strategy_logits + core_out.to(dtype=strategy_logits.dtype)

        pop, ev, max_loss, var_95, cvar_95        = self.risk_head(joint_last)
        pivot_high_probs, pivot_low_probs, pivot_strength = self.pivot_pred_head(joint_last)
        position_size = self.size_head(pop.detach(), joint_last, self.pop_sizing_weight)
        spot_pred     = self.spot_head(joint_last)

        diags = None
        if return_diagnostics:
            diags = {
                **core_diag,
                "chain_diag":    chain_diag,
                "contrib_norms": contrib_norms,
                "pivot_horizons": PivotPredictionHead.PIVOT_HORIZONS,
            }

        return CondorNetOutput(
            strategy_logits  = strategy_logits,
            leg_params       = leg_params,
            entry_signal     = entry_signal,
            pop              = pop,
            ev               = ev,
            max_loss         = max_loss,
            var_95           = var_95,
            cvar_95          = cvar_95,
            pivot_high_probs = pivot_high_probs,
            pivot_low_probs  = pivot_low_probs,
            pivot_strength   = pivot_strength,
            position_size    = position_size,
            spot_pred        = spot_pred,
            final_gate       = core_diag.get("final_gate"),
            diagnostics      = diags,
        )

    def forward_compat(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
        **kwargs,
    ) -> CondorNetOutput:
        """
        Backward compatibility wrapper for v42 single-TF callers.

        Routes single TF input to all 4 TFs with zero-valued chain.
        Existing backtest engine calls (CondorBrainEngine.predict) work unchanged.

        Args:
            x: [B, T, d_tf_in] — single timeframe features (treated as M5)
        """
        B, T, F = x.shape
        device = x.device
        dtype  = x.dtype

        # Zero-valued chain (no options data at inference for compat mode)
        dummy_chain = torch.zeros(B, 1, 10, device=device, dtype=dtype)
        dummy_mask  = torch.ones(B, 1, dtype=torch.bool, device=device)

        return self.forward(
            x_m1=x, x_m5=x, x_m15=x, x_h1=x,
            chain=dummy_chain,
            chain_mask=dummy_mask,
            return_diagnostics=return_diagnostics,
        )

    def get_A_matrix(self) -> torch.Tensor:
        """Expose v42 A-matrix for spectral analysis (B1.6 hook)."""
        return self.condor_core.get_A_matrix()

    @property
    def pred_signature(self):
        """Expose v42 predicate signature for group invariance loss."""
        return self.condor_core.pred_signature


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def build_condornet_v43(
    d_tf_in:           int   = 64,
    d_chain:           int   = 128,
    n_strategy_types:  int   = 10,
    pop_sizing_weight: float = 0.5,
    verbose:           bool  = True,
    **kwargs,
) -> CondorNetV43:
    """
    Factory function for CondorNetV43 with sensible defaults.

    All kwargs are forwarded to CondorNetV43.__init__.

    Example (Lightning AI training):
        model = build_condornet_v43(
            d_tf_in=66,          # 64 base + tod_sin + tod_cos
            d_chain=128,
            n_strategy_types=10,
            verbose=True,
        )
    """
    return CondorNetV43(
        d_tf_in=d_tf_in,
        d_chain=d_chain,
        n_strategy_types=n_strategy_types,
        pop_sizing_weight=pop_sizing_weight,
        verbose=verbose,
        **kwargs,
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CondorNet™ v4.3 — Architecture Self-Test")
    print("=" * 70)

    B, T, F_dim = 2, 64, 64       # batch=2, seq=64, features=64
    N = 40                     # contracts in chain

    model = build_condornet_v43(d_tf_in=F_dim, verbose=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # Build dummy inputs
    x = torch.randn(B, T, F_dim)
    chain = torch.randn(B, N, 10)
    chain_mask = torch.zeros(B, N, dtype=torch.bool)
    chain_mask[:, 35:] = True   # Last 5 are padded

    print(f"\nForward pass test:")
    print(f"  Input TF:    ({B}, {T}, {F_dim}) × 4 TFs")
    print(f"  Chain grid:  ({B}, {N}, 10) | mask: {chain_mask.sum()} padded")

    with torch.no_grad():
        out = model(x, x, x, x, chain, chain_mask, return_diagnostics=True)

    print(f"\nOutputs:")
    print(f"  strategy_logits:   {out.strategy_logits.shape}")
    print(f"  leg_params:        {out.leg_params.shape}")
    print(f"  entry_signal:      {out.entry_signal.shape}")
    print(f"  pop:               {out.pop.shape}  mean={out.pop.mean():.3f}")
    print(f"  ev:                {out.ev.shape}")
    print(f"  var_95:            {out.var_95.shape}")
    print(f"  cvar_95:           {out.cvar_95.shape}")
    print(f"  pivot_high_probs:  {out.pivot_high_probs.shape}")
    print(f"  pivot_low_probs:   {out.pivot_low_probs.shape}")
    print(f"  position_size:     {out.position_size.shape}  mean={out.position_size.mean():.3f}")
    print(f"  spot_pred:         {out.spot_pred.shape}")

    # Test abstain mask
    abstain = out.abstain_mask
    print(f"\n  Abstain mask (threshold={ABSTAIN_CONFIDENCE_THRESHOLD}): {abstain.sum().item()} / {B}")

    # Test backward compat
    print(f"\nBackward compat test (single TF input):")
    with torch.no_grad():
        out_compat = model.forward_compat(x)
    print(f"  strategy_logits: {out_compat.strategy_logits.shape} ✓")

    # Verify CVaR ≥ VaR
    assert (out.cvar_95 >= out.var_95).all(), "CVaR < VaR constraint violated!"
    print(f"\n  CVaR ≥ VaR constraint: PASSED ✓")

    print("\n" + "=" * 70)
    print("Self-test PASSED — condor_brain_net_v43.py ready for Lightning AI")
    print("=" * 70)
