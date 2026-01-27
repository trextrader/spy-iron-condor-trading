"""
intelligence/fuzzifier.py

Feature extraction + fuzzification into linguistic membership degrees.
Uses V2.2 Dynamic indicators (Adaptive ADX/RSI/BBands) from the production registry.
Bridges Neural CDE outputs with the MTFuzz™ Sizing Engine.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from core.config import StrategyConfig
from core.dto import MarketSnapshot
from intelligence.features.dynamic_features import compute_all_dynamic_features, compute_all_primitive_features_v22
import intelligence.fuzzy_engine as fe


@dataclass
class Fuzzifier:
    cfg: StrategyConfig

    def extract_features(self, snapshot: MarketSnapshot) -> pd.DataFrame:
        """
        Compute full V2.2 feature set (52 features) from MarketSnapshot.
        Optimized for real-time production flow.
        """
        bars: pd.DataFrame = snapshot.bars
        
        # Guard: Need minimum history for dynamic windowing (64 bars)
        if bars is None or len(bars) < 64:
             return pd.DataFrame()

        # Step 1: Compute V2.1 Dynamic Features (16 cols)
        df = compute_all_dynamic_features(bars.copy(), inplace=True)
        
        # Step 2: Inject Snapshot Metadata (VIX, IVR)
        df["vix"] = snapshot.vix if snapshot.vix is not None else 15.0
        
        # IV Rank handling: If not in snapshot, fallback to mid-rank
        if "ivr" not in df.columns:
            df["ivr"] = 50.0 

        # Step 3: Compute V2.2 Primitive Features (20 cols)
        # This completes the 52-feature matrix required by the CDE Backbone
        df = compute_all_primitive_features_v22(df, inplace=True)
        
        return df

    def fuzzify(self, features_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Convert V2.2 crisp features into linguistic membership degrees.
        Maps the technical 'puzzle pieces' to the Fuzzy Sizing Engine.
        """
        if features_df is None or features_df.empty:
            return {}

        last = features_df.iloc[-1]
        
        # Map features to membership functions defined in fuzzy_engine.py
        return {
            "mtf": {"neutral": last.get("mtf_consensus", 0.5)},
            "iv_rank": {"high": last.get("ivr", 50.0) / 100.0},
            "vix": {"stable": fe.calculate_regime_membership(last.get("vix", 15.0))},
            "rsi": {"neutral": fe.calculate_rsi_membership(last.get("rsi_dyn", 50.0))},
            "adx": {"ranging": fe.calculate_adx_membership(last.get("adx_adaptive", 25.0))},
            "bbands": {"neutral": fe.calculate_bbands_membership(
                last.get("bb_percentile", 50.0), 
                last.get("bw_expansion_rate", 0.0)
            )},
            "stoch": {"neutral": fe.calculate_stoch_membership(last.get("stoch_k_dyn", 50.0))},
            "volume": {"high": fe.calculate_volume_membership(last.get("volume_ratio", 1.0))},
            "sma_dist": {"neutral": fe.calculate_sma_distance_membership(last.get("ret_z", 0.0) * 0.01)},
            "psar": {"favorable": fe.calculate_psar_membership(last.get("psar_adaptive", 0.0))}
        }
