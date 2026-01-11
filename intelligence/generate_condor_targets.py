"""
CondorBrain Target Generator

Runs historical Iron Condor simulations on 1-minute data to generate
backtest-enhanced training targets:
- Optimal strike offsets (what would have been profitable)
- Realized ROI
- Was profitable (binary)
- Max loss experienced
- Regime labels

This creates the "ground truth" for CondorBrain multi-output training.
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
from dataclasses import dataclass

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ICSimConfig:
    """Iron Condor simulation parameters."""
    # Strike offset ranges to test (% from spot)
    call_offsets: list = None  # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    put_offsets: list = None   # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # Wing width options ($)
    wing_widths: list = None   # [3, 5, 7, 10]
    
    # DTE options
    dte_options: list = None   # [2, 7, 14, 30, 45]
    
    # Trading parameters
    credit_per_spread: float = 0.50  # Default credit assumption
    max_loss_multiplier: float = 2.0  # Max loss = wing_width - credit
    
    # Time parameters (in minutes)
    bars_per_day: int = 390
    
    def __post_init__(self):
        if self.call_offsets is None:
            self.call_offsets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        if self.put_offsets is None:
            self.put_offsets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        if self.wing_widths is None:
            self.wing_widths = [3, 5, 7, 10]
        if self.dte_options is None:
            self.dte_options = [2, 7, 14, 30, 45]

# ============================================================================
# IRON CONDOR SIMULATION ENGINE
# ============================================================================

class ICSimulator:
    """Simulates Iron Condor outcomes on historical 1-minute data."""
    
    def __init__(self, config: ICSimConfig = None):
        self.config = config or ICSimConfig()
        
    def simulate_ic(
        self,
        spot_at_entry: float,
        future_prices: np.ndarray,  # Array of future closes for DTE period
        call_offset_pct: float,
        put_offset_pct: float,
        wing_width: float,
        credit: float = 0.50
    ) -> dict:
        """
        Simulate a single Iron Condor trade.
        
        Args:
            spot_at_entry: Spot price when trade entered
            future_prices: Array of future spot prices over DTE
            call_offset_pct: % above spot for short call
            put_offset_pct: % below spot for short put
            wing_width: $ width of wings
            credit: Credit received per contract
            
        Returns:
            dict with simulation results
        """
        # Calculate strikes
        short_call = spot_at_entry * (1 + call_offset_pct / 100)
        short_put = spot_at_entry * (1 - put_offset_pct / 100)
        long_call = short_call + wing_width
        long_put = short_put - wing_width
        
        # Max profit = credit received
        max_profit = credit * 100  # Per contract
        
        # Max loss = wing_width - credit
        max_loss = (wing_width - credit) * 100
        
        # Check if price ever breached strikes during DTE
        if len(future_prices) == 0:
            return {
                'was_profitable': 0.5,
                'realized_roi': 0.0,
                'max_loss_pct': 0.0,
                'breached_call': False,
                'breached_put': False,
                'final_pnl': 0.0
            }
        
        max_price = np.max(future_prices)
        min_price = np.min(future_prices)
        final_price = future_prices[-1]
        
        breached_call = max_price >= short_call
        breached_put = min_price <= short_put
        
        # Calculate P&L at expiry
        if breached_call:
            # Call side loss
            call_intrinsic = max(0, final_price - short_call)
            call_pnl = -min(call_intrinsic * 100, wing_width * 100) + credit * 100
        else:
            call_pnl = credit * 100 / 2  # Half credit from call side
            
        if breached_put:
            # Put side loss
            put_intrinsic = max(0, short_put - final_price)
            put_pnl = -min(put_intrinsic * 100, wing_width * 100) + credit * 100
        else:
            put_pnl = credit * 100 / 2  # Half credit from put side
        
        # Total P&L (simplified)
        if not breached_call and not breached_put:
            # Full profit - both sides expired worthless
            final_pnl = max_profit
        elif breached_call and breached_put:
            # Disaster - both sides breached (unlikely but handle)
            final_pnl = -max_loss
        elif breached_call:
            # Call side hit
            loss = min((max_price - short_call) * 100, wing_width * 100)
            final_pnl = max_profit - loss
        else:
            # Put side hit
            loss = min((short_put - min_price) * 100, wing_width * 100)
            final_pnl = max_profit - loss
        
        # Clamp to max loss
        final_pnl = max(-max_loss, min(max_profit, final_pnl))
        
        # ROI as percentage of max risk
        roi = final_pnl / max_loss if max_loss > 0 else 0.0
        
        return {
            'was_profitable': 1.0 if final_pnl > 0 else 0.0,
            'realized_roi': roi,
            'max_loss_pct': abs(min(0, final_pnl)) / max_loss if max_loss > 0 else 0.0,
            'breached_call': breached_call,
            'breached_put': breached_put,
            'final_pnl': final_pnl
        }
    
    def find_optimal_strikes(
        self,
        spot_at_entry: float,
        future_prices: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Find optimal strike offsets that would have been most profitable.
        
        Returns:
            (best_call_offset, best_put_offset, best_wing_width, best_roi)
        """
        best_roi = -float('inf')
        best_params = (2.0, 2.0, 5.0, 0.0)  # Defaults
        
        for call_off in self.config.call_offsets:
            for put_off in self.config.put_offsets:
                for wing in self.config.wing_widths:
                    result = self.simulate_ic(
                        spot_at_entry, future_prices,
                        call_off, put_off, wing
                    )
                    
                    if result['realized_roi'] > best_roi:
                        best_roi = result['realized_roi']
                        best_params = (call_off, put_off, wing, result['realized_roi'])
        
        return best_params

# ============================================================================
# TARGET GENERATION
# ============================================================================

def generate_condor_targets(
    df: pd.DataFrame,
    config: ICSimConfig = None,
    output_path: str = None,
    sample_rate: int = 100  # Only simulate every Nth row to speed up
) -> pd.DataFrame:
    """
    Generate backtest-enhanced targets for CondorBrain training.
    
    Args:
        df: Input DataFrame with 1m institutional data
        config: IC simulation config
        output_path: Optional path to save enhanced DataFrame
        sample_rate: Simulate every Nth row (1 = all rows, 100 = 1% of rows)
        
    Returns:
        DataFrame with added target columns
    """
    config = config or ICSimConfig()
    simulator = ICSimulator(config)
    
    print(f"[TargetGen] Starting Iron Condor simulations...")
    print(f"[TargetGen] Input rows: {len(df):,}, Sample rate: 1/{sample_rate}")
    
    # Ensure datetime index
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], utc=True)
        df = df.sort_values('dt')
    
    # Get spot close series (deduplicated by dt for spot timeline)
    if 'close' in df.columns:
        spot_df = df.drop_duplicates('dt')[['dt', 'close', 'high', 'low']].set_index('dt')
    else:
        print("[Error] No 'close' column found")
        return df
    
    # Initialize target columns
    n = len(df)
    target_call_offset = np.full(n, 2.0)
    target_put_offset = np.full(n, 2.0)
    target_wing_width = np.full(n, 5.0)
    target_dte = np.full(n, 14.0)
    was_profitable = np.full(n, 0.5)
    realized_roi = np.zeros(n)
    realized_max_loss = np.full(n, 0.2)
    confidence_target = np.full(n, 0.5)
    
    # Get unique timestamps for simulation
    unique_dts = df['dt'].unique()
    simulated = 0
    
    for i, dt in enumerate(unique_dts):
        if i % sample_rate != 0:
            continue
            
        # Get spot at this time
        if dt not in spot_df.index:
            continue
            
        spot = spot_df.loc[dt, 'close']
        if isinstance(spot, pd.Series):
            spot = spot.iloc[0]
        
        # For each DTE option, simulate
        for dte_days in config.dte_options:
            # Get future prices for this DTE
            dte_bars = dte_days * config.bars_per_day
            future_mask = (spot_df.index > dt) & (spot_df.index <= dt + timedelta(days=dte_days))
            future_closes = spot_df.loc[future_mask, 'close'].values
            
            if len(future_closes) < dte_bars * 0.5:  # Need at least half the expected data
                continue
            
            # Find optimal strikes
            best_call, best_put, best_wing, best_roi = simulator.find_optimal_strikes(
                spot, future_closes
            )
            
            # Simulate with optimal strikes
            result = simulator.simulate_ic(
                spot, future_closes,
                best_call, best_put, best_wing
            )
            
            # Update rows matching this dt
            mask = df['dt'] == dt
            idx = df.index[mask]
            
            target_call_offset[idx] = best_call
            target_put_offset[idx] = best_put
            target_wing_width[idx] = best_wing
            target_dte[idx] = dte_days
            was_profitable[idx] = result['was_profitable']
            realized_roi[idx] = result['realized_roi']
            realized_max_loss[idx] = result['max_loss_pct']
            
            # Confidence = how clear-cut the outcome was
            confidence_target[idx] = abs(result['realized_roi'])
            
            simulated += 1
            break  # Use first valid DTE
        
        if simulated % 1000 == 0:
            print(f"[TargetGen] Simulated {simulated:,} rows...")
    
    print(f"[TargetGen] Completed {simulated:,} simulations")
    
    # Add columns to DataFrame
    df['target_call_offset'] = target_call_offset
    df['target_put_offset'] = target_put_offset
    df['target_wing_width'] = target_wing_width
    df['target_dte'] = target_dte
    df['was_profitable'] = was_profitable
    df['realized_roi'] = realized_roi
    df['realized_max_loss'] = realized_max_loss
    df['confidence_target'] = confidence_target
    
    # Regime labeling based on IVR
    if 'ivr' in df.columns:
        df['regime_label'] = pd.cut(
            df['ivr'], 
            bins=[-0.1, 30, 70, 101], 
            labels=[0, 1, 2]
        ).astype(int)
    else:
        df['regime_label'] = 1
    
    # Save if path provided
    if output_path:
        print(f"[TargetGen] Saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[TargetGen] Saved enhanced dataset with {len(df):,} rows")
    
    return df

# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CondorBrain Training Targets")
    parser.add_argument("--input", type=str, required=True, help="Path to institutional 1m CSV")
    parser.add_argument("--output", type=str, help="Output path for enhanced CSV")
    parser.add_argument("--sample-rate", type=int, default=100, help="Simulate 1 in N rows")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit input rows for testing")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CONDORBRAIN TARGET GENERATOR")
    print("="*60)
    
    # Load data
    print(f"Loading {args.input}...")
    if args.max_rows > 0:
        df = pd.read_csv(args.input, nrows=args.max_rows)
    else:
        df = pd.read_csv(args.input)
    
    print(f"Loaded {len(df):,} rows")
    
    # Generate targets
    output = args.output or args.input.replace('.csv', '_targets.csv')
    df = generate_condor_targets(df, output_path=output, sample_rate=args.sample_rate)
    
    print("="*60)
    print("TARGET GENERATION COMPLETE")
    print(f"Output: {output}")
    print("="*60)

if __name__ == "__main__":
    main()
