#!/usr/bin/env python3
"""
SyntheticOptionsEngine.py

Option A (On-demand, 1-minute):
- Gaussian trades on 1m bars.
- We price options ON-DEMAND per minute for requested strikes/expirations.
- We can also DUMP a micro-chain to CSV for a chosen start/stop window (for debugging/validation),
  OR automatically dump the LAST 6 MONTHS from the latest underlying timestamp.

Outputs per-row fields:
timestamp,date,symbol,option_symbol,strike,expiration,contract_type,
bid,ask,last_price,bid_size,ask_size,volume,open_interest,
delta,gamma,theta,vega,implied_volatility

CLI:
  1) Build full-range 1m underlying cache:
     py -3.12 SyntheticOptionsEngine.py --build-spot --symbol SPY

  2) Dump full-fields options marks to CSV over a custom time window (micro-chain):
     py -3.12 SyntheticOptionsEngine.py --dump-marks --symbol SPY --start 2024-01-02T14:30:00Z --end 2024-01-02T15:30:00Z

  3) Dump full-fields options marks for LAST 6 MONTHS from the latest underlying timestamp:
     py -3.12 SyntheticOptionsEngine.py --dump-marks-6m --symbol SPY

Optional tuning for dump:
  --every 5     sample every N minutes (default 1)
  --band 0.05   +/- strikes around spot (default 0.05)
  --step 5      strike step (default 5)
  --weeks 8     number of weekly expirations (default 8)
  --out path.csv
  --max-rows 200000

NOTE ON SCALE:
Dumping 6 months at 1-minute resolution with many strikes/expirations can produce VERY large files.
Use --every (e.g. 5 or 15) and/or reduce --weeks/--band/--step.
"""

import math
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# Normal CDF/PDF (no scipy)
# ======================================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# ======================================================================================
# Black–Scholes: price + greeks (theta per-day)
# ======================================================================================

def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _bs_d2(d1: float, T: float, sigma: float) -> float:
    return d1 - sigma * math.sqrt(T)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    if T <= 0:
        return max(0.0, S - K) if right == "C" else max(0.0, K - S)

    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(d1, T, sigma)
    if right == "C":
        return max(0.0, S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2))
    else:
        return max(0.0, K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return _norm_cdf(d1) if right == "C" else (_norm_cdf(d1) - 1.0)


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T) / 100.0


def _bs_theta_per_day(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(d1, T, sigma)

    first = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if right == "C":
        second = -r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        second = r * K * math.exp(-r * T) * _norm_cdf(-d2)

    theta_annual = first + second
    return theta_annual / 365.0


# ======================================================================================
# OCC-like option symbol similar to your sample: SPY251031C00636.00
# ======================================================================================

def _format_option_symbol(symbol: str, expiration: dt.date, right: str, strike: float) -> str:
    sym = symbol.upper()
    yymmdd = expiration.strftime("%y%m%d")
    cp = "C" if right.upper() == "C" else "P"
    strike_str = f"{strike:0>7.2f}"  # "0636.00"
    if len(strike_str) < 8:
        strike_str = strike_str.rjust(8, "0")  # "00636.00"
    return f"{sym}{yymmdd}{cp}{strike_str}"


# ======================================================================================
# Params
# ======================================================================================

@dataclass
class SyntheticParams:
    risk_free_rate: float = 0.01

    # synthetic IV surface
    base_iv: float = 0.20
    iv_skew: float = 0.25
    iv_term: float = 0.10
    min_iv: float = 0.05
    max_iv: float = 1.00

    # bid/ask model
    min_spread: float = 0.01
    spread_pct: float = 0.05

    # microstructure proxies (synthetic, deterministic)
    default_bid_size: int = 100
    default_ask_size: int = 100
    oi_base: int = 500
    oi_scale: int = 20000
    vol_base: int = 0
    vol_scale: int = 500

    # output rounding
    price_decimals: int = 2
    iv_decimals: int = 4
    greek_decimals: int = 6


# ======================================================================================
# Engine
# ======================================================================================

class SyntheticOptionsEngine:
    def __init__(self, params: Optional[SyntheticParams] = None, seed: int = 1337):
        self.params = params or SyntheticParams()
        self.seed = int(seed)

        self._marks_cache: Dict[dt.date, Dict[Tuple, Dict]] = {}
        self._spot_cache: Optional[pd.DataFrame] = None

        # project root resolution (script is in data_factory/)
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.reports_dir = self.PROJECT_ROOT / "reports" / "spy"
        self.data_dir = self.PROJECT_ROOT / "data" / "synthetic_options"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Underlying cache --------------------

    def build_underlying_cache(
        self,
        symbol: str = "SPY",
        tf: str = "1",
        start: Optional[str] = None,
        end: Optional[str] = None,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        sym = symbol.lower()
        tf = str(tf)

        candidates = [
            self.reports_dir / f"{sym}_{tf}.csv",
            self.reports_dir / f"{symbol.upper()}_{tf}.csv",
            self.PROJECT_ROOT / "reports" / symbol.upper() / f"{symbol.upper()}_{tf}.csv",
            self.PROJECT_ROOT / "reports" / sym / f"{sym}_{tf}.csv",
        ]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            raise FileNotFoundError(
                f"Could not find underlying CSV for {symbol} tf={tf} in reports/spy or legacy folders."
            )

        df = pd.read_csv(src, parse_dates=["timestamp"])
        if df.empty:
            raise ValueError(f"Underlying CSV is empty: {src}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        if start:
            df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
        if end:
            df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]

        if "close" not in df.columns:
            raise ValueError("Underlying CSV must contain 'close' column.")

        spot_cols = ["timestamp", "close"]
        if "volume" in df.columns:
            spot_cols.append("volume")

        spot = df[spot_cols].copy()
        spot["close"] = pd.to_numeric(spot["close"], errors="coerce").round(self.params.price_decimals)
        if "volume" in spot.columns:
            spot["volume"] = (
                pd.to_numeric(spot["volume"], errors="coerce")
                .fillna(0)
                .round(0)
                .clip(lower=0)
                .astype("int64")
            )

        self._spot_cache = spot

        if save_csv:
            out_path = self.data_dir / f"{sym}_spot_1m.csv"
            spot.to_csv(out_path, index=False)

        return spot

    def load_underlying_cache(self, symbol: str = "SPY") -> pd.DataFrame:
        sym = symbol.lower()
        p = self.data_dir / f"{sym}_spot_1m.csv"
        if not p.exists():
            return self.build_underlying_cache(symbol=symbol, tf="1", save_csv=True)

        df = pd.read_csv(p, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        self._spot_cache = df
        return df

    def _last_six_month_window(self, symbol: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Returns (start_ts, end_ts) covering the last 6 calendar months
        based on the underlying 1-minute dataset timestamps.
        """
        spot = self._spot_cache if self._spot_cache is not None else self.load_underlying_cache(symbol)

        end_ts = spot["timestamp"].max()
        if pd.isna(end_ts):
            raise ValueError("Underlying cache has no timestamps.")

        start_ts = end_ts - pd.DateOffset(months=6)
        return start_ts, end_ts

    # -------------------- On-demand full schema --------------------

    def get_marks_full(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        underlying_price: float,
        expirations: Iterable[pd.Timestamp],
        strikes_by_right: Dict[str, Iterable[float]],
    ) -> pd.DataFrame:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        S = float(underlying_price)
        trade_date = ts.date()

        if trade_date not in self._marks_cache:
            self._marks_cache[trade_date] = {}

        rows = []
        for exp in expirations:
            e = pd.Timestamp(exp)
            if e.tzinfo is None:
                e = e.tz_localize("UTC")
            else:
                e = e.tz_convert("UTC")

            exp_date = e.date()
            dte_days = max((exp_date - trade_date).days, 0)
            T = max(dte_days / 365.0, 1e-6)

            for right, strikes in strikes_by_right.items():
                R = right.upper()
                if R not in ("C", "P"):
                    continue

                for K in strikes:
                    Kf = float(K)
                    key = (ts.value, e.value, R, round(Kf, 2), symbol.upper())
                    cached = self._marks_cache[trade_date].get(key)
                    if cached:
                        rows.append(cached)
                        continue

                    iv = self._compute_iv(S, Kf, T, R)
                    mid = _bs_price(S, Kf, T, self.params.risk_free_rate, iv, R)
                    delta = _bs_delta(S, Kf, T, self.params.risk_free_rate, iv, R)
                    gamma = _bs_gamma(S, Kf, T, self.params.risk_free_rate, iv)
                    vega = _bs_vega(S, Kf, T, self.params.risk_free_rate, iv)
                    theta = _bs_theta_per_day(S, Kf, T, self.params.risk_free_rate, iv, R)

                    spread = max(self.params.min_spread, mid * self.params.spread_pct)
                    bid = max(0.0, mid - spread / 2.0)
                    ask = mid + spread / 2.0

                    last_price = self._jitter_last(mid, trade_date, exp_date, R, Kf)
                    bid_size, ask_size = self._synthetic_sizes(trade_date, exp_date, R, Kf)
                    vol = self._synthetic_volume(trade_date, exp_date, R, Kf, mid, dte_days)
                    oi = self._synthetic_open_interest(exp_date, R, Kf, dte_days)

                    option_symbol = _format_option_symbol(symbol, exp_date, R, Kf)

                    row = {
                        "timestamp": ts.to_pydatetime(),
                        "date": trade_date.isoformat(),
                        "symbol": symbol.upper(),
                        "option_symbol": option_symbol,
                        "strike": round(Kf, 2),
                        "expiration": dt.datetime.combine(exp_date, dt.time(16, 0)),
                        "contract_type": "call" if R == "C" else "put",
                        "bid": round(bid, self.params.price_decimals),
                        "ask": round(ask, self.params.price_decimals),
                        "last_price": round(last_price, self.params.price_decimals),
                        "bid_size": int(bid_size),
                        "ask_size": int(ask_size),
                        "volume": int(vol),
                        "open_interest": int(oi),
                        "delta": round(delta, 4),
                        "gamma": round(gamma, self.params.greek_decimals),
                        "theta": round(theta, self.params.greek_decimals),
                        "vega": round(vega, self.params.greek_decimals),
                        "implied_volatility": round(iv, self.params.iv_decimals),
                    }

                    self._marks_cache[trade_date][key] = row
                    rows.append(row)

        return pd.DataFrame(rows)

    # -------------------- Micro-chain dump to CSV (integrated) --------------------

    def dump_marks_to_csv(
        self,
        symbol: str,
        start: str,
        end: str,
        every_n_minutes: int = 1,
        moneyness_band: float = 0.05,
        strike_step: float = 5.0,
        expiries_weeks: int = 8,
        out_path: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> str:
        """
        Dump full-field options rows over a time window by sampling the underlying 1m timeline.
        This is for validation/debugging and for creating a manageable “options file” if desired.

        WARNING: This can still get large. Use --every and/or reduce --weeks/--band.
        """
        spot = self._spot_cache if self._spot_cache is not None else self.load_underlying_cache(symbol)

        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)

        df = spot[(spot["timestamp"] >= start_ts) & (spot["timestamp"] <= end_ts)].copy()
        df = df.sort_values("timestamp")
        if df.empty:
            raise ValueError("No underlying rows found in the requested start/end window.")

        if every_n_minutes < 1:
            every_n_minutes = 1

        df = df.iloc[::every_n_minutes, :].reset_index(drop=True)

        sym = symbol.lower()
        out = Path(out_path) if out_path else (self.data_dir / f"{sym}_options_marks.csv")
        if out.exists():
            out.unlink()

        wrote = 0
        header_written = False

        for _, row in df.iterrows():
            ts = row["timestamp"]
            S = float(row["close"])

            expirations = self._weekly_expirations(ts.date(), expiries_weeks)

            low = S * (1.0 - moneyness_band)
            high = S * (1.0 + moneyness_band)
            k0 = math.floor(low / strike_step) * strike_step
            k1 = math.ceil(high / strike_step) * strike_step
            strikes = np.arange(k0, k1 + strike_step, strike_step).tolist()

            marks = self.get_marks_full(
                timestamp=ts,
                symbol=symbol,
                underlying_price=S,
                expirations=expirations,
                strikes_by_right={"P": strikes, "C": strikes},
            )

            marks.to_csv(out, mode="a", header=(not header_written), index=False)
            header_written = True

            wrote += len(marks)
            if max_rows and wrote >= max_rows:
                break

        return str(out)

    # -------------------- Internals --------------------

    def _compute_iv(self, S: float, K: float, T: float, right: str) -> float:
        m = (K / S) - 1.0
        skew = self.params.iv_skew * abs(m)
        term = self.params.iv_term * min(T * 2.0, 1.0)
        iv = self.params.base_iv + skew + term
        return float(np.clip(iv, self.params.min_iv, self.params.max_iv))

    def _stable_rng(self, *parts) -> np.random.Generator:
        h = 0
        for p in parts:
            h = (h * 1315423911) ^ hash(p)
        seed = (self.seed + (h & 0xFFFFFFFF)) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def _synthetic_sizes(self, trade_date: dt.date, exp_date: dt.date, right: str, strike: float) -> Tuple[int, int]:
        rng = self._stable_rng("sz", trade_date, exp_date, right, round(strike, 2))
        bid_sz = int(max(1, rng.normal(self.params.default_bid_size, self.params.default_bid_size * 0.15)))
        ask_sz = int(max(1, rng.normal(self.params.default_ask_size, self.params.default_ask_size * 0.15)))
        bid_sz = int(round(bid_sz / 10) * 10)
        ask_sz = int(round(ask_sz / 10) * 10)
        return bid_sz, ask_sz

    def _synthetic_open_interest(self, exp_date: dt.date, right: str, strike: float, dte_days: int) -> int:
        rng = self._stable_rng("oi", exp_date, right, round(strike, 2))
        decay = max(0.15, 1.0 - (dte_days / 60.0))
        base = self.params.oi_base + int(self.params.oi_scale * decay)
        oi = int(max(0, rng.normal(base, base * 0.25)))
        return oi

    def _synthetic_volume(self, trade_date: dt.date, exp_date: dt.date, right: str, strike: float, mid: float, dte_days: int) -> int:
        rng = self._stable_rng("vol", trade_date, exp_date, right, round(strike, 2))
        near = max(0.2, 1.0 - (dte_days / 45.0))
        price_factor = 1.0 if 0.2 <= mid <= 10 else 0.4
        lam = max(0.0, self.params.vol_base + self.params.vol_scale * near * price_factor)
        return int(rng.poisson(lam))

    def _jitter_last(self, mid: float, trade_date: dt.date, exp_date: dt.date, right: str, strike: float) -> float:
        rng = self._stable_rng("last", trade_date, exp_date, right, round(strike, 2))
        jitter = rng.normal(0.0, max(0.01, mid * 0.01))
        return max(0.0, mid + jitter)

    def _weekly_expirations(self, current_date: dt.date, weeks: int) -> List[pd.Timestamp]:
        expirations = []
        days_to_friday = (4 - current_date.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        next_friday = current_date + dt.timedelta(days=days_to_friday)
        for w in range(weeks):
            d = next_friday + dt.timedelta(weeks=w)
            expirations.append(pd.Timestamp(dt.datetime.combine(d, dt.time(16, 0)), tz="UTC"))
        return expirations


# ======================================================================================
# CLI
# ======================================================================================

def _usage():
    print("""
SyntheticOptionsEngine.py

Build full-range 1m underlying cache:
  py -3.12 SyntheticOptionsEngine.py --build-spot --symbol SPY

Dump full-field options marks to CSV (micro-chain) for a custom window:
  py -3.12 SyntheticOptionsEngine.py --dump-marks --symbol SPY --start 2024-01-02T14:30:00Z --end 2024-01-02T15:30:00Z

Dump full-field options marks to CSV (micro-chain) for LAST 6 MONTHS from latest underlying timestamp:
  py -3.12 SyntheticOptionsEngine.py --dump-marks-6m --symbol SPY

Options (dump):
  --every 5     sample every 5 minutes
  --band 0.05   +/- moneyness band
  --step 5      strike step
  --weeks 8     weekly expirations count
  --out path.csv
  --max-rows 200000
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")

    parser.add_argument("--build-spot", action="store_true")
    parser.add_argument("--dump-marks", action="store_true")
    parser.add_argument("--dump-marks-6m", action="store_true",
                        help="Dump full-field options marks for the last 6 months from the latest underlying timestamp")

    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)

    parser.add_argument("--every", type=int, default=1, help="sample every N minutes (dump)")
    parser.add_argument("--band", type=float, default=0.05, help="+/- moneyness band (dump)")
    parser.add_argument("--step", type=float, default=5.0, help="strike step (dump)")
    parser.add_argument("--weeks", type=int, default=8, help="num weekly expirations (dump)")
    parser.add_argument("--out", default=None, help="output csv path (dump)")
    parser.add_argument("--max-rows", type=int, default=None, help="stop after N rows (dump)")

    args = parser.parse_args()
    eng = SyntheticOptionsEngine()

    if args.build_spot:
        spot = eng.build_underlying_cache(symbol=args.symbol, tf="1", start=args.start, end=args.end, save_csv=True)
        out = eng.data_dir / f"{args.symbol.lower()}_spot_1m.csv"
        print(f"[OK] Built spot cache rows={len(spot)} range={spot['timestamp'].min()} -> {spot['timestamp'].max()}")
        print(f"Saved: {out}")

    elif args.dump_marks or args.dump_marks_6m:
        eng.load_underlying_cache(args.symbol)

        if args.dump_marks_6m:
            start_ts, end_ts = eng._last_six_month_window(args.symbol)
            start = start_ts.isoformat()
            end = end_ts.isoformat()
            print(f"[INFO] Dumping last 6 months: {start} -> {end}")
        else:
            if not args.start or not args.end:
                raise SystemExit("dump-marks requires --start and --end (UTC ISO timestamps).")
            start = args.start
            end = args.end

        out_path = eng.dump_marks_to_csv(
            symbol=args.symbol,
            start=start,
            end=end,
            every_n_minutes=args.every,
            moneyness_band=args.band,
            strike_step=args.step,
            expiries_weeks=args.weeks,
            out_path=args.out,
            max_rows=args.max_rows,
        )
        print(f"[OK] Wrote full-field synthetic options marks to: {out_path}")

    else:
        _usage()
