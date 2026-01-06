# data_factory/alpaca_data_client.py
"""
Alpaca-based market data client with full options chain support.
Provides spot prices, VIX, IV rank, and option chains with Greeks.
"""
import datetime as dt
from typing import List, Optional
from dataclasses import dataclass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, OptionChainRequest
from alpaca.trading.client import TradingClient


@dataclass
class AlpacaOptionQuote:
    """Option quote with full Greeks from Alpaca"""
    symbol: str
    expiration: dt.date
    strike: float
    is_call: bool
    bid: float
    ask: float
    mid: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    
    @property
    def mark(self) -> float:
        """Alias for mid price (backtest engine compatibility)"""
        return self.mid


class AlpacaDataClient:
    """
    Market data client using Alpaca APIs exclusively.
    Replaces PolygonClient for live/paper trading.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)
        self.option_client = OptionHistoricalDataClient(api_key, api_secret)
        self.trading_client = TradingClient(api_key, api_secret, paper=True)
        self._api_key = api_key
        self._api_secret = api_secret
    
    def get_spot(self, symbol: str) -> float:
        """Get current spot price for a symbol"""
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.stock_client.get_stock_latest_quote(req)
        quote = quotes.get(symbol)
        if quote:
            return (quote.bid_price + quote.ask_price) / 2
        return 0.0
    
    def get_vix(self) -> float:
        """Get current VIX level"""
        # Note: VIX is not directly available via Alpaca stock data
        # Using VIXY as a proxy or we can fetch from the account data
        try:
            return self.get_spot("VIX")
        except Exception:
            # VIX is an index, not tradeable via Alpaca directly
            # Return a reasonable default - in production, use a VIX ETF or external source
            return 18.0  # Placeholder
    
    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Calculate IV Rank based on current IV vs historical range.
        For now, derive from option chain average IV.
        """
        try:
            # Get a sample of options to calculate current IV
            chain = self._get_raw_option_chain(symbol)
            ivs = [c.implied_volatility for c in chain.values() 
                   if c.implied_volatility and c.implied_volatility > 0]
            if ivs:
                current_iv = sum(ivs) / len(ivs)
                # Estimate IV rank (would need historical data for accurate calc)
                # Using a simplified model: assume IV ranges from 10% to 40%
                iv_rank = max(0, min(100, (current_iv - 0.10) / 0.30 * 100))
                return iv_rank
        except Exception as e:
            print(f"[AlpacaDataClient] IV rank calc error: {e}")
        return 30.0  # Default
    
    def get_expirations(self, symbol: str) -> List[dt.date]:
        """Get available option expiration dates"""
        chain = self._get_raw_option_chain(symbol)
        expirations = set()
        for sym in chain.keys():
            # Parse expiration from OCC symbol: SPY260120C00480000
            # Format: SYMBOL + YYMMDD + C/P + STRIKE (8 digits)
            try:
                # Find where the date starts (after underlying symbol)
                # SPY = 3 chars, then YYMMDD
                base_len = len(symbol)
                date_str = sym[base_len:base_len + 6]
                exp_date = dt.datetime.strptime(date_str, "%y%m%d").date()
                expirations.add(exp_date)
            except Exception:
                continue
        return sorted(list(expirations))
    
    def get_option_chain(self, symbol: str, expiration: dt.date) -> List[AlpacaOptionQuote]:
        """
        Get option chain for a specific expiration date.
        Returns list of AlpacaOptionQuote objects with full Greeks.
        """
        raw_chain = self._get_raw_option_chain(symbol)
        exp_str = expiration.strftime("%y%m%d")
        
        quotes = []
        for occ_symbol, snapshot in raw_chain.items():
            # Filter by expiration
            try:
                base_len = len(symbol)
                date_str = occ_symbol[base_len:base_len + 6]
                if date_str != exp_str:
                    continue
                
                # Parse strike and type
                opt_type = occ_symbol[base_len + 6]
                strike_str = occ_symbol[base_len + 7:]
                strike = float(strike_str) / 1000  # OCC format has 3 decimal places implied
                is_call = (opt_type == 'C')
                
                # Get quote data
                quote = snapshot.latest_quote
                if not quote:
                    continue
                    
                bid = float(quote.bid_price) if quote.bid_price else 0.0
                ask = float(quote.ask_price) if quote.ask_price else 0.0
                mid = (bid + ask) / 2 if (bid > 0 or ask > 0) else 0.0
                
                # Get Greeks
                greeks = snapshot.greeks
                delta = float(greeks.delta) if greeks and greeks.delta else 0.0
                gamma = float(greeks.gamma) if greeks and greeks.gamma else 0.0
                theta = float(greeks.theta) if greeks and greeks.theta else 0.0
                vega = float(greeks.vega) if greeks and greeks.vega else 0.0
                iv = float(snapshot.implied_volatility) if snapshot.implied_volatility else 0.0
                
                # Skip options with no Greeks or quotes
                if mid == 0.0 or delta == 0.0:
                    continue
                
                quotes.append(AlpacaOptionQuote(
                    symbol=occ_symbol,
                    expiration=expiration,
                    strike=strike,
                    is_call=is_call,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    delta=abs(delta) if not is_call else delta,  # Puts have negative delta
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    iv=iv
                ))
            except Exception as e:
                continue
        
        return quotes
    
    def _get_raw_option_chain(self, symbol: str):
        """Get raw option chain from Alpaca API"""
        req = OptionChainRequest(underlying_symbol=symbol)
        return self.option_client.get_option_chain(req)
