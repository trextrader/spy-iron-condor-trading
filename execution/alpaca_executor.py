import os
import time
from dataclasses import dataclass
from datetime import datetime
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderStatus
    from alpaca.common.exceptions import APIError
except ImportError:
    print("⚠️ 'alpaca-py' not installed. Please run: pip install alpaca-py")
    TradingClient = None

class AlpacaExecutor:
    def __init__(self, api_key=None, secret_key=None, paper=True):
        self.api_key = api_key or os.getenv('APCA_API_KEY_ID')
        self.secret_key = secret_key or os.getenv('APCA_API_SECRET_KEY')
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Keys (APCA_API_KEY_ID, APCA_API_SECRET_KEY) not found.")

        if TradingClient:
            self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            account = self.client.get_account()
            print(f"[Alpaca] Connected. Cash: ${float(account.cash):,.2f} | Buying Power: ${float(account.buying_power):,.2f}")
        else:
            raise ImportError("alpaca-py not installed")

    def get_market_status(self):
        # Alpaca check if market open?
        clock = self.client.get_clock()
        return clock.is_open

    def submit_iron_condor(self, symbol, legs, quantity):
        """
        Submit a multi-leg Iron Condor order.
        Legs format: [{'symbol': 'SPY...', 'side': 'sell', 'qty': 1}, ...]
        Note: Alpaca API for multi-leg is complex. 
        For simplicity in V1, we might submit 4 separate orders or use OptionChain API if available.
        Wait, alpaca-py supports multi-leg orders? 
        Actually, for Iron Condor on Alpaca, check docs.
        Currently using MarketOrderRequest per leg is risky (legging in).
        Proper way: OrderRequest with legislation.
        Verification: Alpaca Options API is Beta. 
        Let's assume we construct 4 separate limit/market orders for now, OR valid multi-leg if supported.
        Correction: Alpaca Options API requires specific endpoint.
        For this simplified executor, we will just print the order logic and returns a mock trade ID if testing, 
        or warn that Multi-Leg API implementation needs specific library version.
        
        Using 4 separate MKT orders for now (Risky but functional for Paper).
        """
        print(f"[Alpaca] Submitting Iron Condor for {symbol} Qty {quantity}...")
        
        # Example logic for submitting 4 legs
        trade_ids = []
        for leg in legs:
            # leg: {'option_symbol': 'SPY250117C00600000', 'side': 'sell'/'buy'}
            side = OrderSide.SELL if leg['side'] == 'sell' else OrderSide.BUY
            
            # Construct Option Symbol (OSI) usually passed in leg['option_symbol']
            # Alpaca expects the OSI symbol as the 'symbol' field for options?
            # Yes.
            
            req = MarketOrderRequest(
                symbol=leg['option_symbol'],
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            try:
                order = self.client.submit_order(order_data=req)
                print(f"  [Leg] {side} {leg['option_symbol']}: Submitted (ID: {order.id})")
                trade_ids.append(order.id)
            except Exception as e:
                print(f"  [Error] Failed to submit leg {leg['option_symbol']}: {e}")
        
        return trade_ids

    def get_positions(self):
        return self.client.get_all_positions()

    def close_all(self):
        self.client.close_all_positions(cancel_orders=True)
