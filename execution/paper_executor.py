from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class PaperTrade:
    trade_id: str
    symbol: str
    option_symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    status: str = "OPEN"
    exit_price: float = None
    exit_time: datetime = None
    pnl: float = 0.0
    reason: str = ""

class PaperExecutor:
    def __init__(self, run_config, trading_client=None, slippage_bps=10):
        self.run_config = run_config
        self.trades = []
        self.trading_client = trading_client  # Alpaca TradingClient
        self.slippage_bps = slippage_bps  # Simulate institutional friction (10bps default)

    def open_trade(self, trade_decision):
        """Legacy method for simple paper trades (stub)."""
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            symbol=trade_decision.symbol,
            option_symbol=trade_decision.option_symbol,
            quantity=trade_decision.quantity,
            entry_price=trade_decision.entry_price,
            entry_time=datetime.utcnow(),
            reason=trade_decision.rationale
        )
        self.trades.append(trade)
        return trade

    def apply_slippage(self, credit, qty=1, side="entry"):
        """
        Calculates institutional fill price after slippage.
        If entry: credit is reduced (sell for less).
        If exit: cost is increased (buy for more/sell for less).
        """
        penalty = credit * (self.slippage_bps / 10000.0)
        
        if side == "entry":
            # Selling condor: we get less credit
            return max(0.01, credit - penalty)
        else:
            # Closing condor (buying back or selling wings): we pay more / get less
            return credit + penalty

    def close_trade(self, trade, exit_price):
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.pnl = (exit_price - trade.entry_price) * trade.quantity * 100
        trade.status = "CLOSED"

    def submit_iron_condor(self, symbol, legs, quantity=1):
        """
        Submit an Iron Condor order.
        legs: List of dicts with 'symbol' (OSI) and 'side' ('sell'/'buy').
              Order: [Short Put, Long Put, Short Call, Long Call] or similar.
        """
        # If we have a real Alpaca client, execute LIVE PAPER
        if self.trading_client:
            try:
                from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
                from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce

                print(f"🚀 Submitting LIVE PAPER Iron Condor: {[l['option_symbol'] for l in legs]}")
                
                order_legs = []
                for leg in legs:
                    side = OrderSide.SELL if leg['side'] == 'sell' else OrderSide.BUY
                    order_legs.append(OptionLegRequest(
                        symbol=leg['option_symbol'],
                        side=side,
                        ratio_qty=1
                    ))

                # MLEG Order
                req = MarketOrderRequest(
                    qty=quantity,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    legs=order_legs
                )
                
                res = self.trading_client.submit_order(req)
                print(f"✅ Order Submitted! Client Order ID: {res.client_order_id}")
                return [res.id] # Return ID as list
                
            except Exception as e:
                print(f"❌ Execution Failed: {e}")
                return None
        else:
            # Mock Execution (Legacy internal log only)
            print("⚠️ Dry Run (No Trading Client). Mocking fill...")
            t_ids = []
            for leg in legs:
                tid = str(uuid.uuid4())
                print(f"  Mock Fill: {leg['side'].upper()} {leg['symbol']} (ID: {tid})")
                t_ids.append(tid)
            return t_ids
    def close_iron_condor(self, legs, quantity=1):
        """
        Close an Iron Condor by reversing the legs.
        legs: List of dicts with 'option_symbol' and 'side' ('sell'/'buy').
        """
        if self.trading_client:
            try:
                from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
                from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce

                print(f"📉 Closing Iron Condor Legs: {[l['option_symbol'] for l in legs]}")
                
                order_legs = []
                for leg in legs:
                    # REVERSE SIDE: if it was 'sell', now 'buy' to close.
                    orig_side = leg.get('side', 'sell')
                    side = OrderSide.BUY if orig_side == 'sell' else OrderSide.SELL
                    order_legs.append(OptionLegRequest(
                        symbol=leg['option_symbol'],
                        side=side,
                        ratio_qty=1
                    ))

                # MLEG Order to Close
                req = MarketOrderRequest(
                    qty=quantity,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    legs=order_legs
                )
                
                res = self.trading_client.submit_order(req)
                print(f"✅ Order Submitted for Close! ID: {res.id}")
                return True
            except Exception as e:
                print(f"❌ Close Execution Failed: {e}")
                return False
        else:
            print("⚠️ Dry Run (No Trading Client). Mocking exit fill...")
            return True
