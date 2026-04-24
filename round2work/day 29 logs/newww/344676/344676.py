"""Copy of ``potential2_osmium_only.py`` with take ``edge`` = 1.0 (was 1.5).

Osmium-only (pepper / ``BuyAndHold`` removed). See ``potential2_osmium_only.py``.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Optional, Type
import json
import numpy as np

UNIT_TIMESTAMP = 100

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
}

class Logger:
    def __init__(self, state: TradingState):
        self.logs = {"TIMESTAMP": state.timestamp}
        self.positions = state.position

    def new_product(self, product: str):
        self.product = product
        self.log("POSITION", self.positions.get(product, 0))

    def log(self, key: str, value: str | dict | int | float | list):
        self.logs.setdefault(self.product, {})[key] = value # type: ignore

    def log_orders(self, orders: list[Order]):
        buy_orders = [{"price": order.price, "quantity": order.quantity} for order in orders if order.quantity > 0]
        sell_orders = [{"price": order.price, "quantity": order.quantity} for order in orders if order.quantity < 0]
        self.log("BUY_ORDERS", buy_orders)
        self.log("SELL_ORDERS", sell_orders)

    def log_error(self, message: str):
        self.logs.setdefault(self.product, {}).setdefault("ERRORS", []).append(message) # type: ignore

    def log_warning(self, message: str):
        self.logs.setdefault(self.product, {}).setdefault("WARNINGS", []).append(message) # type: ignore
    
    def dump(self):
        print(json.dumps(self.logs))

class BaseTrader:
    def __init__(self, state: TradingState, product: str, logger: Logger):
        self.product = product
        self.state = state
        self.logger = logger
        self.orders: list[Order] = []

        if product not in POSITION_LIMITS:
            self.logger.log_error(f"No position limit defined for {product}")

        self.pos_limit = POSITION_LIMITS.get(product, 0)

        self.position = state.position.get(product, 0)

        self.bids, self.asks = self.get_order_book()

        self.best_bid: Optional[int] = max(self.bids) if self.bids else None
        self.best_ask: Optional[int] = min(self.asks) if self.asks else None

        if not self.bids and not self.asks:
            self.logger.log_warning("Missing bids and asks — empty order book")

        self.spread: Optional[int] = (self.best_ask - self.best_bid) if self.best_bid and self.best_ask else None

        self.deep_bid: Optional[int] = min(self.bids) if self.bids else None
        self.deep_ask: Optional[int] = max(self.asks) if self.asks else None
        self.mid: Optional[float] = (self.deep_bid + self.deep_ask) / 2 if self.deep_bid and self.deep_ask else None
        self.fair_price: Optional[float] = (self.deep_bid + self.deep_ask) / 2 if self.deep_bid and self.deep_ask else None

        self.buy_capacity = self.pos_limit - self.position if self.pos_limit else 0  
        self.sell_capacity = self.pos_limit + self.position if self.pos_limit else 0   # max short position is negative, e.g. -80

        # Persistent state from last tick
        self.saved_data: dict = self.load_saved_data()

    def get_order_book(self):
        bids, asks = {}, {}
        try:
            depth: OrderDepth = self.state.order_depths[self.product]
            bids  = {p: abs(v) for p, v in sorted(depth.buy_orders.items(), reverse=True)}
            asks = {p: abs(v) for p, v in sorted(depth.sell_orders.items())}
        except KeyError:
            self.logger.log_error(f"No order depth for {self.product}")

        return bids, asks

    
    def load_saved_data(self) -> dict:
        try:
            if not self.state.traderData:
                return {}
            return json.loads(self.state.traderData).get(self.product, {})
        except (json.JSONDecodeError, AttributeError):
            self.logger.log_error("Json failed to parse saved data")
            return {}

    # ── Order helpers ───

    def buy(self, price: int, max_quantity: int):
        quantity = min(max_quantity, self.buy_capacity)

        if quantity > 0:
            self.orders.append(Order(self.product, price, quantity))
            self.buy_capacity -= quantity
        
        return quantity

    def sell(self, price: int, max_quantity: int):
        quantity = min(max_quantity, self.sell_capacity)
        
        if quantity > 0:
            self.orders.append(Order(self.product, price, -quantity))
            self.sell_capacity -= quantity
        
        return quantity
    
    def save_trader_data(self) -> str:
            try:
                # 1. Load the existing global state string from the platform
                current_global_data = {}
                if self.state.traderData:
                    current_global_data = json.loads(self.state.traderData)
                
                # 2. Update only this product's entry
                current_global_data[self.product] = self.saved_data
                
                # 3. Serialize back to a JSON string
                return json.dumps(current_global_data)
                
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.log_error(f"Failed to save data: {e}")
                return self.state.traderData or ""


    # ── Market Orders for taking ───

    def market_buy(self, max_price: Optional[float] = None, max_quantity: Optional[int] = None):
        filled_value = 0
        filled_qty = 0
        for price, quantity in self.asks.items():
            if max_price is not None and price > max_price:
                break

            # Limit order if it takes us over our max quantity
            if max_quantity is not None:
                quantity = min(quantity, max_quantity - filled_qty)

            if quantity <= 0:
                break

            traded_volume = self.buy(price, quantity)
            filled_value += traded_volume * price
            filled_qty += traded_volume
        
        if filled_qty > 0:
            avg_price = filled_value / filled_qty
            self.logger.log("MARKET_BUY", {
                "quantity": filled_qty,
                "avg_price": round(avg_price, 2),
                "slippage": round(avg_price - self.best_ask, 2) if self.best_ask else None,
                "max_price": max_price,
            })

    def market_sell(self, min_price: Optional[float] = None, max_quantity: Optional[int] = None):
        filled_value = 0
        filled_qty = 0
        for price, quantity in self.bids.items():
            if min_price is not None and price < min_price:
                break

            # Limit order if it takes us over our max quantity
            if max_quantity is not None:
                quantity = min(quantity, max_quantity - filled_qty)

            if quantity <= 0:
                break

            traded_volume = self.sell(price, quantity)
            filled_value += traded_volume * price
            filled_qty += traded_volume

        if filled_qty > 0:
            avg_price = filled_value / filled_qty
            self.logger.log("MARKET_SELL", {
                "quantity": filled_qty,
                "avg_price": round(avg_price, 2),
                "slippage": round(self.best_bid - avg_price, 2) if self.best_bid else None,
                "min_price": min_price,
            })
    
    def get_ewm(self, name: str, value: int | float | None, half_life: float):
        alpha = 1 - np.power(2, -1/half_life)

        if value:
            ewm_value = alpha * self.saved_data[name] + (1 - alpha) * value if self.saved_data and name in self.saved_data else value
        else:
            if self.saved_data is None or name not in self.saved_data:
                return None
            
            ewm_value = self.saved_data[name]

        self.saved_data[name] = ewm_value
        return ewm_value
    
    def override_best_bid(self, max_quantity, fair_price, gap_spread, mid_price):
        if self.best_bid and self.best_bid < fair_price - 1:
            self.buy(self.best_bid + 1, max_quantity=max_quantity)
        elif not self.best_bid:
            self.buy(round(mid_price - gap_spread), max_quantity=max_quantity)
        
    def override_best_ask(self, max_quantity, fair_price, gap_spread, mid_price):
        if self.best_ask and self.best_ask > fair_price + 1:
            self.sell(self.best_ask - 1, max_quantity=max_quantity)
        elif not self.best_ask:
            self.sell(round(mid_price + gap_spread), max_quantity=max_quantity)
    
    # ── Strategy to Override ──
    def get_orders(self):
        return {self.product: self.orders}
    

class FairPriceCalculator(BaseTrader):
    def get_orders(self):
        self.market_buy(max_quantity=1)
        if self.position != 0:
            self.market_sell(max_quantity=1)

        return {self.product: self.orders}

# ──── Trading Strategies ─────
class MeanReversionMarketMaker(BaseTrader):
    def get_orders(self):
        MU_PRIOR = 10_000
        eplison = 0.65
        gap_spread = 99
        edge = 1.0
        PRIOR_STRENGTH = 2000

        mid_price = self.get_ewm(name='mid_price', value=self.mid, half_life=2)

        if mid_price is None:
            long_run_mean = self.get_ewm(name='long_run_mean', value=MU_PRIOR, half_life=10_000)
            return {self.product: self.orders}
        
        # --- Long-run empirical mean (slow EWM) ---
        long_run_mean = self.get_ewm(name='long_run_mean', value=mid_price, half_life=10_000)

        num_timestamps = self.state.timestamp / UNIT_TIMESTAMP

        # --- Bayesian blend: prior fades as obs_count grows ---
        # When obs_count << PRIOR_STRENGTH  →  MU ≈ MU_PRIOR
        # When obs_count >> PRIOR_STRENGTH  →  MU ≈ long_run_mean
        if long_run_mean is not None:
            prior_weight = PRIOR_STRENGTH / (PRIOR_STRENGTH + num_timestamps)
            data_weight = 1 - prior_weight
            MU = prior_weight * MU_PRIOR + data_weight * long_run_mean
        else:
            MU = MU_PRIOR  

        # Mean reverting fair price skew
        real_fair_price = MU + eplison * (mid_price - MU)
        
        # Take any mispriced orders
        self.market_sell(min_price=real_fair_price + edge, max_quantity=self.pos_limit)
        self.market_buy(max_price=real_fair_price - edge, max_quantity=self.pos_limit)
        
        # Flatten inventory
        if self.position > 0:
            self.market_sell(min_price=real_fair_price, max_quantity=abs(self.position))
        elif self.position < 0:
            self.market_buy(max_price=real_fair_price, max_quantity=abs(self.position))

        # Market make
        self.override_best_bid(max_quantity=self.pos_limit, fair_price=real_fair_price, gap_spread=gap_spread, mid_price=mid_price)
        self.override_best_ask(max_quantity=self.pos_limit, fair_price=real_fair_price, gap_spread=gap_spread, mid_price=mid_price)
        
        if long_run_mean is not None:
            self.logger.log('Dynamic Mean', MU)

        return {self.product: self.orders}


# ──── Trader Factory Function ─────

def make_trader(product: str, trader_cls: Type['BaseTrader']):
    class _Trader(trader_cls):
        def __init__(self, state, logger):
            super().__init__(state, product, logger)
    return _Trader

TRADERS = {
    "ASH_COATED_OSMIUM": MeanReversionMarketMaker,
}

# ──── Trader Class ─────
class Trader:
    def run(self, state: TradingState):

        logger = Logger(state)
        all_orders = {}
        
        for product in TRADERS:
            logger.new_product(product)
            TraderCls = make_trader(product, TRADERS[product])
            trader = TraderCls(state, logger)
            all_orders.update(trader.get_orders())
            state.traderData = trader.save_trader_data()
            logger.log_orders(trader.orders)

        logger.dump()
        return all_orders, 0, state.traderData