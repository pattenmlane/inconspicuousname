
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from datamodel import Order, OrderDepth

UNIT_TIMESTAMP = 100
SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80


class _MiniState:
    """Subset of ``TradingState`` fields used by ``BaseTrader`` / MM."""

    def __init__(
        self,
        trader_data: str,
        timestamp: int,
        order_depths: dict,
        position: dict,
    ) -> None:
        self.traderData = trader_data
        self.timestamp = int(timestamp)
        self.order_depths = order_depths
        self.position = position


class Logger:
    def __init__(self, state: Any):
        self.logs: dict = {"TIMESTAMP": state.timestamp}
        self.positions = state.position

    def new_product(self, product: str) -> None:
        self.product = product
        self.log("POSITION", self.positions.get(product, 0))

    def log(self, key: str, value: str | dict | int | float | list) -> None:
        self.logs.setdefault(self.product, {})[key] = value  # type: ignore[union-attr]

    def log_orders(self, orders: list[Order]) -> None:
        pass

    def log_error(self, message: str) -> None:
        pass

    def log_warning(self, message: str) -> None:
        pass

    def dump(self) -> None:
        pass


class BaseTrader:
    def __init__(self, state: Any, product: str, logger: Logger):
        self.product = product
        self.state = state
        self.logger = logger
        self.orders: list[Order] = []

        self.pos_limit = POSITION_LIMIT
        self.position = state.position.get(product, 0)

        self.bids, self.asks = self.get_order_book()

        self.best_bid: Optional[int] = max(self.bids) if self.bids else None
        self.best_ask: Optional[int] = min(self.asks) if self.asks else None

        self.spread: Optional[int] = (self.best_ask - self.best_bid) if self.best_bid and self.best_ask else None

        self.deep_bid: Optional[int] = min(self.bids) if self.bids else None
        self.deep_ask: Optional[int] = max(self.asks) if self.asks else None
        self.mid: Optional[float] = (
            (self.deep_bid + self.deep_ask) / 2 if self.deep_bid and self.deep_ask else None
        )
        self.fair_price: Optional[float] = self.mid

        self.buy_capacity = self.pos_limit - self.position if self.pos_limit else 0
        self.sell_capacity = self.pos_limit + self.position if self.pos_limit else 0

        self.saved_data: dict = self.load_saved_data()

    def get_order_book(self) -> tuple[dict, dict]:
        bids, asks = {}, {}
        try:
            depth: OrderDepth = self.state.order_depths[self.product]
            bids = {p: abs(v) for p, v in sorted(depth.buy_orders.items(), reverse=True)}
            asks = {p: abs(v) for p, v in sorted(depth.sell_orders.items())}
        except KeyError:
            self.logger.log_error(f"No order depth for {self.product}")
        return bids, asks

    def load_saved_data(self) -> dict:
        try:
            if not self.state.traderData:
                return {}
            data = json.loads(self.state.traderData)
            return data.get(self.product, {}) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, AttributeError):
            return {}

    def buy(self, price: int, max_quantity: int) -> int:
        quantity = min(max_quantity, self.buy_capacity)
        if quantity > 0:
            self.orders.append(Order(self.product, price, quantity))
            self.buy_capacity -= quantity
        return quantity

    def sell(self, price: int, max_quantity: int) -> int:
        quantity = min(max_quantity, self.sell_capacity)
        if quantity > 0:
            self.orders.append(Order(self.product, price, -quantity))
            self.sell_capacity -= quantity
        return quantity

    def save_trader_data(self) -> str:
        try:
            current_global_data: dict = {}
            if self.state.traderData:
                current_global_data = json.loads(self.state.traderData)
            if not isinstance(current_global_data, dict):
                current_global_data = {}
            current_global_data[self.product] = self.saved_data
            return json.dumps(current_global_data)
        except (json.JSONDecodeError, TypeError):
            return self.state.traderData or "{}"

    def market_buy(self, max_price: Optional[float] = None, max_quantity: Optional[int] = None) -> None:
        filled_qty = 0
        for price, quantity in self.asks.items():
            if max_price is not None and price > max_price:
                break
            if max_quantity is not None:
                quantity = min(quantity, max_quantity - filled_qty)
            if quantity <= 0:
                break
            traded_volume = self.buy(price, quantity)
            filled_qty += traded_volume

    def market_sell(self, min_price: Optional[float] = None, max_quantity: Optional[int] = None) -> None:
        filled_qty = 0
        for price, quantity in self.bids.items():
            if min_price is not None and price < min_price:
                break
            if max_quantity is not None:
                quantity = min(quantity, max_quantity - filled_qty)
            if quantity <= 0:
                break
            traded_volume = self.sell(price, quantity)
            filled_qty += traded_volume

    def get_ewm(self, name: str, value: int | float | None, half_life: float) -> float | None:
        alpha = 1 - np.power(2, -1 / half_life)
        if value:
            ewm_value = (
                alpha * self.saved_data[name] + (1 - alpha) * value
                if self.saved_data and name in self.saved_data
                else value
            )
        else:
            if not self.saved_data or name not in self.saved_data:
                return None
            ewm_value = self.saved_data[name]
        self.saved_data[name] = ewm_value
        return ewm_value

    def override_best_bid(self, max_quantity: int, fair_price: float, gap_spread: float, mid_price: float | None) -> None:
        if self.best_bid and self.best_bid < fair_price - 1:
            self.buy(self.best_bid + 1, max_quantity=max_quantity)
        elif not self.best_bid and mid_price is not None:
            self.buy(round(mid_price - gap_spread), max_quantity=max_quantity)

    def override_best_ask(self, max_quantity: int, fair_price: float, gap_spread: float, mid_price: float | None) -> None:
        if self.best_ask and self.best_ask > fair_price + 1:
            self.sell(self.best_ask - 1, max_quantity=max_quantity)
        elif not self.best_ask and mid_price is not None:
            self.sell(round(mid_price + gap_spread), max_quantity=max_quantity)


class MeanReversionMarketMaker(BaseTrader):
    def get_orders(self) -> dict[str, list[Order]]:
        MU_PRIOR = 10_000
        eplison = 0.65
        gap_spread = 99
        edge = 1.5
        PRIOR_STRENGTH = 2000

        mid_price = self.get_ewm(name="mid_price", value=self.mid, half_life=2)

        if mid_price is None:
            self.get_ewm(name="long_run_mean", value=MU_PRIOR, half_life=10_000)
            return {self.product: self.orders}

        long_run_mean = self.get_ewm(name="long_run_mean", value=mid_price, half_life=10_000)

        num_timestamps = self.state.timestamp / UNIT_TIMESTAMP

        if long_run_mean is not None:
            prior_weight = PRIOR_STRENGTH / (PRIOR_STRENGTH + num_timestamps)
            data_weight = 1 - prior_weight
            MU = prior_weight * MU_PRIOR + data_weight * long_run_mean
        else:
            MU = MU_PRIOR

        real_fair_price = MU + eplison * (mid_price - MU)

        self.market_sell(min_price=real_fair_price + edge, max_quantity=self.pos_limit)
        self.market_buy(max_price=real_fair_price - edge, max_quantity=self.pos_limit)

        if self.position > 0:
            self.market_sell(min_price=real_fair_price, max_quantity=abs(self.position))
        elif self.position < 0:
            self.market_buy(max_price=real_fair_price, max_quantity=abs(self.position))

        self.override_best_bid(
            max_quantity=self.pos_limit,
            fair_price=real_fair_price,
            gap_spread=gap_spread,
            mid_price=mid_price,
        )
        self.override_best_ask(
            max_quantity=self.pos_limit,
            fair_price=real_fair_price,
            gap_spread=gap_spread,
            mid_price=mid_price,
        )

        if long_run_mean is not None:
            self.logger.log("Dynamic Mean", MU)

        return {self.product: self.orders}


def osmium_step(
    depth: OrderDepth,
    position: int,
    timestamp: int,
    trader_data: str = "",
) -> tuple[list[Order], str]:
    """
    Run one MeanReversionMarketMaker tick for osmium only.

    ``trader_data`` is the full JSON string the platform uses; only the
    ``ASH_COATED_OSMIUM`` key is read/written for EWM state.
    """
    state = _MiniState(
        trader_data or "",
        int(timestamp),
        {SYMBOL: depth},
        {SYMBOL: int(position)},
    )
    logger = Logger(state)
    logger.new_product(SYMBOL)
    mm: MeanReversionMarketMaker = MeanReversionMarketMaker(state, SYMBOL, logger)
    orders_map = mm.get_orders()
    out_td = mm.save_trader_data()
    return orders_map.get(SYMBOL, []), out_td
