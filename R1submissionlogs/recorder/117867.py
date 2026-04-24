from datamodel import TradingState
from typing import Dict, List, Optional, Set
import jsonpickle


class Trader:
    """
    No-trade recorder trader.

    Prints order book snapshots in long CSV format:
    day;timestamp;product;side;level;price;volume

    - Default is PRODUCT_FILTER = None to record every product.
    - Optionally set PRODUCT_FILTER = {"PRODUCT_A", "PRODUCT_B"} to limit output.
    """

    PRODUCT_FILTER: Optional[Set[str]] = None

    def _allowed(self, product: str) -> bool:
        return self.PRODUCT_FILTER is None or product in self.PRODUCT_FILTER

    def _load_memory(self, trader_data: str) -> Dict:
        if not trader_data:
            return {"day": 0, "last_timestamp": -1, "header_printed": False}
        try:
            decoded = jsonpickle.decode(trader_data)
            if isinstance(decoded, dict):
                decoded.setdefault("day", 0)
                decoded.setdefault("last_timestamp", -1)
                decoded.setdefault("header_printed", False)
                return decoded
        except Exception:
            pass
        return {"day": 0, "last_timestamp": -1, "header_printed": False}

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)

        # New day detection: timestamp resets between days/simulations.
        if memory["last_timestamp"] >= 0 and state.timestamp < memory["last_timestamp"]:
            memory["day"] += 1

        if not memory["header_printed"]:
            print("day;timestamp;product;side;level;price;volume")
            memory["header_printed"] = True

        for product in sorted(state.order_depths.keys()):
            if not self._allowed(product):
                continue

            depth = state.order_depths[product]

            # Buy side: level 1 is highest bid.
            for level, (price, volume) in enumerate(
                sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True), start=1
            ):
                print(f"{memory['day']};{state.timestamp};{product};bid;{level};{price};{abs(int(volume))}")

            # Sell side: level 1 is lowest ask.
            for level, (price, volume) in enumerate(
                sorted(depth.sell_orders.items(), key=lambda x: x[0]), start=1
            ):
                print(f"{memory['day']};{state.timestamp};{product};ask;{level};{price};{abs(int(volume))}")

        memory["last_timestamp"] = state.timestamp

        # No trading: return empty orders + zero conversions.
        result: Dict[str, List] = {}
        conversions = 0
        return result, conversions, jsonpickle.encode(memory)