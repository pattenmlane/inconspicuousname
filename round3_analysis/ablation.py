"""Run the bot with each piece toggled to attribute PnL."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "imc-prosperity-4-backtester"))
import prosperity4bt.datamodel as dm
sys.modules.setdefault("datamodel", dm)


def reload_bot():
    if "round3_bot" in sys.modules:
        del sys.modules["round3_bot"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("round3_bot", HERE / "round3_bot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["round3_bot"] = mod
    return mod


def reload_backtest():
    if "backtest" in sys.modules:
        del sys.modules["backtest"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("backtest", HERE / "backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_run(disable: tuple[str, ...]):
    """Monkey-patch round3_bot.Trader to skip pieces in `disable`."""
    bot = reload_bot()
    Trader = bot.Trader
    orig_run = Trader.run

    def run(self, state):
        result = {}
        try:
            if "hydrogel" not in disable:
                self._trade_hydrogel(state, result)
        except Exception as e:
            print("h err", e)
        try:
            if "options" not in disable:
                self._trade_options(state, result)
            elif "underlying_only" in disable:
                pass
            else:
                # still market-make underlying even if vouchers disabled
                self._make_underlying(state, result)
        except Exception as e:
            print("o err", e)
        return result, 0, state.traderData or ""

    Trader.run = run
    return bot


def main():
    scenarios = [
        ("hydrogel ONLY",       ("options",)),
        ("underlying mm ONLY",  ("hydrogel",)),  # _trade_options without takes is hard; we keep for ref
        ("vouchers ONLY (no hydrogel)", ("hydrogel",)),
        ("everything",          ()),
    ]
    for label, disable in scenarios:
        print(f"\n>>> SCENARIO: {label}  (disabled: {disable or 'none'})")
        patch_run(disable)
        bt = reload_backtest()
        total, _ = bt.run_backtest(verbose=False)
        print(f"<<< {label}: TOTAL = {total:.2f}\n")


if __name__ == "__main__":
    main()
