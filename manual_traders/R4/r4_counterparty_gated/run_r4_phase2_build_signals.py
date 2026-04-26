"""Build compact signal keys for live trader: Mark 67 aggressive buy on extract (day,timestamp)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"


def load_trades() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("trades_round_4_day_*.csv")):
        day = int(p.stem.replace("trades_round_4_day_", ""))
        df = pd.read_csv(p, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_px() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("prices_round_4_day_*.csv")):
        day = int(p.stem.replace("prices_round_4_day_", ""))
        df = pd.read_csv(p, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_trades()
    px = load_px()
    u = px[px["product"] == "VELVETFRUIT_EXTRACT"][
        ["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]
    ].copy()
    u["bid1"] = pd.to_numeric(u["bid_price_1"], errors="coerce")
    u["ask1"] = pd.to_numeric(u["ask_price_1"], errors="coerce")
    u = u.rename(columns={"mid_price": "mid"})
    u["mid"] = pd.to_numeric(u["mid"], errors="coerce")
    m = tr.merge(u, on=["day", "timestamp"], how="inner")
    m = m[m["symbol"] == "VELVETFRUIT_EXTRACT"]
    m["price"] = pd.to_numeric(m["price"], errors="coerce")
    ag = (m["price"] >= m["ask1"] - 1e-9) & (m["buyer"] == "Mark 67")
    keys = [f"{int(r.day)}:{int(r.timestamp)}" for r in m[ag].itertuples()]
    (OUT / "signals_mark67_aggr_extract_buy.json").write_text(
        json.dumps({"keys": sorted(set(keys)), "n": len(set(keys))}, indent=2),
        encoding="utf-8",
    )
    print("Wrote", OUT / "signals_mark67_aggr_extract_buy.json", "n=", len(set(keys)))


if __name__ == "__main__":
    main()
