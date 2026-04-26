"""
Round 4 — initial counterparty / “Mark bot” sketch from public tapes.

Reads ``Prosperity4Data/ROUND_4/trades_round_4_day_{1,2,3}.csv`` (semicolon-separated;
``buyer`` / ``seller`` populated — the change vs Round 3).

Run from repo root:

  python3 round4work/r4_counterparty_initial_sketch.py

Writes ``round4work/outputs/r4_counterparty_sketch.txt``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    days = [1, 2, 3]
    frames: list[pd.DataFrame] = []
    for d in days:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    if not frames:
        raise SystemExit(f"No trade files under {DATA}")

    tr = pd.concat(frames, ignore_index=True)
    tr["qty"] = tr["quantity"].astype(int)

    lines: list[str] = []
    def ln(s: str = "") -> None:
        lines.append(s)

    buyers, sellers = tr["buyer"].astype(str), tr["seller"].astype(str)
    allp = pd.concat([buyers, sellers])
    ln("=== Round 4 counterparty sketch ===")
    ln(f"Rows: {len(tr):,} | days: {sorted(tr['day'].unique().tolist())}")
    ln(f"Unique names (buy or sell): {allp.nunique()}")
    ln()

    c = Counter()
    for _, r in tr.iterrows():
        c[str(r["buyer"])] += 1
        c[str(r["seller"])] += 1
    ln("Trade-side appearances (each row counts buyer + seller once):")
    for name, n in c.most_common():
        ln(f"  {name:16s} {n:5d}")
    ln()

    def vol_side(name: str) -> tuple[int, int]:
        vb = int(tr.loc[tr["buyer"] == name, "qty"].sum())
        vs = int(tr.loc[tr["seller"] == name, "qty"].sum())
        return vb, vs

    ln("Aggressive buy qty vs sell qty (lot-sum):")
    for name, _ in c.most_common():
        b, s = vol_side(name)
        ln(f"  {name:16s}  buy {b:5d}  sell {s:5d}  net {b - s:+d}")
    ln()

    pairs = tr.groupby(["buyer", "seller"]).size().sort_values(ascending=False).head(15)
    ln("Top buyer → seller pair counts:")
    for (b, s), n in pairs.items():
        ln(f"  {b} → {s}: {n}")
    ln()

    burst = tr.groupby(["day", "timestamp"]).size()
    big = burst[burst >= 4].sort_values(ascending=False).head(10)
    ln("Largest same-(day,timestamp) bursts (≥4 prints):")
    for (d, ts), sz in big.items():
        sub = tr[(tr["day"] == d) & (tr["timestamp"] == ts)]
        syms = sub["symbol"].tolist()
        bu = "mixed" if sub["buyer"].nunique() > 1 else str(sub["buyer"].iloc[0])
        se = "mixed" if sub["seller"].nunique() > 1 else str(sub["seller"].iloc[0])
        ln(f"  day{d} t={ts} n={sz} buyer={bu} seller={se} | {syms[:6]}{'...' if len(syms) > 6 else ''}")
    ln()

    for sym in ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK", "VEV_5200", "VEV_5300"]:
        sub = tr[tr["symbol"] == sym]
        if sub.empty:
            continue
        bc = sub["buyer"].value_counts(normalize=True).head(4)
        sc = sub["seller"].value_counts(normalize=True).head(4)
        ln(f"{sym} (n={len(sub)})")
        ln(f"  buyer share:  {bc.round(3).to_dict()}")
        ln(f"  seller share: {sc.round(3).to_dict()}")
        ln()

    out_path = OUT / "r4_counterparty_sketch.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
