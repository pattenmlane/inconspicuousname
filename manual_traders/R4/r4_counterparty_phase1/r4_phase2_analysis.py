#!/usr/bin/env python3
"""
Round 4 Phase 2 tape analysis (orthogonal to Phase 1 tables).

Prereq: outputs/r4_p1_trades_enriched.csv from phase 1.
Writes r4_p2_* under outputs/.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
DAYS = [1, 2, 3]
W_BURST = 500
K5200 = 5200.0
T_IV = 4.0 / 365.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def _implied_vol(S: float, price: float) -> float:
    if price <= max(S - K5200, 0.0) + 1e-6:
        return float("nan")
    lo, hi = 1e-4, 3.0
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if _bs_call(S, K5200, T_IV, m) > price:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    pr = pd.concat(frames, ignore_index=True)
    pr = pr.rename(columns={"product": "symbol"})
    for c in ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1", "mid_price"]:
        pr[c] = pd.to_numeric(pr[c], errors="coerce")
    pr["spread"] = pr["ask_price_1"] - pr["bid_price_1"]
    bv, av = pr["bid_volume_1"].fillna(0), pr["ask_volume_1"].fillna(0)
    den = bv + av
    pr["microprice"] = np.where(
        den > 0,
        (pr["bid_price_1"] * av + pr["ask_price_1"] * bv) / den,
        pr["mid_price"],
    )
    pr["micro_minus_mid"] = pr["microprice"] - pr["mid_price"]
    return pr


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t["day"] = d
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def burst_01_22_exact(tr: pd.DataFrame) -> set[tuple[int, int]]:
    """Same (day,timestamp): all rows Mark01 buyer, Mark22 seller, >=3 symbols."""
    s = set()
    for (d, ts), sub in tr.groupby(["day", "timestamp"]):
        if len(sub) < 2:
            continue
        if not sub["buyer"].eq("Mark 01").all():
            continue
        if not sub["seller"].eq("Mark 22").all():
            continue
        if sub["symbol"].nunique() >= 3:
            s.add((int(d), int(ts)))
    return s


def burst_near(burst_exact: set[tuple[int, int]]) -> set[tuple[int, int]]:
    by_day: dict[int, list[int]] = defaultdict(list)
    for d, ts in burst_exact:
        by_day[d].append(ts)
    near = set(burst_exact)
    for d, ts in burst_exact:
        for ots in by_day[d]:
            if abs(ots - ts) <= W_BURST:
                near.add((d, ots))
    return near


def main() -> None:
    en_path = OUT / "r4_p1_trades_enriched.csv"
    en = pd.read_csv(en_path)
    pr = load_prices()
    tr = load_trades()

    # Microprice vs mid on extract + forward sum |Δmid| over 5 rows
    rows = []
    for d in DAYS:
        sub = pr[(pr["day"] == d) & (pr["symbol"] == "VELVETFRUIT_EXTRACT")].sort_values("timestamp")
        m = sub["mid_price"].to_numpy()
        mm = sub["micro_minus_mid"].to_numpy()
        for i in range(len(sub)):
            fv = float(np.sum(np.abs(np.diff(m[i : min(i + 6, len(m))])))) if i + 5 < len(m) else float("nan")
            rows.append(
                {
                    "day": d,
                    "timestamp": int(sub["timestamp"].iloc[i]),
                    "micro_minus_mid": float(mm[i]),
                    "fwd_absdmid_sum5": fv,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "r4_p2_extract_micro_fwdvol.csv", index=False)

    burst_ex = burst_01_22_exact(tr)
    burst_nr = burst_near(burst_ex)
    pd.DataFrame(list(burst_ex), columns=["day", "timestamp"]).to_csv(OUT / "r4_p2_burst_01_22_exact.csv", index=False)

    u = pr[pr["symbol"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].sort_values(["day", "timestamp"])
    fwd_u20 = []
    for d in DAYS:
        sub = u[u["day"] == d].reset_index(drop=True)
        mids = sub["mid_price"].to_numpy()
        ts = sub["timestamp"].to_numpy().astype(int)
        imap = {int(ts[i]): i for i in range(len(ts))}
        for _, r in en[en["day"] == d].iterrows():
            t = int(r["timestamp"])
            if t not in imap:
                fwd_u20.append(np.nan)
                continue
            i = imap[t]
            fwd_u20.append(float(mids[i + 20] - mids[i]) if i + 20 < len(mids) else np.nan)
    en = en.copy()
    en["fwd_u_20"] = fwd_u20
    en["burst_exact"] = en.apply(lambda r: 1 if (int(r["day"]), int(r["timestamp"])) in burst_ex else 0, axis=1)
    en["burst_near"] = en.apply(lambda r: 1 if (int(r["day"]), int(r["timestamp"])) in burst_nr else 0, axis=1)

    lines = ["=== Burst-conditioned forward extract mid +20 rows ==="]
    for label, mask in [
        ("burst_exact", en["burst_exact"] > 0),
        ("burst_near_W500", (en["burst_near"] > 0) & (en["burst_exact"] == 0)),
        ("isolated", (en["burst_near"] == 0)),
    ]:
        sub = en.loc[mask, "fwd_u_20"].dropna()
        if len(sub) > 5:
            lines.append(f"{label}: n={len(sub)} mean={sub.mean():.5f} median={sub.median():.5f}")
    Path(OUT / "r4_p2_burst_u_forward.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Lead-lag extract ret vs Mark01 buy - Mark22 sell on VEV_5300
    tr530 = tr[tr["symbol"] == "VEV_5300"].copy()
    tr530["signed"] = np.where(tr530["buyer"] == "Mark 01", tr530["quantity"], 0) - np.where(
        tr530["seller"] == "Mark 22", tr530["quantity"], 0
    )
    flow = tr530.groupby(["day", "timestamp"])["signed"].sum().reset_index()
    corrs = []
    for d in DAYS:
        u2 = pr[(pr["day"] == d) & (pr["symbol"] == "VELVETFRUIT_EXTRACT")].sort_values("timestamp")
        ret = u2["mid_price"].diff().fillna(0.0)
        u2 = u2.assign(ret=ret.values)
        m = u2.merge(flow[flow["day"] == d], on=["day", "timestamp"], how="left").fillna({"signed": 0.0})
        for lag in range(0, 6):
            s = m["signed"].shift(lag)
            c = m["ret"].corr(s) if m["ret"].std() > 0 and s.std() > 0 else float("nan")
            corrs.append({"day": d, "lag": lag, "corr": float(c) if c == c else float("nan")})
    pd.DataFrame(corrs).to_csv(OUT / "r4_p2_leadlag_extract_vs_5300flow.csv", index=False)

    # IV on VEV_5200 trades by pair
    uwide = pr[pr["symbol"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].rename(columns={"mid_price": "S"})
    miv = tr[tr["symbol"] == "VEV_5200"].merge(uwide, on=["day", "timestamp"])
    miv["iv"] = [_implied_vol(float(r["S"]), float(r["price"])) for _, r in miv.iterrows()]
    miv["pair"] = miv["buyer"].astype(str) + "->" + miv["seller"].astype(str)
    miv.dropna(subset=["iv"]).groupby("pair")["iv"].agg(["count", "mean"]).sort_values("mean", ascending=False).to_csv(
        OUT / "r4_p2_iv5200_mean_by_pair.csv"
    )

    # Cumulative Mark 01 net (bought minus sold as seller) per day along tape
    inv = []
    for d in DAYS:
        sub = tr[tr["day"] == d].sort_values("timestamp")
        cum = 0
        for _, r in sub.iterrows():
            if str(r["buyer"]) == "Mark 01":
                cum += int(r["quantity"])
            if str(r["seller"]) == "Mark 01":
                cum -= int(r["quantity"])
            inv.append({"day": d, "timestamp": int(r["timestamp"]), "cum_net_Mark01": cum})
    pd.DataFrame(inv).to_csv(OUT / "r4_p2_inventory_proxy_Mark01_cumnet.csv", index=False)

    # Lookup for live trader: timestamps within W of any Mark01->Mark22 multi-VEV burst
    lookup = {str(d): sorted({ts for (dd, ts) in burst_nr if dd == d}) for d in DAYS}
    (OUT.parent / "burst_near_timestamps_by_day.json").write_text(json.dumps(lookup), encoding="utf-8")

    print("Wrote r4_p2_* to", OUT)


if __name__ == "__main__":
    main()
