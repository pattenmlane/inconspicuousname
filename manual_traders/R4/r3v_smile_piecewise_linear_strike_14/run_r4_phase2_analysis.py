#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal to Phase 1 depth: burst×pair conditioning, microprice/spread,
signed-flow lead–lag, joint 5200+5300 tight gate interaction with Mark prints.

Tape only (CSV); outputs under analysis_outputs/.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("Prosperity4Data/ROUND_4")
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = (1, 2, 3)
S5200, S5300 = "VEV_5200", "VEV_5300"
U = "VELVETFRUIT_EXTRACT"
TH_TIGHT = 2


def load_prices(day: int) -> dict[str, list[tuple[int, list[int], list[int], list[int], list[int], float]]]:
    """sym -> sorted list of (ts, bid_px, bid_vol, ask_px, ask_vol, mid)"""
    by: dict[str, list] = defaultdict(list)
    with open(DATA / f"prices_round_4_day_{day}.csv", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if int(r["day"]) != day:
                continue
            sym = r["product"]
            ts = int(r["timestamp"])
            bp = []
            bv = []
            ap = []
            av = []
            for i in (3, 5, 7):
                if r.get(f"bid_price_{(i-1)//2 + 1}", "") == "":
                    break
                bp.append(int(float(r[f"bid_price_{(i-1)//2 + 1}"])))
                bv.append(int(float(r[f"bid_volume_{(i-1)//2 + 1}"])))
            for i in (9, 11, 13):
                k = (i - 7) // 2
                if r.get(f"ask_price_{k}", "") == "":
                    break
                ap.append(int(float(r[f"ask_price_{k}"])))
                av.append(int(float(r[f"ask_volume_{k}"])))
            mid = float(r["mid_price"])
            by[sym].append((ts, bp, bv, ap, av, mid))
    for sym in by:
        by[sym].sort(key=lambda x: x[0])
    return dict(by)


def snap_row(rows, ts):
    tss = [x[0] for x in rows]
    i = bisect.bisect_right(tss, ts) - 1
    return rows[i] if i >= 0 else None


def fwd_mid(rows, ts, k):
    tss = [x[0] for x in rows]
    i = bisect.bisect_right(tss, ts)
    j = i + k - 1
    return rows[j][5] if j < len(rows) else None


def microprice(row) -> float | None:
    _, bp, bv, ap, av, mid = row
    if not bp or not ap:
        return None
    bb, ba = bp[0], ap[0]
    qb = bv[0] if bv else 0
    qa = av[0] if av else 0
    if qb + qa == 0:
        return None
    return (bb * qa + ba * qb) / (qb + qa)


def load_trades(day):
    rows = []
    with open(DATA / f"trades_round_4_day_{day}.csv", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(
                {
                    "day": day,
                    "ts": int(r["timestamp"]),
                    "buyer": (r.get("buyer") or "").strip(),
                    "seller": (r.get("seller") or "").strip(),
                    "sym": r["symbol"],
                    "price": float(r["price"]),
                    "qty": float(r["quantity"]),
                }
            )
    return rows


def joint_tight_at(pr_by_day, day, ts) -> bool | None:
    r52 = snap_row(pr_by_day[day][S5200], ts)
    r53 = snap_row(pr_by_day[day][S5300], ts)
    if not r52 or not r53:
        return None
    sp52 = r52[4][0] - r52[1][0] if r52[1] and r52[4] else 999
    sp53 = r53[4][0] - r53[1][0] if r53[1] and r53[4] else 999
    return sp52 <= TH_TIGHT and sp53 <= TH_TIGHT


def main() -> None:
    pr = {d: load_prices(d) for d in DAYS}
    all_tr = []
    for d in DAYS:
        all_tr.extend(load_trades(d))

    # --- 1) Burst at ts: >=3 trades; M01->M22 burst: any trade in burst with that pair on VEV
    by_dt = defaultdict(list)
    for tr in all_tr:
        by_dt[(tr["day"], tr["ts"])].append(tr)
    burst_ts = {k for k, v in by_dt.items() if len(v) >= 3}

    def is_m01_m22_vev_burst(day, ts):
        for tr in by_dt.get((day, ts), []):
            if tr["buyer"] == "Mark 01" and tr["seller"] == "Mark 22" and tr["sym"].startswith("VEV_"):
                return True
        return False

    m01_m22_burst = {(d, t) for d, t in burst_ts if is_m01_m22_vev_burst(d, t)}

    # Forward VEV_5300 mid K=20 after M01-M22 burst ticks vs other burst ticks
    ser5300 = {d: pr[d][S5300] for d in DAYS}
    fwd_b = []
    fwd_ob = []
    for d, ts in m01_m22_burst:
        sn = snap_row(ser5300[d], ts)
        if not sn:
            continue
        fm = fwd_mid(ser5300[d], ts, 20)
        if fm is None:
            continue
        fwd_b.append(fm - sn[5])
    for d, ts in burst_ts - m01_m22_burst:
        sn = snap_row(ser5300[d], ts)
        if not sn:
            continue
        fm = fwd_mid(ser5300[d], ts, 20)
        if fm is None:
            continue
        fwd_ob.append(fm - sn[5])

    def mean(xs):
        return statistics.mean(xs) if len(xs) >= 5 else None

    (OUT / "r4_p2_burst_m01_m22_vev_fwd5300_k20.json").write_text(
        json.dumps(
            {
                "n_m01_m22_burst": len(fwd_b),
                "mean_fwd5300": mean(fwd_b),
                "n_other_burst": len(fwd_ob),
                "mean_fwd5300_other_burst": mean(fwd_ob),
            },
            indent=2,
        )
    )

    # Joint tight × M01-M22 burst on extract fwd20
    fwd_ext_tight = []
    fwd_ext_wide = []
    for d, ts in m01_m22_burst:
        jt = joint_tight_at(pr, d, ts)
        if jt is None:
            continue
        ser = pr[d][U]
        sn = snap_row(ser, ts)
        if not sn:
            continue
        fm = fwd_mid(ser, ts, 20)
        if fm is None:
            continue
        dlt = fm - sn[5]
        if jt:
            fwd_ext_tight.append(dlt)
        else:
            fwd_ext_wide.append(dlt)

    (OUT / "r4_p2_joint_gate_m01m22burst_extract_fwd20.json").write_text(
        json.dumps(
            {
                "tight_n": len(fwd_ext_tight),
                "tight_mean": mean(fwd_ext_tight),
                "wide_n": len(fwd_ext_wide),
                "wide_mean": mean(fwd_ext_wide),
            },
            indent=2,
        )
    )

    # --- 2) Microprice vs mid by spread bin (extract), pooled days
    mp_res = []
    for d in DAYS:
        for row in pr[d][U]:
            ts, _, _, _, _, mid = row
            if not row[1] or not row[3]:
                continue
            sp = row[3][0] - row[1][0]
            mp = microprice(row)
            if mp is None:
                continue
            fm = fwd_mid(pr[d][U], ts, 10)
            if fm is None:
                continue
            mp_res.append((sp, abs(mp - mid), fm - mid))

    by_sp = defaultdict(list)
    for sp, imp, fd in mp_res:
        b = 0 if sp <= 2 else (1 if sp <= 5 else 2)
        by_sp[b].append((imp, fd))

    micro_out = {}
    for b, xs in by_sp.items():
        imps = [a for a, _ in xs]
        fds = [c for _, c in xs]
        micro_out[str(b)] = {
            "spread_bin": ["<=2", "3-5", ">=6"][b],
            "n": len(xs),
            "mean_abs_mp_minus_mid": mean(imps) if imps else None,
            "mean_fwd10_mid": mean(fds) if fds else None,
        }
    (OUT / "r4_p2_extract_microprice_spread_fwd10.json").write_text(
        json.dumps(micro_out, indent=2)
    )

    # --- 3) Signed flow lead-lag: per tick net buy notional on extract from trades
    # correlate with fwd extract mid change same tick (contemp) and +5,+20 lags as simple vector
    net_by_d_ts = defaultdict(float)
    for tr in all_tr:
        if tr["sym"] != U:
            continue
        k = (tr["day"], tr["ts"])
        q = tr["price"] * tr["qty"]
        if tr["buyer"] and tr["seller"]:
            net_by_d_ts[k] += q  # crude signed notional: buys add (assume buyer initiated - not perfect)

    # Better: use aggressor from tape price vs snap
    net2 = defaultdict(float)
    for tr in all_tr:
        if tr["sym"] != U:
            continue
        d, ts = tr["day"], tr["ts"]
        ser = pr[d][U]
        sn = snap_row(ser, ts)
        if not sn:
            continue
        px = int(round(tr["price"]))
        q = tr["price"] * tr["qty"]
        if sn[3] and px >= sn[3][0]:
            net2[(d, ts)] += abs(q)
        elif sn[1] and px <= sn[1][0]:
            net2[(d, ts)] -= abs(q)

    # correlate net2 with fwd_mid(ts,20)-mid(ts) across ticks that have trade
    xs_n = []
    ys_fd = []
    for (d, ts), nv in net2.items():
        ser = pr[d][U]
        sn = snap_row(ser, ts)
        if not sn:
            continue
        fm = fwd_mid(ser, ts, 20)
        if fm is None:
            continue
        xs_n.append(nv)
        ys_fd.append(fm - sn[5])
    corr = None
    if len(xs_n) > 30:
        mx = statistics.mean(xs_n)
        my = statistics.mean(ys_fd)
        vx = statistics.pstdev(xs_n)
        vy = statistics.pstdev(ys_fd)
        if vx > 1e-9 and vy > 1e-9:
            corr = sum((a - mx) * (b - my) for a, b in zip(xs_n, ys_fd)) / (len(xs_n) * vx * vy)

    (OUT / "r4_p2_extract_signed_flow_fwd20_corr.json").write_text(
        json.dumps(
            {
                "n_ticks_with_extract_trade": len(xs_n),
                "corr_net_aggr_flow_vs_fwd20_mid": corr,
                "note": "net = +|notional| if trade price>=ask else -|notional| if price<=bid",
            },
            indent=2,
        )
    )

    burst_pairs = sorted([d, t] for d, t in m01_m22_burst)
    (OUT / "r4_p2_m01_m22_burst_pairs.json").write_text(json.dumps(burst_pairs, indent=2))

    meta = {"outputs": sorted(p.name for p in OUT.glob("r4_p2_*"))}
    (OUT / "r4_phase2_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
