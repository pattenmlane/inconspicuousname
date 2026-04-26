#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic **joint tight gate** on Round 4 tapes (same convention as
round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py):
inner-join timestamps for VEV_5200, VEV_5300, VELVETFRUIT_EXTRACT; **tight** =
(s5200 <= TH) & (s5300 <= TH), TH=2.

Outputs (manual_traders/R4/.../analysis_outputs/):
  r4_phase3_joint_panel_sample.csv          (first 5000 rows pooled for audit)
  r4_phase3_sonic_forward_extract_k20.csv  (Welch-style means tight vs not, R4)
  r4_phase3_spread_spread_corr_by_day.csv  (inclineGod: corr(s5200,s5300), vs s_ext)
  r4_phase3_spread_price_diffcorr.csv      (corr of 1-step diffs: dSpread vs dMid)
  r4_phase3_pair_markout_by_gate.csv       (Mark01→22 VEV_5300 etc.: tight vs wide)
  r4_phase3_mark67_extract_by_gate.csv
  r4_phase3_three_way_mark01_22_vev5300.csv (explicit 3-way table)
  r4_phase3_run_log.txt
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K_FWD = 20
KS = (5, 20, 100)

VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"


def load_prices_long() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        if "day" not in df.columns:
            df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def one_product(px: pd.DataFrame, day: int, prod: str) -> pd.DataFrame:
    v = px[(px["day"] == day) & (px["product"] == prod)].copy()
    v = v.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "day": day,
            "timestamp": v["timestamp"].astype(int),
            "spread": (ask - bid).astype(float),
            "mid": mid.astype(float),
        }
    )


def aligned_panel(px: pd.DataFrame, day: int) -> pd.DataFrame:
    a = one_product(px, day, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(px, day, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(px, day, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on=["day", "timestamp"], how="inner").merge(
        e[["day", "timestamp", "s_ext", "m_ext"]], on=["day", "timestamp"], how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["m_ext_f20"] = m["m_ext"].shift(-K_FWD)
    m["fwd20_ext"] = m["m_ext_f20"] - m["m_ext"]
    return m


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            t = pd.read_csv(p, sep=";")
            t["day"] = d
            frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    return tr


def prep_mid_lookup(px: pd.DataFrame) -> dict[tuple[int, int, str], float]:
    px = px.copy()
    px["mid"] = pd.to_numeric(px["mid_price"], errors="coerce")
    lk: dict[tuple[int, int, str], float] = {}
    for _, r in px.iterrows():
        lk[(int(r["day"]), int(r["timestamp"]), str(r["product"]))] = float(r["mid"])
    return lk


def ts_sorted_by_day(px: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for d in DAYS:
        ts = np.sort(px[px["day"] == d]["timestamp"].unique())
        out[d] = ts
    return out


def fwd_mid(mid_lk: dict, ts_sort: dict, d: int, ts: int, sym: str, k: int) -> float | None:
    tsu = ts_sort[d]
    i = bisect.bisect_left(tsu, ts)
    if i >= len(tsu) or tsu[i] != ts:
        return None
    j = i + k
    if j >= len(tsu):
        return None
    t2 = int(tsu[j])
    a = mid_lk.get((d, ts, sym))
    b = mid_lk.get((d, t2, sym))
    if a is None or b is None or (not math.isfinite(a)) or (not math.isfinite(b)):
        return None
    return float(b - a)


def tstat_mean_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"),) * 4
    tt = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(a.mean()), float(b.mean()), float(tt.statistic), float(tt.pvalue)


def main() -> int:
    log: list[str] = []

    def ln(s: str) -> None:
        print(s)
        log.append(s)

    px = load_prices_long()
    panels = [aligned_panel(px, d) for d in DAYS]
    panel = pd.concat(panels, ignore_index=True)
    panel.head(5000).to_csv(OUT / "r4_phase3_joint_panel_sample.csv", index=False)
    ln("Wrote r4_phase3_joint_panel_sample.csv")

    # --- Sonic forward extract K=20 on R4 ---
    rows = []
    for d in DAYS:
        sub = panel[panel["day"] == d]
        t = sub.loc[sub["tight"], "fwd20_ext"].dropna().values
        n = sub.loc[~sub["tight"], "fwd20_ext"].dropna().values
        mt, mn, tst, pv = tstat_mean_diff(t, n)
        rows.append(
            {
                "day": d,
                "p_tight": float(sub["tight"].mean()),
                "n_tight": int(np.isfinite(sub.loc[sub["tight"], "fwd20_ext"]).sum()),
                "n_wide": int(np.isfinite(sub.loc[~sub["tight"], "fwd20_ext"]).sum()),
                "mean_fwd20_tight": mt,
                "mean_fwd20_not": mn,
                "welch_t": tst,
                "p_value": pv,
            }
        )
    sonic = pd.DataFrame(rows)
    sonic.to_csv(OUT / "r4_phase3_sonic_forward_extract_k20.csv", index=False)
    ln(f"Sonic K=20 extract forward (R4):\n{sonic}")

    # --- inclineGod spread-spread and spread vs price (diff corr) ---
    ss_rows = []
    for d in DAYS:
        sub = panel[panel["day"] == d].dropna(subset=["s5200", "s5300", "s_ext"])
        if len(sub) < 50:
            continue
        ss_rows.append(
            {
                "day": d,
                "corr_s5200_s5300": float(sub["s5200"].corr(sub["s5300"])),
                "corr_s5200_s_ext": float(sub["s5200"].corr(sub["s_ext"])),
                "corr_s5300_s_ext": float(sub["s5300"].corr(sub["s_ext"])),
                "n": len(sub),
            }
        )
    pd.DataFrame(ss_rows).to_csv(OUT / "r4_phase3_spread_spread_corr_by_day.csv", index=False)
    ln("Wrote r4_phase3_spread_spread_corr_by_day.csv")

    diff_rows = []
    for d in DAYS:
        sub = panel[panel["day"] == d].copy()
        sub["ds5200"] = sub["s5200"].diff()
        sub["ds5300"] = sub["s5300"].diff()
        sub["dse"] = sub["s_ext"].diff()
        sub["dm"] = sub["m_ext"].diff()
        sub = sub.dropna()
        for label, mask in [("all", slice(None)), ("tight_only", sub["tight"]), ("wide_only", ~sub["tight"])]:
            g = sub.loc[mask] if label != "all" else sub
            if len(g) < 30:
                continue
            diff_rows.append(
                {
                    "day": d,
                    "cohort": label,
                    "n": len(g),
                    "corr_ds5200_dm": float(g["ds5200"].corr(g["dm"])),
                    "corr_ds5300_dm": float(g["ds5300"].corr(g["dm"])),
                    "corr_dse_dm": float(g["dse"].corr(g["dm"])),
                }
            )
    pd.DataFrame(diff_rows).to_csv(OUT / "r4_phase3_spread_price_diffcorr.csv", index=False)
    ln("Wrote r4_phase3_spread_price_diffcorr.csv")

    # --- Merge trades with tight flag at print timestamp ---
    gate = panel[["day", "timestamp", "tight", "s5200", "s5300", "s_ext"]].copy()
    tr = load_trades()
    tr = tr.merge(gate, on=["day", "timestamp"], how="left")
    tr["tight_gate"] = tr["tight"].fillna(False).astype(bool)

    mid_lk = prep_mid_lookup(px)
    ts_sort = ts_sorted_by_day(px)

    def markout_rows(sym_filter: str | None = None, buyer=None, seller=None):
        out = []
        sub = tr
        if sym_filter:
            sub = sub[sub["symbol"] == sym_filter]
        if buyer:
            sub = sub[sub["buyer"] == buyer]
        if seller:
            sub = sub[sub["seller"] == seller]
        for _, r in sub.iterrows():
            d, ts, sym = int(r["day"]), int(r["timestamp"]), str(r["symbol"])
            tg = bool(r["tight_gate"])
            for K in KS:
                dm = fwd_mid(mid_lk, ts_sort, d, ts, sym, K)
                if dm is None:
                    continue
                de = fwd_mid(mid_lk, ts_sort, d, ts, EXTRACT, K)
                out.append(
                    {
                        "day": d,
                        "timestamp": ts,
                        "symbol": sym,
                        "buyer": str(r["buyer"]),
                        "seller": str(r["seller"]),
                        "tight_gate": tg,
                        "K": K,
                        "d_mid": dm,
                        "d_ext": de if de is not None else float("nan"),
                    }
                )
        return pd.DataFrame(out)

    m122 = markout_rows("VEV_5300", "Mark 01", "Mark 22")
    if not m122.empty:
        gsum = (
            m122.groupby(["tight_gate", "K"])["d_mid"]
            .agg(n="count", mean="mean")
            .reset_index()
        )
        gsum.to_csv(OUT / "r4_phase3_pair_markout_by_gate.csv", index=False)
        ln(f"Mark01→Mark22 VEV_5300 markout by gate:\n{gsum}")

        # Mark01→22 on VEV_5300 only occurs under **joint tight** in R4 days 1–3
        # (empirically: ~0 VEV_5300 prints with gate off). Document; skip invalid Welch.
        n_off_5300 = int(((~tr["tight_gate"]) & (tr["symbol"] == "VEV_5300")).sum())
        note530 = (
            "Mark01→Mark22 VEV_5300 prints are only observable under joint tight in this slice; "
            f"VEV_5300 trades with joint gate off: n={n_off_5300} (too few for tight-vs-off Welch on same strike)."
        )
        (OUT / "r4_phase3_mark01_22_vev5300_gate_note.txt").write_text(note530 + "\n", encoding="utf-8")
        pd.DataFrame([{"note": note530, "n_off_vev5300": n_off_5300}]).to_csv(
            OUT / "r4_phase3_three_way_mark01_22_vev5300.csv", index=False
        )
        ln(note530)

    # Mark 67 buyer extract aggressive (price >= ask): need row-level BBO from px
    bb_ask = px[px["product"] == EXTRACT].copy()
    bb_ask["bb"] = pd.to_numeric(bb_ask["bid_price_1"], errors="coerce")
    bb_ask["ba"] = pd.to_numeric(bb_ask["ask_price_1"], errors="coerce")
    ba_lk = {
        (int(r["day"]), int(r["timestamp"])): float(r["ba"])
        for _, r in bb_ask.iterrows()
        if pd.notna(r["ba"])
    }

    m67 = []
    for _, r in tr[(tr["buyer"] == "Mark 67") & (tr["symbol"] == EXTRACT)].iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        ba = ba_lk.get((d, ts))
        if ba is None or float(r["price"]) < ba:
            continue
        tg = bool(r["tight_gate"])
        for K in (5,):
            dm = fwd_mid(mid_lk, ts_sort, d, ts, EXTRACT, K)
            if dm is None:
                continue
            m67.append({"tight_gate": tg, "K": K, "d_mid": dm, "day": d})
    if m67:
        m67df = pd.DataFrame(m67)
        m67df.groupby(["tight_gate", "day"])["d_mid"].agg(["count", "mean"]).reset_index().to_csv(
            OUT / "r4_phase3_mark67_extract_by_gate.csv", index=False
        )
        ln(
            "Mark67 extract aggressive K=5 by gate/day:\n"
            + str(m67df.groupby(["tight_gate", "day"])["d_mid"].agg(["count", "mean"]))
        )
        a = m67df.loc[m67df["tight_gate"], "d_mid"].values.astype(float)
        b = m67df.loc[~m67df["tight_gate"], "d_mid"].values.astype(float)
        mt, mn, tst, pv = tstat_mean_diff(a, b)
        pd.DataFrame(
            [
                {
                    "K": 5,
                    "n_tight": int(np.isfinite(a).sum()),
                    "n_joint_off": int(np.isfinite(b).sum()),
                    "mean_tight": mt,
                    "mean_joint_off": mn,
                    "welch_t": tst,
                    "p_value": pv,
                }
            ]
        ).to_csv(OUT / "r4_phase3_mark67_extract_gate_welch_k5.csv", index=False)
        ln(f"Mark67 aggressive extract K=5: tight vs joint-off Welch → mean_tight={mt}, mean_off={mn}, p={pv}")

    # Mark01→22 on VEV_5500: only strike with **any** joint-gate-off prints in R4 slice (day 3 has 1).
    m550 = markout_rows("VEV_5500", "Mark 01", "Mark 22")
    if not m550.empty and m550["tight_gate"].nunique() > 1:
        g550 = (
            m550.groupby(["tight_gate", "K"])["d_mid"]
            .agg(n="count", mean="mean")
            .reset_index()
        )
        g550.to_csv(OUT / "r4_phase3_mark01_22_vev5500_by_gate.csv", index=False)
        ln(f"Mark01→Mark22 VEV_5500 by gate:\n{g550}")
        w550 = []
        for K in KS:
            a = m550[(m550["tight_gate"]) & (m550["K"] == K)]["d_mid"].values.astype(float)
            b = m550[(~m550["tight_gate"]) & (m550["K"] == K)]["d_mid"].values.astype(float)
            na, nb = int(np.isfinite(a).sum()), int(np.isfinite(b).sum())
            if na < 2 or nb < 2:
                w550.append(
                    {
                        "K": K,
                        "n_tight": na,
                        "n_off": nb,
                        "mean_tight": float("nan"),
                        "mean_off": float("nan"),
                        "welch_t": float("nan"),
                        "p_value": float("nan"),
                        "note": "insufficient_n_for_welch",
                    }
                )
                continue
            mt, mn, tst, pv = tstat_mean_diff(a, b)
            w550.append(
                {
                    "K": K,
                    "n_tight": na,
                    "n_off": nb,
                    "mean_tight": mt,
                    "mean_off": mn,
                    "welch_t": tst,
                    "p_value": pv,
                    "note": "",
                }
            )
        pd.DataFrame(w550).to_csv(OUT / "r4_phase3_mark01_22_vev5500_gate_welch.csv", index=False)

    (OUT / "r4_phase3_run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'r4_phase3_run_log.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
