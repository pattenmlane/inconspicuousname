"""
Round 4 Phase 3 — Sonic joint gate on R4 + inclineGod spread–spread + gate×counterparty.

Gate: s5200<=TH and s5300<=TH at same (day, timestamp), inner-join 5200/5300/extract like
round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py (plus day column for R4).

Run: python3 manual_traders/R4/r3v_smile_sticky_slow_16_round4/analyze_r4_phase3_sonic_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

TH = 2
K = 20
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"


def days() -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in DATA.glob("prices_round_4_day_*.csv"))


def one_product(df: pd.DataFrame, prod: str) -> pd.DataFrame:
    v = (
        df[df["product"] == prod]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(
        spread=(ask - bid).astype(float),
        mid=mid,
        bb=bid,
        ba=ask,
    )[["day", "timestamp", "spread", "mid", "bb", "ba"]].copy()


def aligned_panel(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    a = one_product(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(df, EX).rename(columns={"spread": "s_ext", "mid": "m_ext", "bb": "bb_ex", "ba": "ba_ex"})
    m = a.merge(b, on=["day", "timestamp"], how="inner").merge(
        e[["day", "timestamp", "m_ext", "s_ext", "bb_ex", "ba_ex"]], on=["day", "timestamp"], how="inner"
    )
    return m.sort_values(["day", "timestamp"]).reset_index(drop=True)


def add_fwd(m: pd.DataFrame, k: int = K) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= TH) & (out["s5300"] <= TH)
    out["m_ext_f"] = out.groupby("day")["m_ext"].shift(-k)
    out["fwd_k"] = out["m_ext_f"] - out["m_ext"]
    return out


def load_px_mid_bb_ba() -> pd.DataFrame:
    frames = []
    for d in days():
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        df["day"] = int(d)
        bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
        mid = pd.to_numeric(df["mid_price"], errors="coerce")
        frames.append(
            df.assign(mid=mid, bb=bid, ba=ask)[["day", "timestamp", "product", "mid", "bb", "ba"]]
        )
    return pd.concat(frames, ignore_index=True)


def mid_fwd(px: pd.DataFrame, day: int, ts: int, sym: str, k: int) -> float | None:
    g = px[(px["day"] == day) & (px["product"] == sym)].sort_values("timestamp")
    if g.empty:
        return None
    tss = g["timestamp"].astype(int).to_numpy()
    mids = g["mid"].astype(float).to_numpy()
    p = int(np.searchsorted(tss, ts, side="left"))
    if p >= len(tss) or tss[p] != ts:
        return None
    j = p + k
    if j >= len(tss):
        return None
    return float(mids[j] - mids[p])


def main() -> None:
    panels = [add_fwd(aligned_panel(d)) for d in days()]
    pan = pd.concat(panels, ignore_index=True)
    pan_valid = pan[np.isfinite(pan["fwd_k"])]

    tight = pan_valid[pan_valid["tight"]]["fwd_k"].to_numpy()
    loose = pan_valid[~pan_valid["tight"]]["fwd_k"].to_numpy()
    tt = stats.ttest_ind(tight, loose, equal_var=False, nan_policy="omit") if len(tight) > 10 and len(loose) > 10 else None

    by_day = {}
    for d in days():
        sub = pan_valid[pan_valid["day"] == d]
        t = sub[sub["tight"]]["fwd_k"].to_numpy()
        l = sub[~sub["tight"]]["fwd_k"].to_numpy()
        by_day[str(d)] = {
            "n_tight": int(len(t)),
            "n_loose": int(len(l)),
            "mean_tight": float(np.mean(t)) if len(t) else None,
            "mean_loose": float(np.mean(l)) if len(l) else None,
        }

    (OUT / "r4_phase3_gate_fwd_extract.json").write_text(
        json.dumps(
            {
                "TH": TH,
                "K": K,
                "pooled": {
                    "n_tight": int(len(tight)),
                    "n_loose": int(len(loose)),
                    "mean_fwd_tight": float(np.mean(tight)) if len(tight) else None,
                    "mean_fwd_loose": float(np.mean(loose)) if len(loose) else None,
                    "welch_t": float(tt.statistic) if tt else None,
                    "p": float(tt.pvalue) if tt else None,
                },
                "by_day": by_day,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pan["dm_ext"] = pan.groupby("day")["m_ext"].diff().abs()
    corr_ss = pan[["s5200", "s5300", "s_ext"]].corr().round(4).to_dict()
    corr_sp = {
        "corr_s5200_abs_dm_ext": float(pan["s5200"].corr(pan["dm_ext"])) if pan["dm_ext"].notna().any() else None,
        "corr_s5300_abs_dm_ext": float(pan["s5300"].corr(pan["dm_ext"])) if pan["dm_ext"].notna().any() else None,
        "corr_s_ext_abs_dm_ext": float(pan["s_ext"].corr(pan["dm_ext"])) if pan["dm_ext"].notna().any() else None,
    }
    (OUT / "r4_phase3_spread_spread_matrix.json").write_text(
        json.dumps({"corr_matrix_s5200_s5300_s_ext": corr_ss, "spread_vs_abs_price_move_extract": corr_sp}, indent=2),
        encoding="utf-8",
    )

    px = load_px_mid_bb_ba()
    gate_map = pan.set_index(["day", "timestamp"])["tight"]

    def is_tight(d: int, ts: int) -> bool | None:
        key = (int(d), int(ts))
        if key not in gate_map.index:
            return None
        return bool(gate_map.loc[key])

    tr_frames = []
    for d in days():
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            t = pd.read_csv(p, sep=";")
            t["day"] = int(d)
            tr_frames.append(t)
    tr = pd.concat(tr_frames, ignore_index=True)

    burst_b = set()
    for (d, ts), g in tr.groupby(["day", "timestamp"]):
        if len(g) < 2:
            continue
        if not ((g["buyer"] == "Mark 01").all() and (g["seller"] == "Mark 22").all()):
            continue
        syms = set(g["symbol"].astype(str))
        if len({s for s in syms if s.startswith("VEV_")}) >= 2:
            burst_b.add((int(d), int(ts)))

    def burst_stats(rows: set[tuple[int, int]], tight_filter: bool | None) -> dict:
        vals = []
        for d, ts in rows:
            tg = is_tight(d, ts)
            if tight_filter is True and tg is not True:
                continue
            if tight_filter is False and tg is not False:
                continue
            v = mid_fwd(px, d, ts, EX, K)
            if v is not None:
                vals.append(v)
        arr = np.asarray(vals, float)
        return {"n": int(len(arr)), "mean": float(np.mean(arr)) if len(arr) else None}

    (OUT / "r4_phase3_typeB_burst_gated.json").write_text(
        json.dumps(
            {
                "n_burst_timestamps": len(burst_b),
                "extract_fwd20_all": burst_stats(burst_b, None),
                "extract_fwd20_tight_only": burst_stats(burst_b, True),
                "extract_fwd20_wide_only": burst_stats(burst_b, False),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mvev = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22") & (tr["symbol"].astype(str).str.startswith("VEV_"))]

    def print_fwd(sub: pd.DataFrame, tight_filter: bool | None) -> dict:
        vals = []
        for _, r in sub.iterrows():
            d, ts = int(r["day"]), int(r["timestamp"])
            sym = str(r["symbol"])
            tg = is_tight(d, ts)
            if tight_filter is True and tg is not True:
                continue
            if tight_filter is False and tg is not False:
                continue
            v = mid_fwd(px, d, ts, sym, K)
            if v is not None:
                vals.append(v)
        arr = np.asarray(vals, float)
        return {"n": int(len(arr)), "mean": float(np.mean(arr)) if len(arr) else None}

    (OUT / "r4_phase3_m01_m22_vev_gated.json").write_text(
        json.dumps(
            {
                "n_prints": len(mvev),
                "fwd20_same_symbol_all": print_fwd(mvev, None),
                "fwd20_same_symbol_tight": print_fwd(mvev, True),
                "fwd20_same_symbol_wide": print_fwd(mvev, False),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sub520 = mvev[mvev["symbol"] == VEV_5200]
    (OUT / "r4_phase3_three_way_mark01_m22_vev5200.json").write_text(
        json.dumps({"n": len(sub520), "fwd20_tight": print_fwd(sub520, True), "fwd20_wide": print_fwd(sub520, False)}, indent=2),
        encoding="utf-8",
    )

    ex_px = px[px["product"] == EX][["day", "timestamp", "bb", "ba"]]
    ex_tr = tr[tr["symbol"] == EX].merge(ex_px, on=["day", "timestamp"], how="inner")
    ex_tr = ex_tr.merge(pan[["day", "timestamp", "tight", "fwd_k"]], on=["day", "timestamp"], how="inner")
    rows = []
    for _, r in ex_tr.iterrows():
        bp, ap = r["bb"], r["ba"]
        if pd.isna(bp) or pd.isna(ap):
            continue
        pr, q = float(r["price"]), int(r["quantity"])
        if pr >= float(ap):
            sgn = q
        elif pr <= float(bp):
            sgn = -q
        else:
            continue
        rows.append({"tight": bool(r["tight"]), "sgn": sgn, "fwd_ex": float(r["fwd_k"])})
    sf = pd.DataFrame(rows)
    out_s = {}
    if len(sf) > 50:
        for lab, g in sf.groupby("tight"):
            arr = g["sgn"].to_numpy()
            nxt = g["fwd_ex"].to_numpy()
            if len(arr) > 30 and float(np.std(arr)) > 1e-9 and float(np.std(nxt)) > 1e-9:
                out_s[str(lab)] = {"n": int(len(arr)), "corr_sgn_fwd_ex": float(np.corrcoef(arr, nxt)[0, 1])}
    (OUT / "r4_phase3_signed_flow_gated.json").write_text(json.dumps(out_s, indent=2), encoding="utf-8")

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
