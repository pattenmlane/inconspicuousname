"""
Tape-only: inner-join VEV_5200 and VEV_5300 on (day, timestamp); joint_tight_TH = both spr<=TH.

Compare TH in {1, 2}: frequency of joint rows and mean extract mid fwd_20 (K=20 next price rows)
conditional on joint (same definition as Phase 3 panel).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
DAYS = [1, 2, 3]
EXTRACT = "VELVETFRUIT_EXTRACT"
V5200, V5300 = "VEV_5200", "VEV_5300"


def inner_sprices() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        ddf = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1"],
        )
        a = ddf[ddf["product"] == V5200].drop_duplicates("timestamp", keep="first")
        b = ddf[ddf["product"] == V5300].drop_duplicates("timestamp", keep="first")
        bid1 = pd.to_numeric(a["bid_price_1"], errors="coerce")
        ask1 = pd.to_numeric(a["ask_price_1"], errors="coerce")
        bid2 = pd.to_numeric(b["bid_price_1"], errors="coerce")
        ask2 = pd.to_numeric(b["ask_price_1"], errors="coerce")
        a = a.assign(
            s5200=(ask1 - bid1).astype(float),
            day=ddf["day"].iloc[0] if "day" in ddf.columns and len(ddf) else int(day),
        )
        b = b.assign(s5300=(ask2 - bid2).astype(float))
        if "day" not in a.columns:
            a["day"] = int(day)
        if "day" not in b.columns:
            b["day"] = int(day)
        j = a[["day", "timestamp", "s5200"]].merge(
            b[["day", "timestamp", "s5300"]], on=["day", "timestamp"], how="inner"
        )
        rows.append(j)
    return pd.concat(rows, ignore_index=True)


def extract_fwd20() -> pd.DataFrame:
    u_list = []
    for day in DAYS:
        ddf = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "mid_price"],
        )
        u = ddf[ddf["product"] == EXTRACT].drop_duplicates("timestamp", keep="first")
        u_list.append(u)
    u = pd.concat(u_list, ignore_index=True)
    u = u.sort_values(["day", "timestamp"])
    u["mid"] = pd.to_numeric(u["mid_price"], errors="coerce")
    u["fwd_20"] = u.groupby("day")["mid"].transform(lambda s: s.shift(-20) - s)
    return u[["day", "timestamp", "fwd_20"]]


def main() -> None:
    m = inner_sprices().merge(extract_fwd20(), on=["day", "timestamp"], how="left")
    by = []
    for th in (1, 2):
        joint = (m["s5200"] <= th) & (m["s5300"] <= th)
        m2 = m.assign(joint=joint)
        for d, g in m2.groupby("day"):
            jm = g["joint"]
            sub = g.loc[g["joint"] & g["fwd_20"].notna(), "fwd_20"]
            by.append(
                {
                    "day": int(d),
                    "th": th,
                    "n_ts": len(g),
                    "p_joint": float(jm.mean()) if len(g) else 0.0,
                    "n_joint": int(jm.sum()),
                    "mean_ext_fwd20_when_joint": float(sub.mean()) if len(sub) else float("nan"),
                }
            )
    pd.DataFrame(by).sort_values(["th", "day"]).to_csv(OUT / "r4_sonic_gate_threshold_by_day.csv", index=False)
    pool = []
    for th in (1, 2):
        joint = (m["s5200"] <= th) & (m["s5300"] <= th)
        sub = m.loc[joint & m["fwd_20"].notna(), "fwd_20"]
        pool.append(
            {
                "th": th,
                "n_ts": len(m),
                "p_joint": float(joint.mean()),
                "n_joint": int(joint.sum()),
                "mean_ext_fwd20_when_joint": float(sub.mean()) if len(sub) else float("nan"),
            }
        )
    pd.DataFrame(pool).to_csv(OUT / "r4_sonic_gate_joint_pooled_by_th.csv", index=False)
    print("wrote", OUT / "r4_sonic_gate_joint_pooled_by_th.csv")


if __name__ == "__main__":
    main()
