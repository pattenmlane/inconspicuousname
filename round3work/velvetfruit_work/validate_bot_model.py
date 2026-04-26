"""
Reconstruct the VELVETFRUIT_EXTRACT order book from the true FV stream alone,
using the calibrated bot models, and measure accuracy vs the real book.

Bot A: bid = floor(FV - 0.1) - 2,  ask = ceil(FV + 0.1) + 2
Bot B: bid = floor(FV) - 3,         ask = ceil(FV) + 3
       Presence depends on frac(FV) — see bot_calibration.txt for full pattern.

Run from repo root:
  python3 round3work/velvetfruit_work/validate_bot_model.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("round3work/fairs/VELVETFRUIT_EXTRACTfair/364578")

# ── Load data ─────────────────────────────────────────────────────────────────
fv   = pd.read_csv(BASE / "VELVETFRUIT_EXTRACT_true_fv_day39.csv", sep=";")
book = pd.read_csv(BASE / "prices_round_3_day_39.csv", sep=";")
df   = book.merge(fv[["timestamp", "true_fv"]], on="timestamp", how="left")
df["frac_fv"] = df["true_fv"] % 1
N = len(df)
print(f"Ticks: {N}\n")


# ── Bot models ────────────────────────────────────────────────────────────────
def bot_a_bid(fv: float) -> int:
    return int(np.floor(fv - 0.1)) - 2

def bot_a_ask(fv: float) -> int:
    return int(np.ceil(fv + 0.1)) + 2

def bot_b_bid(fv: float) -> int:
    return int(np.floor(fv)) - 3

def bot_b_ask(fv: float) -> int:
    return int(np.ceil(fv)) + 3


# ── Apply models ──────────────────────────────────────────────────────────────
df["pred_bid_a"] = df["true_fv"].apply(bot_a_bid)
df["pred_ask_a"] = df["true_fv"].apply(bot_a_ask)
df["pred_bid_b"] = df["true_fv"].apply(bot_b_bid)
df["pred_ask_b"] = df["true_fv"].apply(bot_b_ask)

# Bot A ticks: offset ∈ {-2, -3} for bid, {+2, +3} for ask
bid1_off = (df["bid_price_1"] - df["true_fv"]).round(0)
ask1_off = (df["ask_price_1"] - df["true_fv"]).round(0)
bot_a_bid_mask = bid1_off.isin([-2.0, -3.0])
bot_a_ask_mask = ask1_off.isin([2.0, 3.0])


# ── Accuracy helpers ──────────────────────────────────────────────────────────
def match_rate(pred: pd.Series, actual: pd.Series, mask: pd.Series, label: str) -> float:
    valid = mask & actual.notna()
    matches = (pred[valid] == actual[valid]).sum()
    total = valid.sum()
    pct = matches / total * 100
    print(f"  {label:35s}: {matches:4d}/{total}  ({pct:.1f}%)")
    return pct

def offset_stats(pred: pd.Series, actual: pd.Series, label: str) -> None:
    valid = actual.notna()
    err = actual[valid] - pred[valid]
    print(f"  {label:35s}: mean_err={err.mean():+.3f}  std={err.std():.3f}  "
          f"max_abs={err.abs().max():.0f}")


# ── Bot A validation ──────────────────────────────────────────────────────────
print("=" * 65)
print("BOT A  (level 1, formula: floor(FV-0.1)-2 / ceil(FV+0.1)+2)")
print("=" * 65)

print("\nPresence (ticks where offset ∈ {-2,-3}):")
print(f"  bid1: {bot_a_bid_mask.sum()}/{N}  ask1: {bot_a_ask_mask.sum()}/{N}")

print("\nPrice match on Bot A ticks:")
match_rate(df["pred_bid_a"], df["bid_price_1"], bot_a_bid_mask, "bid1 (Bot A ticks)")
match_rate(df["pred_ask_a"], df["ask_price_1"], bot_a_ask_mask, "ask1 (Bot A ticks)")

print("\nResidual errors (all ticks):")
offset_stats(df["pred_bid_a"], df["bid_price_1"], "bid1 error")
offset_stats(df["pred_ask_a"], df["ask_price_1"], "ask1 error")

bid1_err = (df["bid_price_1"] - df["pred_bid_a"]).dropna()
ask1_err = (df["ask_price_1"] - df["pred_ask_a"]).dropna()
print(f"\nBid1 error distribution: {dict(bid1_err.round(0).value_counts().sort_index())}")
print(f"Ask1 error distribution: {dict(ask1_err.round(0).value_counts().sort_index())}")

# Failure frac analysis
both_a = (df["pred_bid_a"] == df["bid_price_1"]) & (df["pred_ask_a"] == df["ask_price_1"])
wrong_a = df[~both_a].copy()
wrong_a["frac_fv"] = wrong_a["true_fv"] % 1
print(f"\nLevel 1 failures: {(~both_a).sum()} ticks")
if len(wrong_a) > 0:
    print(f"  Frac(FV) at failure — mean={wrong_a.frac_fv.mean():.3f}  std={wrong_a.frac_fv.std():.3f}")
    near_boundary = (wrong_a.frac_fv < 0.12) | (wrong_a.frac_fv > 0.88)
    print(f"  Failures where frac(FV) near 0.1 or 0.9 boundary: {near_boundary.sum()} / {len(wrong_a)}")


# ── Bot B validation ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print("BOT B  (level 2, conditional presence by frac(FV))")
print("=" * 65)

# Use ticks where Bot B should be at level 2 AND prices differ from Bot A
bid_coincide = df["pred_bid_a"] == df["pred_bid_b"]
ask_coincide = df["pred_ask_a"] == df["pred_ask_b"]

c3_bid_mask = ~bid_coincide & df["bid_price_2"].notna()
c3_ask_mask = ~ask_coincide & df["ask_price_2"].notna()

print(f"\nBot B ticks (differ from Bot A, level 2 present):")
print(f"  bid: {c3_bid_mask.sum()} ticks, ask: {c3_ask_mask.sum()} ticks")

print("\nPrice match (when Bot B at level 2, prices differ from Bot A):")
match_rate(df["pred_bid_b"], df["bid_price_2"], c3_bid_mask, "bid2 (Bot B at level 2)")
match_rate(df["pred_ask_b"], df["ask_price_2"], c3_ask_mask, "ask2 (Bot B at level 2)")

bid2_err = (df.loc[c3_bid_mask, "bid_price_2"] - df.loc[c3_bid_mask, "pred_bid_b"])
ask2_err = (df.loc[c3_ask_mask, "ask_price_2"] - df.loc[c3_ask_mask, "pred_ask_b"])
print(f"\nBid2 error distribution: {dict(bid2_err.round(0).value_counts().sort_index())}")
print(f"Ask2 error distribution: {dict(ask2_err.round(0).value_counts().sort_index())}")

# Presence prediction by frac rule
frac = df["frac_fv"]
pred_bid2_present = ~bid_coincide  # Bot B bid at level 2 when prices differ
pred_ask2_present = ~ask_coincide  # Bot B ask at level 2 when prices differ
actual_bid2_present = df["bid_price_2"].notna()
actual_ask2_present = df["ask_price_2"].notna()

print(f"\nBot B presence prediction (prices differ → at level 2):")
bid2_p = (pred_bid2_present == actual_bid2_present).mean()
ask2_p = (pred_ask2_present == actual_ask2_present).mean()
print(f"  bid2 presence accuracy: {bid2_p*100:.1f}%")
print(f"  ask2 presence accuracy: {ask2_p*100:.1f}%")


# ── Level 1 volume bimodality ─────────────────────────────────────────────────
print()
print("=" * 65)
print("LEVEL 1 VOLUME BIMODALITY (Bot A+B merging)")
print("=" * 65)

bot_b_at_l1_bid = bid_coincide   # Bot B merged at bid level 1
bot_b_at_l1_ask = ask_coincide   # Bot B merged at ask level 1

v_merged = df.loc[bot_b_at_l1_bid & ~df["bid_price_2"].isna() == False, "bid_volume_1"].dropna()
# Simplify: just use the coincide mask
v_merged = df.loc[bot_b_at_l1_bid, "bid_volume_1"].dropna()
v_alone  = df.loc[~bot_b_at_l1_bid, "bid_volume_1"].dropna()

print(f"\nBot A+B merged at level 1 (coincide): {bot_b_at_l1_bid.sum()} ticks")
if len(v_merged) > 0:
    print(f"  vol: mean={v_merged.mean():.1f}, range=[{v_merged.min():.0f},{v_merged.max():.0f}]")
print(f"\nBot A alone at level 1 (differ): {(~bot_b_at_l1_bid).sum()} ticks")
if len(v_alone) > 0:
    print(f"  vol: mean={v_alone.mean():.1f}, range=[{v_alone.min():.0f},{v_alone.max():.0f}]")


# ── Wall mid accuracy ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("WALL MID ACCURACY (FV proxy)")
print("=" * 65)

bid_wall = df[["bid_price_1", "bid_price_2"]].min(axis=1)
ask_wall = df[["ask_price_1", "ask_price_2"]].max(axis=1)
df["wall_mid"] = (bid_wall + ask_wall) / 2.0
wm_err = df["wall_mid"] - df["true_fv"]
print(f"\n  wall_mid error: mean={wm_err.mean():+.4f}  std={wm_err.std():.4f}")
print(f"  (Compare hydrogel wall_mid std=0.35)")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)

ba_bid_acc = (df.loc[bot_a_bid_mask, "pred_bid_a"] == df.loc[bot_a_bid_mask, "bid_price_1"]).mean()
ba_ask_acc = (df.loc[bot_a_ask_mask, "pred_ask_a"] == df.loc[bot_a_ask_mask, "ask_price_1"]).mean()
bb_bid_acc = (df.loc[c3_bid_mask, "pred_bid_b"] == df.loc[c3_bid_mask, "bid_price_2"]).mean() if c3_bid_mask.sum() > 0 else 0.0
bb_ask_acc = (df.loc[c3_ask_mask, "pred_ask_b"] == df.loc[c3_ask_mask, "ask_price_2"]).mean() if c3_ask_mask.sum() > 0 else 0.0

print(f"""
  Bot A price accuracy  : bid {ba_bid_acc*100:.1f}%  ask {ba_ask_acc*100:.1f}%
  Bot B price accuracy  : bid {bb_bid_acc*100:.1f}%  ask {bb_ask_acc*100:.1f}%  (when at level 2)
  Wall mid std from FV  : {wm_err.std():.4f}

  Bot A: floor(FV-0.1)-2 / ceil(FV+0.1)+2  (boundary-aware rounding at frac ≈ 0.1/0.9)
  Bot B: floor(FV)-3 / ceil(FV)+3  (conditional presence by frac, single-sided mostly)

  Remaining errors are almost entirely at frac(FV) ≈ 0.1 or 0.9 boundary.
""")
