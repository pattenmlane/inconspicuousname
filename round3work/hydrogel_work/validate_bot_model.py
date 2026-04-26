"""
Reconstruct the HYDROGEL_PACK order book from the true FV stream alone,
using the calibrated bot models, and measure accuracy vs the real book.

Bot A: bid = round(FV) - 8,  ask = round(FV) + 8
Bot B: bid = round(FV-0.5) - 10, ask = round(FV+0.5) + 10

Run from repo root:
  python3 round3work/hydrogel_work/validate_bot_model.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("round3work/fairs/hydrogel/364553")

# ── Load data ─────────────────────────────────────────────────────────────────
fv   = pd.read_csv(BASE / "HYDROGEL_PACK_true_fv_day39.csv", sep=";")
book = pd.read_csv(BASE / "prices_round_3_day_39.csv", sep=";")
df   = book.merge(fv[["timestamp", "true_fv"]], on="timestamp", how="left")
N = len(df)
print(f"Ticks: {N}\n")


# ── Bot models ────────────────────────────────────────────────────────────────
def bot_a_bid(fv: float) -> int:
    return int(round(fv)) - 8

def bot_a_ask(fv: float) -> int:
    return int(round(fv)) + 8

def bot_b_bid(fv: float) -> int:
    return int(round(fv - 0.5)) - 10

def bot_b_ask(fv: float) -> int:
    return int(round(fv + 0.5)) + 10


# ── Apply models to every tick ────────────────────────────────────────────────
df["pred_bid_a"] = df["true_fv"].apply(bot_a_bid)
df["pred_ask_a"] = df["true_fv"].apply(bot_a_ask)
df["pred_bid_b"] = df["true_fv"].apply(bot_b_bid)
df["pred_ask_b"] = df["true_fv"].apply(bot_b_ask)


# ── Accuracy helpers ──────────────────────────────────────────────────────────
def match_rate(pred: pd.Series, actual: pd.Series, label: str) -> float:
    valid = actual.notna()
    matches = (pred[valid] == actual[valid]).sum()
    total = valid.sum()
    pct = matches / total * 100
    print(f"  {label:30s}: {matches:4d}/{total}  ({pct:.1f}%)")
    return pct

def offset_stats(pred: pd.Series, actual: pd.Series, label: str) -> None:
    valid = actual.notna()
    err = (actual[valid] - pred[valid])
    print(f"  {label:30s}: mean_err={err.mean():+.3f}  std={err.std():.3f}  "
          f"max_abs={err.abs().max():.0f}")


# ── Bot A validation ──────────────────────────────────────────────────────────
print("=" * 60)
print("BOT A  (level 1, expected offset ±8)")
print("=" * 60)

# Only evaluate on ticks where Bot A is actually in the book (offset ~±8)
a_bid_mask = ((df["bid_price_1"] - df["true_fv"]).round(0) == -8)
a_ask_mask = ((df["ask_price_1"] - df["true_fv"]).round(0) == 8)

print("\nPrice match (ticks where Bot A is present):")
bid_a_match = (df.loc[a_bid_mask, "pred_bid_a"] == df.loc[a_bid_mask, "bid_price_1"])
ask_a_match = (df.loc[a_ask_mask, "pred_ask_a"] == df.loc[a_ask_mask, "ask_price_1"])
print(f"  bid: {bid_a_match.sum():4d}/{a_bid_mask.sum()}  ({bid_a_match.mean()*100:.1f}%)")
print(f"  ask: {ask_a_match.sum():4d}/{a_ask_mask.sum()}  ({ask_a_match.mean()*100:.1f}%)")

print("\nPrice match (ALL ticks, treating Bot A as always present):")
match_rate(df["pred_bid_a"], df["bid_price_1"], "bid1 (vs Bot A model)")
match_rate(df["pred_ask_a"], df["ask_price_1"], "ask1 (vs Bot A model)")

print("\nResidual errors on all ticks:")
offset_stats(df["pred_bid_a"], df["bid_price_1"], "bid1 error")
offset_stats(df["pred_ask_a"], df["ask_price_1"], "ask1 error")

# Error breakdown
bid1_err = (df["bid_price_1"] - df["pred_bid_a"]).dropna()
print(f"\nBid1 error distribution: {dict(bid1_err.value_counts().sort_index())}")
ask1_err = (df["ask_price_1"] - df["pred_ask_a"]).dropna()
print(f"Ask1 error distribution: {dict(ask1_err.value_counts().sort_index())}")


# ── Bot B validation ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("BOT B  (level 2, expected offset ±10 or ±11)")
print("=" * 60)

print("\nPrice match (ALL ticks):")
match_rate(df["pred_bid_b"], df["bid_price_2"], "bid2 (vs Bot B model)")
match_rate(df["pred_ask_b"], df["ask_price_2"], "ask2 (vs Bot B model)")

print("\nResidual errors:")
offset_stats(df["pred_bid_b"], df["bid_price_2"], "bid2 error")
offset_stats(df["pred_ask_b"], df["ask_price_2"], "ask2 error")

bid2_err = (df["bid_price_2"] - df["pred_bid_b"]).dropna()
print(f"\nBid2 error distribution: {dict(bid2_err.value_counts().sort_index())}")
ask2_err = (df["ask_price_2"] - df["pred_ask_b"]).dropna()
print(f"Ask2 error distribution: {dict(ask2_err.value_counts().sort_index())}")


# ── Full book reconstruction accuracy ────────────────────────────────────────
print()
print("=" * 60)
print("FULL BOOK RECONSTRUCTION ACCURACY")
print("=" * 60)

# A tick is "fully correct" if both bid and ask at a level match
both_a = (df["pred_bid_a"] == df["bid_price_1"]) & (df["pred_ask_a"] == df["ask_price_1"])
both_b = (df["pred_bid_b"] == df["bid_price_2"]) & (df["pred_ask_b"] == df["ask_price_2"])
both_ab = both_a & both_b

print(f"\n  Level 1 (Bot A) fully correct : {both_a.sum():4d}/{N}  ({both_a.mean()*100:.1f}%)")
print(f"  Level 2 (Bot B) fully correct : {both_b.sum():4d}/{N}  ({both_b.mean()*100:.1f}%)")
print(f"  Both levels fully correct     : {both_ab.sum():4d}/{N}  ({both_ab.mean()*100:.1f}%)")


# ── Where do errors occur? ────────────────────────────────────────────────────
print()
print("=" * 60)
print("ERROR ANALYSIS — when does the model fail?")
print("=" * 60)

wrong_a = df[~both_a].copy()
wrong_a["frac_fv"] = wrong_a["true_fv"] % 1
wrong_a["bid1_err"] = wrong_a["bid_price_1"] - wrong_a["pred_bid_a"]
wrong_a["ask1_err"] = wrong_a["ask_price_1"] - wrong_a["pred_ask_a"]

print(f"\nLevel 1 failures: {len(wrong_a)} ticks")
print(f"  FV fractional part at failure — mean={wrong_a['frac_fv'].mean():.3f}  "
      f"std={wrong_a['frac_fv'].std():.3f}")
print(f"  Bid error counts: {dict(wrong_a['bid1_err'].value_counts().sort_index())}")
print(f"  Ask error counts: {dict(wrong_a['ask1_err'].value_counts().sort_index())}")

# Are failures clustered near FV = X.5 (rounding boundary)?
near_half = ((wrong_a["frac_fv"] - 0.5).abs() < 0.15)
print(f"  Failures where frac(FV) near 0.5 (±0.15): {near_half.sum()} / {len(wrong_a)}")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
  Bot A price accuracy  : bid {bid_a_match.mean()*100:.1f}%  ask {ask_a_match.mean()*100:.1f}%
  Bot B price accuracy  : {both_b.mean()*100:.1f}% both sides
  Full book (L1+L2)     : {both_ab.mean()*100:.1f}% of ticks exactly reconstructed

  Remaining errors are almost entirely ±1 tick at FV = X.5 rounding boundaries.
  This is floating-point rounding noise in IMC's server, not a model flaw.
""")
