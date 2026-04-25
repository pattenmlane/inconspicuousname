#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'analysis_outputs' / 'regime_shock_response.json'
STRIKES = (4500, 5000, 5100, 5200, 5300)


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(s: float, k: float, t: float, sigma: float) -> float:
    if t <= 1e-12 or sigma <= 1e-12:
        return max(s - k, 0.0)
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / v
    d2 = d1 - v
    return s * ncdf(d1) - k * ncdf(d2)

def implied_vol_bisect(price: float, s: float, k: float, t: float) -> float | None:
    intrinsic = max(s - k, 0.0)
    if price <= intrinsic + 1e-6 or price >= s - 1e-6 or s <= 0 or k <= 0 or t <= 1e-12:
        return None
    lo, hi = 1e-4, 12.0
    flo = bs_call_price(s, k, t, lo) - price
    fhi = bs_call_price(s, k, t, hi) - price
    if flo > 0 or fhi < 0:
        return None
    for _ in range(28):
        md = 0.5 * (lo + hi)
        if bs_call_price(s, k, t, md) >= price:
            hi = md
        else:
            lo = md
    return 0.5 * (lo + hi)

def dte(day: int, ts: int) -> float:
    return max((8.0 - float(day)) - ((int(ts) // 100) / 10000.0), 1e-6)

def regime_from_z(z: float) -> str:
    if z <= -0.8:
        return 'calm'
    if z >= 0.8:
        return 'stressed'
    return 'neutral'

def run_day(day: int) -> dict:
    df = pd.read_csv(REPO / 'Prosperity4Data' / 'ROUND_3' / f'prices_round_3_day_{day}.csv', sep=';')
    p = df.pivot_table(index='timestamp', columns='product', values='mid_price', aggfunc='first')
    s = p['VELVETFRUIT_EXTRACT'].astype(float)
    ds = s.diff()
    shock_thr = float(ds.abs().quantile(0.9))

    # build near-ATM IV median using 3 closest in STRIKES universe each ts
    iv_atm = []
    prev = None
    rv_var = None
    rv_ann = []
    idxs = list(s.index)
    for ts in idxs:
        sv = float(s.loc[ts])
        if prev is None:
            rv_ann.append(0.0)
            prev = sv
        else:
            r = math.log(sv / prev) if prev > 0 and sv > 0 else 0.0
            rv_var = (r * r) if rv_var is None else (1 - 0.06) * rv_var + 0.06 * (r * r)
            rv_ann.append(math.sqrt(max(rv_var, 0.0) * 10000.0))
            prev = sv

        tt = dte(day, int(ts)) / 365.0
        pairs = []
        for k in STRIKES:
            sym = f'VEV_{k}'
            if sym not in p.columns:
                continue
            vv = p.at[ts, sym]
            if pd.isna(vv):
                continue
            iv = implied_vol_bisect(float(vv), sv, float(k), tt)
            if iv is not None:
                pairs.append((abs(float(k) - sv), iv))
        pairs.sort(key=lambda x: x[0])
        vals = [iv for _, iv in pairs[:3]]
        iv_atm.append(float(sum(vals) / len(vals)) if vals else 0.25)

    m = pd.DataFrame({'s': s, 'ds': ds, 'rv_ann': rv_ann, 'iv_atm': iv_atm}, index=s.index)
    spread = m['rv_ann'] - m['iv_atm']
    mu = spread.ewm(alpha=0.01, adjust=False).mean()
    var = (spread - mu).pow(2).ewm(alpha=0.01, adjust=False).mean().clip(lower=1e-9)
    z = (spread - mu) / var.pow(0.5)
    m['regime'] = z.map(regime_from_z)

    # core+near dv response
    for k in STRIKES:
        sym = f'VEV_{k}'
        m[f'dv_{k}'] = p[sym].astype(float).diff()

    out = {'shock_abs_du_p90': shock_thr, 'by_regime': {}}
    for rg in ('calm', 'neutral', 'stressed'):
        rg_mask = m['regime'] == rg
        shock = rg_mask & (m['ds'].abs() >= shock_thr)
        calm = rg_mask & (m['ds'].abs() < shock_thr)
        row = {'n_shock': int(shock.sum()), 'n_calm': int(calm.sum())}
        for k in STRIKES:
            col = f'dv_{k}'
            row[f'mean_abs_dv_shock_{k}'] = float(m.loc[shock, col].abs().mean()) if shock.any() else None
            row[f'mean_abs_dv_calm_{k}'] = float(m.loc[calm, col].abs().mean()) if calm.any() else None
        row['mean_abs_dv_shock_core_near'] = float(
            pd.concat([m.loc[shock, f'dv_{k}'].abs() for k in STRIKES], axis=1).mean().mean()
        ) if shock.any() else None
        row['mean_abs_dv_calm_core_near'] = float(
            pd.concat([m.loc[calm, f'dv_{k}'].abs() for k in STRIKES], axis=1).mean().mean()
        ) if calm.any() else None
        out['by_regime'][rg] = row
    return out


def main() -> None:
    by_day = {str(d): run_day(d) for d in (0, 1, 2)}
    # pooled medians
    pooled = {}
    for rg in ('calm', 'neutral', 'stressed'):
        sh = []
        ca = []
        for d in by_day.values():
            b = d['by_regime'][rg]
            if b['mean_abs_dv_shock_core_near'] is not None:
                sh.append(b['mean_abs_dv_shock_core_near'])
            if b['mean_abs_dv_calm_core_near'] is not None:
                ca.append(b['mean_abs_dv_calm_core_near'])
        med_sh = float(pd.Series(sh).median()) if sh else None
        med_ca = float(pd.Series(ca).median()) if ca else None
        pooled[rg] = {
            'median_mean_abs_dv_shock_core_near': med_sh,
            'median_mean_abs_dv_calm_core_near': med_ca,
            'shock_over_calm': (med_sh / med_ca) if med_sh and med_ca and med_ca > 1e-9 else None,
        }

    out = {
        'method': 'Recreate v9/v15 RV-IV z regime (calm/neutral/stressed). Inside each regime, compare |dV| on extract-shock rows (|dU|>=day p90) vs calm for strikes 4500/5000/5100/5200/5300. ATM-IV from BS inversion on three closest strikes.',
        'by_day': by_day,
        'pooled_median_by_regime': pooled,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'wrote {OUT}')

if __name__ == '__main__':
    main()
