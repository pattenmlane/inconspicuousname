from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/workspace')
DATA = REPO / 'Prosperity4Data' / 'ROUND_3'
OUT = REPO / 'manual_traders' / 'R3_VEV' / 'r3v_jump_gap_filter_17'

EX = 'VELVETFRUIT_EXTRACT'
VEVS = [
    'VEV_4000','VEV_4500','VEV_5000','VEV_5100','VEV_5200',
    'VEV_5300','VEV_5400','VEV_5500','VEV_6000','VEV_6500',
]


def load_day(d: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f'prices_round_3_day_{d}.csv', sep=';')
    return df[df['product'].isin([EX] + VEVS)].copy()


def main() -> None:
    per_obs = []
    for d in (0, 1, 2):
        df = load_day(d)
        piv = df.pivot_table(index='timestamp', columns='product', values='mid_price', aggfunc='last').sort_index()

        spread = {}
        for sym in VEVS:
            s = df[df['product'] == sym][['timestamp', 'ask_price_1', 'bid_price_1']].copy()
            s['sp'] = s['ask_price_1'] - s['bid_price_1']
            spread[sym] = s.set_index('timestamp')['sp']
        sp_df = pd.DataFrame(spread).sort_index()

        ex = piv[EX].ffill()
        dS = ex.diff()

        # shock threshold per-day for signed windows
        thr = float(dS.abs().quantile(0.95))

        for t in piv.index[1:-1]:
            ds0 = float(dS.loc[t]) if pd.notna(dS.loc[t]) else 0.0
            if abs(ds0) < 1e-9:
                continue
            for sym in VEVS:
                if sym not in piv.columns:
                    continue
                v = piv[sym].ffill()
                if t not in v.index:
                    continue
                # lead/lag local responses
                dv_t = float(v.loc[t] - v.shift(1).loc[t]) if pd.notna(v.shift(1).loc[t]) else np.nan
                dv_tp1 = float(v.shift(-1).loc[t] - v.loc[t]) if pd.notna(v.shift(-1).loc[t]) else np.nan

                sp0 = float(sp_df[sym].ffill().loc[t]) if t in sp_df.index else np.nan
                sp1 = float(sp_df[sym].ffill().shift(-1).loc[t]) if t in sp_df.index else np.nan
                dsp = sp1 - sp0 if np.isfinite(sp0) and np.isfinite(sp1) else np.nan

                per_obs.append({
                    'day': d, 'ts': int(t), 'sym': sym, 'K': int(sym.split('_')[1]),
                    'dS_t': ds0, 'abs_dS_t': abs(ds0),
                    'dV_t': dv_t, 'dV_tp1': dv_tp1,
                    'spread_t': sp0, 'spread_tp1': sp1, 'dspread_tp1': dsp,
                    'is_shock95': int(abs(ds0) >= thr),
                    'sign': 'up' if ds0 > 0 else 'down',
                })

    obs = pd.DataFrame(per_obs)
    obs = obs[np.isfinite(obs['dV_t']) & np.isfinite(obs['dV_tp1'])]

    # Pooled lead/lag beta by strike and sign on shock95 windows
    sh = obs[obs['is_shock95'] == 1].copy()
    out = {
        'meta': {
            'n_obs_total': int(len(obs)),
            'n_obs_shock95': int(len(sh)),
            'method': 'local lead/lag response and next-tick spread change conditioned on signed extract shock95',
        },
        'by_strike_sign': {},
        'ranked_follow_through_gap': []
    }

    for K in sorted(sh['K'].unique()):
        block = sh[sh['K'] == K]
        if block.empty:
            continue
        bysign = {}
        for sg in ('up', 'down'):
            b = block[block['sign'] == sg]
            n = int(len(b))
            if n < 30:
                continue
            x = b['dS_t'].to_numpy(dtype=float)
            y0 = b['dV_t'].to_numpy(dtype=float)
            y1 = b['dV_tp1'].to_numpy(dtype=float)
            beta0 = float(np.dot(x, y0) / (np.dot(x, x) + 1e-12))
            beta1 = float(np.dot(x, y1) / (np.dot(x, x) + 1e-12))

            sp_up = float(b['dspread_tp1'].mean()) if b['dspread_tp1'].notna().any() else 0.0
            bysign[sg] = {
                'n': n,
                'beta_same_tick': beta0,
                'beta_next_tick': beta1,
                'follow_through_ratio_abs': float(abs(beta1) / (abs(beta0) + 1e-9)),
                'mean_dspread_next': sp_up,
            }
        out['by_strike_sign'][str(int(K))] = bysign

        if 'up' in bysign and 'down' in bysign:
            gap = bysign['down']['follow_through_ratio_abs'] - bysign['up']['follow_through_ratio_abs']
            out['ranked_follow_through_gap'].append({'K': int(K), 'down_minus_up_follow_ratio': float(gap)})

    out['ranked_follow_through_gap'] = sorted(out['ranked_follow_through_gap'], key=lambda x: x['down_minus_up_follow_ratio'], reverse=True)

    path = OUT / 'analysis_underlying_leadlag_spread_asym.json'
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
