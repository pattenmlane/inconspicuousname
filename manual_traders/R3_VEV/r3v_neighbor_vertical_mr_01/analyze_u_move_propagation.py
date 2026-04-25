#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from pathlib import Path
from collections import defaultdict

ROOT = Path('/workspace')
DATA = ROOT/'Prosperity4Data'/'ROUND_3'
OUT = ROOT/'manual_traders/R3_VEV/r3v_neighbor_vertical_mr_01'/'analysis_u_move_propagation.json'
SYMS=[f'VEV_{k}' for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]]
UNDER='VELVETFRUIT_EXTRACT'


def load_day(path: Path):
    by_ts=defaultdict(dict)
    with path.open() as f:
        r=csv.DictReader(f,delimiter=';')
        for row in r:
            by_ts[int(row['timestamp'])][row['product']]=row
    return by_ts


def mid(row):
    try: return float(row['mid_price'])
    except: return None

def spread(row):
    try:
        bb=float(row['bid_price_1']); ba=float(row['ask_price_1'])
        return ba-bb
    except: return None


def corr(xs, ys):
    n=len(xs)
    if n<3 or n!=len(ys): return None
    mx=sum(xs)/n; my=sum(ys)/n
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=(sum((x-mx)**2 for x in xs))**0.5
    dy=(sum((y-my)**2 for y in ys))**0.5
    if dx<1e-12 or dy<1e-12: return None
    return num/(dx*dy)


def ols_beta(x,y):
    n=len(x)
    if n<3: return None
    mx=sum(x)/n; my=sum(y)/n
    v=sum((xi-mx)**2 for xi in x)
    if v<1e-12: return None
    c=sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    return c/v


def mean(lst):
    return sum(lst)/len(lst) if lst else None


def main():
    du=[]
    dopt={s:[] for s in SYMS}
    du_pos=[]; du_neg=[]
    dopt_pos={s:[] for s in SYMS}
    dopt_neg={s:[] for s in SYMS}
    spread_norm={s:[] for s in SYMS}
    spread_shock={s:[] for s in SYMS}

    for day in (0,1,2):
        by=load_day(DATA/f'prices_round_3_day_{day}.csv')
        tss=sorted(by.keys())
        prev_u=None
        prev_m={s:None for s in SYMS}
        for ts in tss:
            row=by[ts]
            if UNDER not in row: continue
            u=mid(row[UNDER])
            if u is None: continue
            curr_m={s:(mid(row[s]) if s in row else None) for s in SYMS}
            curr_sp={s:(spread(row[s]) if s in row else None) for s in SYMS}
            if prev_u is not None:
                d_u=u-prev_u
                du.append(d_u)
                if d_u>0: du_pos.append(d_u)
                if d_u<0: du_neg.append(d_u)
                is_shock = abs(d_u)>=4.0
                for s in SYMS:
                    if prev_m[s] is None or curr_m[s] is None: continue
                    d_o=curr_m[s]-prev_m[s]
                    dopt[s].append(d_o)
                    if d_u>0: dopt_pos[s].append(d_o)
                    if d_u<0: dopt_neg[s].append(d_o)
                    if curr_sp[s] is not None:
                        if is_shock: spread_shock[s].append(curr_sp[s])
                        else: spread_norm[s].append(curr_sp[s])
            prev_u=u
            prev_m=curr_m

    out={
        'method':'Per-step dU and dOption using mids from ROUND_3 day0-2. Strike-wise OLS beta dOption~beta*dU, corr(dU,dOption), asymmetry by sign(dU), and spread after |dU|>=4 shocks.',
        'shock_threshold_abs_dU':4.0,
        'by_symbol':{}
    }

    for s in SYMS:
        n=min(len(du),len(dopt[s]))
        x=du[:n]; y=dopt[s][:n]
        beta=ols_beta(x,y)
        c=corr(x,y)
        beta_up=ols_beta(du_pos[:len(dopt_pos[s])], dopt_pos[s]) if len(dopt_pos[s])>=3 else None
        beta_dn=ols_beta(du_neg[:len(dopt_neg[s])], dopt_neg[s]) if len(dopt_neg[s])>=3 else None
        out['by_symbol'][s]={
            'n':n,
            'beta_dopt_per_du':beta,
            'corr_du_dopt':c,
            'mean_dopt_upmove':mean(dopt_pos[s]),
            'mean_dopt_downmove':mean(dopt_neg[s]),
            'beta_up':beta_up,
            'beta_down':beta_dn,
            'mean_spread_normal':mean(spread_norm[s]),
            'mean_spread_after_shock':mean(spread_shock[s]),
            'spread_shock_ratio': (mean(spread_shock[s])/mean(spread_norm[s]) if mean(spread_norm[s]) and mean(spread_shock[s]) else None)
        }

    OUT.write_text(json.dumps(out,indent=2))
    print('Wrote',OUT)

if __name__=='__main__':
    main()
