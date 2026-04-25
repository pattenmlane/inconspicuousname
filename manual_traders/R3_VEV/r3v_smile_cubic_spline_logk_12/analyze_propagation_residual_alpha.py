#!/usr/bin/env python3
"""Propagation residual alpha by strike:
res_t = dV_t - beta*dS_t (beta from same-day fit), test corr(res_t, dV_{t+1}).
Also record spread-conditioned signal strength.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
EX = "VELVETFRUIT_EXTRACT"
VEVS = ["VEV_4000","VEV_4500","VEV_5000","VEV_5100","VEV_5200","VEV_5300","VEV_5400","VEV_5500","VEV_6000","VEV_6500"]

def f(x):
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None

def beta(x,y):
    if len(x) < 10:
        return None
    xx, yy = np.array(x), np.array(y)
    vx = float(np.var(xx))
    if vx <= 1e-14:
        return None
    return float(np.cov(xx, yy, ddof=0)[0,1] / vx)

def corr(x,y):
    if len(x) < 10:
        return None
    xx, yy = np.array(x), np.array(y)
    if float(np.std(xx)) <= 1e-14 or float(np.std(yy)) <= 1e-14:
        return None
    return float(np.corrcoef(xx,yy)[0,1])

def run_day(day:int):
    rows = list(csv.DictReader((DATA/f"prices_round_3_day_{day}.csv").open(), delimiter=';'))
    by_ts = defaultdict(dict)
    for r in rows:
        by_ts[int(r['timestamp'])][r['product']] = r
    ts = sorted(by_ts.keys())

    ex = [f(by_ts[t].get(EX,{}).get('mid_price','')) for t in ts]
    dex = [None]
    for i in range(1,len(ts)):
        dex.append(None if ex[i] is None or ex[i-1] is None else ex[i]-ex[i-1])

    out=[]
    for sym in VEVS:
        vm=[]; sp=[]
        for t in ts:
            r=by_ts[t].get(sym,{})
            vm.append(f(r.get('mid_price','')))
            b,a=f(r.get('bid_price_1','')),f(r.get('ask_price_1',''))
            sp.append(None if b is None or a is None else a-b)
        dv=[None]
        for i in range(1,len(ts)):
            dv.append(None if vm[i] is None or vm[i-1] is None else vm[i]-vm[i-1])

        xs=[]; ys=[]
        for i in range(1,len(ts)):
            if dex[i] is None or dv[i] is None: continue
            xs.append(dex[i]); ys.append(dv[i])
        b = beta(xs,ys)
        if b is None:
            out.append({'day':day,'symbol':sym,'beta':None,'corr_res_to_next':None,'corr_res_to_next_tight':None,'n':0,'n_tight':0})
            continue

        res=[]; nxt=[]; res_t=[]; nxt_t=[]
        for i in range(1,len(ts)-1):
            if dex[i] is None or dv[i] is None or dv[i+1] is None: continue
            r = dv[i] - b*dex[i]
            res.append(r); nxt.append(dv[i+1])
            if sp[i] is not None and sp[i] <= 2.0:
                res_t.append(r); nxt_t.append(dv[i+1])
        out.append({
            'day':day,
            'symbol':sym,
            'beta':b,
            'corr_res_to_next':corr(res,nxt),
            'corr_res_to_next_tight':corr(res_t,nxt_t),
            'n':len(res),
            'n_tight':len(res_t),
        })
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for d in (0,1,2): rows.extend(run_day(d))
    p = OUT/'propagation_residual_alpha.csv'
    with p.open('w', newline='', encoding='utf-8') as fobj:
        w=csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote', p)

if __name__=='__main__':
    main()
