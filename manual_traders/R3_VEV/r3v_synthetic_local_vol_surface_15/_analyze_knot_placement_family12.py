
#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'Prosperity4Data' / 'ROUND_3'
OUT = Path(__file__).resolve().parent / 'knot_placement_residual_stability.json'

STRIKES = [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
VOUCHERS = [f'VEV_{k}' for k in STRIKES]
CORE = {5000,5100,5200,5300,5400,5500}
U='VELVETFRUIT_EXTRACT'

def t_years(day:int, ts:int)->float:
    dte=max(8-int(day) - (int(ts)//100)/10000.0,1e-6)
    return dte/365.0

def bba(row:dict)->tuple[float|None,float|None]:
    bids=[]; asks=[]
    for i in (1,2,3):
        bp,bv=row.get(f'bid_price_{i}'),row.get(f'bid_volume_{i}')
        ap,av=row.get(f'ask_price_{i}'),row.get(f'ask_volume_{i}')
        if bp and bv and int(float(bv))>0: bids.append(float(bp))
        if ap and av and int(float(av))>0: asks.append(float(ap))
    if not bids or not asks: return None,None
    return max(bids),min(asks)

def bs_call(S,K,T,s):
    if T<=0 or s<=1e-12: return max(S-K,0.0)
    v=s*math.sqrt(T)
    d1=(math.log(S/K)+0.5*s*s*T)/v; d2=d1-v
    return S*norm.cdf(d1)-K*norm.cdf(d2)

def vega(S,K,T,s):
    if T<=0 or s<=1e-12: return 0.0
    v=s*math.sqrt(T)
    d1=(math.log(S/K)+0.5*s*s*T)/v
    return S*norm.pdf(d1)*math.sqrt(T)

def iv_newton(mid,S,K,T,g=0.45):
    intr=max(S-K,0.0)
    if mid<=intr+1e-6 or mid>=S-1e-9: return float('nan')
    s=max(min(g,6.0),0.04)
    for _ in range(12):
        e=bs_call(S,K,T,s)-mid
        if abs(e)<1e-4: break
        vg=vega(S,K,T,s)
        if vg<1e-8: return float('nan')
        s-=e/vg
        s=max(min(s,8.0),0.03)
    return s if abs(bs_call(S,K,T,s)-mid)<0.05 else float('nan')

def fit_family(xs,ivs,family):
    o=np.argsort(xs)
    x=np.asarray(xs)[o]; y=np.asarray(ivs)[o]
    if len(x)<4: return None
    # base spline through observed knots
    try:
        base=CubicSpline(x,y,bc_type='natural',extrapolate=True)
    except Exception:
        return None
    xlo,xhi=float(x[0]),float(x[-1])
    if family=='uniform':
        grid=np.linspace(xlo,xhi,12)
    else:
        # core-biased: dense around ATM logk~0, sparse wings
        # clamp to observed range
        core=np.array([-0.08,-0.05,-0.03,-0.015,0.0,0.015,0.03,0.05,0.08])
        wings=np.array([xlo, xlo*0.6, xhi*0.6, xhi])
        grid=np.unique(np.clip(np.concatenate([wings,core]),xlo,xhi))
    yg=base(grid)
    try:
        return CubicSpline(grid,yg,bc_type='natural',extrapolate=True)
    except Exception:
        return None

metrics={
  'uniform': defaultdict(lambda: defaultdict(lambda: {'sum_abs_res':0.0,'sum_abs_dres':0.0,'n':0,'prev':None})),
  'core_biased': defaultdict(lambda: defaultdict(lambda: {'sum_abs_res':0.0,'sum_abs_dres':0.0,'n':0,'prev':None})),
}

for day in (0,1,2):
    by_ts=defaultdict(dict)
    with (DATA/f'prices_round_3_day_{day}.csv').open() as f:
        r=csv.DictReader(f,delimiter=';')
        for row in r:
            p=row['product']
            if p==U or p in VOUCHERS:
                by_ts[int(row['timestamp'])][p]=row
    for ts in sorted(by_ts.keys())[::20]:
        snap=by_ts[ts]
        if U not in snap: continue
        bb,ba=bba(snap[U])
        if bb is None or ba<=bb: continue
        S=0.5*(bb+ba); T=t_years(day,ts)
        xs=[]; ivs=[]; mids={}
        for k,v in zip(STRIKES,VOUCHERS):
            if v not in snap: continue
            b,o=bba(snap[v])
            if b is None or o<=b: continue
            mid=0.5*(b+o)
            iv=iv_newton(mid,S,k,T)
            if not math.isfinite(iv): continue
            x=math.log(max(k/S,1e-6))
            xs.append(x); ivs.append(iv); mids[k]=mid
        if len(xs)<4: continue
        for fam in ('uniform','core_biased'):
            spl=fit_family(xs,ivs,fam)
            if spl is None: continue
            for k in CORE:
                if k not in mids: continue
                xk=math.log(max(k/S,1e-6))
                iv_hat=float(np.clip(spl(xk),0.03,7.5))
                theo=bs_call(S,k,T,iv_hat)
                resid=mids[k]-theo
                m=metrics[fam][day][k]
                m['sum_abs_res']+=abs(resid)
                if m['prev'] is not None:
                    m['sum_abs_dres']+=abs(resid-m['prev'])
                m['prev']=resid
                m['n']+=1

out={'timing':{'dte_open':'8-csv_day','intraday':'dte_eff=dte_open-(ts//100)/10000','T':'dte_eff/365','r':0.0},'by_family':{}}
for fam in metrics:
    out['by_family'][fam]={}
    for day in sorted(metrics[fam].keys()):
        rows=[]
        for k in sorted(metrics[fam][day].keys()):
            m=metrics[fam][day][k]; n=max(m['n'],1)
            rows.append({'strike':k,'samples':m['n'],'mean_abs_resid':m['sum_abs_res']/n,'mean_abs_delta_resid':m['sum_abs_dres']/n})
        out['by_family'][fam][str(day)]=rows
OUT.write_text(json.dumps(out,indent=2))
print(OUT)
