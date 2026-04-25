
#!/usr/bin/env python3
"""
Shock-conditional residual edge analysis for v19-style surface on core strikes.
Computes per-strike residual bucket edge after underlying shocks vs normal ticks,
and spread behavior after shocks.
"""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'Prosperity4Data'/'ROUND_3'
OUT=Path(__file__).resolve().parent/'shock_conditional_residual_edge.json'

U='VELVETFRUIT_EXTRACT'
CORE=[5000,5100,5200,5300,5400,5500]
ALL=[4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
VOUS=[f'VEV_{k}' for k in ALL]
WING={4000,4500,6000,6500}
WING_W=0.58
SHOCK=2.5


def t_years(day,ts):
    dte=max(8-int(day)-(int(ts)//100)/10000.0,1e-6)
    return dte/365.0

def bba(r):
    bids=[];asks=[]
    for i in (1,2,3):
        bp,bv=r.get(f'bid_price_{i}'),r.get(f'bid_volume_{i}')
        ap,av=r.get(f'ask_price_{i}'),r.get(f'ask_volume_{i}')
        if bp and bv and int(float(bv))>0: bids.append(float(bp))
        if ap and av and int(float(av))>0: asks.append(float(ap))
    if not bids or not asks:return None,None
    return max(bids),min(asks)

def bs(S,K,T,s):
    if T<=0 or s<=1e-12: return max(S-K,0.0)
    v=s*math.sqrt(T); d1=(math.log(S/K)+0.5*s*s*T)/v; d2=d1-v
    return S*norm.cdf(d1)-K*norm.cdf(d2)

def vega(S,K,T,s):
    if T<=0 or s<=1e-12:return 0.0
    v=s*math.sqrt(T); d1=(math.log(S/K)+0.5*s*s*T)/v
    return S*norm.pdf(d1)*math.sqrt(T)

def iv_newton(mid,S,K,T,g=0.45):
    intr=max(S-K,0)
    if mid<=intr+1e-6 or mid>=S-1e-9:return float('nan')
    s=max(min(g,6),0.04)
    for _ in range(10):
        e=bs(S,K,T,s)-mid
        if abs(e)<1e-4: break
        vg=vega(S,K,T,s)
        if vg<1e-8:return float('nan')
        s=max(min(s-e/vg,8),0.03)
    return s if abs(bs(S,K,T,s)-mid)<0.05 else float('nan')

def fit_sigma(xs,ivs,ws):
    o=np.argsort(xs)
    x=np.asarray(xs)[o]; y=np.asarray(ivs)[o]; w=np.asarray(ws)[o]
    if len(x)<4:return None
    try:
        base=UnivariateSpline(x,y,w=w,k=3,s=0.08)
        g=np.linspace(float(x[0]),float(x[-1]),12)
        yg=np.asarray(base(g),dtype=float)
        return CubicSpline(g,yg,bc_type='natural',extrapolate=True)
    except Exception:
        return None

# stats[strike][bucket] -> sums
def mk():
    return {'n':0,'sum_abs_res':0.0,'sum_abs_fwd':0.0,'sum_sign_fwd':0.0,'sum_hsp':0.0}
st={k:{'shock':mk(),'normal':mk()} for k in CORE}

for day in (0,1,2):
    by=defaultdict(dict)
    with (DATA/f'prices_round_3_day_{day}.csv').open() as f:
        rd=csv.DictReader(f,delimiter=';')
        for r in rd:
            p=r['product']
            if p==U or p in VOUS:
                by[int(r['timestamp'])][p]=r
    tss=sorted(by.keys())
    prev_um=None
    # prebuild mids for forward returns
    cache=[]
    for ts in tss:
        s=by[ts]
        if U not in s:
            cache.append((ts,None,{})); continue
        b,u=bba(s[U])
        if b is None or u<=b:
            cache.append((ts,None,{})); continue
        um=0.5*(b+u)
        vm={}
        for k in CORE:
            v=f'VEV_{k}'
            if v not in s: continue
            bb,aa=bba(s[v])
            if bb is None or aa<=bb: continue
            vm[k]=(0.5*(bb+aa),0.5*(aa-bb))
        cache.append((ts,um,vm))

    for i,(ts,um,vm) in enumerate(cache[:-1]):
        if um is None: continue
        if prev_um is None:
            prev_um=um
            continue
        dS=um-prev_um
        prev_um=um
        bucket='shock' if abs(dS)>=SHOCK else 'normal'

        T=t_years(day,ts)
        # build smile from all strikes at this ts
        s=by[ts]
        xs=[];ivs=[];ws=[]
        for k in ALL:
            v=f'VEV_{k}'
            if v not in s: continue
            bb,aa=bba(s[v])
            if bb is None or aa<=bb: continue
            mid=0.5*(bb+aa)
            iv=iv_newton(mid,um,k,T)
            if not math.isfinite(iv): continue
            xs.append(math.log(max(k/um,1e-6))); ivs.append(iv); ws.append(WING_W if k in WING else 1.0)
        spl=fit_sigma(xs,ivs,ws)
        if spl is None: continue

        _,um_nxt,vm_nxt=cache[i+1]
        for k in CORE:
            if k not in vm or k not in vm_nxt: continue
            mid,hsp=vm[k]
            x=math.log(max(k/um,1e-6)); sig=float(np.clip(spl(x),0.03,7.5))
            theo=bs(um,k,T,sig)
            resid=mid-theo
            fwd=vm_nxt[k][0]-mid
            a=st[k][bucket]
            a['n']+=1
            a['sum_abs_res']+=abs(resid)
            a['sum_abs_fwd']+=abs(fwd)
            a['sum_sign_fwd']+= (1.0 if resid*fwd>0 else 0.0)
            a['sum_hsp']+=hsp

out={'shock_threshold_abs_dS':SHOCK,'by_strike':{}}
for k in CORE:
    out['by_strike'][str(k)]={}
    for b in ('normal','shock'):
        a=st[k][b]; n=max(a['n'],1)
        out['by_strike'][str(k)][b]={
            'samples':a['n'],
            'mean_abs_resid':a['sum_abs_res']/n,
            'mean_abs_fwd_mid_move':a['sum_abs_fwd']/n,
            'sign_agreement_rate_resid_vs_fwd':a['sum_sign_fwd']/n,
            'mean_half_spread':a['sum_hsp']/n,
        }
OUT.write_text(json.dumps(out,indent=2))
print(OUT)
