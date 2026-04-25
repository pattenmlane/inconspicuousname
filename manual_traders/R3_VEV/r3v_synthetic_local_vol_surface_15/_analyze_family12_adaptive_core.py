
#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'Prosperity4Data'/'ROUND_3'
OUT=Path(__file__).resolve().parent/'family12_adaptive_vs_uniform_stability.json'
STRIKES=[4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
VOUS=[f'VEV_{k}' for k in STRIKES]
CORE={5000,5100,5200,5300,5400,5500}
WING={4000,4500,6000,6500}


def t(day,ts):
    return max(8-int(day)-(int(ts)//100)/10000.0,1e-6)/365.0

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
    if T<=0 or s<=1e-12: return max(S-K,0)
    v=s*math.sqrt(T)
    d1=(math.log(S/K)+0.5*s*s*T)/v
    d2=d1-v
    return S*norm.cdf(d1)-K*norm.cdf(d2)

def vg(S,K,T,s):
    if T<=0 or s<=1e-12:return 0.0
    v=s*math.sqrt(T)
    d1=(math.log(S/K)+0.5*s*s*T)/v
    return S*norm.pdf(d1)*math.sqrt(T)

def iv(mid,S,K,T,g=0.45):
    intr=max(S-K,0)
    if mid<=intr+1e-6 or mid>=S-1e-9:return float('nan')
    s=max(min(g,6),0.04)
    for _ in range(12):
        e=bs(S,K,T,s)-mid
        if abs(e)<1e-4: break
        v=vg(S,K,T,s)
        if v<1e-8:return float('nan')
        s=max(min(s-e/v,8),0.03)
    return s if abs(bs(S,K,T,s)-mid)<0.05 else float('nan')

def fit_base(xs,ivs,ws):
    o=np.argsort(xs)
    x=np.asarray(xs)[o]; y=np.asarray(ivs)[o]; w=np.asarray(ws)[o]
    if len(x)<4:return None
    return UnivariateSpline(x,y,w=w,k=3,s=0.08)

def family_grid(xmin,xmax,fam,core_density=1.0):
    if fam=='uniform': return np.linspace(xmin,xmax,12)
    scale=float(np.clip(core_density,0.6,1.6))
    core=np.array([-0.08,-0.05,-0.03,-0.015,0.0,0.015,0.03,0.05,0.08])/scale
    wings=np.array([xmin,xmin*0.7,xmax*0.7,xmax])
    return np.unique(np.clip(np.concatenate([wings,core]),xmin,xmax))

def fit_family(xs,ivs,ws,fam,core_density=1.0):
    base=fit_base(xs,ivs,ws)
    if base is None:return None
    o=np.argsort(xs); xo=np.asarray(xs)[o]
    g=family_grid(float(xo[0]),float(xo[-1]),fam,core_density)
    yg=np.asarray(base(g),dtype=float)
    return CubicSpline(g,yg,bc_type='natural',extrapolate=True)

met={fam:defaultdict(lambda: {'sum_abs_res':0.0,'sum_abs_dres':0.0,'n':0,'prev':None}) for fam in ('uniform','adaptive_core')}
for day in (0,1,2):
    by=defaultdict(dict)
    with (DATA/f'prices_round_3_day_{day}.csv').open() as f:
        rd=csv.DictReader(f,delimiter=';')
        for r in rd:
            p=r['product']
            if p=='VELVETFRUIT_EXTRACT' or p in VOUS:
                by[int(r['timestamp'])][p]=r
    for ts in sorted(by.keys())[::20]:
        s=by[ts]
        if 'VELVETFRUIT_EXTRACT' not in s: continue
        bb,ba=bba(s['VELVETFRUIT_EXTRACT'])
        if bb is None or ba<=bb: continue
        S=0.5*(bb+ba); T=t(day,ts)
        xs=[];ivs=[];ws=[];mid={};hsp=[]
        for k,v in zip(STRIKES,VOUS):
            if v not in s: continue
            b,o=bba(s[v])
            if b is None or o<=b: continue
            m=0.5*(b+o)
            ivk=iv(m,S,k,T)
            if not math.isfinite(ivk): continue
            x=math.log(max(k/S,1e-6))
            xs.append(x);ivs.append(ivk);ws.append(0.58 if k in WING else 1.0)
            mid[k]=m
            if k in CORE: hsp.append(0.5*(o-b))
        if len(xs)<4: continue
        core_density=1.0+np.clip((1.5-(float(np.mean(hsp)) if hsp else 1.5))/2.0,-0.35,0.45)
        spl_u=fit_family(xs,ivs,ws,'uniform',1.0)
        spl_a=fit_family(xs,ivs,ws,'adaptive_core',core_density)
        for fam,spl in [('uniform',spl_u),('adaptive_core',spl_a)]:
            if spl is None: continue
            for k in CORE:
                if k not in mid: continue
                xk=math.log(max(k/S,1e-6))
                sig=float(np.clip(spl(xk),0.03,7.5))
                r=mid[k]-bs(S,k,T,sig)
                m=met[fam][day]
                m['sum_abs_res']+=abs(r)
                if m['prev'] is not None: m['sum_abs_dres']+=abs(r-m['prev'])
                m['prev']=r; m['n']+=1

out={'timing':{'dte_open':'8-csv_day','intraday':'dte_eff=dte_open-(ts//100)/10000','T':'dte_eff/365'},'families':{}}
for fam,dmap in met.items():
    out['families'][fam]={}
    for day,m in dmap.items():
        n=max(1,m['n'])
        out['families'][fam][str(day)]={'samples':m['n'],'mean_abs_resid':m['sum_abs_res']/n,'mean_abs_delta_resid':m['sum_abs_dres']/n}
OUT.write_text(json.dumps(out,indent=2))
print(OUT)
