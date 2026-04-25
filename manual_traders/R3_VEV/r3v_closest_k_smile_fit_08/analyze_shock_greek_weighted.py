#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'analysis_outputs'/'shock_greek_weighted.json'
STRIKES=(4000,4500,5000,5100,5200,5300,5400,5500,6000,6500)

def ncdf(x:float)->float:
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def npdf(x:float)->float:
    return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def cprice(s,k,t,sig):
    if t<=1e-12 or sig<=1e-12: return max(s-k,0.0)
    v=sig*math.sqrt(t); d1=(math.log(s/k)+0.5*sig*sig*t)/v; d2=d1-v
    return s*ncdf(d1)-k*ncdf(d2)

def vega(s,k,t,sig):
    if t<=1e-12 or sig<=1e-12: return 0.0
    v=sig*math.sqrt(t); d1=(math.log(s/k)+0.5*sig*sig*t)/v
    return s*npdf(d1)*math.sqrt(t)

def iv_bisect(price,s,k,t):
    intrinsic=max(s-k,0.0)
    if price<=intrinsic+1e-6 or price>=s-1e-6 or s<=0 or k<=0 or t<=1e-12: return None
    lo,hi=1e-4,12.0
    flo=cprice(s,k,t,lo)-price; fhi=cprice(s,k,t,hi)-price
    if flo>0 or fhi<0: return None
    for _ in range(30):
        md=0.5*(lo+hi)
        if cprice(s,k,t,md)>=price: hi=md
        else: lo=md
    return 0.5*(lo+hi)

def dte(day,timestamp):
    return max((8.0-day)-((int(timestamp)//100)/10000.0),1e-6)

def run_day(day:int):
    df=pd.read_csv(REPO/'Prosperity4Data'/'ROUND_3'/f'prices_round_3_day_{day}.csv',sep=';')
    piv=df.pivot_table(index='timestamp',columns='product',values='mid_price',aggfunc='first')
    ts=piv.index
    s=piv['VELVETFRUIT_EXTRACT'].astype(float)
    ds=s.diff()
    thr=float(ds.abs().quantile(0.90))
    out={}
    for k in STRIKES:
        sym=f'VEV_{k}'
        if sym not in piv.columns: continue
        v=piv[sym].astype(float)
        dv=v.diff()
        m=pd.DataFrame({'s':s,'ds':ds,'v':v,'dv':dv}).dropna()
        if m.empty: continue
        shock=m['ds'].abs()>=thr
        ad=m['dv'].abs()
        mean_sh=float(ad[shock].mean()) if shock.any() else None
        mean_ca=float(ad[~shock].mean()) if (~shock).any() else None

        # greek-weighted response using tape IV
        vegas=[]
        for tstamp,row in m.iloc[::25].iterrows():
            tt=dte(day,int(tstamp))/365.0
            sig=iv_bisect(float(row['v']),float(row['s']),float(k),tt)
            if sig is None: continue
            vg=vega(float(row['s']),float(k),tt,sig)
            vegas.append(vg)
        med_vega=float(pd.Series(vegas).median()) if vegas else 0.0
        shock_energy=(mean_sh or 0.0)*(1.0+med_vega/300.0)
        out[str(k)]={
            'mean_abs_dv_shock':mean_sh,
            'mean_abs_dv_calm':mean_ca,
            'shock_over_calm':(mean_sh/mean_ca if mean_sh and mean_ca and mean_ca>1e-9 else None),
            'median_vega_proxy':med_vega,
            'shock_energy_score':shock_energy,
        }
    return {'shock_abs_ds_p90':thr,'per_strike':out}

res={'method':'Per day, shock=|dU|>=p90. For each strike compute |dV| shock/calm and a greek-weighted shock energy = mean|dV|_shock*(1+median_vega/300). Tape IV from BS inversion sampled every 25 ticks for stability.',
     'by_day':{str(d):run_day(d) for d in (0,1,2)}}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(res,indent=2),encoding='utf-8')
print('wrote',OUT)
