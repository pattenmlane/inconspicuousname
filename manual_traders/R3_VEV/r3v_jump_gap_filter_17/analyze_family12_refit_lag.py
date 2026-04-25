#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'round3work'/'plotting'/'original_method'/'combined_analysis'))
from plot_iv_smile_round3 import implied_vol_call, t_years_effective

DATA=REPO/'Prosperity4Data'/'ROUND_3'
STRIKES=np.array([4000,4500,5000,5100,5200,5300,5400,5500,6000,6500],dtype=float)

def fit_spline(S, ivs_map):
    xs=[]; ys=[]
    for k in STRIKES:
        iv=ivs_map.get(int(k))
        if iv is None or not math.isfinite(iv) or iv<=0: continue
        xs.append(math.log(float(k)/S)); ys.append(iv)
    if len(xs)<4: return None
    xs=np.array(xs); ys=np.array(ys)
    o=np.argsort(xs)
    return CubicSpline(xs[o], ys[o], extrapolate=True)

def prep_day(day:int):
    df=pd.read_csv(DATA/f'prices_round_3_day_{day}.csv',sep=';')
    return df.pivot(index='timestamp',columns='product',values='mid_price').sort_index()

def run_policy(cadence=40,jump_refit=False,jump_ds=3.0):
    errs=[]; post=[]
    for day in [0,1,2]:
        pv=prep_day(day)
        s= pv['VELVETFRUIT_EXTRACT'].astype(float)
        model=None; last_ref=-10**9; prev_s=None; jump_until=-1
        for ts,row in pv.iterrows():
            S=float(row['VELVETFRUIT_EXTRACT'])
            if not math.isfinite(S):
                continue
            dS=0.0 if prev_s is None else abs(S-prev_s)
            prev_s=S
            T=t_years_effective(day,int(ts))
            ivs={}
            for k in STRIKES.astype(int):
                sym=f'VEV_{k}'
                m=row.get(sym)
                if m is None or not math.isfinite(float(m)):
                    continue
                iv=implied_vol_call(float(m),S,float(k),float(T),0.0)
                if math.isfinite(iv):
                    ivs[int(k)]=float(iv)
            tick=int(ts)//100
            need=(model is None) or (tick-last_ref>=cadence) or (jump_refit and dS>=jump_ds)
            if need:
                m=fit_spline(S,ivs)
                if m is not None:
                    model=m; last_ref=tick
                if dS>=jump_ds:
                    jump_until=tick+120
            if model is None:
                continue
            k_atm=int(STRIKES[np.argmin(np.abs(STRIKES-S))])
            obs=ivs.get(k_atm)
            if obs is None:
                continue
            pred=float(model(math.log(k_atm/S)))
            if not math.isfinite(pred):
                continue
            e=abs(pred-obs)
            errs.append(e)
            if tick<=jump_until:
                post.append(e)
    return {'n':len(errs),'mae':float(np.mean(errs)),'post_jump_n':len(post),'post_jump_mae':float(np.mean(post)) if post else None}

base=run_policy(cadence=40,jump_refit=False,jump_ds=3.0)
jump=run_policy(cadence=40,jump_refit=True,jump_ds=3.0)
out={'method':'smile_cubic_spline_logk from tape IVs; fixed cadence vs immediate jump refit','baseline':base,'jump_aware':jump}
Path(__file__).resolve().parent.joinpath('analysis_family12_refit_lag.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
