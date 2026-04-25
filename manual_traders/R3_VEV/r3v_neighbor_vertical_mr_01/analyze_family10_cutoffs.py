#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from pathlib import Path
from statistics import NormalDist

N = NormalDist()
ROOT = Path('/workspace')
DATA = ROOT/'Prosperity4Data'/'ROUND_3'
OUT = ROOT/'manual_traders/R3_VEV/r3v_neighbor_vertical_mr_01'/'analysis_family10_extrinsic_cutoff.json'
STRIKES=[4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
SYMS=[f'VEV_{k}' for k in STRIKES]
UNDER='VELVETFRUIT_EXTRACT'
CUTOFFS=[0.5,1.5,3.0]


def tte_years(day:int)->float:
    return max(8-day,1)/365.0

def bs_call(S,K,T,sig):
    if T<=0 or sig<=0 or S<=0: return max(S-K,0.0)
    st=math.sqrt(T)
    d1=(math.log(S/K)+0.5*sig*sig*T)/(sig*st)
    d2=d1-sig*st
    return S*N.cdf(d1)-K*N.cdf(d2)

def implied_vol(mid,S,K,T):
    ins=max(S-K,0.0)
    if mid<=ins+1e-9 or T<=0: return None
    lo,hi=1e-4,4.0
    for _ in range(45):
        m=0.5*(lo+hi)
        p=bs_call(S,K,T,m)
        if p>mid: hi=m
        else: lo=m
    return 0.5*(lo+hi)

def load(path):
    by_ts={}
    with path.open() as f:
        r=csv.DictReader(f,delimiter=';')
        for row in r:
            ts=int(row['timestamp'])
            by_ts.setdefault(ts,{})[row['product']]=row
    return by_ts

def wls_quad(xs,ys,ws):
    # solve (X'WX)b = X'Wy for [1,x,x^2]
    s00=s01=s02=s11=s12=s22=0.0
    t0=t1=t2=0.0
    for x,y,w in zip(xs,ys,ws):
        x2=x*x
        s00+=w
        s01+=w*x
        s02+=w*x2
        s11+=w*x*x
        s12+=w*x2*x
        s22+=w*x2*x2
        t0+=w*y
        t1+=w*x*y
        t2+=w*x2*y
    # gaussian elimination 3x3
    A=[[s00,s01,s02,t0],[s01,s11,s12,t1],[s02,s12,s22,t2]]
    for i in range(3):
        piv=max(range(i,3),key=lambda r:abs(A[r][i]))
        if abs(A[piv][i])<1e-12: return None
        A[i],A[piv]=A[piv],A[i]
        p=A[i][i]
        for c in range(i,4): A[i][c]/=p
        for r in range(3):
            if r==i: continue
            f=A[r][i]
            for c in range(i,4): A[r][c]-=f*A[i][c]
    return [A[0][3],A[1][3],A[2][3]]

def main():
    extr=[]
    evals={str(c):{'n_fit_points':0,'n_ticks':0,'rmse_iv':0.0,'rmse_price':0.0,'n_resid':0} for c in CUTOFFS}
    for day in (0,1,2):
        T=tte_years(day)
        by_ts=load(DATA/f'prices_round_3_day_{day}.csv')
        for ts,row in by_ts.items():
            if UNDER not in row: continue
            try: S=float(row[UNDER]['mid_price'])
            except: continue
            if S<=0: continue
            mids=[];sp=[];ivs=[];xs=[]
            ok=True
            for k,sym in zip(STRIKES,SYMS):
                r=row.get(sym)
                if not r: ok=False; break
                try:
                    m=float(r['mid_price']); bb=float(r['bid_price_1']); ba=float(r['ask_price_1'])
                except: ok=False; break
                ext=m-max(S-k,0.0)
                extr.append(ext)
                iv=implied_vol(m,S,float(k),T)
                mids.append(m); sp.append(max(ba-bb,0.5)); ivs.append(iv); xs.append(math.log(k/S))
            if not ok: continue
            for c in CUTOFFS:
                fi=[]
                for i,k in enumerate(STRIKES):
                    if ivs[i] is None: continue
                    ext=mids[i]-max(S-k,0.0)
                    if ext<c: continue
                    fi.append(i)
                if len(fi)<4: continue
                x=[xs[i] for i in fi]; y=[ivs[i] for i in fi]; w=[1.0/(sp[i]*sp[i]) for i in fi]
                b=wls_quad(x,y,w)
                if b is None: continue
                ev=evals[str(c)]
                ev['n_ticks']+=1
                ev['n_fit_points']+=len(fi)
                for i in range(len(STRIKES)):
                    if ivs[i] is None: continue
                    pred=max(0.01,b[0]+b[1]*xs[i]+b[2]*xs[i]*xs[i])
                    ev['rmse_iv']+=(pred-ivs[i])**2
                    mp=bs_call(S,float(STRIKES[i]),T,pred)
                    ev['rmse_price']+=(mp-mids[i])**2
                    ev['n_resid']+=1
    extr_sorted=sorted(extr)
    def q(p):
        if not extr_sorted: return None
        i=min(len(extr_sorted)-1,max(0,int(p*(len(extr_sorted)-1))))
        return extr_sorted[i]
    for ev in evals.values():
        if ev['n_resid']>0:
            ev['rmse_iv']=(ev['rmse_iv']/ev['n_resid'])**0.5
            ev['rmse_price']=(ev['rmse_price']/ev['n_resid'])**0.5
            ev['avg_fit_points']=ev['n_fit_points']/max(ev['n_ticks'],1)
    out={
      'tte_rule':'TTE_days = 8 - csv_day using round3description mapping (csv days 0,1,2).',
      'extrinsic_quantiles':{'q10':q(0.1),'q25':q(0.25),'q50':q(0.5),'q75':q(0.75),'q90':q(0.9)},
      'cutoff_eval_wls_quad_logm':evals,
      'notes':'Weights 1/spread^2, fit IV = a + b*log(K/S) + c*log(K/S)^2 on points with extrinsic>=cutoff.'
    }
    OUT.write_text(json.dumps(out,indent=2))
    print('Wrote',OUT)

if __name__=='__main__':
    main()
