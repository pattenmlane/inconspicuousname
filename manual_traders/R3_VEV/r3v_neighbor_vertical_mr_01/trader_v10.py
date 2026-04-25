"""Family10 dual-cutoff consensus + empirical beta-scaled hedge.

Uses analysis_u_move_propagation-derived strike betas (dOption/dU) to scale hedge leg:
hedge_units ~= -q * min(delta_fit, beta_empirical[strike]) * HEDGE_MULT
This reflects observed underlier pass-through per strike under historical tapes.
"""
from __future__ import annotations
import json, math
from datamodel import Order, TradingState
from statistics import NormalDist

N = NormalDist()
STRIKES=[4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
SYMS=[f"VEV_{k}" for k in STRIKES]
UNDER='VELVETFRUIT_EXTRACT'
HYDRO='HYDROGEL_PACK'
LIMITS={UNDER:200,HYDRO:200,**{s:300 for s in SYMS}}

ALPHA=0.03
WARMUP=100
TRADE_EVERY=5
OPEN_Z=3.2
CLOSE_Z=0.8
MAX_CLIP=8
EDGE_SCALE=12.0
CUTOFF_LO=0.5
CUTOFF_HI=1.5
HEDGE_MULT=0.4

BETA_EMP={
 'VEV_4000':0.745,'VEV_4500':0.662,'VEV_5000':0.654,'VEV_5100':0.577,'VEV_5200':0.437,
 'VEV_5300':0.273,'VEV_5400':0.129,'VEV_5500':0.055,'VEV_6000':0.0,'VEV_6500':0.0
}

def _tte_years(day_idx:int)->float: return max(8-max(0,min(day_idx,2)),1)/365.0

def _mid(d):
    if not d.buy_orders or not d.sell_orders: return None
    return 0.5*(max(d.buy_orders)+min(d.sell_orders))

def _half(d):
    if not d.buy_orders or not d.sell_orders: return None
    return 0.5*(min(d.sell_orders)-max(d.buy_orders))

def _bs_call(S,K,T,sig):
    if T<=0 or sig<=0 or S<=0: return max(S-K,0.0)
    st=math.sqrt(T); d1=(math.log(S/K)+0.5*sig*sig*T)/(sig*st); d2=d1-sig*st
    return S*N.cdf(d1)-K*N.cdf(d2)

def _bs_delta(S,K,T,sig):
    if T<=0 or sig<=0 or S<=0: return 0.0
    st=math.sqrt(T); d1=(math.log(S/K)+0.5*sig*sig*T)/(sig*st)
    return N.cdf(d1)

def _iv(mid,S,K,T):
    ins=max(S-K,0.0)
    if mid<=ins+1e-9 or T<=0: return None
    lo,hi=1e-4,4.0
    for _ in range(45):
        m=0.5*(lo+hi)
        p=_bs_call(S,K,T,m)
        if p>mid: hi=m
        else: lo=m
    return 0.5*(lo+hi)

def _fit_wls(xs,ys,ws):
    s00=s01=s02=s11=s12=s22=0.0; t0=t1=t2=0.0
    for x,y,w in zip(xs,ys,ws):
        x2=x*x
        s00+=w; s01+=w*x; s02+=w*x2; s11+=w*x*x; s12+=w*x2*x; s22+=w*x2*x2
        t0+=w*y; t1+=w*x*y; t2+=w*x2*y
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
    return (A[0][3],A[1][3],A[2][3])

def _fit_cutoff(cutoff,S,T,mids,hs,ivs,xs):
    fx=[];fy=[];fw=[]
    for sym,K in zip(SYMS,STRIKES):
        iv=ivs[sym]
        if iv is None: continue
        ext=mids[sym]-max(S-K,0.0)
        if ext<cutoff: continue
        fx.append(xs[sym]); fy.append(iv); fw.append(1.0/(hs[sym]*hs[sym]))
    if len(fx)<4: return None
    return _fit_wls(fx,fy,fw)

class Trader:
    def run(self,state:TradingState):
        try: td=json.loads(state.traderData) if state.traderData else {}
        except: td={}
        prev=td.get('prev_ts'); ts=state.timestamp
        if prev is not None and ts<prev:
            td['day_idx']=int(td.get('day_idx',0))+1
            td['ticks']=0
            td['mu']={s:0.0 for s in SYMS}; td['var']={s:100.0 for s in SYMS}
        td['prev_ts']=ts
        ticks=int(td.get('ticks',0))+1; td['ticks']=ticks
        T=_tte_years(int(td.get('day_idx',0)))

        orders={p:[] for p in LIMITS}; d=state.order_depths
        if UNDER not in d: return orders,0,json.dumps(td)
        S=_mid(d[UNDER])
        if S is None or S<=0: return orders,0,json.dumps(td)

        mids={}; hs={}; ivs={}; xs={}
        for sym,K in zip(SYMS,STRIKES):
            if sym not in d: return orders,0,json.dumps(td)
            m=_mid(d[sym]); h=_half(d[sym])
            if m is None or h is None: return orders,0,json.dumps(td)
            mids[sym]=m; hs[sym]=max(h,0.5); xs[sym]=math.log(K/S)
            ivs[sym]=_iv(m,S,float(K),T)

        c0=_fit_cutoff(CUTOFF_LO,S,T,mids,hs,ivs,xs)
        c1=_fit_cutoff(CUTOFF_HI,S,T,mids,hs,ivs,xs)
        if c0 is None or c1 is None: return orders,0,json.dumps(td)

        mu=td.get('mu',{s:0.0 for s in SYMS}); var=td.get('var',{s:100.0 for s in SYMS})
        for s in SYMS:
            mu.setdefault(s,0.0); var.setdefault(s,100.0)

        res={}; agree={}
        for sym,K in zip(SYMS,STRIKES):
            x=xs[sym]
            iv0=max(0.01,c0[0]+c0[1]*x+c0[2]*x*x)
            iv1=max(0.01,c1[0]+c1[1]*x+c1[2]*x*x)
            f0=_bs_call(S,float(K),T,iv0); f1=_bs_call(S,float(K),T,iv1)
            r0=mids[sym]-f0; r1=mids[sym]-f1
            agree[sym]=(r0*r1)>0
            r=0.5*(r0+r1)
            res[sym]=r
            m0=float(mu[sym]); v0=float(var[sym])
            mu[sym]=(1-ALPHA)*m0+ALPHA*r
            dv=r-m0
            var[sym]=max((1-ALPHA)*v0+ALPHA*dv*dv,1.0)
        td['mu']=mu; td['var']=var

        if ticks<WARMUP or (ticks%TRADE_EVERY)!=0: return orders,0,json.dumps(td)

        best=None; best_z=0.0
        for s in SYMS:
            if not agree[s]: continue
            z=(res[s]-float(mu[s]))/max(math.sqrt(float(var[s])),1e-6)
            if abs(z)>abs(best_z): best_z=z; best=s
        if best is None: return orders,0,json.dumps(td)

        pos=state.position.get(best,0)
        if abs(best_z)<CLOSE_Z and pos!=0:
            dd=d[best]
            if pos>0 and dd.buy_orders:
                orders[best].append(Order(best,max(dd.buy_orders),-min(pos,MAX_CLIP)))
            elif pos<0 and dd.sell_orders:
                orders[best].append(Order(best,min(dd.sell_orders),min(-pos,MAX_CLIP)))
            return orders,0,json.dumps(td)

        if abs(best_z)<OPEN_Z: return orders,0,json.dumps(td)

        idx=SYMS.index(best); K=float(STRIKES[idx]); x=xs[best]
        ivf=max(0.01,0.5*(c0[0]+c1[0])+0.5*(c0[1]+c1[1])*x+0.5*(c0[2]+c1[2])*x*x)
        delta=_bs_delta(S,K,T,ivf)
        beta=min(delta,BETA_EMP.get(best,delta))
        qty=max(1,min(MAX_CLIP,int(MAX_CLIP*min(abs(best_z)/EDGE_SCALE,2.0)*(2.0/hs[best]))))
        pos_opt=state.position.get(best,0); pos_u=state.position.get(UNDER,0)

        if best_z>0:
            q=min(qty,LIMITS[best]+pos_opt)
            if q>0 and d[best].buy_orders:
                orders[best].append(Order(best,max(d[best].buy_orders),-q))
                hedge=int(round(-q*beta*HEDGE_MULT))
                if hedge>0 and d[UNDER].sell_orders:
                    bq=min(hedge,LIMITS[UNDER]-pos_u)
                    if bq>0: orders[UNDER].append(Order(UNDER,min(d[UNDER].sell_orders),bq))
                elif hedge<0 and d[UNDER].buy_orders:
                    sq=min(-hedge,LIMITS[UNDER]+pos_u)
                    if sq>0: orders[UNDER].append(Order(UNDER,max(d[UNDER].buy_orders),-sq))
        else:
            q=min(qty,LIMITS[best]-pos_opt)
            if q>0 and d[best].sell_orders:
                orders[best].append(Order(best,min(d[best].sell_orders),q))
                hedge=int(round(-q*beta*HEDGE_MULT))
                if hedge>0 and d[UNDER].sell_orders:
                    bq=min(hedge,LIMITS[UNDER]-pos_u)
                    if bq>0: orders[UNDER].append(Order(UNDER,min(d[UNDER].sell_orders),bq))
                elif hedge<0 and d[UNDER].buy_orders:
                    sq=min(-hedge,LIMITS[UNDER]+pos_u)
                    if sq>0: orders[UNDER].append(Order(UNDER,max(d[UNDER].buy_orders),-sq))

        return orders,0,json.dumps(td)
