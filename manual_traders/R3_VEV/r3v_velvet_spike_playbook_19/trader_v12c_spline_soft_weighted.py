
"""
Family 12 variation C (distinct): smile cubic spline log-k with SOFT liquidity weighting.

Delta vs v12b:
- Calibration uses all cluster strikes with weighted cubic fit (weight ~ depth/spread quality),
  instead of hard include/exclude thresholds.
- Execution uses dynamic quality floor by regime (stricter in burst), not same hard gate as v12b.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import numpy as np

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

STRIKES=[4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
UNDERLYING='VELVETFRUIT_EXTRACT'

Z_WINDOW=50
Z_SPIKE=2.2
ABS_DS_SPIKE=7.0
WARMUP_DIV100=10
ATM_REL_BURST=0.24
ATM_REL_CALM=0.10
BURST_MAX_LOT=80

EDGE_CALM=0.9
EDGE_BURST=1.6

EXEC_MIN_Q_CALM=2.5
EXEC_MIN_Q_BURST=4.0
EXEC_MAX_SPREAD=8.0


def _symbol_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, 'listings', {}) or {}
    for sym,lst in listings.items():
        if getattr(lst,'product',None)==product:
            return sym
    return None


def t_years(csv_day:int, ts:int)->float:
    d0=8-int(csv_day)
    prog=(int(ts)//100)/10000.0
    return max((d0-prog)/365.0,1e-6)


def _parse_td(raw:str|None)->dict[str,Any]:
    if not raw:
        return {}
    try:
        o=json.loads(raw)
        return o if isinstance(o,dict) else {}
    except Exception:
        return {}


def _rolling_z(series:list[float], window:int)->float:
    if len(series)<max(window,5):
        return 0.0
    a=np.asarray(series[-window:],dtype=float)
    return float(abs(a[-1]-a.mean())/(a.std()+1e-9))


def _norm_cdf(x:float)->float:
    return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))


def _norm_pdf(x:float)->float:
    return math.exp(-0.5*x*x)/math.sqrt(2.0*math.pi)


def bs_call(S:float,K:float,T:float,sigma:float)->tuple[float,float,float]:
    if T<=0 or sigma<=1e-9:
        intrinsic=max(S-K,0.0)
        delta=1.0 if S>K else 0.0
        return intrinsic,delta,0.0
    v=sigma*math.sqrt(T)
    d1=(math.log(S/K)+0.5*sigma*sigma*T)/v
    d2=d1-v
    price=S*_norm_cdf(d1)-K*_norm_cdf(d2)
    delta=_norm_cdf(d1)
    vega=S*_norm_pdf(d1)*math.sqrt(T)
    return float(price),float(delta),float(vega)


def implied_vol_call(market:float,S:float,K:float,T:float)->float:
    intrinsic=max(S-K,0.0)
    if not (S>0 and K>0 and T>0):
        return float('nan')
    if market<=intrinsic+1e-9 or market>=S-1e-9:
        return float('nan')
    lo,hi=1e-4,5.0
    flo=bs_call(S,K,T,lo)[0]-market
    fhi=bs_call(S,K,T,hi)[0]-market
    if flo*fhi>0:
        return float('nan')
    for _ in range(60):
        mid=0.5*(lo+hi)
        fm=bs_call(S,K,T,mid)[0]-market
        if abs(fm)<1e-7:
            return float(mid)
        if flo*fm<=0:
            hi=mid; fhi=fm
        else:
            lo=mid; flo=fm
    return float(0.5*(lo+hi))


def _weighted_cubic_fit(xs:list[float], ys:list[float], ws:list[float])->tuple[list[float],list[float]]:
    x=np.asarray(xs,dtype=float)
    y=np.asarray(ys,dtype=float)
    w=np.asarray(ws,dtype=float)
    deg=3 if len(x)>=4 else (2 if len(x)>=3 else 1)
    coef=np.polyfit(x,y,deg,w=w)
    return coef.tolist(), [float(x.min()), float(x.max())]


def _eval_poly(coef:list[float], x:float)->float:
    return float(np.polyval(np.asarray(coef,dtype=float), float(x)))


def _quality(bb:int,ba:int,bv:int,av:int)->tuple[float,float,float]:
    spread=float(ba-bb)
    depth=float(max(0,bv)+max(0,av))
    q=depth/max(spread,1e-9)
    return spread,depth,q


class Trader:
    def bid(self)->int:
        return 0

    def run(self,state:TradingState):
        store=_parse_td(getattr(state,'traderData',None))
        hist=store.get('s_hist') if isinstance(store.get('s_hist'),list) else []
        hist=[float(x) for x in hist if isinstance(x,(int,float))][-120:]
        dlog=store.get('dlog_hist') if isinstance(store.get('dlog_hist'),list) else []
        dlog=[float(x) for x in dlog if isinstance(x,(int,float))][-200:]

        depths=getattr(state,'order_depths',{}) or {}
        sym_u=_symbol_for_product(state,UNDERLYING)
        if sym_u is None or sym_u not in depths:
            return {},0,json.dumps(store,separators=(',',':'))
        du=depths[sym_u]
        bu=getattr(du,'buy_orders',{}) or {}
        su=getattr(du,'sell_orders',{}) or {}
        if not bu or not su:
            return {},0,json.dumps(store,separators=(',',':'))
        ubb=max(bu.keys()); uba=min(su.keys())
        S=0.5*(float(ubb)+float(uba))

        abs_dS=abs(S-hist[-1]) if hist else 0.0
        hist.append(S)
        if len(hist)>=2 and hist[-2]>0:
            dlog.append(math.log(hist[-1]/hist[-2]))
        hist=hist[-120:]; dlog=dlog[-200:]
        z=_rolling_z(dlog,Z_WINDOW)
        burst=(z>=Z_SPIKE) or (abs_dS>=ABS_DS_SPIKE)
        atm_rel=ATM_REL_BURST if burst else ATM_REL_CALM
        lot_cap=BURST_MAX_LOT if burst else 10_000
        exec_min_q=EXEC_MIN_Q_BURST if burst else EXEC_MIN_Q_CALM

        ts=int(getattr(state,'timestamp',0))
        if ts//100 < WARMUP_DIV100:
            store['s_hist']=hist; store['dlog_hist']=dlog
            return {},0,json.dumps(store,separators=(',',':'))

        T=t_years(int(getattr(state,'csv_day',0)), ts)

        xs=[]; ys=[]; ws=[]
        info={}  # k -> dict
        gated_exec=0
        for k in STRIKES:
            if abs(k/S-1.0)>atm_rel:
                continue
            p=f'VEV_{k}'
            sym=_symbol_for_product(state,p)
            if sym is None or sym not in depths:
                continue
            d=depths[sym]
            b=getattr(d,'buy_orders',{}) or {}
            s=getattr(d,'sell_orders',{}) or {}
            if not b or not s:
                continue
            bb=max(b.keys()); ba=min(s.keys())
            bv=abs(int(b.get(bb,0))); av=abs(int(s.get(ba,0)))
            spread,depth,q=_quality(int(bb),int(ba),int(bv),int(av))
            m=0.5*(float(bb)+float(ba))
            iv=implied_vol_call(m,S,float(k),T)
            if not (math.isfinite(iv) and 0.01<=iv<=3.0):
                continue
            x=math.log(float(k)/S)
            xs.append(x); ys.append(iv)
            ws.append(max(0.2, min(30.0, q/6.0)))
            info[k]={'sym':sym,'bb':int(bb),'ba':int(ba),'bv':int(bv),'av':int(av),'q':float(q),'spread':float(spread)}

        if len(xs)<3:
            store['s_hist']=hist; store['dlog_hist']=dlog; store['gated_exec_last']=0
            return {},0,json.dumps(store,separators=(',',':'))

        coef,(xlo,xhi)=_weighted_cubic_fit(xs,ys,ws)
        pos=getattr(state,'position',{}) or {}
        out={}
        edge=EDGE_BURST if burst else EDGE_CALM

        for k,meta in info.items():
            if meta['spread']>EXEC_MAX_SPREAD or meta['q']<exec_min_q:
                gated_exec += 1
                continue
            x=min(max(math.log(float(k)/S),xlo),xhi)
            iv_hat=max(0.01,min(3.0,_eval_poly(coef,x)))
            theo,delta,vega=bs_call(S,float(k),T,iv_hat)
            sym=meta['sym']; bb=meta['bb']; ba=meta['ba']; bv=meta['bv']; av=meta['av']
            pos_k=int(pos.get(sym,0)); lim=300
            buy_cap=min(lim-pos_k,lot_cap); sell_cap=min(lim+pos_k,lot_cap)
            bid_px=int(math.floor(theo-edge)); ask_px=int(math.ceil(theo+edge))
            orders=[]
            if ba<=bid_px and buy_cap>0:
                q=max(0,min(buy_cap,av))
                if q>0:
                    orders.append(Order(sym,int(ba),int(q))); buy_cap-=q
            if bb>=ask_px and sell_cap>0:
                q=max(0,min(sell_cap,bv))
                if q>0:
                    orders.append(Order(sym,int(bb),-int(q))); sell_cap-=q
            if buy_cap>0:
                orders.append(Order(sym, int(min(bb+1,bid_px)), int(min(buy_cap,lot_cap))))
            if sell_cap>0:
                orders.append(Order(sym, int(max(ba-1,ask_px)), -int(min(sell_cap,lot_cap))))
            if orders:
                out[sym]=orders

        store['s_hist']=hist; store['dlog_hist']=dlog
        store['last_burst']=bool(burst)
        store['fit_coef']=coef
        store['gated_exec_last']=int(gated_exec)
        return out,0,json.dumps(store,separators=(',',':'))
