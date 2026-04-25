
#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'Prosperity4Data' / 'ROUND_3'
OUT = Path(__file__).resolve().parent / 'underlying_propagation_strike_metrics.json'
U='VELVETFRUIT_EXTRACT'
STRIKES=[4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]
VOUS=[f'VEV_{k}' for k in STRIKES]

def bba(row):
    bids=[]; asks=[]
    for i in (1,2,3):
        bp,bv=row.get(f'bid_price_{i}'),row.get(f'bid_volume_{i}')
        ap,av=row.get(f'ask_price_{i}'),row.get(f'ask_volume_{i}')
        if bp and bv and int(float(bv))>0: bids.append(float(bp))
        if ap and av and int(float(av))>0: asks.append(float(ap))
    if not bids or not asks: return None,None
    return max(bids), min(asks)

stats={k:{'dS':[],'dV':[],'dS_up':[],'dV_up':[],'dS_dn':[],'dV_dn':[],'spr_pre':[],'spr_post_shock':[]} for k in STRIKES}

for day in (0,1,2):
    by=defaultdict(dict)
    with (DATA/f'prices_round_3_day_{day}.csv').open() as f:
        r=csv.DictReader(f,delimiter=';')
        for row in r:
            p=row['product']
            if p==U or p in VOUS:
                by[int(row['timestamp'])][p]=row
    tss=sorted(by.keys())
    prev=None
    for ts in tss:
        s=by[ts]
        if U not in s: continue
        bb,ba=bba(s[U])
        if bb is None or ba<=bb: continue
        um=0.5*(bb+ba)
        cur={}
        for k,v in zip(STRIKES,VOUS):
            if v not in s: continue
            b,o=bba(s[v])
            if b is None or o<=b: continue
            cur[k]={'mid':0.5*(b+o),'spr':0.5*(o-b)}
        if prev is not None:
            dS=um-prev['um']
            for k in STRIKES:
                if k in cur and k in prev['v']:
                    dV=cur[k]['mid']-prev['v'][k]['mid']
                    st=stats[k]
                    st['dS'].append(dS); st['dV'].append(dV)
                    if dS>0:
                        st['dS_up'].append(dS); st['dV_up'].append(dV)
                    elif dS<0:
                        st['dS_dn'].append(-dS); st['dV_dn'].append(-dV)
                    st['spr_pre'].append(prev['v'][k]['spr'])
                    # shock event uses underlying jump magnitude threshold
                    if abs(dS)>=2.5:
                        st['spr_post_shock'].append(cur[k]['spr'])
        prev={'um':um,'v':cur}

out={'method':'per-tick response of voucher mids/spreads to underlying mid changes','days':[0,1,2],'by_strike':{}}
for k in STRIKES:
    st=stats[k]
    def slope(x,y):
        if len(x)<20: return None
        X=np.asarray(x); Y=np.asarray(y)
        den=float(np.dot(X,X))
        if den<1e-9: return None
        return float(np.dot(X,Y)/den)
    beta_all=slope(st['dS'],st['dV'])
    beta_up=slope(st['dS_up'],st['dV_up'])
    beta_dn=slope(st['dS_dn'],st['dV_dn'])
    spr_pre=float(np.mean(st['spr_pre'])) if st['spr_pre'] else None
    spr_post=float(np.mean(st['spr_post_shock'])) if st['spr_post_shock'] else None
    widen=(spr_post/spr_pre) if (spr_pre and spr_post) else None
    out['by_strike'][str(k)]={
      'samples':len(st['dS']),
      'beta_all':beta_all,
      'beta_up':beta_up,
      'beta_down':beta_dn,
      'beta_asym_up_minus_down': (beta_up-beta_dn) if (beta_up is not None and beta_dn is not None) else None,
      'mean_half_spread':spr_pre,
      'mean_half_spread_after_shock_abs_dS_ge_2p5':spr_post,
      'spread_widen_ratio_after_shock':widen,
    }

OUT.write_text(json.dumps(out,indent=2))
print(OUT)
