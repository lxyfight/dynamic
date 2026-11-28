"""Spread required delivery trips using only their known D-only timetable.

The first trip keeps its departure. Each additional trip targets the previous
trip's scheduled return, capped by known hard-delivery slack and dispatch cutoff.
There is no fitted offset, no added vehicle/trip, and no hidden-demand lookup.
"""
from copy import deepcopy
from math import floor
from e1_sim.quantities import dec
from e1_sim.kernel import require
from e1_sim.constraints import distance
from e1_sim.initializer import digest
from profit_dispatch_timing import delivery_replay
from e1_sim.initial_timing import load_end,flight_end

def stagger(ins):
    ins=deepcopy(ins)
    trips=sorted((a for a in ins['input']['actions'] if a['kind']=='truck_dispatch'),key=lambda a:(dec(a['at']),a['actor']))
    if len(trips)<2:
        ins['initial_stagger']=dict(active=False,offsets={},public_D_only=True,static_cost_unchanged=True)
        return ins
    base,raw=delivery_replay(ins,ins['input']);cfg=ins['configuration']
    slack=max(0,floor(min(o['deadline']-o['completed_at'] for o in base.orders.values())))
    offsets={};previous_return=dec(0)
    for i,trip in enumerate(trips):
        shift=dec(0) if i==0 else min(dec(slack),max(dec(0),previous_return-dec(trip['at'])),dec(cfg['truck_dispatch_cutoff'])-dec(trip['at']))
        offsets[trip['actor']]=shift
        previous_return=base.trucks[trip['actor']]['return_time']+shift
    actions=ins['input']['actions']
    for a in actions:
        if a.get('actor') in offsets:a['at']=str(dec(a['at'])+offsets[a['actor']])
    orders={o['id']:o for o in raw['orders']};ready={o:dec(0) for o in orders}
    for a in actions:
        if a['kind']=='truck_unload':
            end=dec(a['at'])+dec(cfg['truck_handling_setup'])+dec(cfg['truck_handling_per_kg'])*sum(a['quantities'].values())
            for oid in a['quantities']:ready[oid]=max(ready[oid],end)
    for drone in ins['input']['drones']:
        free=dec(0);home=ins['network']['stations'][drone['home']]
        for a in sorted((a for a in actions if a.get('actor')==drone['id']),key=lambda a:dec(a['at'])):
            if a['kind']=='drone_load':
                start=max(dec(a['at']),free,max(ready[o] for o in a['orders']))
                free=load_end(cfg,orders,a['orders'],start)
            elif a['kind']=='drone_sortie':
                start=max(dec(a['at']),free)
                free,_=flight_end(cfg,ins['network'],orders,drone['home'],a['visit_order'],start)
            else:raise ValueError('unexpected I0 drone command')
            a['at']=str(start)
    final,shifted=delivery_replay(ins,ins['input'])
    require(all(abs(v-final.ledger[k])<dec('1e-7') for k,v in base.ledger.items()),'staggering changed static ledger')
    ins['base_i0_hash']=ins['i0_hash'];ins['i0_hash']=digest(shifted);ins['input_hash']=digest(ins['input'])
    ins['initial_stagger']=dict(active=True,offsets={k:str(v) for k,v in offsets.items()},public_D_only=True,
        static_cost_unchanged=True,minimum_delivery_slack=int(slack))
    return ins
