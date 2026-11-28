"""Initial departure timing uses mandatory deliveries only, never hidden P.

Translate the whole D-only timetable by a common nonnegative offset bounded by
the minimum hard-delivery slack and the last legal truck-dispatch time. The
requested offset is a policy parameter fixed before a confirmation scenario.
"""
from copy import deepcopy
from math import floor
from e1_sim.online import prepare_online
from e1_sim.kernel import COSTS,require
from e1_sim.audit import audit_ledger
from e1_sim.quantities import dec,plain
from e1_sim.initializer import digest
from e1_sim.constraints import distance
from e1_sim.initial_timing import load_end,flight_end

def delivery_replay(ins,raw):
    raw=deepcopy(raw);raw['orders']=[o for o in raw['orders'] if o['kind']=='D']
    raw['external_events']=[];raw['public_information']=dict(future_count=0,revelation_closed=True)
    state,_=prepare_online(raw,ins['configuration'],ins['network'],full_day=True)
    state._record_snapshots=False;state.run(stop_at_terminal=True)
    require(state.is_terminal(),'D-only initialization did not terminate')
    require(all(o['completed_at'] is not None and o['completed_at']<=o['deadline'] for o in state.orders.values()),'delayed I0 missed delivery')
    initial=deepcopy(state.initial);initial['stop_time']=state.now
    audit=audit_ledger(ins['configuration'],ins['network'],initial,state.trace)
    require(not audit['errors'],'delayed I0 ledger errors')
    require(all(abs(dec(v)-dec(audit['ledger'][k]))<dec('1e-7') for k,v in state.ledger.items()),'delayed I0 ledger mismatch')
    return state,raw

def with_departure_delay(ins,requested):
    require(requested>=0,'negative departure delay')
    ins=deepcopy(ins);base,delivery_input=delivery_replay(ins,ins['input'])
    slack=min(float(o['deadline']-o['completed_at']) for o in base.orders.values())
    cutoff=float(ins['configuration']['truck_dispatch_cutoff'])
    last=max(float(a['at']) for a in delivery_input['actions'] if a['kind']=='truck_dispatch')
    delay=min(float(requested),float(max(0,floor(slack))),cutoff-last)
    # Keep exact decimal times: binary float translation can place a takeoff
    # a fraction before its loading-completion event and break commitment.
    actions=ins['input']['actions'];cfg=ins['configuration']
    for action in actions:action['at']=str(dec(action['at'])+dec(delay))
    # Recompute dependent drone starts from their translated prerequisites.
    # Decimal addition is not associative at finite precision: blindly adding
    # the same offset to stored end times can leave a start 1e-25 before return.
    orders={o['id']:o for o in delivery_input['orders']};ready={o:dec(0) for o in orders}
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
            else:raise ValueError('unexpected initial drone command')
            a['at']=str(start)
    final,shifted=delivery_replay(ins,ins['input'])
    for key,value in base.ledger.items():
        require(abs(value-final.ledger[key])<dec('1e-7'),'departure shift changed static cost')
    ins['base_i0_hash']=ins['i0_hash'];ins['i0_hash']=digest(shifted);ins['input_hash']=digest(ins['input'])
    ins['initial_timing']=dict(requested=requested,actual=delay,minimum_delivery_slack=slack,
        public_D_only=True,static_cost_unchanged=True)
    return ins
