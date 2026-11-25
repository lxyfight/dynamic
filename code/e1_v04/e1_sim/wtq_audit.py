"""Independent trace acceptance for threshold closure and immediate continuation."""
def audit_wtq(initial,trace,T,Q,capacity,batch='BM'):
    orders={o['id']:o for o in initial['orders']};homes={u['id']:u['home'] for u in initial['drones']}
    available={};loads=[];errors=[];stops=[]
    for e in trace:
        if e['kind']=='drone_handling_done' and not e['loading']:available[e['order']]=(homes[e['actor']],e['at'])
        if e['kind']=='truck_load_started':loads.append(e)
    def ready(station,at,tid=None):
        taken={o for e in loads if e['at']<at-1e-7 or (tid is not None and abs(e['at']-at)<1e-7 and e['actor']<tid) for o in e['orders']}
        return {o for o,(s,t) in available.items() if s==station and t<=at+1e-7 and o not in taken}
    for leave in [e for e in trace if e['kind']=='truck_left_station']:
        tid,station,end=leave['actor'],leave['station'],leave['at']
        arrival=max(e['at'] for e in trace if e['kind']=='truck_arrived' and e['actor']==tid and e['destination']==station and e['at']<=end)
        events=sorted({arrival,arrival+T}|{t for s,t in available.values() if s==station and arrival<=t<=end})
        close=None;snapshot=set()
        for at in events:
            loaded={o for e in loads if e['actor']==tid and e['station']==station and arrival<=e['at']<at-1e-7 for o in e['orders']}
            ids=ready(station,at)
            if at>=arrival+T-1e-7 or sum(orders[o]['weight'] for o in ids|loaded)>=Q:
                close=at;snapshot=ids;break
        if close is None or end<close-1e-7:errors.append(dict(truck=tid,station=station,error='left before closure'));continue
        operations=[e for e in trace if e.get('actor')==tid and e['kind'] in ('truck_load_started','truck_unload_started') and arrival<=e['at']<=end]
        free=max([close]+[e['end'] for e in operations if (e['at']<close-1e-7 or (e['kind']=='truck_unload_started' and e['at']<=close+1e-7)) and e['end']>close])
        post=[e for e in operations if e['kind']=='truck_load_started' and e['at']>=close-1e-7]
        for e in post:
            if abs(e['at']-free)>1e-7:errors.append(dict(truck=tid,station=station,error='delayed batch after closure',expected=free,actual=e['at']))
            if not set(e['orders'])<=snapshot:errors.append(dict(truck=tid,station=station,error='new goods after closure'))
            free=e['end']
        if abs(end-free)>1e-7:errors.append(dict(truck=tid,station=station,error='delayed departure after closure',expected=free,actual=end))
        # Capacity at departure: remaining D plus all pickup batches already
        # completed on this one-circuit trip, including earlier stations.
        pending_d=sum(sum(e['quantities'].values()) for e in trace if e['kind']=='truck_unload_started' and e['actor']==tid and e['end']>end+1e-7)
        onboard_p=sum(orders[o]['weight'] for e in loads if e['actor']==tid and e['end']<=end+1e-7 for o in e['orders'])
        used_batches=sum(e['actor']==tid and e['station']==station and arrival<=e['at']<=end for e in loads)
        can_load=batch!='B1' or used_batches==0
        leftover=ready(station,end,tid)&snapshot
        if can_load and any(orders[o]['weight']<=capacity-pending_d-onboard_p for o in leftover):
            errors.append(dict(truck=tid,station=station,error='left loadable snapshot goods'))
        stops.append(dict(truck=tid,station=station,arrival=arrival,closing=close,snapshot=sorted(snapshot),departure=end))
    return dict(passed=not errors,errors=errors,stops=stops)
