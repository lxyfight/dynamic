"""Compile a complete known-task candidate through the actual WTQ state machine.

All quantities and release/ready events come from the public candidate. Stranded
accepted pickups receive explicit additional recovery trips before acceptance.
The returned command list is rechecked with the ordinary physical oracle.
"""
from copy import deepcopy
from .kernel import Replay,require
from .quantities import dec,InvalidAction
from .online import install_plan,evaluate_plan,_bound,_new_truck,_road_prefix

class RecordingReplay(Replay):
    def apply(self,action):
        super().apply(action)
        self.executed_commands.append(deepcopy(action))

class ClosingController:
    def __init__(self,T,Q,stops):
        self.T,self.Q=dec(T),dec(Q);self.stops=stops;self.records={};self.log=[]
        require(self.T>=0 and self.Q>0,'invalid WTQ thresholds')
    def historical_closure(self,s,tid,station,arrival):
        def ready_at(at):
            taken={o for e in s.trace if e['kind']=='truck_load_started' and dec(e['at'])<at for o in e['orders']}
            return {oid for oid,o in s.orders.items() if o['kind']=='P' and o['station']==station and o['available_at'] is not None and dec(o['available_at'])<=at and oid not in taken}
        candidates={arrival}
        if arrival+self.T<=s.now:candidates.add(arrival+self.T)
        candidates.update(dec(o['available_at']) for o in s.orders.values() if o['available_at'] is not None and arrival<=dec(o['available_at'])<=s.now)
        for at in sorted(candidates):
            loaded={o for e in s.trace if e['kind']=='truck_load_started' and e['actor']==tid and e['station']==station and arrival<=dec(e['at'])<at for o in e['orders']}
            ready=ready_at(at)
            if at>=arrival+self.T or sum(s.orders[o]['weight'] for o in loaded|ready)>=self.Q:
                return dict(closed=at,snapshot=ready)
        return dict(closed=None,snapshot=set())
    def on_event(self,s):
        for tid in sorted(s.trucks):
            truck=s.trucks[tid];station=truck['location']
            if station not in self.stops.get(tid,set()):continue
            arrivals=[dec(e['at']) for e in s.trace if e['kind']=='truck_arrived' and e['actor']==tid and e['destination']==station]
            arrival=max(arrivals,default=s.start);key=(tid,station,arrival)
            if key not in self.records:self.records[key]=self.historical_closure(s,tid,station,arrival)
            row=self.records[key]
            ready={l['order'] for l in s.lots if l['state']=='available' and l['holder']==station and s.orders[l['order']]['kind']=='P'}
            loaded={o for e in s.trace if e['kind']=='truck_load_started' and e['actor']==tid and e['station']==station and dec(e['at'])>=arrival for o in e['orders']}
            weight=sum(s.orders[o]['weight'] for o in ready|loaded)
            if row['closed'] is None and (s.now>=arrival+self.T or weight>=self.Q):
                row.update(closed=s.now,snapshot=set(ready))
                self.log.append(dict(at=s.now,truck=tid,station=station,snapshot=sorted(ready),weight=weight))
            if row['closed'] is None and arrival+self.T>s.now:s._actions.setdefault(arrival+self.T,[])
            if truck['active_operation'] is not None:continue
            # A committed delivery replenishment must finish before recovery.
            if truck['unloading_phase']=='not_started':continue
            allowed=ready if row['closed'] is None else ready&row['snapshot']
            capacity=s.cfg['truck_capacity']-truck['onboard_weight'];chosen=[]
            if s.batch_permission!='B1' or truck['completed_batches']==0:
                for oid in sorted(allowed):
                    q=s.orders[oid]['weight']
                    if q<=capacity:chosen.append(oid);capacity-=q
            a=dict(at=s.now,actor=tid,station=station)
            if chosen:a.update(kind='truck_load',orders=chosen)
            elif row['closed'] is not None:a.update(kind='truck_continue_circuit')
            else:continue
            s._actions.setdefault(s.now,[]).append(a)

def _run(known,plan,T,Q,extra):
    s=deepcopy(known);s._has_run=False;s._record_snapshots=False;s._snapshots={};s.start=s.now;s.full_day=False
    work=deepcopy(plan)
    work['new_trucks']+=deepcopy(extra['trucks']);work['actions']+=deepcopy(extra['actions'])
    install_plan(s,work)
    stops=deepcopy(s._stops)
    for tid,t in s.trucks.items():
        if t['location'] in s.network['stations']:stops.setdefault(tid,set()).add(t['location'])
    for tid,station in extra['stops']:stops.setdefault(tid,set()).add(station)
    # Preserve intended stops while removing obsolete recovery reservations.
    for t in list(s._actions):
        s._actions[t]=[a for a in s._actions[t] if a['kind'] not in ('truck_load','truck_continue_circuit','truck_depart')]
    s._stops=stops
    s.public=dict(future_count=0,revelation_closed=True)
    s.stop=_bound(s)+dec(T)*(len(s.network['stations'])+1)*(len(s.trucks)+1)+100
    s.__class__=RecordingReplay;s.executed_commands=[]
    controller=ClosingController(T,Q,stops)
    s.run(decision_hook=controller.on_event,stop_at_terminal=True)
    return s,controller

def compile_candidate(known,candidate,T,Q):
    plan=candidate['plan'];extra=dict(trucks=[],actions=[],stops=[])
    try:
        first,controller=_run(known,plan,T,Q,extra)
        left=[o for o in first.orders.values() if o['kind']=='P' and o['accepted'] is True and o['completed_at'] is None]
        used=set(first.trucks)|set(first.drones)
        for o in left:
            require(o['available_at'] is not None,'no complete drone recovery in WTQ candidate')
            station=o['station'];departures=[dec(e['at']) for e in first.trace if e['kind']=='road_started' and e['source']==station]
            arrival=max(dec(o['available_at']),max(departures,default=known.now))
            depart=max(known.now,arrival-_road_prefix(known,station))
            require(depart<=dec(known.cfg['truck_dispatch_cutoff']),'WTQ fallback dispatch would exceed cutoff')
            number=1
            while 'WQ%05d'%number in used:number+=1
            tid='WQ%05d'%number;used.add(tid)
            extra['trucks'].append(_new_truck(tid));extra['stops'].append((tid,station))
            extra['actions'].append(dict(at=depart,kind='truck_dispatch',actor=tid,delivery_orders=[]))
        final,controller=_run(known,plan,T,Q,extra) if left else (first,controller)
        require(final.is_terminal(),'WTQ finite recovery did not complete')
        compiled=deepcopy(plan)
        compiled['new_trucks']+=extra['trucks']
        # Online event barriers expose a default departure at pass-through
        # stations. The ordinary oracle traverses those stations automatically;
        # recording that default as an additional command would start the same
        # road leg twice. Preserve departures only at the compiler's real stops.
        compiled['actions']=[a for a in final.executed_commands
            if not (a['kind']=='truck_continue_circuit'
                    and a.get('station') not in final._stops.get(a['actor'],set()))]
        compiled['wakeups']=sorted({arrival+dec(T) for tid,station,arrival in controller.records if arrival+dec(T)>=known.now})
        checked=evaluate_plan(known,compiled,'WK')
        require(checked['feasible'],'compiled WTQ plan failed ordinary replay: '+str(checked.get('reason')))
        checked['wtq_closures']=controller.log
        return checked
    except (InvalidAction,KeyError) as exc:
        return dict(feasible=False,allowed=False,reason=str(exc),plan=plan,id=candidate['id'])
