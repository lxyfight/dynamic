"""WTQ closing state machine and local execution fixtures.

Local fixtures may carry external recovery contracts. They are not full-day
feasibility certificates and are never passed to the Myopic completion oracle.
"""
from copy import deepcopy
from .kernel import Replay, require
from .quantities import dec, plain
from .audit import audit_ledger


class WTQController:
    def __eq__(self, other):
        return isinstance(other, WTQController) and self.__dict__==other.__dict__

    def __init__(self, policy, controlled, inherited=None):
        self.policy=deepcopy(policy)
        require(dec(policy['threshold_time'])>=0 and dec(policy['threshold_weight'])>0,'invalid WTQ threshold')
        self.states={tid:deepcopy((inherited or {}).get(tid,{})) for tid in controlled}
        for row in self.states.values():
            row.setdefault('closing_since',None);row.setdefault('snapshot',[])
            row.setdefault('loaded_at_this_stop',[])
            row.setdefault('stop_arrival',policy.get('stop_arrival'))

    def view(self,s,tid):
        row=deepcopy(self.states[tid]);t=s.trucks[tid]
        ready={l['order'] for l in s.lots if l['state']=='available' and l['holder']==t['location'] and s.orders[l['order']]['kind']=='P'}
        loaded=set(row['loaded_at_this_stop'])
        loaded.update(oid for e in s.trace if e['kind']=='truck_load_started' and e['actor']==tid for oid in e['orders'])
        if t['active_batch']:loaded.update(t['active_batch']['orders'])
        row.update(cumulative_weight=sum(s.orders[o]['weight'] for o in loaded|ready),
                   executable_snapshot=sorted(set(row['snapshot'])&ready),threshold_weight=self.policy['threshold_weight'])
        return row,ready

    def on_event(self,s):
        for tid in sorted(self.states):
            t=s.trucks[tid]
            if t['location'] not in s.network['stations']:continue
            row,ready=self.view(s,tid)
            require(row['stop_arrival'] is not None,'missing station arrival time')
            if row['closing_since'] is None and (s.now>=dec(row['stop_arrival'])+dec(self.policy['threshold_time']) or row['cumulative_weight']>=dec(self.policy['threshold_weight'])):
                self.states[tid].update(closing_since=s.now,snapshot=sorted(ready))
                row,_=self.view(s,tid)
            if row['closing_since'] is None or t['active_operation'] is not None:continue
            # Explicit fixture actions keep their specified role; do not duplicate.
            if any(a.get('actor')==tid for a in s._actions.get(s.now,[])):continue
            ids=row['executable_snapshot']
            if s.batch_permission=='B1' and t['completed_batches']:ids=[]
            capacity=s.cfg['truck_capacity']-t['onboard_weight']
            chosen=[]
            for oid in ids:
                q=s.orders[oid]['weight']
                if q<=capacity:chosen.append(oid);capacity-=q
            action=dict(at=s.now,actor=tid,station=t['location'])
            if chosen:action.update(kind='truck_load',orders=chosen)
            else:action.update(kind='truck_continue_circuit')
            s._actions.setdefault(s.now,[]).append(action)


class ControlledReplay(Replay):
    def observations(self):
        result=super().observations()
        if hasattr(self,'controller'):
            result['controller']={tid:self.controller.view(self,tid)[0] for tid in self.controller.states}
        for tid,t in result['truck'].items():t['remaining_capacity']=self.cfg['truck_capacity']-t['onboard_weight']
        loads=[oid for e in self.trace if e['kind']=='truck_load_started' for oid in e['orders']]
        result['events']['duplicate_pickup_loads']=sorted({oid for oid in loads if loads.count(oid)>1})
        return plain(result)


def build_controller_replay(package,case):
    raw=deepcopy(case['input']);data=deepcopy(raw)
    cfg=deepcopy(package['configuration']);cfg.update(case.get('parameter_overrides',{}))
    inherited=[]
    for t in data['trucks']:
        op=t['active_operation']
        if op:
            require(op['kind']=='truck_load','unsupported inherited WTQ operation')
            inherited.append((t['id'],t['location'],op));t['active_operation']=None
            for l in data['inventory']:
                if l['order'] in op['orders']:l.update(state='available',holder=t['location'])
    data['external_events']=[e for e in data['external_events'] if e['kind']!='threshold_timer']
    for a in data['actions']:
        if a['kind']=='truck_load':a['station']=next(t['location'] for t in data['trucks'] if t['id']==a['actor'])
    stops={t['id']:{t['location']} for t in data['trucks']}
    s=ControlledReplay(cfg,package['networks'][case['network']],data,planned_stops=stops)
    for tid,station,op in inherited:
        weight=sum(s.orders[o]['weight'] for o in op['orders'])
        require(dec(op['end'])-dec(op['start'])==s._handling('truck',weight),'inherited load duration mismatch')
        if dec(op['start'])>s.now:
            s._actions.setdefault(dec(op['start']),[]).append(dict(at=op['start'],kind='truck_load',actor=tid,station=station,orders=op['orders']))
            continue
        require(s.now<=dec(op['end']),'expired inherited batch')
        for oid in op['orders']:s._transfer(oid,s.orders[oid]['weight'],'available',station,'loading',tid)
        t=s.trucks[tid];t.update(active_operation=dict(kind='truck_load',end=dec(op['end'])),
                                active_batch=dict(orders=op['orders'],weight=weight,start=dec(op['start']),end=dec(op['end'])),loading_end=dec(op['end']))
        s._schedule(dec(op['end']),'truck_load_done',dict(actor=tid,orders=op['orders']))
    s.controller=WTQController(raw['policy'],raw['controlled_trucks'],raw.get('controller_state'))
    s.check_invariants()
    return s


def run_controller_case(package,case):
    s=build_controller_replay(package,case)
    ticks={dec(a['at']) for a in case['assertions'] if a['at'] is not None}
    for row in s.controller.states.values():
        deadline=dec(row['stop_arrival'])+dec(s.controller.policy['threshold_time'])
        if s.start<=deadline<=s.stop:ticks.add(deadline)
    s.run(ticks,decision_hook=lambda state:state.controller.on_event(state))
    audit=audit_ledger(s.cfg,s.network,case['input'],s.trace)
    audit['matches_online']=not audit['errors'] and all(abs(dec(v)-dec(s.observations()['ledger'][k]))<=dec('1e-6') for k,v in audit['ledger'].items())
    return s,audit
