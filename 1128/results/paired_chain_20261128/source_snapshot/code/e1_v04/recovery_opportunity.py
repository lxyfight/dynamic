"""A linear one-step policy improvement using expected demand before a truck passes.

Only the training harness can access the completed day's exogenous arrivals.
The fitted coefficient measures CNY per expected pickup opportunity.
"""
from copy import deepcopy
from math import floor
import hashlib
import pickle

from e1_sim.audit import audit_ledger
from e1_sim.formal_learning import FormalPolicy
from e1_sim.initializer import digest
from e1_sim.kernel import COSTS, require
from e1_sim.online import prepare_online
from e1_sim.online import evaluate_plan
from e1_sim.quantities import dec, plain


def coverage(known, candidate):
    """Expected distinct arrivals until the last eligible-city truck visit.

    A newly revealed order can be scheduled for a truck even when drone return
    needs a legal delay. Coverage is a proxy; feasibility is always replayed.
    Same-city stations and overlapping trips are counted only once.
    """
    if known.public.get('revelation_closed'):
        return 0.
    now=float(known.now)
    dispatches={a['actor'] for a in candidate['plan']['actions'] if a['kind']=='truck_dispatch'
                and not a.get('delivery_orders') and not a.get('quantities')}
    if not dispatches:
        return 0.
    last={};other={}
    for event in candidate['trace']:
        if event['kind']!='road_started':
            continue
        station=event['source']
        if station in known.network['stations']:
            city='C'+station[1:-1]
            times=last if event['actor'] in dispatches else other
            times[city]=max(times.get(city,now),float(event['at']))
    total=0.
    for city,end in last.items():
        for segment in known.public['demand_model']['rates_by_city'][city]:
            lower=max(floor(other.get(city,now))+1,segment['start'])
            upper=min(floor(end),segment['end'])
            total+=max(0,upper-lower+1)*segment['rate']
    return total


class RecoveryPolicy(FormalPolicy):
    def __init__(self,theta,cache=None,levels=()):
        super().__init__(candidate_budget=16,beam_width=4,max_joint_requests=3)
        self.theta=theta
        self.records=[]
        self.cache=cache
        self.levels=levels

    def decide(self,known):
        if self.cache is None:
            return super().decide(known)
        key=hashlib.sha256(pickle.dumps(known.__dict__,protocol=5)).digest()
        if key not in self.cache:
            evaluated,_,_=self.propose(known)
            ranked=self.rank(known,evaluated)
            self.cache[key]={w:self.pick(known,ranked,w) for w in self.levels}
        plan,record=self.cache[key][self.theta]
        self.records.append(record)
        return deepcopy(plan),dict(selected=record['decision'],at=known.now)

    def final_pool(self,known,evaluated):
        rows=super().final_pool(known,evaluated)
        variants=[]
        # Shift a dedicated recovery trip as a whole. Its physical route and all
        # loading batches stay fixed. Only future dispatch commands are eligible.
        for c in evaluated:
            for cmd in c['plan']['actions']:
                if cmd['kind']!='truck_dispatch' or cmd.get('delivery_orders') or cmd.get('quantities'):
                    continue
                who=cmd['actor']
                loads=[oid for a in c['plan']['actions'] if a['kind']=='truck_load' and a['actor']==who for oid in a['orders']]
                if not loads:
                    continue
                slack=min(dec(c['state'].orders[o]['deadline'])-dec(c['state'].orders[o]['completed_at']) for o in loads)
                shift=min(slack,dec(known.cfg['truck_dispatch_cutoff'])-dec(cmd['at']))
                if shift<=0:continue
                plan=deepcopy(c['plan'])
                plan['id']+=f'~defer_{who}'
                for a in plan['actions']:
                    if a['actor']==who:a['at']=dec(a['at'])+shift
                v=evaluate_plan(known,plan,self.waiting)
                if v['feasible'] and v['allowed']:variants.append(v)
        return rows+variants

    def rank(self,known,evaluated):
        return [(float(c['score']),coverage(known,c),c) for c in self.final_pool(known,evaluated)]

    @staticmethod
    def pick(known,ranked,theta):
        j,phi,c=min(ranked,key=lambda v:(-(v[0]+theta*v[1]),v[2]['id']))
        record=dict(t=float(known.now),j=j,x=phi,plan=digest(c['plan']),
                    decision=c['id'],differs=c['id']!=min(ranked,key=lambda v:(-v[0],v[2]['id']))[2]['id'])
        return c['plan'],record

    def choose(self,known,evaluated):
        ranked=self.rank(known,evaluated)
        plan,record=self.pick(known,ranked,self.theta)
        self.records.append(record)
        return next(c for j,x,c in ranked if c['plan'] is plan)


def run(instance,theta,*,fit_samples=False,baseline=False,cache=None,levels=(),policy_override=None,metrics=None):
    policy=FormalPolicy(group='resource',candidate_budget=16) if baseline else RecoveryPolicy(theta,cache,levels)
    if policy_override is not None:
        policy = policy_override
    cfg=instance['configuration']
    state,driver=prepare_online(instance['input'],cfg,instance['network'],full_day=True)
    state._record_snapshots=False
    driver.policy=policy
    state.run(decision_hook=driver.on_event,stop_at_terminal=True)
    require(state.is_terminal() and state.now>=instance['input']['public_information']['revelation_end'],'incomplete service day')
    require(all(o['completed_at'] is not None and o['completed_at']<=o['deadline'] for o in state.orders.values() if o['kind']=='D'),'missed delivery')
    pickups=[o for o in state.orders.values() if o['kind']=='P']
    require(all(o['accepted'] is not None and (not o['accepted'] or o['completed_at'] is not None) for o in pickups),'unresolved pickup')
    initial=deepcopy(state.initial);initial['stop_time']=state.now
    audit=audit_ledger(cfg,state.network,initial,state.trace)
    net=state.ledger['revenue']-sum(state.ledger[k] for k in COSTS)
    ledger=plain(dict(state.ledger,net=net))
    require(not audit['errors'] and all(abs(dec(v)-dec(audit['ledger'][k]))<dec('1e-7') for k,v in ledger.items()),'ledger mismatch')
    records=([dict(t=a['at'],plan=a['plan_hash']) for a in policy.selection_log] if baseline else policy.records)
    row=dict(net=float(net),plans=digest([(r['t'],r['plan']) for r in records]),trace=digest(plain(state.trace)),
        decisions=len(records),accepted=sum(o['accepted'] for o in pickups),ledger=ledger,changes=sum(r.get('differs',False) for r in records))
    if fit_samples:
        # The all-future-rejection loss is common to every candidate at t.
        # Adding its realization here reduces exogenous day-to-day target noise.
        row['samples']=[([r['x']],float(net)-r['j']+sum(float(o['weight'])*cfg['rejection_per_kg']
            for o in pickups if float(o['release'])>r['t'])) for r in records]
    if metrics is not None:
        row['operation_metrics'] = metrics(state, driver, policy)
    return row
