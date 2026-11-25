"""Audited common inputs and residual-value policies for a development comparison."""
from copy import deepcopy
from random import Random
from time import perf_counter
import numpy as np
from recovery_opportunity import RecoveryPolicy, coverage
from e1_sim.resource_features import resource_windows
from e1_sim.information import expected_arrivals, closed
from e1_sim.initializer import digest
from e1_sim.online import prepare_online
from e1_sim.audit import audit_ledger
from e1_sim.kernel import COSTS, require
from e1_sim.quantities import dec, plain

NAMES=['extra_recovery_exposure','time_fraction','expected_remaining_per_city',
       'new_acceptances_per_city','new_trips_per_city','remaining_truck_road_time',
       'remaining_sorties_per_city','pickup_at_customer','pickup_on_drone',
       'pickup_at_station','pickup_on_truck','pickup_handling',
       'minimum_delivery_slack','maximum_pickup_lateness','mean_recovery_lag',
       'station_idle_time','station_capacity_time','drone_tail_availability',
       'station_matched_ready_opportunity','station_arrival_exposure']

def inputs(known,c):
    """Only known state and known-order deterministic continuation; never hidden arrivals."""
    if closed(known.public):return np.zeros(len(NAMES))
    now=float(known.now);H=float(known.public['revelation_end'])
    cities=len(known.public['demand_model']['rates_by_city']);Q=float(known.cfg['truck_capacity'])
    stations=len(known.network['stations']);finished=c['state'];plan=c['plan']
    accepted={oid for oid,o in known.orders.items() if o['kind']=='P' and
              (o['accepted'] is True or plan['decisions'].get(oid) is True) and o['completed_at'] is None}
    bins=[0.]*5
    for lot in known.lots:
        if lot['order'] not in accepted:continue
        state,holder=lot['state'],lot['holder']
        if state=='customer':ix=0
        elif state=='onboard':ix=1 if holder in known.drones else 3
        elif state=='available':ix=2
        elif state in ('loading','unloading'):ix=4
        else:continue
        bins[ix]+=lot['weight']/(Q*cities)
    ps=[o for o in finished.orders.values() if o['kind']=='P' and o['accepted'] is True and float(o['completed_at'])>now]
    ds=[o for o in finished.orders.values() if o['kind']=='D' and float(o['completed_at'])>now]
    windows,available=resource_windows(known,c)
    # Match ready pickup cargo to truck idle windows at its actual station.
    matched=0.
    load_at={oid:float(e['at']) for e in c['trace'] if e['kind']=='truck_load_started' for oid in e['orders']}
    unavailable={l['order'] for l in known.lots if l['order'] in accepted and
                 l['state'] in ('onboard','loading') and l['holder'] in known.trucks}
    for station,a,b,q in windows:
        weight=sum(o['weight'] for o in ps if o['station']==station and o['available_at'] is not None
                   and o['id'] not in unavailable and float(o['available_at'])<=a<load_at.get(o['id'],float('inf')))
        matched+=weight/Q*(b-a)/H*q
    truck_minutes=sum(max(0.,float(e['end'])-max(now,float(e['at']))) for e in c['trace']
                      if e['kind']=='road_started')
    x=[coverage(known,c),now/H,
       sum(expected_arrivals(known.public,city,now) for city in known.public['demand_model']['rates_by_city'])/cities,
       sum(plan['decisions'].values())/cities,len(plan['new_trucks'])/cities,truck_minutes/H/cities,
       sum(a['kind']=='drone_sortie' for a in plan['actions'])/cities]+bins+[
       min((max(0.,float(o['deadline'])-float(o['completed_at']))/H for o in ds),default=0.),
       max((max(0.,float(o['completed_at'])-float(o['deadline']))/H for o in ps),default=0.),
       np.mean([max(0.,float(o['completed_at'])-max(now,float(o['available_at'])))/H for o in ps]) if ps else 0.,
       sum(b-a for s,a,b,q in windows)/H/stations,
       sum((b-a)*q for s,a,b,q in windows)/H/stations,
       sum(available.values())/H/max(1,len(available)),matched/stations,
       sum(expected_arrivals(known.public,'C'+s[1:-1],a,b) for s,a,b,q in windows)/stations]
    require(len(x)==len(NAMES) and np.isfinite(x).all(),'invalid common inputs')
    return np.array(x,dtype=float)

def predict(model,x):
    x=np.asarray(x)
    if model['kind']=='residual':return x[:,0]*model['theta']+model['alpha']*predict(model['correction'],x)
    if model['kind']=='zero':return np.zeros(len(x))
    if model['kind']=='scalar':return x[:,0]*model['theta']
    h=(x-np.asarray(model['mean']))/np.asarray(model['std'])
    for i,(w,b) in enumerate(zip(model['weights'],model['biases'])):
        h=h@np.asarray(w)+np.asarray(b)
        if i<len(model['weights'])-1:h=np.maximum(h,0)
    return h.ravel()*100

class Policy(RecoveryPolicy):
    def __init__(self,model,seed=0,epsilon=0):
        super().__init__(0);self.model=model;self.rng=Random(seed);self.epsilon=epsilon
        self.timings=[];self.rows=[]
    def decide(self,known):
        start=perf_counter();result=super().decide(known);self.timings.append(perf_counter()-start);return result
    def choose(self,known,evaluated):
        pool=self.final_pool(known,evaluated)
        x=np.array([inputs(known,c) for c in pool]);j=np.array([float(c['score']) for c in pool])
        w=np.zeros(len(pool)) if closed(known.public) else predict(self.model,x)
        base=min(range(len(pool)),key=lambda i:(-j[i],pool[i]['id']))
        best=min(range(len(pool)),key=lambda i:(-(j[i]+w[i]),pool[i]['id']))
        selected=self.rng.randrange(len(pool)) if not closed(known.public) and self.rng.random()<self.epsilon else best
        c=pool[selected]
        self.rows.append(dict(t=float(known.now),x=x[selected].tolist(),j=float(j[selected]),w=float(w[selected]),
             closed=closed(known.public),plan=digest(c['plan']),differs=selected!=base,
             candidates=len(pool),unique_inputs=len({tuple(v) for v in x}),w_range=float(np.ptp(w))))
        return c

def execute(instance,model,seed=0,epsilon=0):
    policy=Policy(model,seed,epsilon);cfg=instance['configuration']
    state,driver=prepare_online(instance['input'],cfg,instance['network'],full_day=True)
    state._record_snapshots=False;driver.policy=policy
    begin=perf_counter();state.run(decision_hook=driver.on_event,stop_at_terminal=True);elapsed=perf_counter()-begin
    require(state.is_terminal(),'unfinished day')
    require(all(o['completed_at'] is not None and o['completed_at']<=o['deadline'] for o in state.orders.values() if o['kind']=='D'),'late mandatory delivery')
    require(all(o['completed_at'] is not None for o in state.orders.values() if o['kind']=='P' and o['accepted'] is True),'unfinished pickup')
    raw=deepcopy(state.initial);raw['stop_time']=state.now
    audited=audit_ledger(cfg,state.network,raw,state.trace)
    require(not audited['errors'],'independent accounting trace errors')
    require(all(abs(dec(v)-dec(audited['ledger'][k]))<dec('1e-7') for k,v in state.ledger.items()),'accounting mismatch')
    net=float(state.ledger['revenue']-sum(state.ledger[k] for k in COSTS))
    require(float(state.ledger['drone_activation'])==25*sum(e['kind']=='drone_takeoff' for e in state.trace),'sortie fee mismatch')
    for row in policy.rows:
        row['raw_y']=net-row['j']
        row['future_rejection_baseline']=sum(float(o['weight'])*cfg['rejection_per_kg'] for o in state.orders.values() if o['kind']=='P' and float(o['release'])>row['t'])
        row['y']=row['raw_y']+row['future_rejection_baseline']
    return dict(net=net,accepted=sum(o['kind']=='P' and o['accepted'] is True for o in state.orders.values()),
         rows=policy.rows,timings=policy.timings,seconds=elapsed,plans=digest([r['plan'] for r in policy.rows]),
         input_hash=instance['input_hash'],i0_hash=instance['i0_hash'],ledger=audited['ledger'])

def fit(days,kind,seed,epochs=250):
    """Equal total weight per day; weighted residual least squares, units 100 CNY."""
    rng=np.random.default_rng(seed);X=[];Y=[];SW=[]
    for d in days:
        rows=[r for r in d['rows'] if not r['closed']]
        for r in rows:X.append(r['x']);Y.append(r['y']/100);SW.append(1/max(1,len(rows)))
    x=np.asarray(X);y=np.asarray(Y)[:,None];sw=np.asarray(SW);sw/=sw.sum()
    mean=np.sum(x*sw[:,None],axis=0);std=np.sqrt(np.sum((x-mean)**2*sw[:,None],axis=0));std=np.maximum(std,.05)
    x=(x-mean)/std
    if kind=='linear':
        a=np.column_stack([x,np.ones(len(x))]);reg=np.eye(a.shape[1])*.01;reg[-1,-1]=0
        beta=np.linalg.solve(a.T@(a*sw[:,None])+reg,a.T@(y*sw[:,None]))
        weights=[beta[:-1]];biases=[beta[-1]]
    else:
        dims=[x.shape[1],64,64,1]
        weights=[rng.normal(0,np.sqrt(2/dims[i]),(dims[i],dims[i+1])) for i in range(3)]
        biases=[np.zeros(v) for v in dims[1:]];weights[-1]*=.01
        params=weights+biases;m=[np.zeros_like(p) for p in params];v=[p.copy() for p in m]
        for step in range(1,epochs+1):
            acts=[x]
            for i in range(3):
                z=acts[-1]@weights[i]+biases[i];acts.append(np.maximum(z,0) if i<2 else z)
            delta=2*(acts[-1]-y)*sw[:,None];gw=[];gb=[]
            for i in range(2,-1,-1):
                gw.insert(0,acts[i].T@delta+2e-4*weights[i]);gb.insert(0,delta.sum(axis=0))
                if i:delta=(delta@weights[i].T)*(acts[i]>0)
            for j,(p,g) in enumerate(zip(params,gw+gb)):
                m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
                p-=.002*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
    model=dict(kind=kind,mean=mean.tolist(),std=std.tolist(),weights=[w.tolist() for w in weights],biases=[b.tolist() for b in biases],training_days=len(days),seed=seed)
    model['training_rmse_cny']=float(np.sqrt(np.sum(sw*(predict(model,np.asarray(X))-np.asarray(Y)*100)**2)))
    return model
