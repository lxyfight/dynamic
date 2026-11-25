"""Reproducible development study; not the manuscript's full experiment grid."""
from pathlib import Path
from copy import deepcopy
from random import Random
from collections import defaultdict
import json,hashlib,time,sys,argparse
import numpy as np
from e1_sim.runner import DEFAULT_SPEC,ROOT
from e1_sim.online_examples import make_world
from e1_sim.online import prepare_online
from e1_sim.learning import LearningPolicy
from e1_sim.audit import audit_ledger
from e1_sim.quantities import plain,dec
from e1_sim.kernel import COSTS,require

PACKAGE=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
OUT=ROOT/'实验/VFA开发效果_20261122'

def world(seed):
    rng=Random(seed)
    raw,i0=make_world(PACKAGE,())
    events=defaultdict(list)
    for i in range(6):
        release=rng.randint(1,100)
        # Four public drone home stations; each request has its city's two sides.
        p=dict(id=f'P{i+1}',kind='P',weight=rng.randint(6,19),release=release,deadline=release+240,
               city='C1',xy=[20,rng.choice([-1,1])*rng.randint(1,20)/4],
               eligible_stations=['L1+','L1-'],accepted=None)
        raw['orders'].append(p);events[release].append(p['id'])
    # A compact development scenario, deliberately separate from 1..590 main protocol.
    raw['drones'] += [dict(id='U2',home='L1-',location='L1-',activated=False,onboard_weight=0,takeoff_at=None,active_operation=None)]
    raw['external_events']=[dict(at=t,kind='request_reveal',orders=ids) for t,ids in sorted(events.items())]
    raw['public_information']=dict(future_count=6,revelation_closed=False,revelation_end=100)
    return raw,i0

def episode(seed,policy,batch='BM',save=None):
    raw,i0=world(seed)
    state,controller=prepare_online(raw,PACKAGE['configuration'],PACKAGE['networks']['fixed_two_city'],full_day=True,batch_permission=batch)
    controller.policy=policy
    state.run(decision_hook=controller.on_event,stop_at_terminal=True)
    require(state.is_terminal(),'incomplete episode')
    initial=deepcopy(state.initial);initial['stop_time']=state.now
    audit=audit_ledger(state.cfg,state.network,initial,state.trace)
    net=state.ledger['revenue']-sum(state.ledger[k] for k in COSTS)
    ledger=plain(dict(state.ledger,net=net))
    require(not audit['errors'] and all(abs(dec(audit['ledger'][k])-dec(v))<dec('1e-7') for k,v in ledger.items()),'ledger mismatch')
    accepted=sum(o['kind']=='P' and o['accepted'] is True for o in state.orders.values())
    row=dict(seed=seed,net=float(net),accepted=accepted,completed=sum(o['kind']=='P' and o['completed_at'] is not None for o in state.orders.values()),
             terminal=True,ledger=ledger,decisions=len(policy.timings),max_seconds=max(policy.timings,default=0),
             physical_events=len(state.trace),event_checks=len(controller.event_checks),
             total_seconds=sum(policy.timings),tail=controller.tail.observations(),i0_plan_hash=i0)
    if save:
        save.write_text(json.dumps(dict(summary=row,input=raw,trace=plain(state.trace),decisions=policy.selection_log,
                                        trigger_log=controller.event_checks,audit=audit),ensure_ascii=False,indent=2),encoding='utf8')
    samples=[(x,float(net)-j) for x,j in policy.samples if any(x)]
    return row,samples

def fit(samples,ridge):
    X=np.array([x for x,y in samples]);y=np.array([y for x,y in samples])
    return np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@y).tolist()

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'实验/VFA承诺与触发修订效果_20261122')
    parser.add_argument('--test-start',type=int,default=4000)
    args=parser.parse_args()
    OUT=args.output
    OUT.mkdir(exist_ok=True)
    require(not (OUT/'results.json').exists(),'refuse to overwrite completed study; use a fresh output directory')
    protocol=dict(scope='development only, 1 mandatory D + 6 stochastic P, two-city fixed circuit, release 1..100',
                  training_seeds=list(range(1000,1040)),validation_seeds=list(range(2000,2012)),test_seeds=list(range(args.test_start,args.test_start+24)),
                  training_policy_seeds=[17,29,43],exploration=.25,ridge_candidates=[1,10],scale_candidates=[.5,1],
                  selected_by='validation mean profit; tie by ridge then scale',batch='BM',
                  limitations='Not full manuscript grid; no PI bound or WTQ claims; bounded insertion and adjacent-swap search',
                  commitments='started operations fixed; whole airborne sortie fixed; unflown routes adjustable',
                  triggers='request batches or idle station departure/wait choices; valid reservations execute directly; deterministic tail',
                  previous_test_not_reused='3000..3023 already inspected; new held-out days 4000..4023 by default')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    all_models=[]
    start=time.perf_counter()
    for ps in protocol['training_policy_seeds']:
        samples=[];train=[]
        for i,s in enumerate(protocol['training_seeds']):
            r,ss=episode(s,LearningPolicy(seed=ps*10000+s,explore=.25,candidate_budget=16))
            train.append(r);samples+=ss
            if (i+1)%10==0:print('train',ps,i+1,'samples',len(samples),flush=True)
        validation=[]
        for ridge in protocol['ridge_candidates']:
            w=fit(samples,ridge)
            for scale in protocol['scale_candidates']:
                rows=[episode(s,LearningPolicy(w,scale=scale,candidate_budget=16))[0] for s in protocol['validation_seeds']]
                validation.append(dict(ridge=ridge,scale=scale,weights=w,mean=float(np.mean([r['net'] for r in rows])),rows=rows))
                print('validate',ps,ridge,scale,validation[-1]['mean'],flush=True)
        best=min(validation,key=lambda r:(-r['mean'],r['ridge'],r['scale']))
        model=dict(training_policy_seed=ps,samples=len(samples),weights=best['weights'],scale=best['scale'],ridge=best['ridge'])
        (OUT/f'model_{ps}.json').write_text(json.dumps(dict(model=model,training=train,validation=validation,samples=samples),indent=2),encoding='utf8')
        all_models.append(model)
    # Freeze all weights before opening any test episode.
    (OUT/'frozen_models.json').write_text(json.dumps(all_models,indent=2),encoding='utf8')
    rows=[]
    for i,s in enumerate(protocol['test_seeds']):
        baseline,_=episode(s,LearningPolicy(candidate_budget=16),save=OUT/f'day_{s}_myopic.json')
        trained=[]
        for m in all_models:
            r,_=episode(s,LearningPolicy(m['weights'],scale=m['scale'],candidate_budget=16),save=OUT/f'day_{s}_vfa_{m["training_policy_seed"]}.json')
            trained.append(r)
        rows.append(dict(seed=s,myopic=baseline,vfa=trained))
        print('test',i+1,'delta',np.mean([r['net'] for r in trained])-baseline['net'],flush=True)
    delta=np.array([np.mean([r['net'] for r in day['vfa']])-day['myopic']['net'] for day in rows])
    rng=np.random.default_rng(98765)
    boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
    result=dict(protocol=protocol,rows=rows,mean_difference=float(delta.mean()),paired_day_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                win=int(sum(delta>1e-7)),tie=int(sum(abs(delta)<=1e-7)),loss=int(sum(delta< -1e-7)),seconds=time.perf_counter()-start,
                code_hashes={str(p.relative_to(ROOT/'code/e1_v01c')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code/e1_v01c').rglob('*.py')})
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('DONE',result['mean_difference'],result['paired_day_bootstrap95'],flush=True)

if __name__ == '__main__':
    raise RuntimeError('历史入口已封存；v04请运行verify_unknown_demand.py，不覆盖旧实验')
