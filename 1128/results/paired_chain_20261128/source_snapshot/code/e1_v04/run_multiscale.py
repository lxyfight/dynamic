"""Phase-separated formal multiscale campaign. Test requires a reviewed freeze.

Workers are numerical processes, not agents. Proposal caches only reuse exact
identical public states across policies of one day; isolated timing is separate.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import json,gzip,hashlib,platform,argparse,subprocess
import numpy as np
from e1_sim.runner import ROOT,DEFAULT_SPEC
from e1_sim.multiscale import make_instance
from e1_sim.formal_learning import FormalPolicy
from e1_sim.comparators import Comparator
from e1_sim.online import prepare_online
from e1_sim.audit import audit_ledger
from e1_sim.kernel import COSTS,require
from e1_sim.quantities import plain,dec
from e1_sim.initializer import digest

OUT=ROOT/'tmp/sortie25_formal_campaign'
from e1_sim.cost_model import current_configuration
CFG=current_configuration(json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))['configuration'])
SEEDS=[17,29,43]

def seedof(*v):return int(hashlib.sha256(json.dumps(v).encode()).hexdigest()[:8],16)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)

def specs(split,training_seed=0):
    rows=[]
    cities=[2,4,6] if split in ('train','validation') else [2,4,6,8,10]
    layouts,days=(5,10) if split=='train' else (2,5) if split=='validation' else (3,10)
    # Round-robin cells keep checkpoints balanced to at most one day per cell.
    for day in range(days):
        for layout in range(layouts):
            for c in cities:
                for n in [10,20]:
                    for p in [.2,.4]:
                        family=f'{split}_seed{training_seed}' if split=='train' else split
                        rows.append(dict(cities=c,n=n,p=p,layout_seed=seedof('layout',family,c,layout),day_seed=seedof('day',family,c,layout,day),identity=f'{family}_c{c}_n{n}_p{int(p*100)}_l{layout}_d{day}',layout_id=f'{family}_c{c}_l{layout}'))
    return rows

def episode(instance,model,*,explore=0,explore_seed=0,cache=None):
    kw=dict(candidate_budget=16,beam_width=4,max_joint_requests=3)
    name=model['name']
    if model['kind']=='VFA' or name=='Myopic':
        policy=FormalPolicy(model.get('weights'),group=model.get('group','coordination'),enhance=model.get('enhance',True),seed=explore_seed,explore=explore,**kw)
    else:policy=Comparator(name,rollout_h=10,rollout_seconds=1.,**kw)
    policy.proposal_cache=cache
    state,driver=prepare_online(instance['input'],instance['configuration'],instance['network'],full_day=True)
    state._record_snapshots=False;driver.policy=policy
    begin=perf_counter();state.run(decision_hook=driver.on_event,stop_at_terminal=True);elapsed=perf_counter()-begin
    require(state.is_terminal(),'nonterminal formal episode')
    initial=deepcopy(state.initial);initial['stop_time']=state.now;audit=audit_ledger(state.cfg,state.network,initial,state.trace)
    net=state.ledger['revenue']-sum(state.ledger[k] for k in COSTS);ledger=plain(dict(state.ledger,net=net))
    require(not audit['errors'] and all(abs(dec(v)-dec(audit['ledger'][k]))<dec('1e-7') for k,v in ledger.items()),'formal ledger mismatch')
    accepted=sum(o['kind']=='P' and o['accepted'] is True for o in state.orders.values());completed=sum(o['kind']=='P' and o['completed_at'] is not None for o in state.orders.values())
    require(accepted==completed,'accepted obligation missing')
    summary=dict(id=instance['id'],model=name,net=float(net),accepted=accepted,completed=completed,terminal=True,ledger=ledger,
        i0_hash=instance['i0_hash'],input_hash=instance['input_hash'],seconds=elapsed,decision_seconds=policy.timings,
        cached_proposals=cache is not None,rollout=getattr(policy,'rollout_log',[]),tail=driver.tail.observations(),
        cities=instance['cities'],layout_seed=instance['layout_seed'],orders=len(instance['input']['orders']),
        max_batches=max(t['loading_batches'] for t in state.trucks.values()))
    if instance.get('information_mode')=='unknown_daily_total':
        require(float(state.now)>=instance['input']['public_information']['revelation_end'],'premature terminal')
        require(all(o['completed_at'] is not None and o['completed_at']<=o['deadline']
                    for o in state.orders.values() if o['kind']=='D'),'delivery obligation missed')
        summary.update(deliveries_per_city=instance['deliveries_per_city'],
                       expected_pickups_per_city=instance['expected_pickups_per_city'])
    else:
        summary.update(n=instance['n'],p=instance['p'])
    detail=dict(summary=summary,trace=plain(state.trace),decisions=policy.selection_log,triggers=driver.event_checks,audit=audit)
    samples=[(x,float(net)-j) for x,j in policy.samples if any(x)]
    return summary,detail,samples

def job(task):
    spec,models,phase,explore,ps=task
    result_path=OUT/phase/(spec['identity']+'.json')
    # Resume requires an exact protocol and source freeze, checked by main.
    if result_path.exists():return read(result_path)
    args={k:v for k,v in spec.items() if k!='layout_id'}
    instance=make_instance(CFG,**args)
    ip=OUT/'instances'/(instance['id']+'.json');write(ip,instance)
    cache={} if len(models)>1 else None;rows=[]
    for m in models:
        summary,detail,samples=episode(instance,m,explore=explore,explore_seed=seedof('policy',ps,spec['identity']),cache=cache)
        summary['layout_id']=spec['layout_id'];summary['samples']=samples
        trace_path=OUT/'traces'/phase/(spec['identity']+'__'+m['name']+'.json.gz');trace_path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(trace_path,'wt',encoding='utf8') as f:json.dump(detail,f,ensure_ascii=False)
        rows.append(summary)
    result=dict(id=spec['identity'],rows=rows);write(result_path,result);return result

def parallel(tasks,workers,label):
    results=[];begin=perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(job,t) for t in tasks]
        for i,f in enumerate(as_completed(pending),1):
            results.append(f.result())
            if i%10==0 or i==len(tasks):
                progress=dict(phase=label,completed=i,total=len(tasks),elapsed_seconds=perf_counter()-begin)
                write(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    return sorted(results,key=lambda r:r['id'])

def source_hashes():
    files=list((Path(__file__).parent/'e1_sim').glob('*.py'))+[Path(__file__)]
    return {str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def protocol():
    return dict(version='formal-multiscale-v1',cities=[2,4,6,8,10],orders_per_city=[10,20],pickup_proportions=[.2,.4],
      train_days_per_seed=600,training_seeds=SEEDS,checkpoints=[100,300,600],epsilon=[.2,.1,.05],validation_days=120,test_days=600,
      selected_feature_group='coordination15 fixed from separate development evidence',ridge=1.,fit='cumulative complete-day equal weights, same F1 sum-loss ridge as development; all coefficients penalized',
      selection='highest validation mean (12 cells equally weighted); within 0.01 yuan choose earliest checkpoint; retain all 3 seeds',
      batch_permission='BM',candidate_budget=16,search='same deterministic count-limited proposals for Myopic and VFA; no claimed hard wall-clock limit',
      I0='packed mandatory D only; all methods share exact I0; no hidden P inspected',
      main_comparators=['Greedy','Myopic','RH','VFA17','VFA29','VFA43'],
      rollout='200 preselected test days; H=10, 4 finalists, 1-second cooperative scenario-scoring deadline; only complete shared rounds count, otherwise Myopic fallback; base proposal time separate',
      PI='60 independent small days c=2/3,N=10/15/20; exhaustive feasible-sortie cover relaxation upper bound, not exact Appendix A PI optimum',
      timing='Parallel campaign/cached proposal times are throughput diagnostics. Isolated uncached timing evaluated separately; no 1s end-to-end performance claim.',
      exclusions='none silently; construction failures recorded and stop campaign; no deadline changes',
      test_access='test instances generated only after explicit training-review freeze',
      statistics='paired day differences; hierarchy: layouts then days within city/n/p cells; seen and unseen cities reported separately',
      runtime=dict(python=platform.python_version(),platform=platform.platform(),logical_cpus=os.cpu_count()))

def initialize():
    raise RuntimeError('v03正式训练规格尚未冻结；请使用verify_revision.py验收，不沿用旧训练协议')
    OUT.mkdir(exist_ok=True)
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',protocol());write(OUT/'source_hashes.json',source_hashes())
    else:require(read(OUT/'source_hashes.json')==source_hashes(),'source changed after formal freeze; create a new version, do not mix results')

def fit(days):
    rows=[(x,y,1/len(day)) for day in days if day for x,y in day]
    X=np.asarray([x for x,y,w in rows]);y=np.asarray([y for x,y,w in rows]);w=np.asarray([w for x,y,w in rows])
    theta=np.linalg.solve(X.T@(w[:,None]*X)+np.eye(15),X.T@(w*y))
    return theta.tolist(),float(np.sum(w*(X@theta-y)**2)/w.sum()),len(rows)

def train(workers):
    histories=[];weights={s:None for s in SEEDS};days={s:[] for s in SEEDS};previous=0
    for iteration,(count,eps) in enumerate(zip([100,300,600],[.2,.1,.05]),1):
        tasks=[]
        for s in SEEDS:
            model=dict(name=f'VFA{s}',kind='VFA',weights=weights[s])
            tasks.extend((sp,[model],f'train_r{iteration}',eps,s) for sp in specs('train',s)[previous:count])
        data=parallel(tasks,workers,f'train {count}');models=[]
        for s in SEEDS:
            rows=[r['rows'][0] for r in data if r['id'].startswith(f'train_seed{s}_')]
            days[s].extend(r['samples'] for r in rows);weights[s],mse,n=fit(days[s]);models.append(dict(name=f'VFA{s}',kind='VFA',weights=weights[s],seed=s,checkpoint=count,mse=mse,samples=n))
        write(OUT/f'weights_{count}.json',models)
        validation=parallel([(sp,[dict(name='Myopic',kind='Myopic')]+models,f'validation_{count}',0,0) for sp in specs('validation')],workers,f'validation {count}')
        for m in models:
            vals=[next(r for r in d['rows'] if r['model']==m['name'])['net'] for d in validation]
            base=[next(r for r in d['rows'] if r['model']=='Myopic')['net'] for d in validation]
            histories.append(dict(**m,validation_mean=float(np.mean(vals)),myopic_mean=float(np.mean(base)),paired_improvement=float(np.mean(np.asarray(vals)-base))))
        write(OUT/'training_curves.json',histories);previous=count
    write(OUT/'training_complete.json',dict(completed=True,trained_days=1800,validation_runs=3*120*4,test_opened=False))

def freeze():
    require((OUT/'training_review.json').exists(),'inspect curves and write training_review.json before freezing')
    require(read(OUT/'training_complete.json')['completed'],'training incomplete')
    curves=read(OUT/'training_curves.json');models=[]
    for s in SEEDS:
        options=[r for r in curves if r['seed']==s];best=max(r['validation_mean'] for r in options)
        models.append(min((r for r in options if r['validation_mean']>=best-.01),key=lambda r:r['checkpoint']))
    write(OUT/'frozen_models.json',dict(models=models,selection='validation only; all three seeds retained',review=read(OUT/'training_review.json'),test_opened=False))

def test(workers):
    frozen=read(OUT/'frozen_models.json');models=[dict(name=n,kind=n) for n in ['Myopic','Greedy','RH']]+frozen['models']
    result=parallel([(sp,models,'test',0,0) for sp in specs('test')],workers,'test')
    write(OUT/'test_results.json',result)

def rollout(workers):
    require((OUT/'frozen_models.json').exists(),'models not frozen')
    # 10/cell, layout allocation 4/3/3 (extra layout rotates across cells).
    selected=[]
    for sp in specs('test'):
        layout=int(sp['layout_id'].rsplit('l',1)[1]);day=int(sp['identity'].rsplit('d',1)[1])
        extra=([2,4,6,8,10].index(sp['cities'])*4+[10,20].index(sp['n'])*2+[.2,.4].index(sp['p']))%3
        if day<3+(layout==extra):selected.append(sp)
    assert len(selected)==200
    result=parallel([(sp,[dict(name='Rollout',kind='Rollout')],'rollout',0,0) for sp in selected],workers,'rollout');write(OUT/'rollout_results.json',result)

if __name__ == '__main__':
    raise RuntimeError('历史入口已封存；v04请运行verify_unknown_demand.py，不覆盖旧实验')
