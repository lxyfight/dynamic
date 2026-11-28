"""Batch-screen observable waiting predictors and run a frozen policy out of layout."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from math import sqrt
from pathlib import Path
from statistics import mean
import csv
import hashlib
import numpy as np

from high_demand_features import FeaturePolicy
from recovery_opportunity import run
from run_global_demand_pilot import operation_metrics
from run_multiscale import ROOT, digest, read, write
from run_wait_counterfactual import pure_pairs
from wait_increment_model import GROUPS, wait_features


OUT = ROOT/'tmp/wait_learning_v1'
BASE = ROOT/'tmp/high_demand_screen_v1/freeze.json'
EPS = 1e-7
RIDGE = .25
SPLIT = {'train':list(range(520,524)), 'validation':[524,525], 'test':list(range(526,530))}


def protocol():
    src = Path(__file__).parent
    value = dict(splits=SPLIT,groups=GROUPS,ridge=RIDGE,theta_threshold=0.,
                 label='Whole-day paired gain of one five-minute review with frozen continuation minus known-plan change.',
                 weight='equal layouts, equal cases within each layout, equal decisions within each case',
                 features='Probability uses disclosed eligibility marks; drone uses a tail-time heuristic; truck uses weight fit and deadline slack.',
                 selection='Compare 8 ridge subsets by validation hindsight regret, then fewer predictors, then validation MAE. Prefer depart at tied regret. Test best learned even if depart selected.',
                 execution='Apply frozen decision rule at every strict pure pair on its own trajectory; after the forced first action later valid events may replan.',
                 frozen_six=digest(read(BASE)['model']),
                 source={name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in
                         ('run_wait_increment_learning.py','wait_increment_model.py','run_wait_counterfactual.py')})
    p=OUT/'model_protocol.json'
    if p.exists() and read(p)!=value:
        raise ValueError('Model protocol differs from its frozen copy')
    write(p,value)
    return value


def weights(rows):
    lc=Counter(tuple(r['case']) for r in rows)
    ncase=Counter(c[0] for c in lc)
    return np.array([1/(len(ncase)*ncase[r['case'][0]]*lc[tuple(r['case'])]) for r in rows])


def fit(rows,cols):
    w=weights(rows)
    x=np.array([[r['x'][c] for c in cols] for r in rows],dtype=float)
    y=np.array([r['y']-r['known_plan_delta'] for r in rows])
    center=w@x
    std=np.sqrt(w @ (x-center)**2)
    std[std<1e-10]=1.
    z=(x-center)/std
    intercept=float(w@y)
    beta=np.linalg.solve((z.T*w)@z+RIDGE*np.eye(len(cols)),z.T@(w*(y-intercept)))
    return dict(columns=cols,mean=center.tolist(),scale=std.tolist(),coefficients=beta.tolist(),
                intercept=intercept)


def score(model,x):
    return x['known_plan_delta']+model['intercept']+sum(
        coef*(x[c]-mu)/sd for c,coef,mu,sd in
        zip(model['columns'],model['coefficients'],model['mean'],model['scale']))


def metrics(rows,m):
    w=weights(rows)
    y=np.array([r['y'] for r in rows])
    if m is None:
        action=np.zeros(len(rows),dtype=bool)
    else:
        pred=np.array([score(m,r['x']) for r in rows])
        action=pred>0
    gain=w@(action*y)
    result=dict(n=len(rows), wait=int(action.sum()),regret=float(w@np.maximum(y,0)-gain),
                paired_gain=float(gain))
    if m is not None:
        var=float(w@(y-float(w@y))**2)
        result.update(mae=float(w@abs(pred-y)),r2=1-float(w@(pred-y)**2)/var if var>EPS else None)
    return result


def load_split(stage):
    rows=read(OUT/(stage+'.json'))
    assert {r['case'][0] for r in rows}==set(SPLIT[stage])
    for r in rows:
        assert abs(r['known_plan_delta']-r['x']['known_plan_delta'])<EPS
        assert np.isfinite(r['y']) and all(np.isfinite(v) for v in r['x'].values())
    return rows


def freeze():
    train,val=(load_split(s) for s in ('train','validation'))
    table=[]
    for size in range(4):
        for groups in combinations(GROUPS,size):
            features=[c for g in groups for c in GROUPS[g]]
            m=fit(train,features)
            table.append(dict(groups=list(groups),model=m,training=metrics(train,m),validation=metrics(val,m)))
    best=min(table,key=lambda r:(round(r['validation']['regret'],10),len(r['groups']),r['validation']['mae'],r['groups']))
    baseline=metrics(val,None)
    selected='learned' if best['validation']['regret']<baseline['regret']-EPS else 'depart'
    result=dict(selected=selected,learned=best,validation_depart=baseline,table=table,
                train_sha=digest(train),validation_sha=digest(val),model_protocol=digest(protocol()))
    path=OUT/'wait_model_freeze.json'
    if path.exists():
        assert read(path)==result
    write(path,result)
    print('FROZEN',selected,best['groups'],baseline,best['validation'],flush=True)
    return result


class WithWait(FeaturePolicy):
    def __init__(self,six,model):
        super().__init__(six)
        self.aux=model
        self.probes=[]

    def pick(self,known,ranked,theta):
        chosen,record=super().pick(known,ranked,theta)
        depart=next(c for _,_,c in ranked if c['plan'] is chosen)
        _,matched=pure_pairs(known,ranked,depart)
        if not matched:
            return chosen,record
        truck,station,wait=min(matched,key=lambda t:(t[0],t[2]['id']))
        x=wait_features(known,depart,wait,truck,station)
        value=score(self.aux,x)
        activate=value>0
        self.probes.append(dict(t=float(known.now),truck=truck,station=station,
                                x=x,score=value,wait=activate))
        if activate:
            return wait['plan'],dict(record,plan=digest(wait['plan']),decision=wait['id'],j=float(wait['score']))
        return chosen,record


def case_run(case):
    frozen=read(OUT/'wait_model_freeze.json')
    path=OUT/'online'/('_'.join(map(str,case))+'.json')
    world=read(OUT/'worlds'/(digest(case)+'.json'))
    if not path.exists():
        policy=WithWait(read(BASE)['model'],frozen['learned']['model'])
        r=run(world,28,policy_override=policy,metrics=operation_metrics)
        assert sum(p['wait'] for p in policy.probes)<=r['operation_metrics']['new_wait_chosen']
        r.update(case=case,probes=policy.probes,wait_model_hash=digest(frozen),
                 input_hash=world['input_hash'],i0_hash=world['i0_hash'])
        write(path,r)
    r=read(path)
    assert r['wait_model_hash']==digest(frozen)
    assert r['input_hash']==digest(world['input']) and r['i0_hash']==world['i0_hash']
    return r


def run_online():
    cases=[(i,100 if i%2==0 else 150,s) for i in SPLIT['test'] for s in ('uniform','dual')]
    with ProcessPoolExecutor(max_workers=5) as pool:
        rows=list(pool.map(case_run,cases))
    print('ONLINE FINISHED',len(rows),flush=True)


def report():
    frozen=read(OUT/'wait_model_freeze.json')
    assert digest(load_split('train'))==frozen['train_sha']
    assert digest(load_split('validation'))==frozen['validation_sha']
    test=load_split('test')
    pairs=[]
    for i in SPLIT['test']:
        for shape in ('uniform','dual'):
            case=[i,100 if i%2==0 else 150,shape]
            key='_'.join(map(str,case))
            base=read(OUT/'runs'/(key+'_depart.json'))
            r=read(OUT/'online'/(key+'.json'))
            assert r['input_hash']==base['input_hash'] and r['i0_hash']==base['i0_hash']
            a=sum(p['wait'] for p in r['probes'])
            if not a:
                assert all(r[k]==base[k] for k in ('net','ledger','plans','trace'))
            pairs.append(dict(case=case,net=r['net'],baseline=base['net'],gain=r['net']-base['net'],
                              baseline_accepted=base['accepted'],accepted=r['accepted'],waiting=a,
                              eligible=len(r['probes']),same_trace=r['trace']==base['trace']))
    gains=[r['gain'] for r in pairs]
    learned=dict(mean_net=mean(r['net'] for r in pairs),mean_gain=mean(gains),
                 wins=sum(g>EPS for g in gains),losses=sum(g< -EPS for g in gains),
                 ties=sum(abs(g)<=EPS for g in gains),waits=sum(r['waiting'] for r in pairs),
                 eligible=sum(r['eligible'] for r in pairs),by_case=pairs)
    baseline=dict(mean_net=mean(r['baseline'] for r in pairs),mean_gain=0.,wins=0,losses=0,ties=len(pairs))
    result=dict(selected=frozen['selected'],learned_online=learned,frozen_baseline=baseline,
                selected_online=learned if frozen['selected']=='learned' else baseline,
                heldout_one_action=metrics(test,frozen['learned']['model']),heldout_depart=metrics(test,None))
    write(OUT/'wait_policy_results.json',result)
    with (OUT/'逐日等待验证.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        out=csv.writer(stream)
        out.writerow(['布局','订单数','客户分布','六特征基线净收益','学习等待净收益','学习等待增益',
                      '基线接单','学习等待接单','可比较次数','学习等待次数','物理轨迹相同'])
        for r in pairs:
            i,n,s=r['case']
            out.writerow([i,n,'均匀' if s=='uniform' else '双簇',r['baseline'],r['net'],r['gain'],
                          r['baseline_accepted'],r['accepted'],r['eligible'],r['waiting'],r['same_trace']])
    print(frozen['selected'],{k:v for k,v in learned.items() if k!='by_case'},result['heldout_one_action'])


if __name__=='__main__':
    import sys
    {'protocol':protocol,'freeze':freeze,'online':run_online,'report':report}[sys.argv[1]]()
