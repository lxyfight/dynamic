"""Predeclared extension using all matched development actions, same validation and holdout."""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from statistics import mean
import csv
import hashlib
import sys

from run_wait_increment_learning import (OUT,BASE,GROUPS,EPS,fit,metrics,load_split,
    operation_metrics,run,WithWait,digest,read,write)


E = OUT/'expanded'
CASES=[(i,100 if i%2==0 else 150,s) for i in range(526,530) for s in ('uniform','dual')]


def freeze_model():
    # Bind the historical split and selection rule to this checkout's protocol.
    protocol_path=E/'protocol.json'
    protocol=read(Path(__file__).resolve().parents[2]/'data/fixtures/expanded_wait_protocol.json')
    protocol['base_protocol']=digest(read(OUT/'model_protocol.json'))
    if protocol_path.exists():
        assert read(protocol_path)==protocol
    else:
        write(protocol_path,protocol)
    train=read(OUT/'expanded_train.json')
    assert {r['case'][0] for r in train}==set(protocol['training_layouts'])
    assert len({(tuple(r['case']),r['time']) for r in train})==len(train)
    orig=load_split('train')
    by_id={(tuple(r['case']),r['time']):r for r in train}
    for r in orig:
        matched=by_id[tuple(r['case']),r['time']]
        assert matched['x']==r['x'] and abs(matched['y']-r['y'])<EPS
    assert sum(r['case'][0]<500 for r in train)==37
    val=load_split('validation')
    table=[]
    for n in range(4):
        for groups in combinations(GROUPS,n):
            model=fit(train,[c for g in groups for c in GROUPS[g]])
            table.append(dict(groups=list(groups),model=model,training=metrics(train,model),validation=metrics(val,model)))
    best=min(table,key=lambda r:(round(r['validation']['regret'],10),len(r['groups']),r['validation']['mae'],r['groups']))
    base=metrics(val,None)
    selected='learned' if best['validation']['regret']<base['regret']-EPS else 'depart'
    result=dict(selected=selected,learned=best,table=table,validation_depart=base,train_sha=digest(train),
                validation_sha=digest(val),model_protocol=digest(protocol),
                source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=E/'wait_model_freeze.json'
    if path.exists(): assert read(path)==result
    write(path,result)
    print('EXPANDED FREEZE',len(train),selected,best['groups'],best['validation'],flush=True)


def online_one(case):
    frozen=read(E/'wait_model_freeze.json')
    assert frozen['source']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=E/'online'/('_'.join(map(str,case))+'.json')
    w=read(OUT/'worlds'/(digest(case)+'.json'))
    if not path.exists():
        pol=WithWait(read(BASE)['model'],frozen['learned']['model'])
        r=run(w,28,policy_override=pol,metrics=operation_metrics)
        r.update(case=case,probes=pol.probes,wait_model_hash=digest(frozen),input_hash=w['input_hash'],i0_hash=w['i0_hash'])
        write(path,r)
    row=read(path)
    assert row['wait_model_hash']==digest(frozen) and row['input_hash']==digest(w['input']) and row['i0_hash']==w['i0_hash']
    return row


def report():
    f=read(E/'wait_model_freeze.json')
    test=load_split('test')
    baseline=metrics(test,None)
    learned=metrics(test,f['learned']['model'])
    pairs=[]
    for c in CASES:
        key='_'.join(map(str,c))
        base=read(OUT/'runs'/(key+'_depart.json'))
        new=read(E/'online'/(key+'.json'))
        assert (base['input_hash'],base['i0_hash'])==(new['input_hash'],new['i0_hash'])
        for r in (base,new):
            l=r['ledger']
            assert abs(r['net']-l['revenue']+sum(v for k,v in l.items() if k not in ('revenue','net')))<EPS
        actual=sum(p['wait'] for p in new['probes'])
        assert new['operation_metrics']['new_wait_chosen']>=actual
        if not actual:
            assert all(new[k]==base[k] for k in ('plans','net','ledger','trace'))
        pairs.append(dict(case=c,net=new['net'],baseline=base['net'],gain=new['net']-base['net'],
                          waiting=actual,eligible=len(new['probes']),accepted=new['accepted'],baseline_accepted=base['accepted'],
                          same_trace=base['trace']==new['trace']))
    ds=[r['gain'] for r in pairs]
    group=dict(mean_net=mean(r['net'] for r in pairs),mean_gain=mean(ds),
               wins=sum(d>EPS for d in ds),losses=sum(d< -EPS for d in ds),ties=sum(abs(d)<=EPS for d in ds),
               waits=sum(r['waiting'] for r in pairs),eligible=sum(r['eligible'] for r in pairs),by_case=pairs)
    reference=dict(mean_net=mean(r['baseline'] for r in pairs),mean_gain=0.,wins=0,losses=0,ties=len(pairs))
    result=dict(selected=f['selected'],selected_online=group if f['selected']=='learned' else reference,
                learned_online=group,frozen_baseline=reference,heldout_one_action=learned,heldout_depart=baseline)
    write(E/'wait_policy_results.json',result)
    with (E/'逐日等待验证.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        csvout=csv.writer(stream)
        csvout.writerow(['布局','全网订单数','空间分布','六特征基线净收益','扩充模型净收益','增益','学习等待次数','可比较次数'])
        for p in pairs:
            i,n,s=p['case']
            csvout.writerow([i,n,'均匀' if s=='uniform' else '双簇',p['baseline'],p['net'],p['gain'],p['waiting'],p['eligible']])
    print(f['selected'], {k:v for k,v in group.items() if k!='by_case'},learned,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='freeze': freeze_model()
    elif sys.argv[1]=='online':
        with ProcessPoolExecutor(max_workers=5) as pool:
            print('FINISHED EXPANDED',len(list(pool.map(online_one,CASES))))
    elif sys.argv[1]=='report': report()
