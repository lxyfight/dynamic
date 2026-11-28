"""Independent structural checks and unconditioned marked Poisson sensitivity."""
from concurrent.futures import ProcessPoolExecutor
from random import Random
from math import isclose
from statistics import mean
from run_global_demand_pilot import (ROOT, OUT, CFG, BANDS, TOTALS, SHARES, seed, digest, make_world,
    ReviewPolicy, expected_arrivals, operation_metrics, run, read, write)
from e1_sim.online import prepare_online, known_clone


def audit():
    for c in (2,4):
        for shape in ('uniform','dual'):
            low=make_world(22,c,30,.2,shape)
            high=make_world(22,c,40,.4,shape)
            assert low['i0_hash']==high['i0_hash']
            assert low['input_hash']!=high['input_hash']
            states=[]
            for world,n in ((low,6),(high,16)):
                raw=world['input']
                assert sum(x['kind']=='P' for x in raw['orders'])==n
                assert sum(x['kind']=='D' for x in raw['orders'])==24
                state,_=prepare_online(raw,CFG,world['network'],full_day=True)
                k=known_clone(state)
                assert not k._hidden and all(o['kind']=='D' for o in k.orders.values())
                policy=ReviewPolicy(28,'adaptive')
                plan,_=policy.decide(k)
                states.append((digest(k.public),digest(plan)))
            assert states[0]==states[1]
    for file in (OUT/'runs').glob('*.json'):
        row=read(file)
        assert all(x['operation_metrics']['new_wait_chosen'] <= x['operation_metrics']['decisions_with_new_wait'] for x in row['results'])
    print('Four paired hidden-suffix comparisons verified for 2/4 cities and uniform/dual demand.',flush=True)


def sensitivity(spec):
    block,n,s,pattern=spec
    mu=n*s
    rng=Random(seed('unconditional-count',block,n,s))
    pos=rng.expovariate(1)
    num=0
    while pos<=mu:
        num+=1
        pos+=rng.expovariate(1)
    # A Poisson count followed by conditional iid marked slots gives independent
    # Poisson increments on the minute grid. Fixed D and common count across shapes.
    total=round(n*(1-s))+num
    world=make_world(block,2,total,num/total,pattern)
    assert sum(o['kind']=='P' for o in world['input']['orders'])==num
    results={}
    for mode,weight in [('fixed',0),('fixed',28),('adaptive',28)]:
        result=run(world,weight,policy_override=ReviewPolicy(weight,mode),metrics=operation_metrics)
        results[mode+str(weight)]=result['net']
    return dict(spec=spec,pickups=num,net=results)


if __name__=='__main__':
    audit()
    specs=[(b,n,s,p) for b in (30,31) for n,s in ((30,.2),(40,.4)) for p in ('uniform','dual')]
    with ProcessPoolExecutor(max_workers=3) as pool:
        data=list(pool.map(sensitivity,specs))
    write(ROOT/'tmp/global_poisson_sensitivity_v1.json',dict(data=data,mean_net={
        k:mean(r['net'][k] for r in data) for k in data[0]['net']}))
