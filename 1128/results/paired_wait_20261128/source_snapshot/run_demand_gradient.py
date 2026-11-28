"""Matched-layout demand-load sensitivity, evaluated after the feature-screen freeze."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter
import hashlib
import json
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from high_demand_features import run_case
from run_global_demand_pilot import (BANDS, ROOT, digest, make_world, point, public_law, read,
                                     release_slots, seed, validate_public, write)

OUT = ROOT / 'tmp/demand_gradient_v1'
PARENT = ROOT / 'tmp/high_demand_screen_v1'
LAYOUTS = (452,453)
SHAPES = ('uniform','dual')
SIZES = (40,100,150)
HISTORY_SIZES = (40,75,100,125,150,175)
SHARE = .4


def world_path(case):
    return OUT / 'worlds' / ('_'.join(map(str,case))+'.json')


def public_history(net, pattern):
    rng = Random(seed('gradient-history-v1',net['city_x'],pattern))
    counts = Counter()
    for _ in range(160):
        n = round(rng.choice(HISTORY_SIZES)*rng.choice((.2,.4,.5,.6)))
        for t in release_slots(rng,n):
            city,_ = point(rng,net,pattern)
            counts[city,next(lo for lo,hi,_ in BANDS if lo<=t<=hi)] += 1
    law = public_law(len(net['city_x']),1)
    for city in law['demand_model']['rates_by_city']:
        law['demand_model']['rates_by_city'][city] = [
            dict(start=lo,end=hi,rate=counts[city,lo]/(160*(hi-lo+1))) for lo,hi,_ in BANDS]
    validate_public(law)
    return law


def prepare(case):
    path = world_path(case)
    if path.exists():
        return
    layout,n,shape = case
    w = make_world(layout,2,n,SHARE,shape)
    raw = w['input']
    raw['public_information'] = public_history(w['network'],shape)
    # Attach an independent arrival time to each pickup id before chronological
    # sorting. Small populations now retain all shared ids and their time marks.
    rng = Random(seed('gradient-time-marks-v1',layout))
    slots = range(1,591)
    weights = [weight for lo,hi,weight in BANDS for _ in range(lo,hi+1)]
    releases = rng.choices(slots,weights,k=round(max(SIZES)*SHARE))
    events = defaultdict(list)
    for order in raw['orders']:
        if order['kind']=='P':
            t = releases[int(order['id'][1:])]
            order.update(release=t,deadline=t+360)
            events[t].append(order['id'])
    raw['external_events'] = [dict(at=t,kind='request_reveal',orders=sorted(ids)) for t,ids in sorted(events.items())]
    w['input_hash'] = digest(raw)
    write(path,w)


def validate_worlds(cases):
    worlds = {case:read(world_path(case)) for case in cases}
    all_hashes = {w['input_hash'] for w in worlds.values()}
    assert len(all_hashes)==len(cases)==12
    for layout in LAYOUTS:
        for n in SIZES:
            exogenous = set()
            for shape in SHAPES:
                w = worlds[layout,n,shape]
                raw = w['input']
                assert digest(raw)==w['input_hash']
                orders = raw['orders']
                assert len(orders)==n and sum(o['kind']=='P' for o in orders)==round(n*SHARE)
                p = {o['id']:o['release'] for o in orders if o['kind']=='P'}
                events = [(i,e['at']) for e in raw['external_events'] for i in e['orders']]
                assert len(events)==len(p) and dict(events)==p
                exogenous.add(digest(sorted((o['id'],o['weight'],o['release']) for o in orders)))
            assert len(exogenous)==1
        for shape in SHAPES:
            triplet = [worlds[layout,n,shape] for n in SIZES]
            assert len({digest(w['network']) for w in triplet})==1
            assert len({digest(w['input']['public_information']) for w in triplet})==1
            old = {}
            for w in triplet:
                current = {o['id']:[o[k] for k in ('kind','city','xy','weight','release','deadline','eligible_stations')]
                           for o in w['input']['orders']}
                assert all(current[oid]==mark for oid,mark in old.items())
                old=current
    return worlds


def evaluate(job):
    case,label,model = job
    path = OUT/'runs'/('_'.join(map(str,(*case,label)))+'.json')
    w = read(world_path(case))
    if not path.exists():
        start = perf_counter()
        r = run_case(w,model=model,teacher=0 if label=='Myopic' else 28)
        r.update(case=case,policy=label,model_hash=digest(model),seconds=perf_counter()-start)
        write(path,r)
    r = read(path)
    assert tuple(r['case'])==case and r['policy']==label and r['model_hash']==digest(model)
    assert (r['input_hash'],r['i0_hash'])==(w['input_hash'],w['i0_hash'])
    return r


def paired_summary(cases, policies, rows):
    by = {(tuple(r['case']),r['policy']):r for r in rows}
    assert len(by)==len(cases)*len(policies)
    summary = {}
    for policy in policies:
        if policy=='Myopic':
            continue
        paired = []
        for case in cases:
            base,r = by[case,'Myopic'],by[case,policy]
            assert (r['input_hash'],r['i0_hash'])==(base['input_hash'],base['i0_hash'])
            paired.append(dict(layout=case[0],size=case[1],shape=case[2],myopic=base['net'],
                net=r['net'],gain=r['net']-base['net'],accepted_myopic=base['accepted'],
                accepted=r['accepted'],seconds_myopic=base['seconds'],seconds=r['seconds']))
        measures = {}
        for n in SIZES:
            cell = [r for r in paired if r['size']==n]
            wins = sum(r['gain']>1e-7 for r in cell)
            losses = sum(r['gain']< -1e-7 for r in cell)
            measures[str(n)] = dict(cases=len(cell),wins=wins,ties=len(cell)-wins-losses,losses=losses,
                **{k:mean(r[k] for r in cell) for k in ('net','myopic','gain','accepted','accepted_myopic','seconds','seconds_myopic')})
        changes = {}
        for layout in LAYOUTS:
            values = [mean(r['gain'] for r in paired if r['layout']==layout and r['size']==n) for n in SIZES]
            changes[str(layout)] = dict(zip(('at_40','at_100','at_150'),values))
            changes[str(layout)].update(diff_100_40=values[1]-values[0],diff_150_40=values[2]-values[0],
                                        diff_150_100=values[2]-values[1])
        summary[policy] = dict(by_size=measures,by_layout=changes,paired=paired)
    return summary


def main():
    start = perf_counter()
    freeze = read(PARENT/'freeze.json')
    parent_protocol = read(PARENT/'protocol.json')
    assert not (set(LAYOUTS) & {i for split in parent_protocol['splits'].values() for i in split})
    chosen = freeze['chosen']
    labels = list(dict.fromkeys(['Myopic',chosen]+(['shortlist_1'] if chosen in ('Myopic','scalar') else [])))
    models = {k:freeze['shortlisted'][k] for k in labels}
    cases = [(layout,n,shape) for layout in LAYOUTS for n in SIZES for shape in SHAPES]
    protocol = dict(cases=cases,share=SHARE,history_sizes=HISTORY_SIZES,history_days=160,freeze_hash=digest(freeze),
        parent_source=parent_protocol['source'],source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen=chosen,policies=models,arrival_process='Fixed N and pickup share, independent conditional NHPP time marks nested by order ID.',
        candidate_policy='Frozen model. With frozen Myopic or scalar additionally evaluate predeclared shortlist_1.')
    if (OUT/'protocol.json').exists():
        assert digest(read(OUT/'protocol.json'))==digest(protocol)
    write(OUT/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare,cases))
        validate_worlds(cases)
        rows=[]
        jobs = [(case,k,model) for case in reversed(cases) for k,model in models.items()]
        futures = [pool.submit(evaluate,j) for j in jobs]
        for f in as_completed(futures):
            r=f.result()
            rows.append(r)
            print('gradient',len(rows),'/',len(jobs),r['case'],r['policy'],flush=True)
    summary = paired_summary(cases,models,rows)
    write(OUT/'summary.json',dict(frozen=chosen,cases=len(cases),rollouts=len(rows),elapsed_seconds=perf_counter()-start,
                                  results=summary))
    for label,v in summary.items():
        print(label,json.dumps(v['by_size']),json.dumps(v['by_layout']),flush=True)


if __name__=='__main__':
    main()
