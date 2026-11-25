"""Continue the frozen high-demand linear VFA, using additional completed episodes."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from time import perf_counter
import hashlib
import numpy as np

import run_high_demand_screen as factory
from high_demand_features import run_case
from high_demand_screen import _matrix
from run_global_demand_pilot import ROOT, read, write, digest, seed

PARENT = ROOT / 'tmp/high_demand_screen_v1'
OUT = ROOT / 'tmp/frozen6_iter_v1'
SHAPES = ('uniform', 'dual')
BATCHES = ((480, 481), (482, 483), (484, 485))
VALIDATION = (490, 491)
TRANSFER = (500, 501)


def cases(layouts):
    return [(layout, 100 if layout % 2 == 0 else 150, shape)
            for layout in layouts for shape in SHAPES]


def one(job):
    spec, model, stage, k = job
    # Isolate the cache while using the identical public-forecast generator.
    factory.OUT = OUT
    key = digest([spec, model, stage, k])
    path = OUT / 'runs' / (key + '.json')
    if path.exists():
        return read(path)
    world = factory.world_for(spec)
    training = stage == 'training'
    started = perf_counter()
    row = run_case(world, model=model, collect=training,
                   explore_seed=seed('fixed-six-exploration-v1', *spec) if training else None)
    row.update(case=spec, layout=spec[0], checkpoint=k, stage=stage,
               model_hash=digest(model), seconds=perf_counter()-started)
    write(path, row)
    return row


def batch(pool, jobs):
    tasks = {pool.submit(one, j): i for i, j in enumerate(jobs)}
    rows = [None] * len(tasks)
    for done, future in enumerate(as_completed(tasks), 1):
        row = future.result()
        rows[tasks[future]] = row
        print(row['stage'], row['checkpoint'], row['case'], done, '/', len(jobs), flush=True)
    return rows


def refit(days, original):
    # Fixed, training-only scale keeps the ridge penalty comparable across batches.
    x, y, w = _matrix(days)
    cols = original['columns']
    z = (x[:, cols] - original['mean']) / original['std']
    center = w @ z
    centered = z - center
    y0 = w @ y
    gram = centered.T @ (w[:, None] * centered)
    beta = np.linalg.solve(gram + .01*np.eye(len(cols)), centered.T @ (w*(y-y0)))
    model = dict(original, weights=beta.tolist(), intercept=float(y0-center@beta))
    return model, len(y)


def paired(rows, baseline):
    assert len(rows) == len(baseline) == 4
    pairs = []
    for a, b in zip(rows, baseline):
        assert (tuple(a['case']),a['i0_hash'],a['input_hash'])==(tuple(b['case']),b['i0_hash'],b['input_hash'])
        pairs.append(dict(case=a['case'], net=a['net'], frozen_net=b['net'],
                          gain=a['net']-b['net'],accepted=a['accepted'],frozen_accepted=b['accepted'],
                          same_full_trace=a['trace']==b['trace']))
    gains = [p['gain'] for p in pairs]
    return dict(net=mean(a['net'] for a in rows),gain=mean(gains),wins=sum(v>1e-7 for v in gains),
                ties=sum(abs(v)<=1e-7 for v in gains),losses=sum(v< -1e-7 for v in gains),pairs=pairs)


def main():
    original = read(PARENT/'freeze.json')['model']
    prior_protocol = read(PARENT/'protocol.json')
    assert original['columns']==list(range(6))
    old = [read(p) for p in (PARENT/'runs').glob('*.json')]
    warm_days = sorted([r for r in old if r['policy']=='teacher' and r['layout'] in prior_protocol['splits']['fit']],
                       key=lambda d:d['case'])
    prior_model, warm_samples = refit(warm_days, original)
    assert len(warm_days)==8 and warm_samples==500
    assert np.allclose(prior_model['weights'],original['weights']) and abs(prior_model['intercept']-original['intercept'])<1e-7
    partitions = [set(sum(BATCHES,())),set(VALIDATION),set(TRANSFER),
                  {s for v in prior_protocol['splits'].values() for s in v} | {452,453}]
    assert all(not partitions[i] & partitions[j] for i in range(len(partitions)) for j in range(i))
    for name,signature in prior_protocol['source'].items():
        here=Path(__file__).parent
        src=here/name if (here/name).exists() else here/'e1_sim'/name
        assert hashlib.sha256(src.read_bytes()).hexdigest()==signature, name
    protocol = dict(parent_model_hash=digest(original), parent_sources=prior_protocol['source'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        columns=original['columns'], warm_cases=[r['case'] for r in warm_days], warm_samples=warm_samples,
        additional_training=[cases(batch_ids) for batch_ids in BATCHES],
        validation=cases(VALIDATION),transfer=cases(TRANSFER),checkpoints_new_episodes=[0,4,8,12],
        checkpoints_all_episodes=[8,12,16,20], dynamic_share=.4, training_exploration=.1,
        refit='Cumulative weighted ridge .01, with frozen scale and unpenalized intercept, no validation labels. Same features throughout.',
        transfer_rule='Compare fixed last update and initial freeze on fresh cases regardless of curve. Training pilot; no automatic production replacement.')
    pp = OUT/'protocol.json'
    if pp.exists() and digest(read(pp)) != digest(protocol):
        # An already-frozen campaign retains its original protocol. An independently
        # recorded, exact source-hash pair permits the case-key serialization fix.
        stored=read(pp)
        archived=OUT/'runner_at_freeze.py'
        migration=OUT/'runner_compatibility.json'
        if not archived.is_file() or not migration.is_file():
            raise ValueError('different existing protocol')
        replacement=read(migration)
        current_hash=protocol['code_sha256']
        if (replacement['original']!=stored['code_sha256'] or replacement['corrected']!=current_hash
                or hashlib.sha256(archived.read_bytes()).hexdigest()!=stored['code_sha256']
                or digest(dict(stored,code_sha256=current_hash))!=digest(protocol)):
            raise ValueError('different existing protocol')
        protocol=stored
    if not pp.exists():
        write(pp, protocol)
    checkpoint_models=[original]
    new_days=[]
    counts=[500]
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for k, layouts in enumerate(BATCHES):
            source=checkpoint_models[-1]
            rows = batch(pool,[(sp,source,'training',4*k) for sp in cases(layouts)])
            new_days.extend(rows)
            updated,count=refit(warm_days+new_days,original)
            checkpoint_models.append(updated)
            counts.append(count)
            write(OUT/f'model_ep{len(new_days)}.json',updated)
        all_models={4*i:digest(m) for i,m in enumerate(checkpoint_models)}
        write(OUT/'pre_validation_models.json',dict(models=all_models,last_checkpoint=12))
        all_results=[]
        for i, model in enumerate(checkpoint_models):
            rows=batch(pool,[(sp,model,'validation',4*i) for sp in cases(VALIDATION)])
            if i==0:
                baseline=rows
            all_results.append(dict(additional_episodes=4*i,total_episodes=len(warm_days)+4*i,
                training_samples=counts[i],model_hash=digest(model),validation=paired(rows,baseline)))
            write(OUT/'validation_curve.json',all_results)
        locked=read(OUT/'pre_validation_models.json')
        assert locked['last_checkpoint']==12 and locked['models']['12']==digest(checkpoint_models[-1])
        base=batch(pool,[(sp,original,'transfer',0) for sp in cases(TRANSFER)])
        last=batch(pool,[(sp,checkpoint_models[-1],'transfer',12) for sp in cases(TRANSFER)])
    result=dict(protocol=protocol,validation_curve=all_results,transfer_final_vs_frozen=paired(last,base),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        invocation_elapsed_seconds=perf_counter()-start, total_new_training_samples=counts[-1]-warm_samples)
    write(OUT/'results.json',result)
    print('DONE',[(r['total_episodes'],r['validation']['net']) for r in all_results],
          'transfer',result['transfer_final_vs_frozen']['gain'],flush=True)


if __name__=='__main__':
    main()
