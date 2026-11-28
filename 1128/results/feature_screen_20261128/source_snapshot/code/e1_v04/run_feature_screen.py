"""November 28 development screen. The reserved test bank is never generated.

Uses the existing audited simulator/trainer; no policy or cost changes.
Run from this directory with the Python environment containing Gurobi.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

FEATURE_SETS = {
    'none': 'none',
    'chain': 'existing_chain_opportunity',
    'chain_late': 'existing_chain_opportunity,future_pickup_lateness',
    'full': 'full',
}
SEARCH = dict(candidate_budget=32, enhancement_budget=8,
              action_solver='gurobi', mip_pattern_limit=32)
WORLD_OPTIONS = dict(arrival_mode='p2', improvement_budget=8, history_days=160,
                     temporal='peak')
SETTING = dict(h=5, review_intervals=[2, 5, 10], feature_horizon=5)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def pilot_case(spec, output):
    from uniform_wait_worlds import make_world
    from run_unified_vfa import audited_episode
    start = time.perf_counter()
    world = make_world(**spec, **WORLD_OPTIONS)
    row, _ = audited_episode(world, **SETTING, **SEARCH)
    row['wall_seconds'] = time.perf_counter() - start
    row['generation'] = world['generation']
    row['pickup_examples'] = [dict(id=o['id'], release=o['release'],
        allowance=o['deadline']-o['release'], deadline=o['deadline'])
        for o in world['input']['orders'] if o['kind'] == 'P'][:3]
    name = 'L{layout_id}_d{day_id}_N{total}_{pattern}'.format(**spec)
    write(Path(output) / 'pilot' / (name + '.json'), row)
    return row


def training_group(alias, output):
    target = Path(output) / 'training' / alias
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'run_unified_vfa.py', 'train',
        '--review-mode', 'adaptive', '--adaptive-intervals', '2', '5', '10',
        '--arrival-mode', 'p2', '--action-solver', 'gurobi',
        '--mip-pattern-limit', '32', '--candidate-budget', '32',
        '--enhancement-budget', '8', '--initial-budget', '8',
        '--feature-sets', FEATURE_SETS[alias],
        '--orders', '40', '80', '--train-layouts', '112871', '112872',
        '--validation-layouts', '112881', '112882', '--days', '1',
        '--share', '0.6', '--patterns', 'uniform', 'dual', '--temporal', 'peak',
        '--iterations', '2', '--seed', '17', '--explore', '0.15',
        '--ridge', '1', '--history-days', '160', '--output', str(target)]
    write(Path(output) / 'commands' / (alias + '.json'), command)
    start = time.perf_counter()
    with (target.parent / (alias + '.log')).open('w', encoding='utf8') as log:
        result = subprocess.run(command, cwd=str(Path(__file__).resolve().parent),
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    status = dict(group=alias, returncode=result.returncode,
                  wall_seconds=time.perf_counter()-start)
    write(target.parent / (alias + '_status.json'), status)
    if result.returncode:
        raise RuntimeError('Training failed: ' + alias + '; see ' + str(target.parent / (alias+'.log')))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot', 'train', 'all'], default='all')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('../../results/feature_screen_20261128'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import gurobipy as gp
    import numpy as np
    manifest = dict(scope='development_only', test_generated=False, test_evaluated=False,
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                         gurobi=gp.gurobi.version(), platform=platform.platform()),
        split=dict(pilot=dict(layouts=[112870], days=[0,1], orders=[20,40,80]),
                   train=dict(layouts=[112871,112872], days=[0], orders=[40,80]),
                   validation=dict(layouts=[112881,112882], days=[0], orders=[40,80]),
                   reserved_test=dict(layouts=[112891,112892,112893,112894],
                                      days=[0,1,2,3,4], orders=[40,80,120],
                                      temporal=['peak','flat'], generated=False)),
        common=dict(patterns=['uniform','dual'], share=0.6, temporal='peak',
                    cities=2, world_options=WORLD_OPTIONS, search=SEARCH, setting=SETTING),
        training=dict(iterations=2, exploration=0.15, seed=17, ridge=1.0,
                      feature_sets=FEATURE_SETS, selection='validation_mean_net'),
        note='N is nominal: mandatory D=N-round(.6N); pickup count random with expectation round(.6N).')
    write(output/'campaign_manifest.json', manifest)
    if args.stage in ('pilot', 'all'):
        specs = [dict(layout_id=112870, day_id=d, total=n, share=.6, pattern=p)
                 for d in [0,1] for n in [20,40,80] for p in ['uniform','dual']]
        rows = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            tasks = [pool.submit(pilot_case, spec, str(output)) for spec in specs]
            for task in as_completed(tasks):
                row = task.result()
                rows.append(row)
                print(json.dumps(dict(stage='pilot', spec=row['world_spec'], net=row['net'],
                    pickups=row['realized_pickups'], accepted=row['accepted'],
                    search=row['action_search'])), flush=True)
        rows.sort(key=lambda r:(r['world_spec']['day_id'],r['world_spec']['total'],r['world_spec']['pattern']))
        write(output/'pilot_results.json', rows)
    if args.stage in ('train', 'all'):
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            tasks = [pool.submit(training_group, group, str(output)) for group in FEATURE_SETS]
            for task in as_completed(tasks):
                print(json.dumps(task.result()), flush=True)


if __name__ == '__main__':
    main()
