"""Paired high-demand, out-of-layout comparison of frozen VFA and adapted RH."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter
import csv
import hashlib

import run_high_demand_screen as source_campaign
from e1_sim.enhancement import enhance_pool
from e1_sim.initializer import digest
from e1_sim.quantities import dec
from high_demand_features import FeaturePolicy
from recovery_opportunity import run
from run_global_demand_pilot import ROOT, read, write, operation_metrics


OUT = ROOT / 'tmp/frozen_reopt_comparison_v1'
PARENT = ROOT / 'tmp/high_demand_screen_v1'
LAYOUTS = (460, 461, 462, 463)
SIZES = (150, 100)
SHAPES = ('uniform', 'dual')
CASES = list(product(LAYOUTS, SIZES, SHAPES))
POLICIES = ('VFA', 'RH', 'Myopic')


def prepare(case):
    # Routing the input cache to this process's study directory preserves the
    # frozen experiment and reuses its generator and forecast verbatim.
    source_campaign.OUT = OUT
    world = source_campaign.world_for(case)
    raw = world['input']
    assert digest(raw) == world['input_hash']
    assert len(raw['orders']) == case[1]
    assert sum(o['kind'] == 'P' for o in raw['orders']) == round(case[1] * .4)


def physical_key(row):
    """Canonicalize exactly equivalent times without rounding physical starts."""
    def time_text(value):
        number = dec(value)
        if number == 0:
            return '0'
        raw = format(number,'f')
        return raw.rstrip('0').rstrip('.') if '.' in raw else raw

    plan = row['plan']
    ordered = sorted(plan['actions'],key=lambda a:dec(a['at']))
    return digest(dict(actions=[dict(a,at=time_text(a['at'])) for a in ordered],
                       new_trucks=sorted(plan['new_trucks'], key=lambda truck: truck['id']),
                       decisions=plan['decisions'], wait_reviews={k:time_text(v) for k,v in plan.get('wait_reviews', {}).items()},
                       wakeups=[time_text(v) for v in sorted(plan.get('wakeups', []),key=dec)]))


def best(rows):
    return min(rows, key=lambda c: (-float(c['score']), c['id']))


class RollingReoptimization(FeaturePolicy):
    """Reoptimize the verified known-order suffix at every genuine decision."""
    def __init__(self):
        super().__init__(teacher=0)
        self.second_pass = []

    def final_pool(self, known, evaluated):
        common = super().final_pool(known, evaluated)
        # Adapted RH's prescribed second pass (Comparator.choose), applied to
        # the complete current common pool, including deferred recovery trips.
        finalists = sorted(common, key=lambda c: (-float(c['score']), c['id']))[:4]
        augmented, stats = enhance_pool(known, finalists, 8, self.waiting)
        extra = augmented[len(finalists):]
        pool = common + extra
        common_keys = {physical_key(row) for row in common}
        extra_keys = {physical_key(row) for row in extra}
        before, after = best(common), best(pool)
        gain = float(after['score'] - before['score'])
        assert gain >= -1e-7
        self.second_pass.append(dict(at=float(known.now), attempts=stats['attempted'],
            feasible=len(extra), unique_new=len(extra_keys - common_keys),
            selected_new=physical_key(after) not in common_keys,
            selected_different=physical_key(after) != physical_key(before),
            strictly_improved=gain > 1e-7, immediate_gain=gain, by_move=stats['by_move']))
        return pool


def execute(job):
    case, label = job
    # Only final canonical-time diagnostics are published as RH results.
    filename = '_'.join(map(str, (*case, label))) + '.json'
    path = OUT / ('rh_verified' if label == 'RH' else 'runs') / filename
    if path.exists():
        return read(path)
    w = read(OUT / 'worlds' / (digest(case) + '.json'))
    model = read(PARENT / 'freeze.json')['model'] if label == 'VFA' else None
    policy = RollingReoptimization() if label == 'RH' else FeaturePolicy(model=model, teacher=0)
    t0 = perf_counter()
    row = run(w, 0, policy_override=policy, metrics=operation_metrics)
    row.update(case=case, policy=label, input_hash=w['input_hash'], i0_hash=w['i0_hash'],
               policy_seconds=perf_counter() - t0, model_hash=digest(model))
    if label == 'RH':
        row['second_pass'] = policy.second_pass
        assert len(policy.second_pass) == row['decisions']
    write(path, row)
    return row


def summarize(rows, model):
    keyed = {(tuple(r['case']), r['policy']): r for r in rows}
    assert set(keyed) == set(product(CASES, POLICIES)) and len(rows) == len(keyed)
    flattened = []
    for case in CASES:
        w = read(OUT / 'worlds' / (digest(case) + '.json'))
        assert digest(w['input']) == w['input_hash']
        for label in POLICIES:
            row = keyed[case, label]
            assert row['input_hash'] == w['input_hash'] and row['i0_hash'] == w['i0_hash']
            assert row['model_hash'] == digest(model if label == 'VFA' else None)
            ledger = row['ledger']
            reproduced = ledger['revenue'] - sum(v for k, v in ledger.items() if k not in ('net','revenue'))
            assert abs(reproduced - row['net']) < 1e-7
            extra = row.get('second_pass', [])
            totals = {k:sum(r[k] for r in extra) for k in
                      ('attempts','feasible','unique_new','selected_new','selected_different','strictly_improved')}
            flattened.append(dict(layout=case[0],orders=case[1],shape=case[2],policy=label,
                net=row['net'],gain_vs_Myopic=row['net']-keyed[case,'Myopic']['net'],
                gain_vs_RH=row['net']-keyed[case,'RH']['net'],accepted=row['accepted'],
                truck_trips=row['operation_metrics']['trips'],new_waits=row['operation_metrics']['new_wait_chosen'],
                decisions=row['decisions'],seconds=row['policy_seconds'],**totals,
                input_hash=w['input_hash'],i0_hash=w['i0_hash']))
    with (OUT/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=flattened[0])
        writer.writeheader()
        writer.writerows(flattened)
    cells = {}
    for n in ('all',100,150):
        cells[n] = {}
        for label in POLICIES:
            group = [r for r in flattened if r['policy']==label and (n=='all' or r['orders']==n)]
            cells[n][label] = dict(cases=len(group), **{k:mean(r[k] for r in group) for k in
                ('net','gain_vs_Myopic','gain_vs_RH','accepted','truck_trips','new_waits','seconds')})
        for baseline in ('Myopic','RH'):
            diffs = [r['gain_vs_'+baseline] for r in flattened if r['policy']=='VFA' and (n=='all' or r['orders']==n)]
            cells[n]['VFA']['wins_ties_losses_vs_'+baseline] = [
                sum(d>1e-7 for d in diffs),sum(abs(d)<=1e-7 for d in diffs),sum(d< -1e-7 for d in diffs)]
    rh = [r for r in flattened if r['policy']=='RH']
    diagnostics = {k:sum(r[k] for r in rh) for k in ('decisions','attempts','feasible','unique_new',
                   'selected_new','selected_different','strictly_improved')}
    diagnostics['same_full_day_physical_trace_as_Myopic'] = sum(keyed[c,'RH']['trace']==keyed[c,'Myopic']['trace'] for c in CASES)
    write(OUT/'summary.json',dict(groups=cells,second_pass=diagnostics))
    print(cells,diagnostics,flush=True)


def main():
    src = Path(__file__).resolve().parent
    freeze = read(PARENT/'freeze.json')
    earlier = read(PARENT/'protocol.json')
    assert freeze['chosen']=='shortlist_2' and len(freeze['model']['columns'])==6
    assert not set(LAYOUTS) & {x for split in earlier['splits'].values() for x in split}
    assert not set(LAYOUTS) & {452,453}
    for name, expected_hash in earlier['source'].items():
        candidate = src / name
        if not candidate.is_file():
            candidate = src/'e1_sim'/name
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_hash
    protocol = dict(cases=CASES,policies=POLICIES,share=.4,freeze_hash=digest(freeze),
        model=freeze['model'],study_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_sources=earlier['source'],joint_admission_limit=3,insertion_budget=16,beam_width=4,
        common_enhancement_budget=16,wait_minutes=5,rh_parents=4,rh_additional_replay_budget=8,
        rh_objective='Known-order feasible continuation net; full physical replay, replanning at decision events.',
        external_wait_diagnostics=False)
    if (OUT/'protocol.json').exists():
        assert digest(read(OUT/'protocol.json'))==digest(protocol)
    write(OUT/'protocol.json',protocol)
    # Launch policy jobs as soon as their one immutable world is available.
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        pending_worlds = {pool.submit(prepare, case):case for case in CASES}
        futures = []
        for future in as_completed(pending_worlds):
            future.result()
            case = pending_worlds[future]
            futures.extend(pool.submit(execute,(case,policy)) for policy in POLICIES)
            print('world',case,flush=True)
        for f in as_completed(futures):
            r = f.result()
            rows.append(r)
            print('complete',len(rows),len(futures),r['case'],r['policy'],flush=True)
    summarize(rows,freeze['model'])


if __name__ == '__main__':
    main()
