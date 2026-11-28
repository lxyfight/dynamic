"""32-day development pilot for a common demand-dependent waiting interval.

This is a mechanism/coverage experiment, not a trained-policy comparison.
At one predetermined public-state opportunity per day, replay depart and each
fixed/system interval against the identical realized future. Afterwards use
the same frozen dispatcher without new speculative waits. Also collect a
diagnostic trajectory with at most three reviews on the same station visit.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from e1_sim.audit import audit_ledger
from e1_sim.kernel import COSTS, require
from e1_sim.online import OnlineCoordinator, install_plan, prepare_online
from e1_sim.quantities import dec, plain
from e1_sim.system_wait import SystemWaitRule
from e1_sim.unknown_demand import seed
from run_multiscale import ROOT, digest, read, write
from system_wait_policy import SystemIntervalPolicy, paired_candidates
from uniform_wait_worlds import make_world, pilot_cases


DEFAULT_OUT = ROOT / 'results/system_wait_pilot_20261125'
FIXED_INTERVALS = (2, 5, 10, 20, 30)


def default_rule():
    return SystemWaitRule(signal='ready', max_h=30, target_expected=.5)


def source_hashes():
    src = Path(__file__).resolve().parent
    paths = list((src/'e1_sim').glob('*.py')) + [
        src/name for name in ('system_wait_policy.py', 'run_system_wait_pilot.py',
        'uniform_wait_worlds.py', 'high_demand_features.py', 'recovery_opportunity.py',
        'run_global_demand_pilot.py', 'run_multiscale.py', 'nn_refinement.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def terminal_result(state, policy, elapsed):
    require(state.is_terminal(), 'pilot did not finish all obligations')
    require(not state._hidden, 'pilot ended before all requests were revealed')
    require(all(o['completed_at'] is not None and o['completed_at'] <= o['deadline']
                for o in state.orders.values() if o['kind'] == 'D'), 'late mandatory delivery')
    ps = [o for o in state.orders.values() if o['kind'] == 'P']
    require(all(o['accepted'] is not None and
                (not o['accepted'] or o['completed_at'] is not None) for o in ps),
            'unresolved accepted pickup')
    net = state.ledger['revenue'] - sum(state.ledger[k] for k in COSTS)
    ledger = plain(dict(state.ledger, net=net))
    initial = deepcopy(state.initial)
    initial['stop_time'] = state.now
    audit = audit_ledger(state.cfg, state.network, initial, state.trace)
    require(not audit['errors'] and all(abs(dec(v)-dec(audit['ledger'][k])) < dec('1e-7')
            for k,v in ledger.items()), 'independent ledger audit failed')
    history = plain(getattr(state, '_station_wait_history', []))
    return dict(net=float(net), ledger=ledger, accepted=sum(o['accepted'] for o in ps),
                terminal=True, audited=True, seconds=elapsed,
                trace_hash=digest(plain(state.trace)),
                truck_trips=sum(e['kind'] == 'truck_dispatched' for e in state.trace),
                active_wait_minutes=sum(float(m['active_wait_minutes']) for m in history),
                active_reviews=sum(m['review_count'] for m in history),
                station_visits=history, decisions=len(policy.records),
                wait_decisions=policy.wait_decisions)


def save_trace(path, state):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(plain(state.trace), stream, ensure_ascii=False, separators=(',', ':'))


def baseline_run(world, model, rule, *, capture=True):
    policy = SystemIntervalPolicy(model, rule, mode='depart', observe=True)
    state, driver = prepare_online(world['input'], world['configuration'],
                                   world['network'], full_day=True)
    state._record_snapshots = False
    driver.policy = policy
    spec = world['spec']
    target_time = (0, 200, 400)[seed('system-wait-pilot-snapshot-v1',
        spec['layout_id'], spec['day_id'], spec['total'], spec['share']) % 3]
    selected = {}

    def snapshot(known, base, pairs, entry):
        # First eligible opportunity at/after a predeclared time target. Keep
        # the first as fallback if no such opportunity occurs later that day.
        qualifies = float(known.now) >= target_time
        if selected and (selected['target_reached'] or not qualifies):
            return
        pair = min(pairs, key=lambda p: (p['truck'], p['id']))
        selected.clear()
        selected.update(environment=deepcopy(state), known=deepcopy(known),
                        base=deepcopy(base), truck=pair['truck'],
                        visit_id=pair['identity']['visit_id'], target_reached=qualifies,
                        target_time=target_time, at=float(known.now),
                        cadence=deepcopy(entry['cadence']), prefix=digest(plain(state.trace)))

    if capture:
        policy.snapshot_callback = snapshot
    start = perf_counter()
    state.run(decision_hook=driver.on_event, stop_at_terminal=True)
    row = terminal_result(state, policy, perf_counter()-start)
    return row, selected, state


def continue_branch(snapshot, plan, model, rule, *, repeat=False):
    """Resume after completions/revelations, before the first chosen action.

    The environment snapshot retains the identical exogenous event queue.
    It is never passed directly to the policy: subsequent calls use known_clone.
    The current hook is skipped because that one decision is the intervention.
    A zero-intervention replay is required to match the full baseline exactly.
    """
    state = deepcopy(snapshot['environment'])
    policy = SystemIntervalPolicy(model, rule,
        mode='explore' if repeat else 'depart',
        repeat_visit=snapshot['visit_id'], repeat_limit=2 if repeat else 0,
        observe=repeat)
    driver = OnlineCoordinator(policy, extend_horizon=True)
    install_plan(state, deepcopy(plan))
    state._has_run = False
    state.start = state.now
    state._record_snapshots = False
    first = True

    def hook(environment):
        nonlocal first
        if first:
            first = False
            return
        driver.on_event(environment)

    start = perf_counter()
    state.run(decision_hook=hook, stop_at_terminal=True)
    row = terminal_result(state, policy, perf_counter()-start)
    row['continuation_forced_reviews'] = policy.forced_reviews
    return row, state


def one_case(task):
    spec, out_text, expected_sources = task
    out = Path(out_text)
    key = digest(spec)[:20]
    result_path = out/'cases'/(key+'.json')
    if result_path.exists():
        row = read(result_path)
        require(row['source_hash'] == digest(expected_sources), 'stale cached pilot result')
        return row
    require(source_hashes() == expected_sources, 'source changed during the frozen pilot')
    status_path = out/'status'/(key+'.json')
    write(status_path, dict(spec=spec, phase='world'))
    world_path = out/'worlds'/(key+'.json')
    if world_path.exists():
        world = read(world_path)
    else:
        world = make_world(**spec)
        write(world_path, world)
    require(world['input_hash'] == digest(world['input']), 'world hash mismatch')
    model = read(ROOT/'results/frozen_six.json')
    rule = default_rule()
    write(status_path, dict(spec=spec, phase='baseline'))
    baseline, snapshot, base_state = baseline_run(world, model, rule)
    trace_dir = out/'traces'/key
    trace_dir.mkdir(parents=True, exist_ok=True)
    save_trace(trace_dir/'baseline.json.gz', base_state)
    result = dict(spec=spec, source_hash=digest(expected_sources), model_hash=digest(model),
                  input_hash=world['input_hash'], i0_hash=world['i0_hash'],
                  public_hash=world['public_hash'], baseline=baseline, pairs=[],
                  sampled_state=None, repeated_diagnostic=None)
    if snapshot:
        zero, zero_state = continue_branch(snapshot, snapshot['base']['plan'], model, rule)
        require(all(zero[k] == baseline[k] for k in ('net','ledger','accepted','trace_hash')),
                'resumed departure is not identical to original baseline')
        sample = {k:v for k,v in snapshot.items() if k not in ('environment','known','base')}
        sample['base_plan_hash'] = digest(snapshot['base']['plan'])
        sample['zero_intervention_verified'] = True
        result['sampled_state'] = sample
        temporal = SystemWaitRule(signal='arrivals', max_h=30, target_expected=.5).decision(
            snapshot['known'].public, snapshot['known'].now)
        sample['temporal_only_cadence'] = temporal
        intervals = sorted(set(FIXED_INTERVALS + (snapshot['cadence']['h'], temporal['h'])))
        adaptive_pair = None
        for interval in intervals:
            write(status_path, dict(spec=spec, phase='paired_continuation', h=interval,
                                   completed_pairs=len(result['pairs'])))
            candidates, diagnostics = paired_candidates(snapshot['known'], snapshot['base'],
                                                        interval, truck=snapshot['truck'])
            require(len(candidates) <= 1, 'duplicate pair for a single truck/h')
            if not candidates:
                result['pairs'].append(dict(requested_h=interval, feasible=False,
                                            diagnostics=plain(diagnostics)))
                continue
            pair = candidates[0]
            alt, state = continue_branch(snapshot, pair['candidate']['plan'], model, rule)
            delta = {k:alt['ledger'][k]-baseline['ledger'][k] for k in baseline['ledger']}
            gain = alt['net']-baseline['net']
            require(abs(gain-delta['net']) < 1e-7, 'paired accounting discrepancy')
            label = dict(id=digest([world['input_hash'], pair['identity']]),
                         identity=pair['identity'], requested_h=interval, h=pair['h'],
                         is_system_interval=interval == snapshot['cadence']['h'], feasible=True,
                         is_temporal_interval=interval == temporal['h'],
                         known_plan_delta=pair['known_plan_delta'], gain=gain,
                         residual_target=gain-pair['known_plan_delta'],
                         accepted_delta=alt['accepted']-baseline['accepted'],
                         ledger_delta=delta, memory=pair['memory'],
                         outcome=alt, diagnostics=plain(diagnostics))
            result['pairs'].append(label)
            save_trace(trace_dir/f'wait_{interval}.json.gz', state)
            if label['is_system_interval']:
                adaptive_pair = pair
        if adaptive_pair is not None:
            write(status_path, dict(spec=spec, phase='repeated_wait_diagnostic'))
            repeated, state = continue_branch(snapshot, adaptive_pair['candidate']['plan'],
                                              model, rule, repeat=True)
            repeated['gain'] = repeated['net']-baseline['net']
            repeated['interpretation'] = 'Diagnostic exploration, at most three reviews on one visit; not a learned strategy.'
            result['repeated_diagnostic'] = repeated
            save_trace(trace_dir/'repeat_up_to_three.json.gz', state)
    require(source_hashes() == expected_sources, 'source changed while case was executing')
    write(result_path, result)
    write(status_path, dict(spec=spec, phase='complete', paired_runs=sum(p['feasible'] for p in result['pairs'])))
    return result


def summarize(rows, out):
    labels = [p for r in rows for p in r['pairs'] if p['feasible']]
    adaptive = [p for p in labels if p['is_system_interval']]
    temporal = [p for p in labels if p['is_temporal_interval']]
    entries = [e for r in rows for e in r['baseline']['wait_decisions']]
    hist = {}
    for entry in entries:
        h = str(entry['cadence']['h'])
        hist[h] = hist.get(h, 0)+1
    def gains(group):
        return dict(n=len(group), mean_gain=mean(p['gain'] for p in group) if group else None,
                    wins=sum(p['gain']>1e-7 for p in group),
                    ties=sum(abs(p['gain'])<=1e-7 for p in group),
                    losses=sum(p['gain'] < -1e-7 for p in group))
    summary = dict(completed_days=len(rows), independent_layouts=len({r['spec']['layout_id'] for r in rows}),
        days_with_pair=sum(r['sampled_state'] is not None for r in rows),
        audited_runs=len(rows)+len(labels)+sum(r['sampled_state'] is not None for r in rows)
                     +sum(r['repeated_diagnostic'] is not None for r in rows),
        baseline_mean_net=mean(r['baseline']['net'] for r in rows),
        baseline_mean_accepted=mean(r['baseline']['accepted'] for r in rows),
        decision_epochs=len(entries), departure_epochs=sum(bool(e['departure_actors']) for e in entries),
        feasible_pair_epochs=sum(bool(e['pairs']) for e in entries),
        interval_histogram=hist, single_intervention_system=gains(adaptive),
        single_intervention_temporal_only=gains(temporal),
        fixed_interventions={str(h):gains([p for p in labels if p['requested_h']==h]) for h in FIXED_INTERVALS},
        repeated_diagnostic=gains([r['repeated_diagnostic'] for r in rows if r['repeated_diagnostic'] is not None]),
        limitations=['Development diagnostics, no new waiting model trained.',
          'Single-intervention values are realized paired returns, not optimal policy values.',
          'Repeated waits are bounded diagnostic exploration, not a performance claim.',
          'Only two independent layouts; no confirmatory confidence interval.'])
    write(out/'summary.json', summary)
    with (out/'paired_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields=['layout_id','day_id','total','share','pattern','at','requested_h','h','is_system_interval','is_temporal_interval',
                'gain','known_plan_delta','residual_target','accepted_delta','input_hash','i0_hash']
        writer=csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for pair in row['pairs']:
                if not pair['feasible']:
                    continue
                item={**row['spec'], 'at':row['sampled_state']['at'], **pair,
                      'input_hash':row['input_hash'], 'i0_hash':row['i0_hash']}
                writer.writerow({k:item[k] for k in fields})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--smoke', action='store_true', help='One 20-order development case; use a separate output.')
    args=parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    cases=pilot_cases()
    if args.smoke:
        cases=[dict(layout_id=609,day_id=0,total=20,share=.4,pattern='uniform')]
    elif args.limit is not None:
        cases=cases[:args.limit]
    out=args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources=source_hashes()
    protocol=dict(version=1,cases=cases,rule=asdict(default_rule()),fixed_intervals=FIXED_INTERVALS,
        six_model_hash=digest(read(ROOT/'results/frozen_six.json')),sources=sources,
        sampling='One first public eligible event at/after predetermined 0/200/400; first-event fallback.',
        continuation='Same frozen six-feature dispatcher with no new speculative wait after intervention.',
        recurrence='One separate trajectory with at most three reviews on the sampled visit.',
        study='Development pilot; no weight fitting, no validation/test selection, no optimality claim.')
    protocol=json.loads(json.dumps(protocol))
    pp=out/'protocol.json'
    if pp.exists():
        require(read(pp)==protocol, 'protocol/source changed; choose a new output directory')
    write(pp, protocol)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(one_case,(c,str(out),sources)):c for c in cases}
        for future in as_completed(jobs):
            row=future.result()
            rows.append(row)
            write(out/'progress.json',dict(completed=len(rows),expected=len(cases),last_case=row['spec']))
            print('COMPLETE',len(rows),'/',len(cases),row['spec'],
                  'pairs',sum(p['feasible'] for p in row['pairs']),flush=True)
    rows.sort(key=lambda r:json.dumps(r['spec'], sort_keys=True))
    print(json.dumps(summarize(rows,out),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
