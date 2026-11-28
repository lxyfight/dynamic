"""Gate-protected full-day review comparison; admissions/services remain none.

This is a new policy experiment, not a reinterpretation of action-only gains.
It must not execute until the tune-frozen primary passes both dev action gates.
The reserved independent final-test bank is never generated or evaluated.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from e1_sim.information import closed, opportunities_open
from e1_sim.initializer import digest
from e1_sim.online import prepare_online
from e1_sim.quantities import dec, plain
from fit_paired_chain import (ACTIONS, TOL, VERSION, DOWNSTREAM_NAME, cluster_stats,
    file_hash, freeze, model_scores, object_hash, read, write)
from paired_wait_branches import build_branches, qualification
from run_feature_screen import SEARCH, SETTING, WORLD_OPTIONS
from run_paired_chain_campaign import case_id, source_hashes, specs
from run_paired_wait_diagnostic import final_metrics
from unified_vfa_features import FEATURE_NAMES, features
from unified_vfa_policy import UnifiedVFAPolicy, canonical_plan_digest
from uniform_wait_worlds import make_world


def action_menu(mode, fixed_h=None):
    if mode == 'adaptive' and fixed_h is None:
        return ACTIONS
    if mode == 'fixed' and fixed_h in (2,5,10):
        return (0,fixed_h)
    raise ValueError('choose adaptive or fixed with h in 2,5,10')


def menu_choice(scores, menu):
    if len(scores) != 4 or not all(np.isfinite(scores)):
        raise ValueError('four finite known-state scores required')
    best = max(scores[ACTIONS.index(h)] for h in menu)
    return next(h for h in menu if best-scores[ACTIONS.index(h)] <= TOL)


def validate_action_gate(result, frozen, model):
    """Recompute the conjunctive gate, rather than trusting a command flag."""
    if (result.get('stage') != 'augmented' or not result.get('dev_evaluated')
            or not result.get('proceed_to_full_day_model_comparison')):
        raise ValueError('full-day simulation prohibited: frozen primary dev action gate has not passed')
    primary = result.get('primary_family')
    if primary not in ('paired_chain','paired_augmented') or frozen.get('primary_family') != primary:
        raise ValueError('primary family is missing or differs from tune freeze')
    gate = result['candidate_gates'][primary]
    if not all(gate.get(k) for k in ('pass_both_lower95','complete_eight_layouts','selected_primary','authoritative_for_full_day')):
        raise ValueError('full-day simulation prohibited: primary gate incomplete or failed')
    inference = result['reports']['dev']['models'][primary]['layout_inference']
    if any(inference[ref]['n'] != 8 or inference[ref]['t_df'] != 7
           or not np.isfinite(inference[ref]['ci95']).all() or inference[ref]['ci95'][0] <= 0
           for ref in ('depart','fixed10')):
        raise ValueError('full-day simulation prohibited: both eight-layout lower95 bounds must exceed zero')
    if model != frozen['selected_models'][primary]:
        raise ValueError('saved model differs from the tune-frozen model')
    columns = [0,1] if primary == 'paired_chain' else [0,1,7]
    names = list(FEATURE_NAMES)+[DOWNSTREAM_NAME]
    if (model.get('version') != VERSION or model.get('kind') != 'paired'
            or model.get('columns') != columns or model.get('intercept') != 0
            or model.get('feature_names') != [names[c] for c in columns]
            or len(model.get('weights',[])) != len(columns)
            or len(model.get('scales',[])) != len(columns)
            or not all(np.isfinite(model['weights']))
            or not all(np.isfinite(s) and s > 0 for s in model['scales'])):
        raise ValueError('invalid frozen primary differential model')
    return primary


def load_gate(output):
    output = Path(output)
    result_path = output/'augmented/results.json'
    if not result_path.is_file():
        raise ValueError('full-day simulation prohibited: augmented/results.json and passed dev action gate required')
    result = read(result_path)
    # Reject missing/false gate before trying to read models or generate worlds.
    if not result.get('proceed_to_full_day_model_comparison'):
        raise ValueError('full-day simulation prohibited: primary dev action gate is false')
    frozen_path = output/'augmented/frozen_selection.json'
    frozen = read(frozen_path)
    primary = result['primary_family']
    model_path = output/'augmented/models'/(primary+'.json')
    model = read(model_path)
    validate_action_gate(result,frozen,model)
    source = Path(__file__).resolve().parent
    feature_receipts = read(output/'augmented/known_features/train.json')['input_receipts']
    for name,expected in feature_receipts.items():
        if name.startswith('source/') and file_hash(source/name[len('source/'):]) != expected:
            raise ValueError('fitted feature source changed: '+name)
    runtime = source_hashes()
    for split in ('train','tune','dev'):
        protocol = read(output/split/'protocol.json')
        if protocol['source_hashes'] != runtime or protocol['cases'] != specs(split):
            raise ValueError('locked simulator or bank design changed since paired campaign')
        if not (output/split/'evaluation_complete.json').is_file():
            raise ValueError('all three paired banks must be complete before full-day simulation')
    return model, dict(primary_family=primary,result_sha256=file_hash(result_path),
        frozen_selection_sha256=file_hash(frozen_path),model_sha256=file_hash(model_path),
        runtime_source_hashes=runtime)


class PairedReviewPolicy(UnifiedVFAPolicy):
    """A none planner followed by one truck's restricted known-only selector."""
    def __init__(self, action_model, mode, fixed_h=None, **search):
        options = dict(SEARCH)
        options.update(search)
        # Do not give the action-specific ridge model to the original VFA.
        super().__init__(model=None,explore=0.,**SETTING,**options)
        self.action_model = deepcopy(action_model)
        self.action_mode, self.fixed_h = mode, fixed_h
        self.menu = action_menu(mode,fixed_h)
        self.selector_log = []

    def score_branches(self, known, branches, truck):
        xs, js, audits = [], [], []
        for h in ACTIONS:
            branch = branches[h]
            x = list(features(known,branch,5))
            if 7 in self.action_model['columns']:
                from paired_downstream_opportunity import opportunity_from_candidate
                value,audit = opportunity_from_candidate(known,branch,truck)
                x.append(float(value));audits.append(audit)
            xs.append(x);js.append(float(branch['score']))
        row = dict(features=xs,J=js,closed=closed(known.public))
        return model_scores(row,self.action_model).tolist(),xs,audits

    def choose(self, known, evaluated):
        base = super().choose(known,evaluated)
        event = dict(at=float(known.now),trace_prefix_length=len(known.trace),mode=self.action_mode,
            menu=list(self.menu),none_plan_hash=canonical_plan_digest(base['plan']),
            none_selected=base['id'],none_J=float(base['score']),status='none_fallback',
            qualification=[],selected_h=None,selected_plan_hash=canonical_plan_digest(base['plan']))
        if not opportunities_open(known.public,known.now):
            event['reason'] = 'future_opportunities_closed'
            self.selector_log.append(event)
            return base
        for tid in sorted(known.trucks):
            q = qualification(known,base,tid,require_just_unloaded=True)
            if q['just_unloaded'] or q['eligible']:
                event['qualification'].append(q)
        eligible = [q for q in event['qualification'] if q['eligible']]
        if len(eligible) != 1:
            event['reason'] = 'multiple_eligible_trucks' if eligible else 'no_eligible_post_unload_departure'
            self.selector_log.append(event)
            return base
        truck = eligible[0]['truck']
        try:
            branches = build_branches(known,base,truck,require_just_unloaded=True)
        except ValueError as exc:
            event.update(reason='four_branch_feasibility_failed',detail=str(exc),truck=truck)
            self.selector_log.append(event)
            return base
        scores,xs,audits = self.score_branches(known,branches,truck)
        h = menu_choice(scores,self.menu)
        chosen,index = branches[h],ACTIONS.index(h)
        event.update(status='selected',truck=truck,station=eligible[0]['station'],selected_h=h,
            selected_plan_hash=canonical_plan_digest(chosen['plan']),all_four_verified=True,
            branch_audits={str(k):v['intervention'] for k,v in branches.items()},
            branch_scores={str(k):dict(J=float(branches[k]['score']),features=xs[i],
                score=scores[i],delta_vs_depart=scores[i]-scores[0],offered=k in self.menu)
                for i,k in enumerate(ACTIONS)},downstream_audits=audits)
        self.selector_log.append(event)
        self._record_actual_choice(known,chosen,branches,scores,xs,event,index)
        return chosen

    def _record_actual_choice(self, known, chosen, branches, scores, xs, event, index):
        """Parent search costs remain none-search costs; chosen actions are real."""
        entry = self.selection_log[-1]
        prior_reviews = getattr(known,'_wait_reviews',{})
        fresh = {tid:float(dec(at)-known.now) for tid,at in chosen['plan'].get('wait_reviews',{}).items()
                 if dec(prior_reviews.get(tid,known.now)) <= known.now < dec(at)}
        entry['none_selection'] = {k:deepcopy(entry[k]) for k in ('selected','J','W','canonical_plan_hash','chosen_wait_intervals')}
        entry['paired_selector'] = dict(truck=event['truck'],selected_h=event['selected_h'],menu=list(self.menu),
            none_plan_hash=event['none_plan_hash'],selected_plan_hash=event['selected_plan_hash'])
        entry.update(selected=chosen['id'],J=float(chosen['score']),W=scores[index]-float(chosen['score']),
            features=xs[index],feature_names=list(FEATURE_NAMES)+([DOWNSTREAM_NAME] if len(xs[index])==8 else []),
            model_mode='none_base_then_frozen_action_selector',chosen_wait_intervals=fresh,
            plan_hash=digest(chosen['plan']),canonical_plan_hash=canonical_plan_digest(chosen['plan']),
            admissions=deepcopy(chosen['plan']['decisions']),explored=False)
        # Keep the original search pool under its own name; the choice pool
        # exposed by decide must describe the actual restricted selector.
        entry['none_search_pool'] = entry['pool']
        entry['pool'] = [dict(id=branches[h]['id'],J=float(branches[h]['score']),
            W=scores[i]-float(branches[h]['score']),score=scores[i],features=xs[i],
            plan_hash=digest(branches[h]['plan']),decisions=deepcopy(branches[h]['plan']['decisions']),
            wait_reviews=plain(branches[h]['plan'].get('wait_reviews',{})),
            joint_wait=branches[h].get('joint_wait')) for i,h in enumerate(ACTIONS) if h in self.menu]
        entry['selector_feasible_interval_counts'] = {str(h):1 for h in self.menu if h}
        self.records[-1].update(j=float(chosen['score']),x=xs[index],plan=entry['plan_hash'],decision=chosen['id'])
        self.samples[-1].update(x=xs[index],j=float(chosen['score']),plan=entry['plan_hash'],explored=False)

    def selector_metrics(self):
        selected = [r for r in self.selector_log if r['status']=='selected']
        waits = [r for r in selected if r['selected_h'] > 0]
        return dict(epochs=len(self.selector_log),eligible_selection_epochs=len(selected),
            active_wait_events=len(waits),active_wait_interval_counts=dict(Counter(r['selected_h'] for r in waits)),
            depart_selected=sum(r['selected_h']==0 for r in selected),
            fallback_reasons=dict(Counter(r['reason'] for r in self.selector_log if r['status']!='selected')),
            four_branch_replay_count=4*len(selected),
            dependent_drone_timing_events=sum(not r['branch_audits'][str(r['selected_h'])]['strict_non_target_actions_unchanged'] for r in waits),
            retained_none_behavior='All noneligible decisions, admissions, assignments, routes and new trucks use none.')


def attach_actual_departures(trace, selector_log):
    for row in selector_log:
        if row['status'] != 'selected':
            continue
        departures = [e for e in trace[row['trace_prefix_length']:] if e['kind']=='road_started'
            and e.get('actor')==row['truck'] and e.get('source')==row['station']]
        if not departures:
            raise ValueError('selected truck has no subsequent physical departure')
        delay = float(departures[0]['at'])-row['at']
        if delay < -TOL:
            raise ValueError('recorded departure precedes its selector state')
        row.update(actual_departure_delay=delay,actual_first_departure=float(departures[0]['at']),
            early_interruption=bool(row['selected_h'] > 0 and delay < row['selected_h']-TOL))


def full_day_protocol(output, model, gate):
    source = Path(__file__).resolve().parent
    value = dict(version='paired-chain-full-day-v1',gate=gate,model=model,search=SEARCH,setting=SETTING,
        world_options=WORLD_OPTIONS,tune_cases=specs('tune'),dev_cases=specs('dev'),
        seed=read(output/'tune/protocol.json')['seed'],
        selector_source_sha256=file_hash(source/'run_paired_chain_full_day.py'),
        fit_source_sha256=file_hash(source/'fit_paired_chain.py'),
        downstream_source_sha256=file_hash(source/'paired_downstream_opportunity.py'),
        eligibility='one just-unloaded pure-depart truck, all four first-action branches feasible',
        adaptive_menu=[0,2,5,10],fixed_menus=[[0,2],[0,5],[0,10]],
        fixed_selection='tune mean full-day net, tie smallest h; same frozen model for all menus',
        multi_truck_rule='none fallback if multiple eligible trucks',
        independent_test_opened=False,
        scope='exploratory development full-day confirmation after the dev action gate; not independent final test')
    freeze(output/'full_day/protocol.json',value)
    return value


def evaluate_day(spec, output, split, model, mode, fixed_h, run_hash):
    output = Path(output)
    source_protocol = read(output/split/'protocol.json')
    case = case_id(spec,source_protocol['seed'])
    policy_id = 'adaptive' if mode == 'adaptive' else 'fixed'+str(fixed_h)
    path = output/'full_day'/split/policy_id/(case+'.json')
    if path.exists():
        cached = read(path)
        if (cached['run_hash'] != run_hash or cached['spec'] != spec or cached['policy'] != policy_id
                or not cached['terminal'] or not cached['ledger_audited']):
            raise ValueError('cached full-day output differs from the frozen protocol')
        return cached
    captured = read(output/split/'capture'/(case+'.json'))
    world = make_world(**spec,**WORLD_OPTIONS)
    if world['input_hash'] != captured['input_hash'] or world['i0_hash'] != captured['i0_hash']:
        raise ValueError('full-day world differs from captured none world')
    state,driver = prepare_online(world['input'],world['configuration'],world['network'],full_day=True)
    state._record_snapshots = False
    policy = PairedReviewPolicy(model,mode,fixed_h)
    driver.policy = policy
    state.run(decision_hook=driver.on_event,stop_at_terminal=True)
    metrics = final_metrics(state)
    attach_actual_departures(state.trace,policy.selector_log)
    selector = policy.selector_metrics()
    selector['early_interrupted_waits'] = sum(r.get('early_interruption',False) for r in policy.selector_log)
    selector['actual_wait_delay_sum'] = sum(r.get('actual_departure_delay',0.) for r in policy.selector_log if r.get('selected_h'))
    row = dict(metrics,case=case,spec=spec,policy=policy_id,run_hash=run_hash,
        input_hash=world['input_hash'],i0_hash=world['i0_hash'],selector=selector,
        search=policy.search_metrics(),selector_log=policy.selector_log,
        actual_selection_log=plain(policy.selection_log),
        canonical_decisions=digest([(r['at'],r['canonical_plan_hash']) for r in policy.selection_log]),
        none_reference={k:captured['reference'][k] for k in ('net','accepted','trips','trace')},
        net_gain_vs_none=metrics['net']-captured['reference']['net'])
    write(path,row)
    return row


def execute_jobs(jobs, workers):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(evaluate_day,*args) for args in jobs]
        for future in as_completed(pending):
            row = future.result();results.append(row)
            print(json.dumps(dict(case=row['case'],policy=row['policy'],net=row['net'],
                active_waits=row['selector']['active_wait_events'],accepted=row['accepted']),ensure_ascii=False),flush=True)
    return sorted(results,key=lambda r:(r['case'],r['policy']))


def select_fixed(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['policy']].append(row)
    if set(groups) != {'fixed2','fixed5','fixed10'} or any(len(v)!=4 for v in groups.values()):
        raise ValueError('fixed selection needs exactly four tune cases for every h')
    identities = [{r['case'] for r in groups['fixed'+str(h)]} for h in (2,5,10)]
    if any(identity != identities[0] or len(identity)!=4 for identity in identities):
        raise ValueError('fixed policies must share the same four tune cases')
    means = {int(name[5:]):float(np.mean([r['net'] for r in group])) for name,group in groups.items()}
    best = max(means.values())
    selected = min(h for h,value in means.items() if best-value <= TOL)
    return dict(selected_h=selected,tune_mean_net=means,tie_break='smallest h within 1e-9',
        selection_source='tune full-day returns only',dev_returns_used=False)


def dev_comparison(rows, selected_h):
    grouped = defaultdict(dict)
    fixed = 'fixed'+str(selected_h)
    for row in rows:
        if row['policy'] in grouped[row['case']]:
            raise ValueError('duplicate full-day case/policy')
        grouped[row['case']][row['policy']] = row
    if len(grouped) != 16 or any(set(v)!={'adaptive',fixed} for v in grouped.values()):
        raise ValueError('dev requires all sixteen paired adaptive/fixed worlds')
    comparisons,clusters = [],defaultdict(list)
    for case,policies in sorted(grouped.items()):
        a,f = policies['adaptive'],policies[fixed]
        if (a['input_hash'],a['i0_hash'],a['none_reference']) != (f['input_hash'],f['i0_hash'],f['none_reference']):
            raise ValueError('paired full-day worlds or none references differ')
        item = dict(case=case,spec=a['spec'],adaptive_net=a['net'],fixed_net=f['net'],none_net=a['none_reference']['net'],
            adaptive_minus_fixed=a['net']-f['net'],adaptive_minus_none=a['net_gain_vs_none'],
            fixed_minus_none=f['net_gain_vs_none'],
            active_waits={p:r['selector']['active_wait_events'] for p,r in policies.items()},
            waits_by_interval={p:r['selector']['active_wait_interval_counts'] for p,r in policies.items()},
            accepted={p:r['accepted'] for p,r in policies.items()},
            trips={p:r['trips'] for p,r in policies.items()},
            search_truncated_epochs={p:r['search']['truncated_epochs'] for p,r in policies.items()},
            trace_hashes={p:r['trace'] for p,r in policies.items()},
            ledger_audited=all(r['ledger_audited'] and r['terminal'] for r in policies.values()))
        comparisons.append(item);clusters[a['spec']['layout_id']].append(item)
    metrics = ('adaptive_minus_fixed','adaptive_minus_none','fixed_minus_none')
    layout_means = {str(l):{k:float(np.mean([r[k] for r in group])) for k in metrics} for l,group in sorted(clusters.items())}
    return dict(cases=comparisons,selected_fixed_h=selected_h,
        mean_net={k:float(np.mean([r[k] for r in comparisons])) for k in ('adaptive_net','fixed_net','none_net')},
        equal_case_differences={k:float(np.mean([r[k] for r in comparisons])) for k in metrics},
        layout_means=layout_means,layout_inference={k:cluster_stats([r[k] for r in layout_means.values()]) for k in metrics},
        independent_test_opened=False,
        caveat='Dev was previously used for the action gate. These are exploratory full-day results, not independent final-test evidence.')


def run(output, stage, jobs=4):
    output = Path(output).resolve()
    # Both manually requested stages must pass this check before any worker,
    # model rollout, world generation or full-day output mutation occurs.
    model,gate = load_gate(output)
    if jobs < 1:
        raise ValueError('jobs must be positive')
    protocol = full_day_protocol(output,model,gate)
    run_hash = object_hash(protocol)
    if stage == 'tune-fixed':
        tasks = [(spec,str(output),'tune',model,'fixed',h,run_hash) for spec in specs('tune') for h in (2,5,10)]
        rows = execute_jobs(tasks,jobs)
        selection = dict(select_fixed(rows),run_hash=run_hash,primary_family=gate['primary_family'])
        freeze(output/'full_day/selected_fixed.json',selection)
        write(output/'full_day/tune_results.json',dict(selection=selection,cases=rows,independent_test_opened=False))
        return selection
    if stage != 'dev':
        raise ValueError('stage must be tune-fixed or dev')
    selection = read(output/'full_day/selected_fixed.json')
    if selection['run_hash'] != run_hash or selection['primary_family'] != gate['primary_family']:
        raise ValueError('fixed choice was not frozen under the same model/protocol')
    h = selection['selected_h']
    tasks = [(spec,str(output),'dev',model,mode,interval,run_hash) for spec in specs('dev')
             for mode,interval in (('adaptive',None),('fixed',h))]
    rows = execute_jobs(tasks,jobs)
    report = dev_comparison(rows,h)
    report.update(run_hash=run_hash,primary_family=gate['primary_family'])
    write(output/'full_day/dev_results.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['tune-fixed','dev'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    result = run(args.output,args.stage,args.jobs)
    print(json.dumps(dict(stage=args.stage,output=str(args.output/'full_day'),
        selected_h=result.get('selected_h',result.get('selected_fixed_h'))),ensure_ascii=False))


if __name__=='__main__':
    main()
