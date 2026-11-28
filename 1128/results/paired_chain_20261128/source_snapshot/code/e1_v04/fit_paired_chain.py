"""Frozen action-only differential ridge; never run or retrain a day policy.

Train/tune/dev banks come from run_paired_chain_campaign.py. Conditional
suffixes label the four fixed first actions; every continuation is none.
The original suffix (-1) is a replay audit and never a fitting label.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from unified_vfa_features import FEATURE_NAMES

ACTIONS = (0, 2, 5, 10)
ALPHAS = (.001, .01, .1, 1., 10., 100.)
TOL = 1e-9
VERSION = 'paired-action-ridge-v1'
DOWNSTREAM_NAME = 'downstream_profitable_pickup_count'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def choose_action(scores):
    """Ties at declared tolerance prefer depart, then the shortest interval."""
    values = np.asarray(scores, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('four finite scores required')
    best = float(values.max())
    return next(h for h, value in zip(ACTIONS, values) if best-float(value) <= TOL)


def regression_design(rows, columns, kind='paired'):
    """Equal eligible case, state and action; total weights sum to one.

    Averaging conditional suffixes first is algebraically equivalent to
    repeating these fixed x rows with equal suffix weights. No inverse-
    variance weights and no centering/intercept are used.
    """
    if kind not in ('paired', 'level') or not rows:
        raise ValueError('nonempty paired or level training rows required')
    counts = Counter(row['case'] for row in rows)
    x, y, weights = [], [], []
    indices = range(1, 4) if kind == 'paired' else range(4)
    for row in rows:
        phi = np.asarray(row['features'], dtype=float)[:, columns]
        j = np.asarray(row['J'], dtype=float)
        g = np.asarray(row['G'], dtype=float).mean(axis=0)
        if j.shape != (4,) or g.shape != (4,):
            raise ValueError('training outcome/known-score action mismatch')
        for a in indices:
            x.append(phi[a]-phi[0] if kind == 'paired' else phi[a])
            y.append((g[a]-g[0])-(j[a]-j[0]) if kind == 'paired' else g[a]-j[a])
            weights.append(1./(len(counts)*counts[row['case']]*len(indices)))
    return np.asarray(x), np.asarray(y), np.asarray(weights)


def fit_ridge(rows, columns, alpha, kind='paired'):
    x, y, weights = regression_design(rows, columns, kind)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError('finite training data and positive alpha required')
    rms = np.sqrt(np.sum(weights[:, None]*x*x, axis=0))
    scales = np.where(rms < 1e-8, 1., rms)
    z = x/scales
    beta = np.linalg.solve(z.T @ (weights[:, None]*z)+alpha*np.eye(len(columns)),
                           z.T @ (weights*y))
    return dict(version=VERSION, kind=kind, columns=list(columns), alpha=float(alpha),
        weights=beta.tolist(), scales=scales.tolist(), raw_coefficients=(beta/scales).tolist(),
        intercept=0., feature_names=[(list(FEATURE_NAMES)+[DOWNSTREAM_NAME])[c] for c in columns],
        fit=dict(target='mean_suffix[(G_h-G_0)-(J_h-J_0)]' if kind == 'paired' else 'mean_suffix[G_h-J_h]',
            weighting='equal eligible case/state/action; sum weights=1', weight_sum=float(weights.sum()),
            feature_scaling='train-only uncentered weighted RMS; no target scaling',
            rows=len(y), cases=len({r['case'] for r in rows}), states=len(rows),
            zero_rms_columns=[int(columns[i]) for i,v in enumerate(rms) if v < 1e-8],
            weighted_rmse=float(np.sqrt(np.sum(weights*(z@beta-y)**2))),
            matrix_rank=int(np.linalg.matrix_rank(z)),
            singular_values=np.linalg.svd(z*np.sqrt(weights[:, None]), compute_uv=False).tolist()))


def model_scores(row, model):
    phi = np.asarray(row['features'], dtype=float)[:, model['columns']]
    future = (phi/np.asarray(model['scales'])) @ np.asarray(model['weights'])
    # A closed public arrival horizon has zero future opportunity value.
    if row.get('closed', False):
        future[:] = 0.
    return np.asarray(row['J'], dtype=float)+future


def choice_gain(rows, model):
    by_case = defaultdict(list)
    choices = []
    for row in rows:
        h = choose_action(model_scores(row, model))
        g = np.asarray(row['G'], dtype=float)
        by_case[row['case']].append(float(np.mean(g[:, ACTIONS.index(h)]-g[:, 0])))
        choices.append(h)
    if not by_case:
        raise ValueError('no eligible tune states')
    return float(np.mean([np.mean(v) for v in by_case.values()])), choices


def select_model(train, tune, columns, kind='paired', alphas=ALPHAS):
    """There is deliberately no dev argument at the fitting/selection boundary."""
    candidates = []
    for alpha in alphas:
        model = fit_ridge(train, columns, alpha, kind)
        gain, choices = choice_gain(tune, model)
        candidates.append(dict(model=model, tune_equal_case_gain=gain,
                               tune_choice_counts=dict(Counter(choices))))
    best_gain = max(c['tune_equal_case_gain'] for c in candidates)
    tied = [c for c in candidates if best_gain-c['tune_equal_case_gain'] <= TOL]
    chosen = max(tied, key=lambda c:c['model']['alpha'])
    result = dict(chosen['model'])
    result['selection'] = dict(source='tune only', objective='equal-case mean paired gain vs depart',
        tune_equal_case_gain=chosen['tune_equal_case_gain'], alpha_grid=list(alphas),
        tie_break='largest alpha within 1e-9; action ties shortest h',
        tune_choice_counts=chosen['tune_choice_counts'])
    return result, candidates


def select_family(models):
    best = max(m['selection']['tune_equal_case_gain'] for m in models.values())
    tied = [name for name,m in models.items() if best-m['selection']['tune_equal_case_gain'] <= TOL]
    return min(tied, key=lambda name:(len(models[name]['columns']), name))


def sample_stats(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else None
    return dict(n=n, mean=mean, sample_sd=sd, se=sd/np.sqrt(n) if sd is not None else None,
        win=int(np.sum(values > TOL)), tie=int(np.sum(np.abs(values) <= TOL)),
        loss=int(np.sum(values < -TOL)))


def cluster_stats(values):
    """Only independent layout means enter the exploratory interval/test."""
    values = np.asarray(values, dtype=float)
    result = sample_stats(values)
    n = len(values)
    if n < 2:
        result.update(ci95=None, t_df=None, signflip_one_sided_p=None, signflip_two_sided_p=None)
        return result
    half = float(student_t.ppf(.975, n-1)*result['se'])
    observed = result['mean']
    signed_means = np.asarray([np.mean(values*np.asarray(signs)) for signs in product((-1,1), repeat=n)])
    result.update(ci95=[observed-half, observed+half], t_df=n-1,
        signflip_one_sided_p=float(np.mean(signed_means >= observed-TOL)),
        signflip_two_sided_p=float(np.mean(np.abs(signed_means) >= abs(observed)-TOL)),
        signflip_assignments=2**n,
        inference_unit='layout; states and conditional suffixes are not independent layout replicates',
        caveat='Exploratory t interval; exact enumeration is a symmetry-based sign-flip sensitivity, not a randomized trial.')
    return result


def manifest(split_root):
    from run_paired_chain_campaign import RESERVED_TEST, specs
    protocol = read(split_root/'protocol.json')
    split = protocol['split']
    if protocol['cases'] != specs(split):
        raise ValueError('bank cases differ from preregistered campaign')
    sites = read(split_root/'sites.json')
    if not sites or len({s['id'] for s in sites}) != len(sites):
        raise ValueError('empty or duplicated state bank')
    captures = [read(p) for p in sorted((split_root/'capture').glob('*.json'))]
    if any(not c['capture_complete'] for c in captures):
        raise ValueError('incomplete capture')
    if len(captures) != len(protocol['cases']) or {object_hash(c['spec']) for c in captures} != {object_hash(s) for s in protocol['cases']}:
        raise ValueError('capture cases incomplete or extra')
    captured_sites = {s['id']:s for c in captures for s in c['sites']}
    if captured_sites != {s['id']:s for s in sites}:
        raise ValueError('site manifest differs from completed captures')
    for s in sites:
        q = s['qualification']
        if not q['eligible'] or q['stratum'] != 'post_unload_primary' or not q['just_unloaded']:
            raise ValueError('only freshly unloaded primary states allowed')
        if s['spec']['layout_id'] in RESERVED_TEST:
            raise ValueError('reserved final-test layout prohibited')
    return protocol, sites, captures


def known_feature_bank(split_root, destination, augmented):
    """Compute from known_clone only, before opening any offline pair labels."""
    from e1_sim.information import closed
    from e1_sim.initializer import digest
    from e1_sim.online import evaluate_plan, known_clone
    from e1_sim.quantities import plain
    from run_paired_wait_diagnostic import load_pickle
    from unified_vfa_features import features
    from unified_vfa_policy import canonical_plan_digest
    protocol, sites, captures = manifest(split_root)
    source = Path(__file__).resolve().parent
    receipts = {'protocol.json':file_hash(split_root/'protocol.json'),
                'sites.json':file_hash(split_root/'sites.json')}
    source_files = ['fit_paired_chain.py','unified_vfa_features.py','unified_vfa_policy.py','e1_sim/online.py']
    if augmented:
        source_files.append('paired_downstream_opportunity.py')
    receipts.update({'source/'+p:file_hash(source/p) for p in source_files})
    receipts.update({'snapshots/'+s['id']+'.pkl.gz':file_hash(split_root/'snapshots'/(s['id']+'.pkl.gz')) for s in sites})
    if destination.exists():
        cached = read(destination)
        if cached.get('input_receipts') == receipts:
            return cached['states'], protocol, captures
        if (destination.parent.parent/'frozen_selection.json').exists():
            raise ValueError('known feature cache/source changed after selection freeze')
        # Preparation is a known-only derived artifact. Before any fitting
        # freeze it can be regenerated after a source change, never reused
        # silently with stale source receipts.
    rows = []
    for site in sites:
        bundle = load_pickle(split_root/'snapshots'/(site['id']+'.pkl.gz'))
        if bundle['site'] != site or set(bundle['plans']) != set(ACTIONS):
            raise ValueError('snapshot identity/action mismatch')
        known = known_clone(bundle['state'])
        if known._hidden or any(e[3] in ('request_reveal','revelation_end') for e in known._queue):
            raise ValueError('future leaked into known-state feature input')
        if set(o['id'] for o in known.initial['orders']) != set(known.orders):
            raise ValueError('hidden order in initial feature input')
        row = dict(site_id=site['id'], case=site['case'], spec=site['spec'], at=site['at'],
            truck=site['truck'], station=site['station'], closed=closed(known.public),
            known_state_hash=digest(plain(known.initial)), J=[], features=[], plan_hashes=[], downstream_audit=[])
        for h in ACTIONS:
            candidate = evaluate_plan(known, bundle['plans'][h], 'WK')
            if not candidate['feasible'] or not candidate['allowed']:
                raise ValueError('frozen first-action plan no longer feasible')
            phi = list(features(known, candidate, 5))
            if augmented:
                from paired_downstream_opportunity import downstream_opportunity
                value, audit = downstream_opportunity(known, bundle['plans'][h], site['truck'])
                phi.append(float(value))
                row['downstream_audit'].append(audit)
            row['J'].append(float(candidate['score']))
            row['features'].append(phi)
            row['plan_hashes'].append(canonical_plan_digest(bundle['plans'][h]))
        rows.append(row)
    write(destination, dict(input_receipts=receipts, feature_names=list(FEATURE_NAMES)+([DOWNSTREAM_NAME] if augmented else []),
        states=rows, offline_future_read_for_features=False))
    return rows, protocol, captures


def check_base_features(output, split, rows):
    base = read(Path(output)/'base/known_features'/(split+'.json'))['states']
    by_id = {r['site_id']:r for r in base}
    if set(by_id) != {r['site_id'] for r in rows}:
        raise ValueError('base/augmented state set mismatch')
    for row in rows:
        old = by_id[row['site_id']]
        if (old['J'] != row['J'] or old['known_state_hash'] != row['known_state_hash']
                or old['plan_hashes'] != row['plan_hashes']
                or old['features'] != [x[:7] for x in row['features']]):
            raise ValueError('augmented changed a base feature, known score or fixed plan')


def prepare(output, stage):
    """Known-only precomputation; intentionally never calls load_labels."""
    output = Path(output).resolve()
    augmented = stage == 'augmented'
    records = {}
    for split in ('train','tune','dev'):
        rows,protocol,captures = known_feature_bank(output/split,output/stage/'known_features'/(split+'.json'),augmented)
        if augmented:
            check_base_features(output,split,rows)
        records[split] = dict(states=len(rows),cases=len(captures),
            known_feature_sha256=file_hash(output/stage/'known_features'/(split+'.json')),
            layouts=sorted({r['spec']['layout_id'] for r in rows}))
        print(json.dumps(dict(stage='prepare-'+stage,split=split,states=len(rows),future_labels_read=False)),flush=True)
    value = dict(stage=stage,splits=records,future_labels_read=False,independent_test_opened=False)
    write(output/stage/'prepared.json',value)
    return value


def load_labels(split_root, feature_rows, protocol, captures):
    """Require every audit + conditional four-branch pair and known-state hash."""
    samples = protocol['samples']
    if samples < 2:
        raise ValueError('at least two conditional suffixes required')
    expected = {r['site_id']+f'_r{k:02}.json' for r in feature_rows for k in [-1]+list(range(samples))}
    found = {p.name for p in (split_root/'paired').glob('*.json')}
    if found != expected:
        raise ValueError('incomplete/extra paired bank: missing=%s extra=%s' % (len(expected-found),len(found-expected)))
    references = {c['case']:c['reference'] for c in captures}
    labels, receipts = [], {}
    for row in feature_rows:
        values = {name:[] for name in ('G','accepted','trips','truck_time','departure_delay')}
        original = None
        for replica in [-1]+list(range(samples)):
            name = row['site_id']+f'_r{replica:02}.json'
            path = split_root/'paired'/name
            pair = read(path)
            receipts[name] = file_hash(path)
            if pair['site']['id'] != row['site_id'] or pair['replica'] != replica or set(pair['branches']) != set(map(str,ACTIONS)):
                raise ValueError('pair identity/actions mismatch')
            if pair['online_known_hash'] != row['known_state_hash']:
                raise ValueError('online known state changes across future suffixes')
            branches = [pair['branches'][str(h)] for h in ACTIONS]
            for b in branches:
                if not b['terminal'] or not b['ledger_audited'] or b['suffix_hash'] != pair['suffix_hash']:
                    raise ValueError('paired branch terminal/ledger/common-future audit failed')
            if replica == -1:
                reference = references[row['case']]
                if branches[0]['net'] != reference['net'] or branches[0]['trace'] != reference['trace']:
                    raise ValueError('original suffix depart branch failed source-trajectory reproduction')
                original = [float(b['net']) for b in branches]
                continue
            values['G'].append([float(b['net']) for b in branches])
            values['accepted'].append([int(b['accepted']) for b in branches])
            values['trips'].append([int(b['trips']) for b in branches])
            values['truck_time'].append([float(b['ledger']['truck_time']) for b in branches])
            values['departure_delay'].append([float(b['actual_first_departure_delay']) for b in branches])
        labels.append(dict(row, **values, original_G=original))
    return labels, dict(paired_files=len(receipts), conditional_samples_per_state=samples,
        complete_original_reproductions=len(labels), known_hash_matches=len(receipts),
        conditional_only_labels=True, file_receipts=receipts)


def old_chain_model(path):
    from unified_vfa_policy import validate_model
    old = read(path)
    validate_model(old, 5, (2,5,10), 5)
    if old['active_features'] != list(FEATURE_NAMES[:2]):
        raise ValueError('legacy chain must have exactly the first two active features')
    return dict(kind='legacy', columns=list(range(7)), weights=old['weights'], scales=old['scales'],
                source=dict(path=str(path), sha256=file_hash(path)))


def report_split(rows, models, captures):
    state_reports = []
    model_state_means = {name:defaultdict(list) for name in list(models)+['depart','fixed10']}
    for row in rows:
        g = np.asarray(row['G'])
        scores = {name:model_scores(row,model) for name,model in models.items()}
        choices = {name:choose_action(s) for name,s in scores.items()}
        choices.update(depart=0, fixed10=10)
        detail = dict(site_id=row['site_id'], case=row['case'], spec=row['spec'], at=row['at'],
            station=row['station'], J=row['J'], features=row['features'], actions={}, models={})
        for index,h in enumerate(ACTIONS):
            detail['actions'][str(h)] = dict(G_mean=float(g[:,index].mean()),
                gain_vs_depart=sample_stats(g[:,index]-g[:,0]), original_gain_vs_depart=row['original_G'][index]-row['original_G'][0])
        for name in models:
            index = ACTIONS.index(choices[name])
            comparisons = {}
            for ref in ('depart','fixed10','old_chain'):
                ri = ACTIONS.index(choices[ref])
                comparisons[ref] = sample_stats(g[:,index]-g[:,ri])
            item = dict(chosen_h=choices[name], scores={str(h):dict(J=row['J'][i],
                W=float(scores[name][i]-row['J'][i]), total=float(scores[name][i]),
                delta_vs_depart=float(scores[name][i]-scores[name][0])) for i,h in enumerate(ACTIONS)},
                comparisons=comparisons,
                accepted_delta_vs_depart=float(np.mean(np.asarray(row['accepted'])[:,index]-np.asarray(row['accepted'])[:,0])),
                trips_saved_vs_depart=float(np.mean(np.asarray(row['trips'])[:,0]-np.asarray(row['trips'])[:,index])),
                extra_truck_time_cost_vs_depart=float(np.mean(np.asarray(row['truck_time'])[:,index]-np.asarray(row['truck_time'])[:,0])),
                actual_departure_delay=float(np.mean(np.asarray(row['departure_delay'])[:,index])))
            detail['models'][name] = item
            model_state_means[name][row['case']].append(item)
        for name in ('depart','fixed10'):
            index = ACTIONS.index(choices[name])
            model_state_means[name][row['case']].append(dict(chosen_h=choices[name],
                comparisons={ref:sample_stats(g[:,index]-g[:,ACTIONS.index(choices[ref])])
                             for ref in ('depart','fixed10','old_chain')}))
        state_reports.append(detail)
    case_specs = {c['case']:c['spec'] for c in captures}
    empty = sorted(set(case_specs)-{r['case'] for r in rows})
    summaries = {}
    for name, case_items in model_state_means.items():
        cases = {}
        layouts = defaultdict(list)
        for case, items in case_items.items():
            means = {ref:float(np.mean([i['comparisons'][ref]['mean'] for i in items])) for ref in ('depart','fixed10','old_chain')}
            cases[case] = dict(spec=case_specs[case], states=len(items), gains=means,
                choice_counts=dict(Counter(i['chosen_h'] for i in items)))
            layouts[case_specs[case]['layout_id']].append(means)
        layout_rows = {str(l):{ref:float(np.mean([m[ref] for m in items])) for ref in ('depart','fixed10','old_chain')}
                       for l,items in sorted(layouts.items())}
        summaries[name] = dict(choice_counts=dict(Counter(i['chosen_h'] for items in case_items.values() for i in items)),
            equal_case_gains={ref:float(np.mean([c['gains'][ref] for c in cases.values()])) for ref in ('depart','fixed10','old_chain')},
            cases=cases, layout_means=layout_rows,
            layout_inference={ref:cluster_stats([m[ref] for m in layout_rows.values()]) for ref in ('depart','fixed10','old_chain')})
    return dict(states=state_reports, models=summaries, case_coverage=dict(total=len(captures),
        eligible=len(captures)-len(empty), no_qualifying_states=empty),
        averaging='suffix mean, then equal state within case, equal case within layout, equal layout',
        warning='Action-only conditional returns under none continuation; not full-day learned-policy returns.')


def freeze(path, value):
    # JSON object keys are strings; normalize Counter/action keys so a second
    # frozen report pass compares to the representation actually persisted.
    canonical = json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    if path.exists() and read(path) != canonical:
        raise ValueError('frozen selection/input changed; use a new campaign output')
    write(path, canonical)


def complete_layout_coverage(report, protocol):
    """Eligibility is not an outcome: empty cases are excluded, never zeroed.

    All planned captures and all eligible-state paired branches were checked
    by the loaders. Every one of the eight planned dev layouts must have a
    qualifying state; each layout averages its eligible spatial cases only.
    """
    expected = {case['layout_id'] for case in protocol['cases']}
    observed = {row['spec']['layout_id'] for row in report['states']}
    return len(expected) == 8 and observed == expected


def run(output, stage, include_level=True, prior=None, defer_dev=False):
    output = Path(output).resolve()
    destination = output/stage
    prior = Path(prior) if prior else Path(__file__).resolve().parents[2]/'results/feature_screen_20261128/training/chain/selected_model.json'
    augmented = stage == 'augmented'
    banks, audits, captures, feature_rows = {}, {}, {}, {}
    # Train and tune labels are the only labels opened before freezing.
    for split in ('train','tune'):
        feature_rows[split], protocol, captures[split] = known_feature_bank(output/split, destination/'known_features'/(split+'.json'), augmented)
        if augmented:
            check_base_features(output,split,feature_rows[split])
        banks[split], audits[split] = load_labels(output/split, feature_rows[split], protocol, captures[split])
        write(destination/'input_audits'/(split+'.json'), audits[split])
    columns = [0,1,7] if augmented else [0,1]
    family = 'paired_augmented' if augmented else 'paired_chain'
    selected, candidates = select_model(banks['train'],banks['tune'],columns)
    models = {'old_chain':old_chain_model(prior), family:selected}
    write(destination/'models'/(family+'_candidates.json'),candidates)
    if augmented:
        base = read(output/'base/frozen_selection.json')
        for split in ('train','tune'):
            if base['label_receipt_hashes'][split] != object_hash(audits[split]['file_receipts']):
                raise ValueError('base and augmented must use identical paired training/tune outcomes')
        models['paired_chain'] = read(output/'base/models/paired_chain.json')
        if include_level and (output/'base/models/level_chain.json').exists():
            models['level_chain'] = read(output/'base/models/level_chain.json')
        primary = select_family({name:models[name] for name in ('paired_chain','paired_augmented')})
    else:
        primary = None  # family comparison has not been frozen yet
        if include_level:
            level, controls = select_model(banks['train'],banks['tune'],[0,1],kind='level')
            models['level_chain'] = level
            write(destination/'models/level_chain_candidates.json', controls)
    frozen = dict(version=VERSION, stage=stage, primary_family=primary,
        primary_selection='tune equal-case gain only; ties prefer chain2',
        candidate_family=family, selected_models={name:model for name,model in models.items() if name != 'old_chain'},
        label_receipt_hashes={s:object_hash(audits[s]['file_receipts']) for s in ('train','tune')},
        legacy_source=models['old_chain']['source'], dev_outcomes_used_for_selection=False,
        gate='primary family only; exploratory dev layout t 95% lower >0 vs depart AND fixed10; 8 complete layouts')
    freeze(destination/'frozen_selection.json', frozen)
    for name,model in models.items():
        freeze(destination/'models'/(name+'.json'),model)
    reports = {s:report_split(banks[s],models,captures[s]) for s in ('train','tune')}
    if defer_dev:
        result = dict(version=VERSION,stage=stage,primary_family=primary,dev_evaluated=False,
            independent_test_opened=False,feature_names=list(FEATURE_NAMES)+([DOWNSTREAM_NAME] if augmented else []),
            frozen_selection_path=str(destination/'frozen_selection.json'),reports=reports,
            candidate_gates={},proceed_to_full_day_model_comparison=False,
            inference_notes=['Dev labels remain unopened; freeze the tune-selected family before evaluating either family on dev.'])
        write(destination/'results.json',result)
        return result
    # This is the first opening of dev outcomes; all fitting and selection above.
    feature_rows['dev'], protocol, captures['dev'] = known_feature_bank(output/'dev', destination/'known_features/dev.json', augmented)
    if augmented:
        check_base_features(output,'dev',feature_rows['dev'])
    banks['dev'], audits['dev'] = load_labels(output/'dev',feature_rows['dev'],protocol,captures['dev'])
    write(destination/'input_audits/dev.json', audits['dev'])
    reports['dev'] = report_split(banks['dev'],models,captures['dev'])
    candidate_gates = {}
    complete = complete_layout_coverage(reports['dev'],protocol)
    for name in ('paired_chain','paired_augmented'):
        if name not in models:
            continue
        inference = reports['dev']['models'][name]['layout_inference']
        complete8 = complete and inference['depart']['n'] == 8
        passed = complete8 and all(inference[ref]['ci95'][0] > 0 for ref in ('depart','fixed10'))
        candidate_gates[name] = dict(pass_both_lower95=bool(passed), complete_eight_layouts=complete8,
            selected_primary=name == primary, authoritative_for_full_day=name == primary)
    result = dict(version=VERSION, stage=stage, primary_family=primary,dev_evaluated=True,
        independent_test_opened=False, feature_names=list(FEATURE_NAMES)+([DOWNSTREAM_NAME] if augmented else []),
        frozen_selection_path=str(destination/'frozen_selection.json'), reports=reports,
        candidate_gates=candidate_gates,
        proceed_to_full_day_model_comparison=bool(primary and candidate_gates[primary]['pass_both_lower95']),
        inference_notes=['Dev is an exploratory action gate, not final held-out test.',
            'Zero-eligible-state cases are excluded without zero imputation; layout means average eligible cases. Coverage is reported explicitly.',
            'Suffixes share a captured prefix; states share a layout. Only layout means enter t/sign-flip inference.',
            'Both comparators must pass for the tune-selected primary; controls cannot rescue its gate.',
            'The same-data level control addresses training-data changes; it is an attribution audit, not a randomized causal experiment.',
            'Original suffix -1 is excluded from every training, selection and conditional performance mean.'])
    write(destination/'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['base','augmented','prepare-base','prepare-augmented'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-level-control', action='store_true')
    parser.add_argument('--defer-dev', action='store_true',help='fit/freeze and report train/tune without opening dev outcomes')
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args()
    if args.stage.startswith('prepare-'):
        prepare(args.output,args.stage[len('prepare-'):])
        return
    result = run(args.output,args.stage,not args.skip_level_control,args.prior,args.defer_dev)
    print(json.dumps(dict(stage=args.stage,primary_family=result['primary_family'],
        gates=result['candidate_gates'],proceed_to_full_day=result['proceed_to_full_day_model_comparison'],
        results=str(args.output/args.stage/'results.json')),ensure_ascii=False))


if __name__ == '__main__':
    main()
