"""Train candidate feature sets and review intervals on independent layout splits.

Validation selects a single frozen model before any optional held-out test.
Every trajectory is completed and independently ledger-audited by run().
"""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np

from e1_sim.initializer import digest
from recovery_opportunity import run
from uniform_wait_worlds import CFG, PICKUP_DEADLINE_LAGS, make_world
from unified_vfa_features import FEATURE_GROUPS, FEATURE_NAMES, active_feature_names
from unified_vfa_policy import (MODEL_VERSION, UnifiedVFAPolicy, operating_rules,
                                parse_pattern_limit, validate_saved_version)


def fit_model(days, h, ridge=1.0, feature_set='full', *, review_intervals=None,
              arrival_mode='fixed_count', feature_horizon=None, rules=None):
    """Equal total weight per day; uncentered, scaled ridge with no intercept."""
    usable = [day for day in days if day]
    if not usable:
        raise ValueError('No nonterminal training samples; increase training demand')
    x = np.asarray([sample['x'] for day in usable for sample in day], dtype=float)
    y = np.asarray([sample['y'] for day in usable for sample in day], dtype=float)
    w = np.asarray([1.0 / len(day) for day in usable for sample in day])
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid training feature or target')
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be finite and positive')
    active = active_feature_names(feature_set)
    columns = [FEATURE_NAMES.index(name) for name in active]
    scales = np.ones(len(FEATURE_NAMES))
    weights = np.zeros(len(FEATURE_NAMES))
    if columns:
        observed = x[:, columns]
        active_scales = np.sqrt(np.average(observed * observed, axis=0, weights=w))
        active_scales[active_scales < 1e-8] = 1.0
        scales[columns] = active_scales
        z = observed / active_scales
        weights[columns] = np.linalg.solve(z.T @ (w[:, None] * z) + ridge * np.eye(len(columns)),
                                           z.T @ (w * y))
    if feature_horizon is None:
        feature_horizon = h if review_intervals is None else 5
    return dict(version=MODEL_VERSION, h=h, feature_set=feature_set,
                operating_rules=operating_rules(CFG, PICKUP_DEADLINE_LAGS) if rules is None else rules,
                wait_mode='adaptive' if review_intervals is not None else 'fixed',
                review_intervals=list(review_intervals) if review_intervals is not None else None,
                arrival_mode=arrival_mode, feature_horizon=feature_horizon,
                active_features=list(active),
                feature_names=list(FEATURE_NAMES), weights=weights.tolist(), scales=scales.tolist(),
                trained=bool(columns), fit=dict(method='equal_day_scaled_ridge_no_intercept', ridge=ridge,
                days=len(usable), samples=len(y), target='audited_full_day_net_minus_J_known'))


def audited_episode(world, h, model=None, *, review_intervals=None, feature_horizon=None,
                    explore=0.0, seed=0, **budgets):
    arrival_mode = world['spec'].get('arrival_mode', 'fixed_count')
    if model is not None and model.get('arrival_mode', 'fixed_count') != arrival_mode:
        raise ValueError('Model and world use different order arrival laws')
    policy = UnifiedVFAPolicy(h=h, model=model, explore=explore, seed=seed,
                              review_intervals=review_intervals, feature_horizon=feature_horizon,
                              **budgets)
    row = run(world, 0, policy_override=policy)
    # Hidden realized demand is available to the simulator/labeler only.
    samples = [dict(x=list(sample['x']), y=row['net'] - sample['j'])
               for sample in policy.samples if not sample['closed']]
    realized_pickups = sum(order['kind'] == 'P' for order in world['input']['orders'])
    realized_orders = len(world['input']['orders'])
    row.update(h=h, wait_mode='adaptive' if review_intervals is not None else 'fixed',
               review_intervals=review_intervals, realized_pickups=realized_pickups,
               realized_orders=realized_orders,
               realized_pickup_share=realized_pickups / realized_orders if realized_orders else None,
               world_spec=world['spec'], input_hash=world['input_hash'], i0_hash=world['i0_hash'],
               sample_count=len(samples), trained_model=model is not None and model.get('trained', True),
               canonical_decisions=digest([(r['at'], r['canonical_plan_hash'])
                                           for r in policy.selection_log]),
               decision_summary=[dict(at=r['at'], plan_hash=r['canonical_plan_hash'],
                                      admissions=r['admissions'], waits=r['chosen_wait_intervals'],
                                      J=r['J'], W=r['W'], features=list(r['features']))
                                 for r in policy.selection_log],
               action_search=policy.search_metrics())
    return row, samples


def validation_choice(rows):
    if not rows:
        raise ValueError('No validation candidates')
    # Prefer fewer active columns, then smaller h if validation returns tie.
    return min(rows, key=lambda row: (-row['validation_mean_net'],
               len(active_feature_names(row.get('feature_set', 'full'))),
               row.get('wait_mode', 'fixed') != 'fixed', row['h'],
               row.get('feature_set', 'full')))


def validation_residual_fit(days, model):
    """Equal mass per day. Reference is the policy's simulated return, not optimal DP."""
    usable = [day for day in days if day]
    report = dict(reference='on_policy_audited_G_minus_J_known', weighting='equal_nonempty_day',
                  days=len(usable), samples=sum(map(len, usable)), rmse=None, r2=None)
    if not usable:
        return report
    coefficients = np.asarray(model['weights']) / np.asarray(model['scales'])
    targets = np.asarray([s['y'] for day in usable for s in day])
    predictions = np.asarray([s['x'] for day in usable for s in day]) @ coefficients
    weights = [1 / len(day) for day in usable for _ in day]
    errors = np.average((targets - predictions) ** 2, weights=weights)
    report['rmse'] = float(np.sqrt(errors))
    if targets.min() != targets.max():
        center = np.average(targets, weights=weights)
        variance = np.average((targets - center) ** 2, weights=weights)
        report['r2'] = float(1 - errors / variance)
    return report


def validate_layout_split(train, validation, test=()):
    all_ids = [*train, *validation, *test]
    if not train or not validation or len(set(all_ids)) != len(all_ids):
        raise ValueError('Training and validation require nonempty, mutually disjoint layout IDs '
                         'and any test layouts must be new')


def resolve_modes(review_mode, arrival_mode, *, default_mode, fixed_interval_supplied=False,
                  adaptive_menu_supplied=False, model=None):
    """Recover a saved experiment's semantics or select the new experiment defaults."""
    if model is not None:
        validate_saved_version(model)
        trained_mode = model.get('wait_mode', 'fixed')
        trained_arrivals = model.get('arrival_mode', 'fixed_count')
        if review_mode not in (None, trained_mode) or arrival_mode not in (None, trained_arrivals):
            raise ValueError('The saved model requires its training review mode and arrival law')
        return trained_mode, trained_arrivals
    if review_mode is None:
        review_mode = ('fixed' if fixed_interval_supplied and not adaptive_menu_supplied
                       else default_mode)
    return review_mode, arrival_mode or ('fixed_count' if review_mode == 'fixed' else 'p2')


def review_settings(mode, h_values, menu):
    if mode not in ('fixed', 'adaptive', 'compare'):
        raise ValueError('Unknown review mode')
    if not menu or any(type(h) is not int or h <= 0 for h in menu) or len(set(menu)) != len(menu):
        raise ValueError('Adaptive intervals must be distinct positive integers')
    menu = sorted(menu)
    if h_values is None:
        h_values = menu if mode == 'compare' else [5, 10]
    if not h_values or any(type(h) is not int or h <= 0 for h in h_values) or len(set(h_values)) != len(h_values):
        raise ValueError('Fixed intervals must be distinct positive integers')
    settings = []
    if mode != 'adaptive':
        for h in h_values:
            settings.append(dict(h=h, review_intervals=None, feature_horizon=5 if mode == 'compare' else h))
    if mode != 'fixed':
        settings.append(dict(h=5, review_intervals=menu, feature_horizon=5))
    return settings


def train(args):
    validate_layout_split(args.train_layouts, args.validation_layouts, args.test_layouts)
    validation_orders = args.validation_orders or args.orders
    test_orders = args.test_orders or validation_orders
    if args.test_orders and not args.test_layouts:
        raise ValueError('--test-orders requires --test-layouts')
    if args.days < 1 or args.iterations < 1 or not 0 < args.share < 1:
        raise ValueError('Positive days/iterations and mixed delivery/pickup demand required')
    for sizes in (args.orders, validation_orders, test_orders):
        if len(set(sizes)) != len(sizes) or any(not 0 < round(n * args.share) < n for n in sizes):
            raise ValueError('Each split requires distinct sizes and both deliveries and pickups after rounding')
    selected_columns = [active_feature_names(spec) for spec in args.feature_sets]
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError('Each candidate feature set must select a different set of columns')
    args.review_mode, args.arrival_mode = resolve_modes(
        args.review_mode, args.arrival_mode, default_mode='compare',
        fixed_interval_supplied=args.h_values is not None,
        adaptive_menu_supplied=args.adaptive_intervals is not None)
    if args.adaptive_intervals is None:
        args.adaptive_intervals = [2, 5, 10]
    settings = review_settings(args.review_mode, args.h_values, args.adaptive_intervals)
    fixed_grid = [c['h'] for c in settings if c['review_intervals'] is None]
    if not 0 <= args.explore <= 1:
        raise ValueError('explore must be in [0,1]')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; do not overwrite a prior campaign')
    out.mkdir(parents=True, exist_ok=True)
    budgets = dict(candidate_budget=args.candidate_budget,
                   enhancement_budget=args.enhancement_budget,
                   wait_budget_per_base=args.wait_budget, action_solver=args.action_solver,
                   mip_pattern_limit=args.mip_pattern_limit)

    def worlds(layouts, sizes):
        return [make_world(layout, day, total, args.share, pattern,
                           temporal=temporal, history_days=args.history_days, arrival_mode=args.arrival_mode,
                           improvement_budget=args.initial_budget)
                for layout in layouts for day in range(args.days) for total in sizes
                for pattern in args.patterns for temporal in args.temporal]
    banks = {'train': worlds(args.train_layouts, args.orders),
             'validation': worlds(args.validation_layouts, validation_orders)}
    source_dir = Path(__file__).resolve().parent
    source_files = [source_dir / name for name in ('run_unified_vfa.py', 'unified_vfa_policy.py',
                    'unified_vfa_features.py', 'uniform_wait_worlds.py', 'run_global_demand_pilot.py',
                    'recovery_opportunity.py', 'run_multiscale.py')]
    if args.action_solver == 'gurobi':
        source_files.extend(source_dir / name for name in ('joint_action_mip.py', 'service_action_mip.py'))
    source_files += sorted((source_dir / 'e1_sim').glob('*.py'))
    source_hashes = {str(p.relative_to(source_dir)).replace('\\', '/'):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    rules = operating_rules(CFG, PICKUP_DEADLINE_LAGS)
    protocol = dict(version=MODEL_VERSION, operating_rules=rules, h_values=fixed_grid, review_settings=settings,
                    arrival_mode=args.arrival_mode,
                    feature_sets=dict(zip(args.feature_sets, selected_columns)),
                    feature_names=list(FEATURE_NAMES), iterations=args.iterations, explore=args.explore,
                    search=budgets, seed=args.seed, ridge=args.ridge, formal_test_opened=False,
                    source_hashes=source_hashes,
                    training_label='G - J_known, both using the full-day ledger including sunk costs',
                    configuration={k: v for k, v in vars(args).items() if k not in ('output', 'command')},
                    worlds={phase: [dict(spec=w['spec'], input_hash=w['input_hash'],
                                        i0_hash=w['i0_hash'], public_hash=w['public_hash'])
                                    for w in ws] for phase, ws in banks.items()})
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf8')
    candidates = []
    for setting, feature_set in product(settings, args.feature_sets):
        h = setting['h']
        mode = 'adaptive' if setting['review_intervals'] is not None else 'fixed'
        model, days, curve = None, [], []
        for iteration in range(args.iterations):
            for i, world in enumerate(banks['train']):
                row, samples = audited_episode(world, model=model, **setting, explore=args.explore,
                                                seed=args.seed + iteration * 100000 + i, **budgets)
                days.append(samples)
                # Persist completed episodes immediately, including diagnostics;
                # a later interrupted run must not erase earlier audit evidence.
                with (out / 'training_episodes.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(dict(feature_set=feature_set, iteration=iteration + 1,
                                                 day=i + 1, episode=row), ensure_ascii=False) + '\n')
                    stream.flush()
                print(json.dumps(dict(phase='train', h=h, wait_mode=mode, feature_set=feature_set, iteration=iteration + 1,
                                      day=i + 1, net=row['net'])), flush=True)
            model = fit_model(days, ridge=args.ridge, feature_set=feature_set, rules=rules,
                              arrival_mode=args.arrival_mode, **setting)
            validated = [audited_episode(w, model=model, **setting, **budgets) for w in banks['validation']]
            validation, validation_days = zip(*validated)
            curve.append(dict(iteration=iteration + 1,
                              validation_mean_net=mean(r['net'] for r in validation),
                              residual_fit=validation_residual_fit(validation_days, model),
                              validation=validation, model=model))
            checkpoint_name = (f'checkpoint_{mode}_h{h}_'
                               f'{feature_set.replace(",", "__")}_iteration{iteration + 1}.json')
            (out / checkpoint_name).write_text(json.dumps(
                dict(feature_set=feature_set, **setting, **curve[-1]),
                ensure_ascii=False, indent=2), encoding='utf8')
        # Checkpoint and h selection both use validation, never a test outcome.
        best = max(curve, key=lambda r: (r['validation_mean_net'], -r['iteration']))
        item = dict(**setting, wait_mode=mode, feature_set=feature_set, iteration=best['iteration'],
                    validation_mean_net=best['validation_mean_net'],
                    model=best['model'], curve=curve)
        candidates.append(item)
        print(json.dumps(dict(phase='validation', h=h, wait_mode=mode, feature_set=feature_set,
                              mean_net=item['validation_mean_net'])), flush=True)
    chosen = validation_choice(candidates)
    winners = {mode: validation_choice([c for c in candidates if c['wait_mode'] == mode])
               for mode in dict.fromkeys(c['wait_mode'] for c in candidates)}
    protocol_hash = digest(protocol)
    frozen = {}
    for mode, winner in winners.items():
        frozen[mode] = dict(winner['model'], selection=dict(source='validation_only',
            validation_mean_net=winner['validation_mean_net'], candidate_h=fixed_grid,
            adaptive_intervals=args.adaptive_intervals, candidate_feature_sets=args.feature_sets,
            iteration=winner['iteration'], search=budgets,
            world_options=dict(history_days=args.history_days, improvement_budget=args.initial_budget),
            train_layouts=args.train_layouts, validation_layouts=args.validation_layouts,
            protocol_hash=protocol_hash, formal_test_opened=False))
    selected_mode = chosen['wait_mode']
    outputs = {'selected_model.json': frozen[selected_mode], 'results.json': candidates}
    if len(winners) > 1:
        outputs.update({f'selected_{mode}_model.json': model for mode, model in frozen.items()})
    for filename, data in outputs.items():
        (out / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(selected_h=chosen['h'], selected_features=chosen['feature_set'],
                         selected_mode=selected_mode, model=str(out / 'selected_model.json'),
                         formal_test_opened=False)), flush=True)
    if args.test_layouts:
        paired = []
        for world in worlds(args.test_layouts, test_orders):
            evaluated = {}
            for mode, model in frozen.items():
                options = {k: model[k] for k in ('h', 'review_intervals', 'feature_horizon')}
                evaluated[mode], _ = audited_episode(world, model=model, **options, **budgets)
            selected = evaluated[selected_mode]
            selected_options = {k: frozen[selected_mode][k] for k in ('h', 'review_intervals', 'feature_horizon')}
            baseline, _ = audited_episode(world, **selected_options, **budgets)
            pair = dict(spec=world['spec'], input_hash=world['input_hash'], public_hash=world['public_hash'],
                        i0_hash=world['i0_hash'], policy=selected, myopic=baseline,
                        net_gain=selected['net'] - baseline['net'])
            if args.review_mode == 'compare':
                pair.update(fixed=evaluated['fixed'], adaptive=evaluated['adaptive'],
                            adaptive_minus_fixed=evaluated['adaptive']['net'] - evaluated['fixed']['net'])
            paired.append(pair)
        result = dict(model_hash=digest(frozen[selected_mode]), protocol_hash=protocol_hash, pairs=paired,
                      model_hashes={mode: digest(model) for mode, model in frozen.items()},
                      mean_net_gain=mean(row['net_gain'] for row in paired),
                      mean_gain_by_size={n: mean(row['net_gain'] for row in paired
                                                if row['spec']['total'] == n) for n in test_orders})
        if args.review_mode == 'compare':
            result['mean_adaptive_minus_fixed'] = mean(row['adaptive_minus_fixed'] for row in paired)
            result['adaptive_minus_fixed_by_size'] = {
                n: mean(row['adaptive_minus_fixed'] for row in paired if row['spec']['total'] == n)
                for n in test_orders}
        (out / 'heldout_test.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                               encoding='utf8')
        print(json.dumps({k: v for k, v in result.items() if k.startswith('mean_')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train', help='Development training and validation-only feature/interval selection')
    p.add_argument('--review-mode', choices=('fixed', 'adaptive', 'compare'),
                   help='Default compare; explicit fixed interval grid alone preserves fixed mode')
    p.add_argument('--adaptive-intervals', nargs='+', type=int, help='Default online menu: 2 5 10 minutes')
    p.add_argument('--arrival-mode', choices=('fixed_count', 'p2'),
                   help='Defaults to p2 for adaptive/compare and fixed_count for fixed')
    p.add_argument('--h-values', nargs='+', type=int,
                   help='Fixed policy grid, defaults to the adaptive menu in compare mode or 5 10 otherwise')
    p.add_argument('--train-layouts', nargs='+', type=int, default=[710, 711])
    p.add_argument('--validation-layouts', nargs='+', type=int, default=[810, 811])
    p.add_argument('--test-layouts', nargs='+', type=int, default=[],
                   help='Optional untouched layouts to evaluate after model selection')
    p.add_argument('--days', type=int, default=1)
    p.add_argument('--orders', nargs='+', type=int, default=[20], help='Training order counts')
    p.add_argument('--validation-orders', nargs='+', type=int, help='Defaults to training order counts')
    p.add_argument('--test-orders', nargs='+', type=int, help='Defaults to validation order counts')
    p.add_argument('--feature-sets', nargs='+', default=['full'],
                   help='Use none, full, or comma-separated groups per set: ' + ', '.join(FEATURE_GROUPS))
    p.add_argument('--share', type=float, default=.4)
    p.add_argument('--patterns', nargs='+', choices=['uniform', 'dual'], default=['uniform', 'dual'])
    p.add_argument('--temporal', nargs='+', choices=['peak', 'flat'], default=['peak'])
    p.add_argument('--iterations', type=int, default=2)
    p.add_argument('--explore', type=float, default=.15)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--ridge', type=float, default=1.)
    p.add_argument('--history-days', type=int, default=160)
    p.add_argument('--initial-budget', type=int, default=32)
    p.add_argument('--candidate-budget', type=int, default=16)
    p.add_argument('--enhancement-budget', type=int, default=16)
    p.add_argument('--wait-budget', type=int, default=8)
    p.add_argument('--action-solver', choices=['bounded', 'gurobi'], default='bounded')
    p.add_argument('--mip-pattern-limit', type=parse_pattern_limit, default=64,
                   help="Gurobi patterns per search; use 'all' for an uncapped restricted search")
    p.add_argument('--output', default='../../tmp/unified_vfa_development')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
