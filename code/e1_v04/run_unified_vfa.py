"""Train fixed-h unified VFA candidates and select h using validation days only.

This is a development runner, not a prevalidated policy or a final experiment.
Every trajectory is completed and independently ledger-audited by run().
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np

from e1_sim.initializer import digest
from recovery_opportunity import run
from uniform_wait_worlds import make_world
from unified_vfa_features import FEATURE_NAMES
from unified_vfa_policy import UnifiedVFAPolicy


def fit_model(days, h, ridge=1.0):
    """Equal total weight per day; uncentered, scaled ridge with no intercept."""
    usable = [day for day in days if day]
    if not usable:
        raise ValueError('No nonterminal training samples; increase training demand')
    x = np.asarray([sample['x'] for day in usable for sample in day], dtype=float)
    y = np.asarray([sample['y'] for day in usable for sample in day], dtype=float)
    w = np.asarray([1.0 / len(day) for day in usable for sample in day])
    if x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid training feature or target')
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be finite and positive')
    scales = np.sqrt(np.average(x * x, axis=0, weights=w))
    scales[scales < 1e-8] = 1.0
    z = x / scales
    weights = np.linalg.solve(z.T @ (w[:, None] * z) + ridge * np.eye(z.shape[1]),
                              z.T @ (w * y))
    return dict(version='unified-vfa-development-v1', h=h,
                feature_names=list(FEATURE_NAMES), weights=weights.tolist(), scales=scales.tolist(),
                trained=True, fit=dict(method='equal_day_scaled_ridge_no_intercept', ridge=ridge,
                days=len(usable), samples=len(y), target='audited_full_day_net_minus_J_known'))


def audited_episode(world, h, model=None, *, explore=0.0, seed=0, **budgets):
    policy = UnifiedVFAPolicy(h=h, model=model, explore=explore, seed=seed, **budgets)
    row = run(world, 0, policy_override=policy)
    # Hidden realized demand is available to the simulator/labeler only.
    samples = [dict(x=list(sample['x']), y=row['net'] - sample['j'])
               for sample in policy.samples if not sample['closed']]
    row.update(h=h, input_hash=world['input_hash'], i0_hash=world['i0_hash'],
               sample_count=len(samples), trained_model=model is not None)
    return row, samples


def validation_choice(rows):
    if not rows:
        raise ValueError('No validation candidates')
    # No test day is accepted by this selector. Stable tie break prefers smaller h.
    return max(rows, key=lambda row: (row['validation_mean_net'], -row['h']))


def validate_layout_split(train, validation):
    if not train or not validation or set(train) & set(validation):
        raise ValueError('Training and validation require nonempty disjoint layout IDs')


def train(args):
    validate_layout_split(args.train_layouts, args.validation_layouts)
    if args.days < 1 or args.iterations < 1 or args.orders < 2 or not 0 < args.share < 1:
        raise ValueError('Positive days/iterations and mixed delivery/pickup demand required')
    if not 0 < round(args.orders * args.share) < args.orders:
        raise ValueError('orders/share must generate both deliveries and pickups after rounding')
    if not args.h_values or any(h < 1 for h in args.h_values) or len(set(args.h_values)) != len(args.h_values):
        raise ValueError('h-values must be distinct positive integers')
    if not 0 <= args.explore <= 1:
        raise ValueError('explore must be in [0,1]')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; do not overwrite a prior campaign')
    out.mkdir(parents=True, exist_ok=True)
    budgets = dict(candidate_budget=args.candidate_budget,
                   enhancement_budget=args.enhancement_budget,
                   wait_budget_per_base=args.wait_budget)
    def worlds(layouts):
        return [make_world(layout, day, args.orders, args.share, pattern,
                           temporal=temporal, history_days=args.history_days,
                           improvement_budget=args.initial_budget)
                for layout in layouts for day in range(args.days)
                for pattern in args.patterns for temporal in args.temporal]
    banks = {'train': worlds(args.train_layouts), 'validation': worlds(args.validation_layouts)}
    source_dir = Path(__file__).resolve().parent
    source_files = [source_dir / name for name in ('run_unified_vfa.py', 'unified_vfa_policy.py',
                    'unified_vfa_features.py', 'uniform_wait_worlds.py', 'run_global_demand_pilot.py',
                    'recovery_opportunity.py', 'run_multiscale.py')]
    source_files += sorted((source_dir / 'e1_sim').glob('*.py'))
    source_hashes = {str(p.relative_to(source_dir)).replace('\\', '/'):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    protocol = dict(version='unified-vfa-development-v1', h_values=args.h_values,
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
    for h in args.h_values:
        model, days, curve = None, [], []
        for iteration in range(args.iterations):
            for i, world in enumerate(banks['train']):
                row, samples = audited_episode(world, h, model, explore=args.explore,
                                                seed=args.seed + iteration * 100000 + i, **budgets)
                days.append(samples)
                print(json.dumps(dict(phase='train', h=h, iteration=iteration + 1,
                                      day=i + 1, net=row['net'])), flush=True)
            model = fit_model(days, h, args.ridge)
            validation = [audited_episode(w, h, model, **budgets)[0] for w in banks['validation']]
            curve.append(dict(iteration=iteration + 1,
                              validation_mean_net=mean(r['net'] for r in validation),
                              validation=validation, model=model))
        # Checkpoint and h selection both use validation, never a test outcome.
        best = max(curve, key=lambda r: (r['validation_mean_net'], -r['iteration']))
        item = dict(h=h, iteration=best['iteration'], validation_mean_net=best['validation_mean_net'],
                    model=best['model'], curve=curve)
        candidates.append(item)
        print(json.dumps(dict(phase='validation', h=h, mean_net=item['validation_mean_net'])), flush=True)
    chosen = validation_choice(candidates)
    frozen = dict(chosen['model'], selection=dict(source='validation_only',
        validation_mean_net=chosen['validation_mean_net'], candidate_h=args.h_values,
        train_layouts=args.train_layouts, validation_layouts=args.validation_layouts,
        protocol_hash=digest(protocol), formal_test_opened=False))
    for filename, data in [('selected_model.json', frozen), ('results.json', candidates)]:
        (out / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(selected_h=chosen['h'], model=str(out / 'selected_model.json'),
                         formal_test_opened=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train', help='Development training and validation-only h selection')
    p.add_argument('--h-values', nargs='+', type=int, default=[5, 10])
    p.add_argument('--train-layouts', nargs='+', type=int, default=[710, 711])
    p.add_argument('--validation-layouts', nargs='+', type=int, default=[810, 811])
    p.add_argument('--days', type=int, default=1)
    p.add_argument('--orders', type=int, default=20)
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
    p.add_argument('--output', default='../../tmp/unified_vfa_development')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
