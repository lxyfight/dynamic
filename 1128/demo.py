"""Run a fixed or adaptive unified candidate policy; --legacy reproduces the old baseline."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "e1_v04"))

from unified_vfa_policy import parse_pattern_limit
from high_demand_features import run_case
from run_high_demand_screen import world_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=int, default=460, help="Synthetic layout seed")
    parser.add_argument("--orders", type=int, default=20, help="Total demand level (nominal in p2 mode)")
    parser.add_argument("--pattern", choices=("uniform", "dual"), default="uniform")
    parser.add_argument("--temporal", choices=("peak", "flat"), default="peak")
    parser.add_argument("--share", type=float, default=.4, help="Offline pickup proportion")
    parser.add_argument("--h", type=int, default=None, help="Fixed waiting interval in minutes")
    parser.add_argument("--review-mode", choices=("fixed", "adaptive"))
    parser.add_argument("--adaptive-intervals", nargs="+", type=int)
    parser.add_argument("--arrival-mode", choices=("fixed_count", "p2"))
    parser.add_argument("--model", type=Path, help="A trained unified VFA JSON model")
    parser.add_argument("--action-solver", choices=("bounded", "gurobi"), default=argparse.SUPPRESS,
                        help="Action search for the fixed-h unified policy")
    parser.add_argument("--mip-pattern-limit", type=parse_pattern_limit, default=argparse.SUPPRESS,
                        help="New patterns per Gurobi phase and event, or all for tiny restricted instances")
    parser.add_argument("--legacy", action="store_true", help="Reproduce the old frozen-six-feature demo")
    args = parser.parse_args()
    if args.orders < 2:
        parser.error("--orders must be at least 2 to include both deliveries and pickups")
    if not 0 < args.share < 1 or not 0 < round(args.orders * args.share) < args.orders:
        parser.error("--orders and --share must generate both deliveries and pickups")
    if args.h is not None and args.h < 1:
        parser.error("--h must be positive")
    if args.legacy:
        if (hasattr(args, 'action_solver') or hasattr(args, 'mip_pattern_limit') or
                args.review_mode or args.adaptive_intervals or args.arrival_mode):
            parser.error('Search, review and arrival options apply to the unified policy')
        if args.model is not None or args.h is not None or args.share != .4 or args.temporal != 'peak':
            parser.error("--legacy retains its original 40% pickup and peak profile; no new model/h options")
        return legacy_demo(args)

    from uniform_wait_worlds import make_world
    from run_unified_vfa import audited_episode, resolve_modes
    model = None if args.model is None else json.loads(args.model.read_text(encoding='utf8'))
    try:
        mode, arrival_mode = resolve_modes(args.review_mode, args.arrival_mode, default_mode='adaptive',
            fixed_interval_supplied=args.h is not None,
            adaptive_menu_supplied=args.adaptive_intervals is not None, model=model)
    except ValueError as exc:
        parser.error(str(exc))
    h = args.h if args.h is not None else (model['h'] if model is not None else 5)
    if mode == 'fixed' and args.adaptive_intervals is not None:
        parser.error('Use --adaptive-intervals with the adaptive policy')
    if mode == 'adaptive' and args.h is not None:
        parser.error('Use --adaptive-intervals to select the adaptive action menu')
    intervals = args.adaptive_intervals or (model.get('review_intervals') if model else None)
    if mode == 'adaptive':
        intervals = sorted(intervals if intervals is not None else [2, 5, 10])
        if len(intervals) != len(set(intervals)) or min(intervals) <= 0:
            parser.error('Choose distinct positive adaptive intervals')
    horizon = model.get('feature_horizon', h) if model else h
    selection = model.get('selection', {}) if model else {}
    search = dict(action_solver='bounded', mip_pattern_limit=64)
    search.update(selection.get('search', {}))
    search.update({k: vars(args)[k] for k in ('action_solver', 'mip_pattern_limit') if k in vars(args)})
    world = make_world(args.layout, 0, args.orders, args.share, args.pattern, temporal=args.temporal,
                       arrival_mode=arrival_mode, **selection.get('world_options', {}))
    review = f'adaptive review menu {intervals}' if intervals is not None else f'fixed h={h}'
    print(f'{review} min; {args.pattern} space / {args.temporal} time; '
          f'{arrival_mode} arrivals, nominal size {args.orders}.')
    print('All candidate plans share one value function and each truck keeps its own wait clock.')
    if search['action_solver'] == 'gurobi':
        limit = 'all restricted patterns' if search['mip_pattern_limit'] is None else search['mip_pattern_limit']
        print(f'Gurobi search, pattern limit per phase and event: {limit}.')
    if model is None:
        print('No trained model supplied: running the zero-value Myopic baseline.')
    print(f"{'Policy':<18}{'Net (CNY)':>12}{'Accepted':>12}{'Decisions':>12}")
    policies = [('Unified-Myopic', None)]
    if model is not None:
        policies.append(('Unified-VFA', model))
    for label, parameters in policies:
        row, _ = audited_episode(world, h, parameters, review_intervals=intervals, feature_horizon=horizon, **search)
        print(f"{label:<18}{row['net']:>12.2f}{row['accepted']:>12}{row['decisions']:>12}")
        if search['action_solver'] == 'gurobi':
            effort = row['action_search']
            print(f"  Service patterns tried: {effort['service_patterns_attempted']}; "
                  f"wait patterns tried: {effort['wait_patterns_attempted']}; "
                  f"events with truncated search: {effort['truncated_epochs']}.")
    print(f"Realized day: {row['realized_orders']} orders including {row['realized_pickups']} pickups.")
    print('Complete-day feasibility and independent ledger audit passed.')


def legacy_demo(args):

    model = json.loads((ROOT / "results" / "frozen_six.json").read_text(encoding="utf-8"))
    if model["columns"] != list(range(6)):
        raise ValueError("Expected the six features of the frozen policy")
    world = world_for((args.layout, args.orders, args.pattern))
    print(f"Layout {args.layout}, {args.pattern}, {args.orders} orders "
          f"({round(.4 * args.orders)} dynamic pickups)")
    print(f"{'Policy':<12}{'Net (CNY)':>12}{'Accepted':>12}{'Truck trips':>14}")
    profits = {}
    for label, parameters in (("Myopic", None), ("VFA", model)):
        # Each full-day run replays its event trace and audits the final ledger.
        row = run_case(world, model=parameters, teacher=0)
        if (row["input_hash"], row["i0_hash"]) != (world["input_hash"], world["i0_hash"]):
            raise RuntimeError(f"{label} used a different instance or initial plan")
        profits[label] = row["net"]
        print(f"{label:<12}{row['net']:>12.2f}{row['accepted']:>12}"
              f"{row['operation_metrics']['trips']:>14}")
    print(f"Paired VFA gain (CNY): {profits['VFA'] - profits['Myopic']:.2f}")


if __name__ == "__main__":
    main()
