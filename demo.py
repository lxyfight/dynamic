"""Run the fixed-h unified candidate policy; --legacy reproduces the old baseline."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "e1_v04"))

from high_demand_features import run_case
from run_high_demand_screen import world_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=int, default=460, help="Synthetic layout seed")
    parser.add_argument("--orders", type=int, default=20, help="Global order count")
    parser.add_argument("--pattern", choices=("uniform", "dual"), default="uniform")
    parser.add_argument("--temporal", choices=("peak", "flat"), default="peak")
    parser.add_argument("--share", type=float, default=.4, help="Offline pickup proportion")
    parser.add_argument("--h", type=int, default=None, help="Fixed waiting interval in minutes")
    parser.add_argument("--model", type=Path, help="A trained unified VFA JSON model")
    parser.add_argument("--legacy", action="store_true", help="Reproduce the old frozen-six-feature demo")
    args = parser.parse_args()
    if args.orders < 2:
        parser.error("--orders must be at least 2 to include both deliveries and pickups")
    if not 0 < args.share < 1 or not 0 < round(args.orders * args.share) < args.orders:
        parser.error("--orders and --share must generate both deliveries and pickups")
    if args.h is not None and args.h < 1:
        parser.error("--h must be positive")
    if args.legacy:
        if args.model is not None or args.h is not None or args.share != .4 or args.temporal != 'peak':
            parser.error("--legacy retains its original 40% pickup and peak profile; no new model/h options")
        return legacy_demo(args)

    from uniform_wait_worlds import make_world
    from run_unified_vfa import audited_episode
    model = None if args.model is None else json.loads(args.model.read_text(encoding='utf8'))
    h = args.h if args.h is not None else (model['h'] if model is not None else 5)
    world = make_world(args.layout, 0, args.orders, args.share, args.pattern, temporal=args.temporal)
    print(f'Fixed h={h} min; {args.pattern} space / {args.temporal} time; '
          f'{args.orders} orders ({round(args.orders * args.share)} dynamic pickups).')
    print('All candidate plans share one value function and each truck keeps its own wait clock.')
    if model is None:
        print('No trained model supplied: running the zero-value Myopic baseline. '
              'The default h=5 is not a validated optimum.')
    print(f"{'Policy':<18}{'Net (CNY)':>12}{'Accepted':>12}{'Decisions':>12}")
    policies = [('Unified-Myopic', None)]
    if model is not None:
        policies.append(('Unified-VFA', model))
    for label, parameters in policies:
        row, _ = audited_episode(world, h, parameters)
        print(f"{label:<18}{row['net']:>12.2f}{row['accepted']:>12}{row['decisions']:>12}")
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
