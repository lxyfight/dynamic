"""Run Myopic and the frozen six-feature VFA on the same synthetic service day."""
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
    parser.add_argument("--orders", type=int, default=20, help="Global order count (40%% pickups)")
    parser.add_argument("--pattern", choices=("uniform", "dual"), default="uniform")
    args = parser.parse_args()
    if args.orders < 2:
        parser.error("--orders must be at least 2 to include both deliveries and pickups")

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
