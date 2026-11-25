"""Execution-layer E1 adapters, including pure physical boundary predicates."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from .audit import audit_ledger
from .constraints import module_observations
from .kernel import Replay
from .quantities import InvalidAction, UnsupportedCase, dec, plain

REPLAY_IDS = frozenset(("E01", "E02", "E02-R", "E03-BM", "E03-B1", "E04-late", "E04-same",
                       "E05", "E06-over", "E06-exact", "E07-parallel", "E07-conflict", "E08-plain", "E08-split",
                       "E09", "E10", "E16", "E17-BM", "E17-B1", "E21", "E23", "E29"))
MODULE_IDS = frozenset(("E11-exact", "E11-over", "E19", "E20-exact", "E20-late", "E28", "E30",
                       "E33-split", "E33-fraction", "E33-demand", "E33-pickup_demand", "E33-unit",
                       "E33-balance", "E33-capacity", "E33-time", "E33-pickup_split"))
POLICY_IDS = frozenset(("E12-W0", "E12-WK", "E14", "E18", "E22", "E27-success", "E27-failure", "E27-W0_restriction", "E27-unknown_count"))
CONTROL_IDS = frozenset(('E13-BM','E13-B1','E24-new_ready','E24-locked_elsewhere','E25','E26','E26-capacity'))
FORECAST_IDS = frozenset(('E15-integer','E15-fractional','E15-zero'))
SUPPORTED = REPLAY_IDS | MODULE_IDS | POLICY_IDS | CONTROL_IDS | FORECAST_IDS | {'E31','E32'}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = ROOT / "data/fixtures/e1_cases.json"


def build_replay(package, case):
    """Read inputs only. No assertions or expected values enter the executor."""
    if case["scope"] not in ("full_day", "local", "event"):
        raise UnsupportedCase("only full-day/local scripted replay is implemented")
    data = deepcopy(case["input"])
    if set(data.get("policy") or {}) - {"batch_permission"} or data.get("candidates"):
        raise UnsupportedCase("policy/constraint adapters are not implemented")
    config = deepcopy(package["configuration"])
    config.update(case.get("parameter_overrides", {}))
    # Scripted logistics specify future station stops; this is not an online
    # policy looking ahead. Fail if the supplied script leaves a stop ambiguous.
    homes = {u["id"]: u["home"] for u in data["drones"]}
    stations = {}
    for lot in data["inventory"]:
        if lot["state"] == "available":
            stations.setdefault(lot["order"], set()).add(lot["holder"])
    for a in data["actions"]:
        if a["kind"] == "drone_sortie":
            for oid in a["visit_order"]:
                stations.setdefault(oid, set()).add(homes[a["actor"]])
    for e in data["external_events"]:
        if e["kind"] == "pickup_unload_complete":
            for oid in e["orders"]:
                stations.setdefault(oid, set()).add(e["station"])
    stops = {}
    for a in data["actions"]:
        if a["kind"] == "truck_unload":
            stops.setdefault(a["actor"], set()).add(a["station"])
        elif a["kind"] == "truck_load":
            if "station" not in a:
                possibilities = [stations.get(oid, set()) for oid in a["orders"]]
                if not possibilities or any(len(s) != 1 for s in possibilities):
                    raise UnsupportedCase("script must explicitly identify pickup loading station")
                merged = set.union(*possibilities)
                if len(merged) != 1:
                    raise InvalidAction("pickup batch spans multiple stations")
                a["station"] = next(iter(merged))
            stops.setdefault(a["actor"], set()).add(a["station"])
    return Replay(config, package["networks"][case["network"]], data,
                  full_day=case["scope"] == "full_day", planned_stops=stops)


def compare(actual, expected, op):
    if op == "equal":
        if isinstance(expected, bool):
            return isinstance(actual, bool) and actual == expected
        return actual == expected
    if op == "close":
        if isinstance(expected, list):
            return isinstance(actual, list) and len(actual) == len(expected) and all(compare(a, b, op) for a, b in zip(actual, expected))
        if isinstance(expected, dict):
            return isinstance(actual, dict) and actual.keys() == expected.keys() and all(compare(actual[k], expected[k], op) for k in expected)
        return not isinstance(actual, bool) and abs(dec(actual) - dec(expected)) <= dec("0.000001")
    raise UnsupportedCase("assertion operator: " + op)


def execute_case(package, case):
    replay = None
    policy_result = None
    if case['id']=='E31':
        from .pi_adapter import mapping_observations
        observed=mapping_observations(package,case);audit=observed.pop('independent_audit')
    elif case['id']=='E32':
        from .pi_mapping import compression_mapping
        cfg=deepcopy(package['configuration']);cfg.update(case.get('parameter_overrides',{}))
        observed=plain(compression_mapping(cfg,case['input']));audit=None
    elif case['id'] in CONTROL_IDS:
        from .wtq import run_controller_case
        replay,audit=run_controller_case(package,case)
        observed=replay.observations()
    elif case['id'] in FORECAST_IDS:
        from .forecast import sample_releases
        raw=deepcopy(case['input'])
        sample=sample_releases(raw['start_time'],raw['public_information'],raw['random_streams']['forecast_seed'])
        expected=sample.pop('expected_arrivals_at_next')
        sample['real_inventory_mutations']=int(raw!=case['input'])
        observed=dict(sample=sample,distribution=dict(expected_arrivals_at_301=expected),ledger={})
        audit=None
    elif case["id"] in POLICY_IDS:
        from .policy_cases import policy_observations
        policy_result = policy_observations(package, case)
        observed, audit = policy_result["observations"], policy_result["independent_audit"]
    elif case["id"] in MODULE_IDS:
        observed = module_observations(package, case)
        audit = None
    else:
        replay = build_replay(package, case)
        replay.run([a["at"] for a in case["assertions"] if a["at"] is not None])
        observed = replay.observations()
        audit = audit_ledger(replay.cfg, replay.network, replay.initial, replay.trace)
        audit["matches_online"] = not audit["errors"] and all(compare(observed["ledger"][k], v, "close") for k, v in audit["ledger"].items())
    checks = []
    for assertion in case["assertions"]:
        check = deepcopy(assertion)
        try:
            if replay is not None:
                actual = replay.observe(assertion["target"], assertion["at"], assertion["phase"])
            else:
                actual = observed
                for key in assertion["target"].split("."):
                    actual = actual[key]
            check.update(actual=actual, status="PASS" if compare(actual, assertion["expected"], assertion["op"]) else "FAIL")
        except (KeyError, IndexError, UnsupportedCase) as exc:
            check.update(status="NOT_IMPLEMENTED", error=str(exc))
        checks.append(check)
    warnings = []
    declared = case["ledger_scope"]["include"]
    aliases = {"revenue": "realized_revenue", "lateness": "pickup_lateness"}
    if "all" not in declared:
        missing = [k for k, v in observed["ledger"].items() if k != "net" and v != 0
                   and aliases.get(k, k) not in declared]
        if missing:
            warnings.append("Source ledger_scope.include omits physical costs required by assertions: " + ", ".join(missing) +
                            ". All actual costs inside the window are retained; historical costs excluded. Source is unchanged.")
    return dict(id=case["id"], scope=case["scope"], validation_only=replay is None and case['id']!='E31' and not (policy_result and policy_result["executed"]),
                status="PASS" if all(c["status"] == "PASS" for c in checks) and (audit is None or audit["matches_online"]) and not warnings else "FAIL",
                assertions=checks, independent_audit=audit, observations=observed,
                trace=plain(replay.trace) if replay else policy_result["trace"] if policy_result else [],
                diagnostics=policy_result["diagnostics"] if policy_result else {},
                final_public_state=replay.public_snapshot() if replay else None, warnings=warnings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/e1_verification")
    parser.add_argument("--cases", nargs="*", help="default: inventory all 59; execute execution-layer adapters")
    args = parser.parse_args(argv)
    package = json.loads(args.spec.read_text(encoding="utf-8"))
    known_ids = {c["id"] for c in package["cases"]}
    if args.cases and set(args.cases) - known_ids:
        parser.error("unknown case ID")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in package["cases"]:
        if args.cases is not None and case["id"] not in args.cases:
            continue
        if case["id"] not in SUPPORTED:
            result = dict(id=case["id"], status="NOT_IMPLEMENTED", assertion_count=len(case["assertions"]),
                          reason="PI fixed-trajectory mapping not implemented")
        else:
            try:
                result = execute_case(package, case)
            except Exception as exc:
                result = dict(id=case["id"], status="FAIL", error=f"{type(exc).__name__}: {exc}")
            (args.output / (case["id"] + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(f"{result['id']}: {result['status']}" + (" " + result["error"] if "error" in result else ""))
    summary = dict(spec_version=package["spec_version"], spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                   python=sys.version, counts={s: sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "NOT_IMPLEMENTED")},
                   assertions_passed=sum(c["status"] == "PASS" for r in results for c in r.get("assertions", [])),
                   assertions_total=sum(len(c["assertions"]) for c in package["cases"]),
                   results=[{k: v for k, v in r.items() if k in ("id", "status", "reason", "error", "warnings")} for r in results],
                   limits=["Bounded Myopic candidate pool, not globally optimal routing",
                           "WTQ compiler has separate full-day semantic/ledger acceptance; old time-window diagnostic remains rejected",
                           "E31 fixed selected-structure feasibility witness checked; not a free PI optimizer or global optimality certificate",
                           "F1 feature ablation and iterative training have separate result artifacts; full manuscript scale grid remains pending",
                           "Conditional sampler generates city/count/release only, not full geographic request scenarios",
                           "E22 fixture scores and external recovery contracts are transaction-only; local windows are not full-day comparisons"])
    source_dir = Path(__file__).parent
    summary["implementation_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob("*.py"))}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"]), "assertions passed:", summary["assertions_passed"])
    return int(any(r["status"] == "FAIL" for r in results))
