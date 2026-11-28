"""Reproduce the November 28 rule audit without opening a held-out test bank.

From code/e1_v04:
    python verify_screen_rules.py
    python verify_screen_rules.py --output ../../results/feature_screen_20261128/audit/verification_script_check.json

This runs the 20 focused rule tests, generates one development example, and
replays the existing E02 fixture with the current study configuration. It does
not train a model or run the full regression suite.
"""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest

from e1_sim.runner import DEFAULT_SPEC, build_replay
from uniform_wait_worlds import CFG, make_world


CODE_ROOT = Path(__file__).resolve().parent
TEST_PATTERNS = (
    'test_pickup_deadline_distribution.py',
    'test_weight_only_handling.py',
    'test_unified_operating_rules.py',
    'test_unified_vfa_features.py',
    'test_deadline_exposure.py',
)
CONFIGURATION_KEYS = (
    'truck_handling_setup', 'truck_handling_per_kg',
    'drone_handling_setup_per_request', 'drone_handling_per_kg',
    'customer_service_per_request', 'pickup_lateness_per_request_min',
    'truck_dispatch_cutoff',
)
EXPECTED_EXAMPLES = [
    dict(id='P00001', release=29, lag=240, deadline=269),
    dict(id='P00009', release=163, lag=300, deadline=463),
    dict(id='P00000', release=3, lag=360, deadline=363),
]


def pickup_record(order):
    return dict(id=order['id'], release=order['release'],
                lag=order['deadline'] - order['release'],
                deadline=order['deadline'])


def verify():
    test_log = io.StringIO()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.discover(str(CODE_ROOT / 'tests'), pattern=p)
                               for p in TEST_PATTERNS)
    result = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)

    generation_call = dict(layout_id=710, day_id=1, total=30, share=.9,
                           pattern='uniform', arrival_mode='p2', history_days=1,
                           improvement_budget=0)
    world = make_world(**generation_call)
    pickups = [order for order in world['input']['orders'] if order['kind'] == 'P']
    covering = [pickup_record(next(order for order in pickups
                    if order['deadline'] - order['release'] == lag))
                for lag in (240, 300, 360)]

    package = json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
    # Historical fixtures retain their historical defaults. Explicitly apply
    # the current study configuration to all four real handling operations.
    package['configuration'] = CFG
    case = next(case for case in package['cases'] if case['id'] == 'E02')
    state = build_replay(package, case).run()
    event_kinds = {'truck_unload_started', 'truck_load_started',
                   'drone_handling_started', 'service_started'}
    operations = [dict(kind=e['kind'], loading=e.get('loading'),
                       start=str(e['at']), end=str(e['end']),
                       duration=str(e['end'] - e['at']))
                  for e in state.trace if e['kind'] in event_kinds]
    actual_handling = Counter((r['kind'], r['loading'], Decimal(r['duration']))
                              for r in operations if r['kind'] != 'service_started')
    quarter = Decimal('.25')
    expected_handling = Counter([
        ('truck_unload_started', None, quarter),
        ('truck_load_started', None, quarter),
        ('drone_handling_started', True, quarter),
        ('drone_handling_started', False, quarter),
    ])
    checks = dict(
        all_twenty_targeted_tests_pass=(result.testsRun == 20 and
                                       result.wasSuccessful() and not result.skipped),
        actual_pickup_count_matches_original_audit=(len(pickups) == 33),
        three_actual_samples_match_original_audit=(covering == EXPECTED_EXAMPLES),
        four_actual_handling_operations_are_quarter_minute=(actual_handling == expected_handling),
        both_actual_customer_services_are_one_minute=(
            [Decimal(r['duration']) for r in operations if r['kind'] == 'service_started']
            == [Decimal(1), Decimal(1)]),
    )
    return dict(
        python=sys.executable, version=platform.python_version(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='development_rule_verification_only', independent_test_opened=False,
        tests=dict(patterns=list(TEST_PATTERNS), run=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors),
                   skipped=len(result.skipped), successful=result.wasSuccessful(),
                   log=test_log.getvalue()),
        configuration={key: CFG[key] for key in CONFIGURATION_KEYS},
        generation_call=generation_call,
        example_world=dict(id=world['id'], spec=world['spec'],
                           generation=world['generation'], input_hash=world['input_hash'],
                           seeds=world['seeds']),
        pickups=[pickup_record(order) for order in pickups[:3]],
        examples_covering_each_deadline_lag=covering,
        ten_kg_replay=dict(fixture='E02', operations=operations),
        reference_checks=checks, successful=all(checks.values()),
        notes=[
            'Station setup is zero in current CFG; historical fixture defaults are overridden.',
            'The listed pickups are generated data, not manually constructed examples.',
            'No training model or independent test layout is loaded or changed.',
            'The complete 336-test regression is not rerun by this script.',
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=Path('../../results/feature_screen_20261128/audit/rules_check_reproduced.json'),
        help='JSON result path, relative to the current working directory.')
    args = parser.parse_args()
    report = verify()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(output=str(output), successful=report['successful'],
                          tests=report['tests']['run'], reference_checks=report['reference_checks'],
                          examples=report['examples_covering_each_deadline_lag']),
                     ensure_ascii=True))
    return 0 if report['successful'] else 1


if __name__ == '__main__':
    sys.exit(main())
