"""Recover completed supplemental pairs after Windows hard-link denial.

Does not change or restart the original supplementary executor. On Windows,
os.rename provides atomic create-if-absent publication of an independent copy.
The original scheduling failure is preserved and is not called a success.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from run_paired_wait_diagnostic import read, write
from supplement_paired_chain import (DEFAULT_OUTPUT, TIMING_EXCLUSIONS,
    check_sources, completion_ready, declared_tasks, pair_name, scientific_content,
    timing_content, validate_completed_pair)

DEFAULT_RUN = '20261128T191213_37952'


def publish_rename_if_absent(source, destination):
    """Never link source itself; never read or overwrite an existing destination."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(destination.parent),
                prefix='.relay-publish-', suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            with source.open('rb') as original:
                shutil.copyfileobj(original, output)
            output.flush()
            os.fsync(output.fileno())
        try:
            if os.name == 'nt':
                # Windows rename fails if destination exists; never os.replace.
                os.rename(str(temporary), str(destination))
                temporary = None
            else:
                os.link(str(temporary), str(destination))
            return True
        except FileExistsError:
            return False
    finally:
        if temporary is not None:
            temporary.unlink()


def load_context(root, run):
    root, run = Path(root), Path(run)
    scratch, dev = run / 'dev', root / 'dev'
    manifest = read(run / 'manifest.json')
    protocol = read(dev / 'protocol.json')
    check_sources(protocol)
    if manifest['source_hashes'] != protocol['source_hashes']:
        raise ValueError('supplement/runtime source mismatch')
    if read(scratch / 'protocol.json') != protocol:
        raise ValueError('scratch protocol differs')
    sites = read(dev / 'sites.json')
    tasks = {pair_name(s, r): (s, r) for s, r in declared_tasks(sites, protocol['samples'])}
    if not set(manifest['scheduled_names']) <= set(tasks):
        raise ValueError('supplement contains undeclared sample identities')
    return scratch, dev, manifest, protocol, tasks


def relay_once(scratch, dev, manifest, tasks, processed):
    outcomes = []
    for path in sorted((scratch / 'paired').glob('*.json'), reverse=True):
        name = path.name
        if name in processed:
            continue
        if name not in tasks or name not in manifest['scheduled_names']:
            raise ValueError('undeclared supplemental pair: ' + name)
        if (dev / 'paired' / name).exists():
            processed.add(name)
            outcomes.append(dict(name=name, status='existing_not_read'))
            continue
        try:
            row = read(path)
            # The scratch writer is non-atomic. Incomplete JSON is retried later.
            read(scratch / 'offline_futures' / name)
        except (OSError, ValueError):
            continue
        site, replica = tasks[name]
        count = validate_completed_pair(site, replica, scratch, row)
        future_created = publish_rename_if_absent(scratch / 'offline_futures' / name,
                                                 dev / 'offline_futures' / name)
        pair_created = publish_rename_if_absent(path, dev / 'paired' / name)
        processed.add(name)
        outcomes.append(dict(name=name, status='published' if pair_created else 'existing_at_publication',
            future_created=future_created, pair_created=pair_created,
            pair_validation_checks=count, suffix_hash=row['suffix_hash'],
            online_known_hash=row['online_known_hash']))
    return outcomes


def relay(root, run, poll_seconds):
    scratch, dev, manifest, protocol, tasks = load_context(root, run)
    processed, outcomes = set(), []
    while not completion_ready(dev):
        check_sources(protocol)
        new = relay_once(scratch, dev, manifest, tasks, processed)
        outcomes.extend(new)
        if new:
            record = dict(stage='relay', complete=False, observed_pairs=len(processed),
                published=sum(r.get('pair_created', False) for r in outcomes), outcomes=outcomes,
                original_scheduler_status='hard-link publication failed; original executor may still be running')
            write(Path(run) / 'relay_progress.json', record)
            print(json.dumps(dict(observed_pairs=len(processed), newly_published=sum(
                r.get('pair_created', False) for r in new), total_published=record['published'])), flush=True)
        time.sleep(poll_seconds)
    check_sources(protocol)
    record = dict(stage='relay', complete=True, stopped_because='authoritative dev completion observed',
        observed_pairs=len(processed), published=sum(r.get('pair_created', False) for r in outcomes),
        outcomes=outcomes, original_scheduler_success=False,
        recovery_audit_required=True, source_hashes=protocol['source_hashes'])
    write(Path(run) / 'relay_complete.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('outcomes', 'source_hashes')}), flush=True)


def audit_recovery(root, run, original_exited, error_log):
    if not original_exited:
        raise ValueError('confirm original executor has exited with --original-supplement-exited')
    scratch, dev, manifest, protocol, tasks = load_context(root, run)
    if not completion_ready(dev):
        raise ValueError('authoritative dev executor has not completed')
    error_log = Path(error_log)
    error_bytes = error_log.read_bytes()
    error_text = error_bytes.decode('utf8', errors='replace')
    if not ('PermissionError' in error_text and ('WinError 5' in error_text or 'os.link' in error_text)):
        raise ValueError('original error log does not document the hard-link publication failure')
    failures, comparisons = [], []
    for name, expected in manifest['input_hashes'].items():
        for directory in (scratch, dev):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                failures.append(dict(input=name, directory=str(directory), error='input hash changed'))
    source = Path(__file__).resolve().parent / 'supplement_paired_chain.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['scheduler_sha256']:
        failures.append(dict(error='original supplementary source changed'))
    pairs = sorted((scratch / 'paired').glob('*.json'))
    pair_names = {p.name for p in pairs}
    futures = {p.name for p in (scratch / 'offline_futures').glob('*.json')}
    if pair_names != futures:
        failures.append(dict(error='incomplete scratch tasks', pairs_without_future=sorted(pair_names-futures),
            future_without_pair=sorted(futures-pair_names)))
    for path in pairs:
        name = path.name
        if name not in tasks or name not in manifest['scheduled_names']:
            failures.append(dict(pair=name, error='undeclared supplemental sample'))
            continue
        row = read(path)
        site, replica = tasks[name]
        checks = validate_completed_pair(site, replica, scratch, row)
        final = read(dev / 'paired' / name)
        same = scientific_content(row) == scientific_content(final)
        same_future = read(scratch / 'offline_futures' / name) == read(dev / 'offline_futures' / name)
        if not same or not same_future:
            failures.append(dict(pair=name, scientific_equal=same, future_equal=same_future))
        comparisons.append(dict(pair=name, scientific_equal=same, future_equal=same_future,
            pair_validation_checks=checks, scratch_timing=timing_content(row), final_timing=timing_content(final)))
    result = dict(successful=not failures, status='recovered_and_verified' if not failures else 'failed',
        original_scheduler_success=False, original_schedule_complete_present=(Path(run)/'schedule_complete.json').exists(),
        original_executor_exit_confirmed=True, original_failure='PermissionError during os.link publication',
        original_error_log=str(error_log.resolve()), original_error_log_sha256=hashlib.sha256(error_bytes).hexdigest(),
        original_scheduler_sha256=manifest['scheduler_sha256'], source_hashes=protocol['source_hashes'],
        verified_scratch_pairs=len(comparisons), failures=failures, comparisons=comparisons,
        exact_comparison_exceptions=TIMING_EXCLUSIONS, independent_test_opened=False,
        verified_at=datetime.now().isoformat(), scope='Recovery of completed scratch results; not original schedule success.')
    write(Path(run) / 'recovery_completion.json', result)
    write(Path(root) / 'audit' / 'supplement_recovery_integrity.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['relay', 'audit-recovery'], default='relay')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--scratch-run', type=Path)
    p.add_argument('--poll-seconds', type=float, default=5)
    p.add_argument('--original-supplement-exited', action='store_true')
    p.add_argument('--original-error-log', type=Path)
    args = p.parse_args()
    root = args.output.resolve()
    run = args.scratch_run.resolve() if args.scratch_run else root / 'scheduler_scratch' / DEFAULT_RUN
    if not 0 < args.poll_seconds <= 60:
        p.error('--poll-seconds must be in (0,60]')
    if args.stage == 'relay':
        relay(root, run, args.poll_seconds)
    else:
        result = audit_recovery(root, run, args.original_supplement_exited,
                                args.original_error_log or root / 'supplement.log')
        print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','source_hashes')}), flush=True)
        if not result['successful']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
