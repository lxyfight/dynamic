"""Evaluate only missing, predeclared dev pairs in a separate scratch bank.

The original dev executor remains authoritative. Supplemental publication uses
create-if-absent hard links from independent temporary copies, never from the
retained scratch originals. An already running original job may subsequently
rewrite its own result. Run --stage audit after BOTH executors finish to compare
all completed scratch results against the authoritative bank.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from audit_paired_chain_campaign import Checks, audit_pair
from e1_sim.initializer import digest
from e1_sim.online import known_clone
from e1_sim.quantities import plain
from run_paired_chain_campaign import lock_protocol, protocol_for, source_hashes
from run_paired_wait_diagnostic import evaluate_sample, load_pickle, read, write

DEFAULT_OUTPUT = Path('../../results/paired_chain_20261128')
TIMING_EXCLUSIONS = ['wall_seconds', 'branches.*.continuation_search.decision_seconds']


def pair_name(site, replica):
    return site['id'] + f'_r{replica:02}.json'


def declared_tasks(sites, samples):
    if samples < 1 or len({s['id'] for s in sites}) != len(sites):
        raise ValueError('invalid declared sample bank')
    return [(site, replica) for site in sites for replica in [-1] + list(range(samples))]


def missing_reverse_tasks(sites, samples, dev):
    # Existing files may currently be written by the original worker. Do not read.
    return [(s, r) for s, r in reversed(declared_tasks(sites, samples))
            if not (Path(dev) / 'paired' / pair_name(s, r)).exists()]


def completion_ready(directory):
    """A partially written completion marker is not evidence of completion."""
    directory = Path(directory)
    try:
        marker = read(directory / 'evaluation_complete.json')
        protocol = read(directory / 'protocol.json')
        sites = read(directory / 'sites.json')
    except (OSError, ValueError):
        return False
    count = len(sites) * (protocol['samples'] + 1)
    return (marker.get('split') == protocol['split']
            and marker.get('primary_states') == len(sites)
            and marker.get('conditional_replicas_per_state') == protocol['samples']
            and marker.get('original_audit_replicas_per_state') == 1
            and marker.get('paired_samples') == count
            and marker.get('full_day_branches') == 4 * count
            and marker.get('independent_test_opened') is False)


def wait_for_training(root, wait):
    while not all(completion_ready(Path(root) / s) for s in ('train', 'tune')):
        if not wait:
            raise RuntimeError('train/tune are not complete; use --wait-for-training to wait')
        print(json.dumps(dict(status='waiting_for_train_and_tune')), flush=True)
        time.sleep(30)


def check_sources(protocol):
    if source_hashes() != protocol['source_hashes']:
        raise ValueError('locked runtime source hashes changed; no supplemental publication allowed')


def copy_byte_identical(source, destination):
    """Copy, not link: the retained audit original must have its own inode."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = source.read_bytes()
    with destination.open('xb') as stream:
        stream.write(original)
    copied = destination.read_bytes()
    if copied != original:
        raise ValueError('input copy differs: ' + str(source))
    return hashlib.sha256(original).hexdigest()


def publish_copy_if_absent(source, destination):
    """Atomically create the destination, without reading/overwriting existing files.

    The temporary publication copy has a different inode from source. Therefore
    an in-flight original worker rewriting destination cannot mutate source.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(destination.parent),
                prefix='.supplement-publish-', suffix='.tmp', delete=False) as target:
            temporary = Path(target.name)
            with source.open('rb') as original:
                shutil.copyfileobj(original, target)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(str(temporary), str(destination))
            return True
        except FileExistsError:
            return False
    finally:
        if temporary is not None:
            temporary.unlink()


def stage_inputs(dev, scratch, sites, protocol):
    check_sources(protocol)
    files = [Path('protocol.json'), Path('sites.json'), Path('capture_summary.json')]
    files += [Path('capture') / (case + '.json') for case in sorted({s['case'] for s in sites})]
    files += [Path('snapshots') / (s['id'] + '.pkl.gz') for s in sites]
    files += [Path('states') / (s['id'] + '.json') for s in sites]
    hashes = {p.as_posix(): copy_byte_identical(Path(dev) / p, Path(scratch) / p) for p in files}
    check_sources(protocol)
    return hashes


def validate_completed_pair(site, replica, scratch, row):
    scratch = Path(scratch)
    bundle = load_pickle(scratch / 'snapshots' / (site['id'] + '.pkl.gz'))
    if bundle['site'] != site or set(bundle['plans']) != {0, 2, 5, 10}:
        raise ValueError('snapshot identity or intervention set changed')
    if digest(plain(bundle['state'].trace)) != site['prefix_hash']:
        raise ValueError('snapshot prefix changed')
    known_hash = digest(plain(known_clone(bundle['state']).initial))
    metadata = read(scratch / 'offline_futures' / pair_name(site, replica))
    reference = read(scratch / 'capture' / (site['case'] + '.json'))['reference']
    checks = Checks()
    audit_pair(row, metadata, site, replica, known_hash, reference, checks)
    if checks.failures:
        raise ValueError('supplemental pair audit failed: ' + json.dumps(checks.failures))
    return checks.count


def evaluate_and_publish(site, replica, scratch, dev):
    scratch, dev = Path(scratch), Path(dev)
    name = pair_name(site, replica)
    if (dev / 'paired' / name).exists():
        return dict(name=name, status='skipped_existing_without_reading')
    protocol = read(scratch / 'protocol.json')
    check_sources(protocol)
    row = evaluate_sample(site, scratch, replica)
    count = validate_completed_pair(site, replica, scratch, row)
    check_sources(protocol)
    # Preserve the same offline future before making the complete pair visible.
    # Existing metadata may be in-flight; do not read it until main completion.
    future_created = publish_copy_if_absent(scratch / 'offline_futures' / name,
                                           dev / 'offline_futures' / name)
    pair_created = publish_copy_if_absent(scratch / 'paired' / name, dev / 'paired' / name)
    result = dict(name=name, status='published' if pair_created else 'computed_but_existing',
                  pair_created=pair_created, future_created=future_created,
                  pair_validation_checks=count, suffix_hash=row['suffix_hash'],
                  online_known_hash=row['online_known_hash'])
    write(scratch.parent / 'scheduling' / name, result)
    return result


def scientific_content(row):
    """Only perf_counter durations are excluded; all scientific values stay exact."""
    row = deepcopy(row)
    row.pop('wall_seconds', None)
    for branch in row.get('branches', {}).values():
        branch.get('continuation_search', {}).pop('decision_seconds', None)
    return row


def timing_content(row):
    return dict(wall_seconds=row.get('wall_seconds'),
                decision_seconds={h: b.get('continuation_search', {}).get('decision_seconds')
                                  for h, b in row.get('branches', {}).items()})


def audit_supplement(root):
    """Must be invoked after the original dev executor and supplements finish."""
    root = Path(root)
    if not completion_ready(root / 'dev'):
        return dict(successful=False, status='incomplete', reason='main dev completion missing', checked=0)
    runs = sorted((root / 'scheduler_scratch').glob('*/manifest.json'))
    unfinished = [str(p.parent) for p in runs if not (p.parent / 'schedule_complete.json').is_file()]
    if unfinished:
        return dict(successful=False, status='incomplete', unfinished_supplements=unfinished, checked=0)
    failures, compared = [], []
    protocol = read(root / 'dev' / 'protocol.json')
    check_sources(protocol)
    sites = read(root / 'dev' / 'sites.json')
    declared = {pair_name(s, r) for s, r in declared_tasks(sites, protocol['samples'])}
    for path in runs:
        manifest = read(path)
        scratch = path.parent / 'dev'
        if manifest['source_hashes'] != protocol['source_hashes']:
            failures.append(dict(run=str(path.parent), error='runtime hashes differ'))
        for name, expected in manifest['input_hashes'].items():
            for directory in (root / 'dev', scratch):
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                    failures.append(dict(run=str(path.parent), input=name, error='input copy changed'))
        for pair in sorted((scratch / 'paired').glob('*.json')):
            if pair.name not in declared or pair.name not in manifest['scheduled_names']:
                failures.append(dict(pair=pair.name, error='undeclared supplemental sample'))
                continue
            final = root / 'dev' / 'paired' / pair.name
            future = root / 'dev' / 'offline_futures' / pair.name
            if not final.is_file() or not future.is_file():
                failures.append(dict(pair=pair.name, error='authoritative result missing'))
                continue
            original, authoritative = read(pair), read(final)
            same = scientific_content(original) == scientific_content(authoritative)
            same_future = read(scratch / 'offline_futures' / pair.name) == read(future)
            if not same or not same_future:
                failures.append(dict(pair=pair.name, scientific_equal=same, future_equal=same_future))
            compared.append(dict(run=path.parent.name, pair=pair.name, scientific_equal=same,
                future_equal=same_future, scratch_timing=timing_content(original),
                final_timing=timing_content(authoritative)))
    return dict(successful=not failures, status='passed' if not failures else 'failed',
        runs=len(runs), checked=len(compared), failures=failures, comparisons=compared,
        exact_comparison_exceptions=TIMING_EXCLUSIONS, independent_test_opened=False,
        scope='Supplemental duplicate calculations only; full-bank integrity is audited separately.')


def schedule(root, jobs, wait):
    root = Path(root)
    wait_for_training(root, wait)
    dev = root / 'dev'
    protocol, sites = read(dev / 'protocol.json'), read(dev / 'sites.json')
    lock_protocol(dev, protocol_for('dev', protocol['samples'], protocol['seed']), create=False)
    if not all(s['qualification']['eligible'] and s['qualification']['just_unloaded']
               and s['qualification']['stratum'] == 'post_unload_primary' for s in sites):
        raise ValueError('only the declared post-unload dev states may be supplemented')
    tasks = missing_reverse_tasks(sites, protocol['samples'], dev)
    run = root / 'scheduler_scratch' / (datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + str(os.getpid()))
    scratch = run / 'dev'
    inputs = stage_inputs(dev, scratch, sites, protocol)
    manifest = dict(version='paired-chain-supplement-v1', authoritative_dev=str(dev),
        source_hashes=protocol['source_hashes'], input_hashes=inputs, jobs=jobs,
        scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scheduled_names=[pair_name(s, r) for s, r in tasks], scheduled_pairs=len(tasks),
        conditional_samples=protocol['samples'], queue_order='reverse of original declared queue',
        publication='independent temporary copy then os.link create-if-absent; existing files never read',
        final_duplicate_comparison='required after main dev and supplemental executors finish',
        exact_comparison_exceptions=TIMING_EXCLUSIONS, independent_test_opened=False)
    write(run / 'manifest.json', manifest)
    outcomes = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        pending = [pool.submit(evaluate_and_publish, s, r, str(scratch), str(dev)) for s, r in tasks]
        for i, future in enumerate(as_completed(pending), 1):
            result = future.result()
            outcomes.append(result)
            print(json.dumps(dict(done=i, total=len(tasks), **result)), flush=True)
    check_sources(protocol)
    write(run / 'schedule_complete.json', dict(complete=True, scheduled=len(tasks), outcomes=outcomes))
    print(json.dumps(dict(status='supplement_complete', scratch=str(run),
        published=sum(r.get('pair_created', False) for r in outcomes),
        duplicate_audit='Run --stage audit after original dev completion.')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['schedule', 'audit'], default='schedule')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = args.output.resolve()
    if args.stage == 'schedule':
        schedule(root, args.jobs, args.wait_for_training)
    else:
        result = audit_supplement(root)
        write(root / 'audit' / 'supplement_duplicate_integrity.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'comparisons'}), flush=True)
        if not result['successful']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
