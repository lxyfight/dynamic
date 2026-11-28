"""Aggregate completed development episodes; never reads a test bank."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from run_feature_screen import FEATURE_SETS, write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def key(row):
    spec = row['world_spec']
    return (spec['layout_id'], spec['day_id'], spec['total'], spec['pattern'], spec.get('temporal','peak'))


def aggregate(rows):
    intervals, feasible_intervals = Counter(), Counter()
    for row in rows:
        intervals.update(row['action_search']['chosen_wait_interval_counts'])
        feasible_intervals.update(row['action_search']['feasible_wait_interval_counts'])
    result = dict(episodes=len(rows), distinct_scenario_days=len({key(r) for r in rows}),
        distinct_layouts=len({r['world_spec']['layout_id'] for r in rows}),
        mean_net=mean(r['net'] for r in rows),
        pickup_counts=[r['realized_pickups'] for r in rows],
        total_pickups=sum(r['realized_pickups'] for r in rows),
        accepted=sum(r['accepted'] for r in rows),
        waits={str(k):intervals[str(k)] for k in [2,5,10]},
        feasible_wait_candidates={str(k):feasible_intervals[str(k)] for k in [2,5,10]})
    for name in ['epochs','feasible_wait_epochs','truncated_epochs','base_search_truncated_epochs',
                 'enhancement_truncated_epochs','wait_search_truncated_epochs',
                 'service_search_truncated_epochs','forced_reject_requests',
                 'wait_patterns_attempted','wait_patterns_accepted',
                 'service_patterns_attempted','service_patterns_accepted','decision_seconds']:
        result[name] = sum(r['action_search'][name] for r in rows)
    result['features'] = {}
    for name in rows[0]['action_search']['feature_diagnostics']:
        values = [r['action_search']['feature_diagnostics'][name] for r in rows]
        result['features'][name] = dict(minimum=min(v['minimum'] for v in values),
            maximum=max(v['maximum'] for v in values),
            nonzero_candidates=sum(v['nonzero_candidates'] for v in values),
            action_varying_epochs=sum(v['action_varying_epochs'] for v in values))
    return result


def first_divergence(left, right):
    for a,b in zip(left['decision_summary'],right['decision_summary']):
        if (a['at'],a['plan_hash']) != (b['at'],b['plan_hash']):
            return dict(none=a, candidate=b)
    if len(left['decision_summary']) != len(right['decision_summary']):
        index = min(len(left['decision_summary']),len(right['decision_summary']))
        return dict(reason='one_decision_sequence_is_a_strict_prefix',
            none=left['decision_summary'][index] if index<len(left['decision_summary']) else None,
            candidate=right['decision_summary'][index] if index<len(right['decision_summary']) else None)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('../../results/feature_screen_20261128'))
    args = parser.parse_args()
    root = args.output.resolve()
    pilot = read(root/'pilot_results.json')
    pilot_table = [dict(N=n, pattern=p, **aggregate([r for r in pilot
        if r['world_spec']['total']==n and r['world_spec']['pattern']==p]))
        for n in [20,40,80] for p in ['uniform','dual']]
    groups, protocols, banks, train_rows = {}, {}, {}, {}
    for alias in FEATURE_SETS:
        directory = root/'training'/alias
        candidate = read(directory/'results.json')[0]
        protocols[alias] = read(directory/'protocol.json')
        selected = next(c for c in candidate['curve'] if c['iteration']==candidate['iteration'])
        banks[alias] = {key(r):r for r in selected['validation']}
        train_rows[alias] = [read_line['episode'] for read_line in
            (json.loads(line) for line in (directory/'training_episodes.jsonl').read_text(encoding='utf8').splitlines())]
        model = candidate['model']
        groups[alias] = dict(feature_set=candidate['feature_set'], selected_iteration=candidate['iteration'],
            validation=aggregate(selected['validation']), training=aggregate(train_rows[alias]),
            coefficients={name:dict(weight=w,scale=s,beta=w/s) for name,w,s in
                zip(model['feature_names'],model['weights'],model['scales'])},
            residual_fit=selected['residual_fit'],
            iterations=[dict(iteration=c['iteration'], validation_mean_net=c['validation_mean_net'],
                residual_fit=c['residual_fit']) for c in candidate['curve']])
    reference = protocols['none']
    for alias, protocol in protocols.items():
        for field in ['worlds','search','source_hashes','seed','ridge','iterations','explore',
                      'operating_rules','review_settings','h_values','arrival_mode']:
            assert protocol[field] == reference[field], (alias,field)
        assert {k:v for k,v in protocol['configuration'].items() if k!='feature_sets'} == {
            k:v for k,v in reference['configuration'].items() if k!='feature_sets'}, alias
        assert not protocol['formal_test_opened']
        assert not protocol['configuration']['test_layouts']
        assert set(banks[alias]) == set(banks['none'])
    paired = []
    for identity, baseline in sorted(banks['none'].items()):
        row = dict(day=identity, pickups=baseline['realized_pickups'], none_net=baseline['net'], groups={})
        for alias, bank in banks.items():
            candidate = bank[identity]
            assert candidate['input_hash'] == baseline['input_hash']
            assert candidate['i0_hash'] == baseline['i0_hash']
            row['groups'][alias] = dict(net=candidate['net'], difference=candidate['net']-baseline['net'],
                accepted=candidate['accepted'], waits=candidate['action_search']['chosen_wait_interval_counts'],
                same_decisions=candidate['canonical_decisions']==baseline['canonical_decisions'],
                same_trace=candidate['trace']==baseline['trace'],
                first_divergence=first_divergence(baseline,candidate))
        paired.append(row)
    for alias, group in groups.items():
        pairs = [r['groups'][alias] for r in paired]
        differences = [p['difference'] for p in pairs]
        group['paired_vs_none'] = dict(mean=mean(differences), differences=differences,
            wins=sum(d>1e-7 for d in differences), ties=sum(abs(d)<=1e-7 for d in differences),
            losses=sum(d< -1e-7 for d in differences),
            same_decisions=sum(p['same_decisions'] for p in pairs), same_trace=sum(p['same_trace'] for p in pairs))
    winner = min(groups, key=lambda alias:(-groups[alias]['validation']['mean_net'],
        len(read(root/'training'/alias/'selected_model.json')['active_features']),alias))
    comparisons = []
    for left,right in combinations(FEATURE_SETS,2):
        daily = []
        for identity in sorted(banks[left]):
            a,b = banks[left][identity],banks[right][identity]
            daily.append(dict(day=identity, right_minus_left=b['net']-a['net'],
                same_decisions=a['canonical_decisions']==b['canonical_decisions'],
                same_trace=a['trace']==b['trace']))
        comparisons.append(dict(left=left,right=right,days=daily,
            same_profit_days=sum(abs(v['right_minus_left'])<=1e-7 for v in daily),
            same_decision_days=sum(v['same_decisions'] for v in daily),
            same_trace_days=sum(v['same_trace'] for v in daily)))
    summary = dict(pilot=pilot_table, groups=groups, paired_validation=paired,
        all_pairwise_comparisons=comparisons,
        selected_by_validation=winner, fairness_checks_passed=True, independent_test_opened=False,
        validation_worlds=reference['worlds']['validation'],
        caveat='Eight scenario-days cover two layouts; paired layouts/patterns/scales are correlated, not eight independent geometries.')
    write(root/'summary.json', summary)
    write(root/'paired_validation.json', paired)
    write(root/'provisional_validation_winner.json',dict(
        status='development_selection_only_not_final_freeze', group=winner,
        candidate_feature_sets=FEATURE_SETS,
        model=read(root/'training'/winner/'selected_model.json'), independent_test_opened=False))
    lines = ['2026-11-28 first-round feature screen (development only)',
        'Independent test data were neither generated nor evaluated.',
        'See campaign_manifest.json for the preregistered split and commands/*.json for exact commands.', '',
        'PILOT: N / pattern / pickup counts / accepted / feasible wait epochs / chosen 2,5,10 / truncations']
    for row in pilot_table:
        lines.append(f"{row['N']} / {row['pattern']} / {row['pickup_counts']} / {row['accepted']} / {row['feasible_wait_epochs']} / {row['waits']} / {row['truncated_epochs']}/{row['epochs']}")
    lines += ['', 'VALIDATION: group / iteration / mean net / paired delta / accepted / chosen waits / truncations / same decisions / same trace']
    for alias, group in groups.items():
        value, pair = group['validation'], group['paired_vs_none']
        lines.append(f"{alias} / {group['selected_iteration']} / {value['mean_net']:.6f} / {pair['mean']:.6f} / {value['accepted']} / {value['waits']} / {value['truncated_epochs']}/{value['epochs']} / {pair['same_decisions']}/8 / {pair['same_trace']}/8")
    lines += ['', 'DAILY PAIRED DIFFERENCES: day / pickups / none net / chain delta / chain_late delta / full delta']
    for row in paired:
        lines.append(f"{row['day']} / {row['pickups']} / {row['none_net']:.6f} / " + ' / '.join(f"{row['groups'][a]['difference']:.6f}" for a in ['chain','chain_late','full']))
    (root/'RESULT_TABLES.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    source_root = Path(__file__).resolve().parent
    write(root/'artifact_hashes.json', {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
