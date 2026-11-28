"""Cross-study audit of raw ledgers, common inputs, and the locked six-feature model."""
from collections import defaultdict
import csv
from math import isclose
from statistics import mean

from e1_sim.initializer import digest
from run_multiscale import ROOT, read, write


TMP = ROOT/'tmp'
PARENT = TMP/'high_demand_screen_v1'
RH = TMP/'frozen_reopt_comparison_v1'
TRAIN = TMP/'frozen6_iter_v1'
WAIT = TMP/'wait_counterfactual_v1'
OUT = TMP/'frozen_followup_validation'
COSTS = ('truck_trip', 'truck_time', 'drone_activation', 'flight', 'lateness', 'rejection')


def same(a, b):
    assert isclose(a, b, rel_tol=1e-12, abs_tol=1e-7), (a, b)


def load_runs(paths, worlds):
    rows = [read(p) for p in sorted(paths)]
    for r in rows:
        w = worlds[tuple(r['case'])]
        assert (r['input_hash'], r['i0_hash']) == (w['input_hash'], w['i0_hash'])
        ledger = r['ledger']
        assert set(ledger) == {'net','revenue',*COSTS}
        assert min(ledger[k] for k in COSTS) >= 0
        same(ledger['revenue'] - sum(ledger[k] for k in COSTS), r['net'])
        same(ledger['net'], r['net'])
        fee = w['configuration']['drone_sortie_cost']
        assert fee == 25 and w['configuration']['truck_handling_setup'] == 0
        same(ledger['drone_activation'] / fee, round(ledger['drone_activation'] / fee))
    return rows


def load_worlds(path, cases):
    worlds = {}
    for case in cases:
        key = tuple(case)
        w = read(path/'worlds'/(digest(key)+'.json'))
        assert digest(w['input']) == w['input_hash']
        assert w['input']['public_information']['information_mode'] == 'unknown_daily_total'
        assert sum(o['kind']=='P' for o in w['input']['orders']) == round(key[1]*.4)
        assert len(w['input']['orders']) == key[1]
        worlds[key] = w
    assert len({w['input_hash'] for w in worlds.values()}) == len(worlds)
    return worlds


def check_comparison(model):
    proto = read(RH/'protocol.json')
    assert digest(proto['model']) == digest(model)
    worlds = load_worlds(RH, proto['cases'])
    paths = [(RH/('rh_verified' if label=='RH' else 'runs')/
              ('_'.join(map(str,(*case,label)))+'.json'))
             for case in worlds for label in ('VFA','RH','Myopic')]
    rows = load_runs(paths, worlds)
    keyed = {(tuple(r['case']),r['policy']):r for r in rows}
    assert len(keyed) == len(rows) == len(worlds)*3
    per_layout = defaultdict(list)
    count = 0
    for case in worlds:
        v, rh, my = (keyed[case,label] for label in ('VFA','RH','Myopic'))
        assert v['model_hash'] == digest(model)
        assert rh['model_hash'] == my['model_hash'] == digest(None)
        per_layout[case[0]].append(v['net']-rh['net'])
        count += len(rh['second_pass'])
    summary = read(RH/'summary.json')
    same(mean(g for ds in per_layout.values() for g in ds), summary['groups']['all']['VFA']['gain_vs_RH'])
    assert count == summary['second_pass']['decisions']
    scalar_dir = RH/'scalar_reference'
    earlier = load_runs((scalar_dir/'runs').glob('*.json'), worlds)
    assert len(earlier) == len(worlds) == len({tuple(r['case']) for r in earlier})
    assert all(r['theta'] == 28 for r in earlier)
    scalar_gain = mean(keyed[tuple(r['case']),'VFA']['net']-r['net'] for r in earlier)
    same(scalar_gain,read(scalar_dir/'comparison.json')['groups']['all']['gain'])
    return len(rows), {s:mean(ds) for s,ds in sorted(per_layout.items())}, scalar_gain


def check_training(model):
    p = read(TRAIN/'protocol.json')
    assert p['parent_model_hash'] == digest(model)
    cases = sum(p['additional_training'], []) + p['validation'] + p['transfer']
    rows = load_runs((TRAIN/'runs').glob('*.json'), load_worlds(TRAIN, cases))
    lookup = {(tuple(r['case']),r['checkpoint'],r['stage']):r for r in rows}
    models = read(TRAIN/'pre_validation_models.json')['models']
    assert models['0'] == digest(model) and len(rows) == len(lookup)
    result = read(TRAIN/'results.json')
    curve = []
    count = p['warm_samples']
    checked = 0
    for c in result['validation_curve']:
        k = c['additional_episodes']
        if k:
            for case in p['additional_training'][k//4-1]:
                row = lookup[tuple(case),k-4,'training']
                assert row['model_hash'] == models[str(k-4)]
                count += sum(not s['closed'] for s in row['samples'])
                checked += 1
        group = [lookup[tuple(case),k,'validation'] for case in p['validation']]
        assert all(r['model_hash'] == models[str(k)] for r in group)
        same(mean(r['net'] for r in group), c['validation']['net'])
        assert count == c['training_samples']
        curve.append([c['total_episodes'], count, c['validation']['net']])
        checked += len(group)
    gains = []
    for case in p['transfer']:
        a,b = [lookup[tuple(case),k,'transfer'] for k in (0,12)]
        assert (a['model_hash'], b['model_hash']) == (models['0'], models['12'])
        gains.append(b['net']-a['net'])
    checked += 2*len(gains)
    assert checked == len(rows) and [row[0] for row in curve] == p['checkpoints_all_episodes']
    same(mean(gains), result['transfer_final_vs_frozen']['gain'])
    return len(rows), curve, gains


def check_waits():
    p = read(WAIT/'protocol.json')
    reference = load_worlds(PARENT, p['cases'])
    rows = load_runs((WAIT/'runs').glob('*.json'), reference)
    baseline = {tuple(r['case']):r for r in rows if r['target'] is None}
    assert set(baseline) == set(reference)
    sites = {(case, s['time']):s for case,r in baseline.items() for s in r['sites']}
    branches = [r for r in rows if r['target'] is not None]
    keys = [(tuple(r['case']),r['target']['time']) for r in branches]
    assert len(branches) == len(set(keys)) and set(keys) == set(sites)
    gains = []
    details = []
    for key,r in zip(keys,branches):
        case,t = key
        ref = baseline[case]
        assert r['target'] == sites[key]
        metrics = r['operation_metrics']['physical'][t]
        assert metrics['actual_departure_delay'] > 0
        gain = r['net']-ref['net']
        same(gain, r['ledger']['revenue']-ref['ledger']['revenue'] -
             sum(r['ledger'][k]-ref['ledger'][k] for k in COSTS))
        gains.append(gain)
        site = sites[key]
        details.append((case[0],case[1],{'uniform':'均匀','dual':'双簇'}[case[2]],
                        float(t),site['station'],site['truck'],site['known_plan_delta'],
                        site['future_value_delta'],ref['net'],r['net'],gain,r['accepted']-ref['accepted'],
                        metrics['interval_new_station_eligible'],metrics['actual_departure_delay']))
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'逐节点等待收益.csv').open('w',encoding='utf-8-sig',newline='') as out:
        writer = csv.writer(out)
        writer.writerow(('布局','全网订单数','空间分布','决策时刻（分钟）','交接站','卡车',
                         '等待减离站的已知计划收益（元）','等待减离站的VFA未来估值（元）',
                         '立即离站全日净收益（元）','预约等待全日净收益（元）','等待增益（元）',
                         '全日多接取件数','随后5分钟新增本站可服务取件数','实际推迟离站（分钟）'))
        writer.writerows(sorted(details))
    return dict(days=len(reference), offers=sum(r['offer_epochs'] for r in baseline.values()),
                pure_sites=len(sites), branches=len(gains), mean_gain=mean(gains),
                wins=sum(d>1e-7 for d in gains), losses=sum(d< -1e-7 for d in gains),
                ties=sum(abs(d)<=1e-7 for d in gains))


def main():
    freeze = read(PARENT/'freeze.json')
    assert freeze['chosen'] == 'shortlist_2'
    model = freeze['model']
    rh_days, gaps, scalar_gain = check_comparison(model)
    train_days, curve, transfer = check_training(model)
    assert not set(gaps) & {c[0] for phase in ('validation','transfer')
                            for c in read(TRAIN/'protocol.json')[phase]}
    waits = check_waits()
    info = read(OUT/'information.json')
    assert info['frozen_hash'] == digest(model) and {r['policy'] for r in info['cases']} == {'VFA','RH'}
    result = dict(frozen_model_hash=digest(model),comparison_replays=rh_days,
                  scalar_reference_replays=16, gain_over_scalar_reference=scalar_gain,
                  gain_over_RH_by_layout=gaps,training_and_evaluation_replays=train_days,
                  validation_curve=curve,transfer_final_minus_frozen=transfer,
                  waiting=waits, public_information_check=info['cases'])
    write(OUT/'verification.json',result)
    print(rh_days, gaps, train_days, curve, transfer, waits)


if __name__ == '__main__':
    main()
