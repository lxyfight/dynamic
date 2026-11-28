"""Independently verify the batch screen and export per-scenario paired results."""
from itertools import combinations
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import math

OUT = Path(__file__).resolve().parents[2]/'tmp/high_demand_screen_v1'
LABELS = ('Myopic','scalar','shortlist_1','shortlist_2')
SHAPES = {'uniform':'均匀','dual':'双簇'}
TOL = 1e-7


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def digest(obj):
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',',':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def equal(a, b):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=TOL):
        raise ValueError(f'{a} differs from {b}')


def main():
    protocol = load(OUT/'protocol.json')
    rank = load(OUT/'all_feature_groups.json')
    freeze = load(OUT/'freeze.json')
    splits = protocol['splits']
    stages = ('fit','value_validation','policy_validation','confirmation')
    assert set(splits)==set(stages)
    all_layouts = sum((splits[stage] for stage in stages),[])
    assert len(set(all_layouts)) == len(all_layouts)

    groups = tuple(protocol['groups'])
    models = freeze['shortlisted']
    combos = {tuple(c) for n in range(len(groups)+1) for c in combinations(groups,n)}
    scores = rank['scores']
    assert {tuple(s['groups']) for s in scores} == combos and len(scores)==len(combos)
    assert set(models)==set(LABELS) and models['scalar'] is None and models['Myopic'] is None
    first = min(scores,key=lambda s:(s['mae'],len(s['columns']),tuple(s['groups'])))
    second = min((s for s in scores if s['columns']!=first['columns']),
                 key=lambda s:(s['rmse'],len(s['columns']),tuple(s['groups'])))
    for idx, (k,s) in enumerate(zip(LABELS[2:],(first,second))):
        assert models[k]==rank['finalists'][idx] and s['columns']==models[k]['columns']
    assert freeze['model']==models[freeze['chosen']]

    expected_paths = set()
    paired = []
    metrics = {}
    seen_marks = {}
    input_cases = {}
    for phase in stages:
        rows = {}
        open_samples = 0
        for layout in splits[phase]:
            # Read actual world and run artifacts using the keys of the campaign.
            for pattern in SHAPES:
                total = 100 if layout%2==0 else 150
                case = (layout,total,pattern)
                world = load(OUT/'worlds'/(digest(case)+'.json'))
                raw = world['input']
                assert world['input_hash']==digest(raw)
                assert world['input_hash'] not in input_cases, f'Duplicate world {case} and {input_cases.get(world["input_hash"])}'
                input_cases[world['input_hash']] = case
                assert world['configuration']['truck_handling_setup']==0
                orders = raw['orders']
                assert len(orders)==total and sum(o['kind']=='P' for o in orders)==round(protocol['share']*total)
                marks = tuple((o['id'],o['kind'],o['weight'],o['release']) for o in orders)
                assert marks==seen_marks.setdefault(layout,marks)
                bank = phase in ('fit','value_validation')
                for label in (('teacher',) if bank else LABELS):
                    key = digest([case,label,None if bank else models[label],bank])
                    path = OUT/'runs'/(key+'.json')
                    expected_paths.add(path)
                    row = load(path)
                    assert tuple(row['case'])==case and row['layout']==layout and row['policy']==label
                    assert (row['input_hash'],row['i0_hash'])==(world['input_hash'],world['i0_hash'])
                    ledger = row['ledger']
                    equal(row['net'],ledger['revenue']-sum(v for k,v in ledger.items() if k not in ('net','revenue')))
                    equal(row['net'],ledger['net'])
                    if bank:
                        assert len(row['samples'])==row['decisions']
                        open_samples += sum(not s['closed'] for s in row['samples'])
                    else:
                        assert not row['samples']
                    rows[case,label] = row
        if phase in ('fit','value_validation'):
            prefix = 'train' if phase=='fit' else 'validation'
            assert rank[f'{prefix}_days']==len(rows) and rank[f'{prefix}_samples']==open_samples
            continue
        recorded = load(OUT/(phase+'.json'))
        for label in LABELS:
            policy = [row for (case,k),row in rows.items() if k==label]
            for n in ('all','100','150'):
                cell = [r for r in policy if n=='all' or r['case'][1]==int(n)]
                delta = [r['net']-rows[tuple(r['case']),'Myopic']['net'] for r in cell]
                stats = dict(cases=len(cell),net=mean(r['net'] for r in cell),gain=mean(delta),
                             wins=sum(v>TOL for v in delta),losses=sum(v< -TOL for v in delta),
                             ties=sum(abs(v)<=TOL for v in delta))
                saved = recorded[label][n]
                equal(stats['net'],saved['net'])
                equal(stats['gain'],saved['gain'])
                assert (stats['cases'],stats['wins'],stats['losses'])==(saved['n'],saved['wins'],saved['losses'])
                for layout in {r['layout'] for r in cell}:
                    layout_delta = [d for r,d in zip(cell,delta) if r['layout']==layout]
                    equal(mean(layout_delta),saved['layout_gains'][str(layout)])
                metrics.setdefault(phase,{}).setdefault(label,{})[n] = stats
            if phase!='confirmation':
                continue
            for r in policy:
                case = tuple(r['case'])
                base = rows[case,'Myopic']
                diff = r['net']-base['net']
                paired.append({
                    '布局':case[0],'全网订单数':case[1],'动态度':protocol['share'],'空间分布':SHAPES[case[2]],
                    '策略':label,'冻结策略':int(label==freeze['chosen']),
                    '全程净收益（元）':r['net'],'Myopic净收益（元）':base['net'],'配对增益（元）':diff,
                    '与Myopic比较':'胜' if diff>TOL else '负' if diff< -TOL else '平',
                    '取件接单数':r['accepted'],'车趟数':r['operation_metrics']['trips'],
                    '新增主动等待次数':r['operation_metrics']['new_wait_chosen'],
                    '存在新增等待候选的决策次数':r['operation_metrics']['decisions_with_new_wait'],
                    '实例哈希':r['input_hash'],'初始计划哈希':r['i0_hash']})

    assert set((OUT/'runs').glob('*.json'))==expected_paths
    assert freeze['validation']==load(OUT/'policy_validation.json')
    validated = metrics['policy_validation']
    best = max(v['all']['net'] for v in validated.values())
    priority = sorted(LABELS,key=lambda k:(0 if k=='Myopic' else 1 if k=='scalar' else 2,
                                          len((models[k] or {}).get('columns',[]))))
    assert freeze['chosen']==next(k for k in priority if validated[k]['all']['net']>=best-1)
    assert len(paired)==len(splits['confirmation'])*len(SHAPES)*len(LABELS)
    paired.sort(key=lambda r:(r['布局'],r['空间分布'],LABELS.index(r['策略'])))
    path = OUT/'paired_results.csv'
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    temp.replace(path)
    print(json.dumps(dict(frozen=freeze['chosen'],raw_rollouts=len(expected_paths),
        feature_sets=len(combos),csv_rows=len(paired),confirmation=metrics['confirmation']),
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
