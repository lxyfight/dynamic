"""Audit and explain the completed 32-day pilot without rerunning simulations.

No partial report is produced: all 32 case files, the frozen design, progress
and final summary must agree before analysis.json and REPORT.md are written.
This reporting entry point is outside the pilot's numerical-source manifest.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean, median

from e1_sim.initializer import digest
from e1_sim.kernel import COSTS
from e1_sim.system_wait import SystemWaitRule
from run_system_wait_pilot import DEFAULT_OUT, ROOT, source_hashes


TOL = 1e-7
SPEC_FIELDS = ('layout_id', 'day_id', 'total', 'share', 'pattern')
CHINESE_COSTS = dict(revenue='履约收入', truck_trip='卡车里程与通行费',
    truck_time='卡车在外时间费', drone_activation='无人机航次费',
    flight='无人机飞行费', lateness='返仓迟到罚金', rejection='拒单损失')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spec_key(spec):
    return tuple(spec[k] for k in SPEC_FIELDS)


def stats(values):
    values = list(values)
    return dict(n=len(values), mean=mean(values) if values else None,
                median=median(values) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                wins=sum(v > TOL for v in values),
                ties=sum(abs(v) <= TOL for v in values),
                losses=sum(v < -TOL for v in values))


def groups(items, value_key='gain'):
    """Report denominators for marginals and the eight complete design cells."""
    result = {'all': stats(x[value_key] for x in items)}
    for field in ('pattern', 'total', 'share'):
        result[field] = {str(value): stats(x[value_key] for x in items if x['spec'][field] == value)
                         for value in sorted({x['spec'][field] for x in items})}
    result['cells'] = []
    for pattern, total, share in product(('uniform', 'dual'), (100, 150), (.2, .6)):
        subset = [x for x in items if (x['spec']['pattern'], x['spec']['total'], x['spec']['share'])
                  == (pattern, total, share)]
        result['cells'].append(dict(pattern=pattern, total=total, share=share,
                                    **stats(x[value_key] for x in subset)))
    return result


def read_trace(out, case_id, name, expected_hash):
    path = out/'traces'/case_id/name
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        trace = json.load(stream)
    require(digest(trace) == expected_hash, f'trace hash mismatch: {path}')
    return trace


def inspect_extreme(out, row, pair):
    case_id = digest(row['spec'])[:20]
    baseline = read_trace(out, case_id, 'baseline.json.gz', row['baseline']['trace_hash'])
    waited = read_trace(out, case_id, f"wait_{pair['requested_h']}.json.gz", pair['outcome']['trace_hash'])
    def sorties(trace):
        return {(e['actor'], tuple(e['orders'])): e for e in trace if e['kind'] == 'drone_takeoff'}
    before, after = sorties(baseline), sorties(waited)
    owner_before = {oid: key for key in before for oid in key[1]}
    merged = [e for key, e in after.items()
              if len({owner_before[o] for o in key[1] if o in owner_before}) >= 2]
    owner_after = {oid: key for key in after for oid in key[1]}
    split = [e for key, e in before.items()
             if len({owner_after[o] for o in key[1] if o in owner_after}) >= 2]
    accepted_before = {e['order'] for e in baseline if e['kind'] == 'accept'}
    accepted_after = {e['order'] for e in waited if e['kind'] == 'accept'}
    costs = pair['ledger_delta']
    effects = {k: costs[k] if k == 'revenue' else -costs[k]
               for k in ('revenue',) + COSTS}
    return dict(spec=row['spec'], case_id=case_id, sample_at=row['sampled_state']['at'],
        truck=row['sampled_state']['truck'], visit_id=row['sampled_state']['visit_id'],
        requested_h=pair['requested_h'], h=pair['h'], gain=pair['gain'],
        known_plan_delta=pair['known_plan_delta'], residual_target=pair['residual_target'],
        accepted_delta=pair['accepted_delta'], ledger_delta=costs, profit_effects=effects,
        truck_trips_delta=pair['outcome']['truck_trips']-row['baseline']['truck_trips'],
        sorties_delta=len(after)-len(before), newly_accepted=sorted(accepted_after-accepted_before),
        no_longer_accepted=sorted(accepted_before-accepted_after),
        combined_sorties=sorted(merged, key=lambda e: (e['at'], e['actor'])),
        split_baseline_sorties=sorted(split, key=lambda e: (e['at'], e['actor'])),
        baseline_only_sorties=sorted((before[k] for k in before.keys()-after.keys()),
                                    key=lambda e: (e['at'], e['actor'])),
        wait_only_sorties=sorted((after[k] for k in after.keys()-before.keys()),
                                key=lambda e: (e['at'], e['actor'])))


def analyze(out):
    out = Path(out).resolve()
    protocol = read(out/'protocol.json')
    expected = set(product((610, 611), (0, 1), (100, 150), (.2, .6), ('uniform', 'dual')))
    declared = [spec_key(s) for s in protocol['cases']]
    require(len(declared) == 32 and set(declared) == expected, 'protocol is not the complete frozen 32-day design')
    progress = read(out/'progress.json')
    require(progress.get('completed') == progress.get('expected') == 32,
            f"pilot incomplete ({progress.get('completed')}/32); no final artifacts written")
    require((out/'summary.json').exists(), 'final summary is missing; no final artifacts written')
    summary = read(out/'summary.json')
    require(summary.get('completed_days') == 32 and summary.get('independent_layouts') == 2,
            'final summary is not the complete two-layout pilot')
    files = sorted((out/'cases').glob('*.json'))
    require(len(files) == 32, f'expected 32 completed case files, found {len(files)}')
    current_sources = source_hashes()
    require(current_sources == protocol['sources'], 'current numerical sources differ from the frozen protocol')
    model_hash = digest(read(ROOT/'results/frozen_six.json'))
    require(model_hash == protocol['six_model_hash'], 'frozen six-feature model changed')
    source_hash = digest(current_sources)
    rule = SystemWaitRule(**protocol['rule'])
    temporal_rule = replace(rule, signal='arrivals')
    rows, worlds = [], {}
    ledger_error = 0.
    h_checks, cadence_cache, interval_histogram = 0, {}, Counter()
    repeated_rows, system_rows, recurrence = [], [], []
    outcomes_checked, pairs_checked = 0, 0

    def close(a, b, label):
        nonlocal ledger_error
        err = abs(float(a)-float(b))
        ledger_error = max(ledger_error, err)
        require(err <= TOL, f'{label}: {a} != {b}')

    def outcome_check(row, label):
        nonlocal outcomes_checked
        require(row['terminal'] and row['audited'], f'{label}: terminal/audit flag failed')
        ledger = row['ledger']
        close(row['net'], ledger['revenue']-sum(ledger[k] for k in COSTS), label+' ledger')
        close(row['net'], ledger['net'], label+' net')
        close(row['active_wait_minutes'], sum(v['active_wait_minutes'] for v in row['station_visits']),
              label+' active wait minutes')
        require(row['active_reviews'] == sum(v['review_count'] for v in row['station_visits']),
                label+' active review count mismatch')
        outcomes_checked += 1

    def cadence_check(public, recorded, active_rule, label):
        nonlocal h_checks
        key = (digest(public), recorded['at'], active_rule.signal)
        if key not in cadence_cache:
            cadence_cache[key] = active_rule.decision(public, recorded['at'])
        computed = cadence_cache[key]
        for field in ('h', 'mode', 'signal', 'reason', 'opportunities_open'):
            require(computed[field] == recorded[field], f'{label}: public cadence {field} mismatch')
        require(computed['expected_by_h'].keys() == recorded['expected_by_h'].keys(),
                f'{label}: cadence grid mismatch')
        require(abs(computed['forecast_daily_total']-recorded['forecast_daily_total']) < 1e-10,
                f'{label}: public forecast mismatch')
        require(all(abs(v-recorded['expected_by_h'][h]) < 1e-10
                    for h, v in computed['expected_by_h'].items()), f'{label}: public ready counts mismatch')
        h_checks += 1

    def decision_checks(outcome, public, label, count_histogram=False):
        for entry in outcome['wait_decisions']:
            cadence_check(public, entry['cadence'], rule, label)
            if count_histogram:
                interval_histogram[entry['cadence']['h']] += 1
            for pair in entry['pairs']:
                close(pair['requested_h'], entry['cadence']['h'], label+' common requested h')
                close(pair['h'], min(entry['cadence']['h'], public['revelation_end']-entry['at']),
                      label+' common realized h')
            for diagnostic in entry['diagnostics']:
                close(diagnostic['requested_h'], entry['cadence']['h'], label+' diagnostic h')

    for file in files:
        row = read(file)
        require(spec_key(row['spec']) in expected, f'unexpected case: {file.name}')
        case_id = digest(row['spec'])[:20]
        require(file.stem == case_id, f'case filename/spec mismatch: {file.name}')
        world = read(out/'worlds'/file.name)
        worlds[spec_key(row['spec'])] = world
        require(spec_key(world['spec']) == spec_key(row['spec']), f'world/case spec mismatch: {file.name}')
        require(row['source_hash'] == source_hash and row['model_hash'] == model_hash,
                f'source/model mismatch: {file.name}')
        for field in ('input_hash', 'i0_hash', 'public_hash'):
            require(row[field] == world[field], f'world binding mismatch {field}: {file.name}')
        require(world['input_hash'] == digest(world['input']), f'world content mismatch: {file.name}')
        public = world['input']['public_information']
        require(world['public_hash'] == digest(public), f'public law hash mismatch: {file.name}')
        times_weights = [(o['id'], o['weight'], o['release']) for o in world['input']['orders']]
        require(world['times_weights_hash'] == digest(times_weights), f'time/weight hash mismatch: {file.name}')
        outcome_check(row['baseline'], file.name+' baseline')
        require(row['baseline']['active_reviews'] == 0, f'baseline created active waits: {file.name}')
        decision_checks(row['baseline'], public, file.name+' baseline', count_histogram=True)
        sample = row['sampled_state']
        if sample is None:
            require(not row['pairs'] and row['repeated_diagnostic'] is None, 'unbound interventions without sampled state')
        else:
            require(sample['zero_intervention_verified'], f'zero-intervention verification missing: {file.name}')
            cadence_check(public, sample['cadence'], rule, file.name+' sampled')
            cadence_check(public, sample['temporal_only_cadence'], temporal_rule, file.name+' temporal')
            observed_intervals = [pair['requested_h'] for pair in row['pairs']]
            required_intervals = set(protocol['fixed_intervals']) | {
                sample['cadence']['h'], sample['temporal_only_cadence']['h']}
            require(len(observed_intervals) == len(set(observed_intervals)) and
                    set(observed_intervals) == required_intervals, f'incomplete intervention grid: {file.name}')
        system = []
        for pair in row['pairs']:
            if not pair['feasible']:
                continue
            outcome_check(pair['outcome'], file.name+f" h={pair['requested_h']}")
            close(pair['gain'], pair['outcome']['net']-row['baseline']['net'], 'pair net difference')
            for component in ('net', 'revenue') + COSTS:
                close(pair['ledger_delta'][component],
                      pair['outcome']['ledger'][component]-row['baseline']['ledger'][component],
                      'pair ledger component '+component)
            close(pair['gain'], pair['ledger_delta']['revenue']-
                  sum(pair['ledger_delta'][k] for k in COSTS), 'pair decomposed profit')
            close(pair['residual_target'], pair['gain']-pair['known_plan_delta'], 'paired residual target')
            require(pair['accepted_delta'] == pair['outcome']['accepted']-row['baseline']['accepted'],
                    'accepted count difference mismatch')
            require(pair['id'] == digest([world['input_hash'], pair['identity']]), 'pair input/identity binding mismatch')
            identity = pair['identity']
            require(identity['truck'] == sample['truck'] and identity['visit_id'] == sample['visit_id']
                    and identity['base_plan'] == sample['base_plan_hash'], 'pair not from the common sampled base')
            close(identity['at'], sample['at'], 'pair sampled clock')
            close(identity['h'], pair['h'], 'pair interval identity')
            close(pair['h'], min(pair['requested_h'], public['revelation_end']-sample['at']), 'pair horizon clipping')
            require(pair['is_system_interval'] == (pair['requested_h'] == sample['cadence']['h'])
                    and pair['is_temporal_interval'] == (pair['requested_h'] == sample['temporal_only_cadence']['h']),
                    'interval role mismatch')
            if pair['is_system_interval']:
                system.append(pair)
                system_rows.append(dict(spec=row['spec'], case_id=case_id, gain=pair['gain']))
            pairs_checked += 1
        require(len(system) == (1 if sample else 0), f'sampled state lacks unique feasible system intervention: {file.name}')
        repeated = row['repeated_diagnostic']
        if repeated is not None:
            outcome_check(repeated, file.name+' repeated')
            decision_checks(repeated, public, file.name+' repeated')
            close(repeated['gain'], repeated['net']-row['baseline']['net'], 'repeated gain')
            visits = [m for m in repeated['station_visits'] if m['visit_id'] == sample['visit_id']]
            require(len(visits) == 1, 'repeated diagnostic lacks a unique actual station visit')
            selected, observed_counts = [], set()
            for entry in repeated['wait_decisions']:
                for pair in entry['pairs']:
                    if pair['identity']['visit_id'] == sample['visit_id']:
                        observed_counts.add(pair['memory']['review_count'])
                if entry['selected'] is not None:
                    chosen = next(p for p in entry['pairs'] if p['id'] == entry['selected'])
                    require(chosen['identity']['visit_id'] == sample['visit_id'], 'repeat changed station visit')
                    selected.append(dict(at=entry['at'], prior_reviews=chosen['memory']['review_count'],
                        prior_active_minutes=chosen['memory']['active_wait_minutes']))
            require(repeated['continuation_forced_reviews'] == len(selected) <= 2,
                    'repeated diagnostic exceeds declared continuation budget')
            require(visits[0]['review_count'] == 1+len(selected) <= 3,
                    'actual installed reviews do not match repeated interventions')
            require([e['prior_reviews'] for e in selected] == list(range(1, 1+len(selected))),
                    'missing/nonsequential repeated-wait memory')
            recurrence.append(dict(spec=row['spec'], case_id=case_id,
                actual_reviews=visits[0]['review_count'], selected_states=selected,
                observed_prior_review_counts=sorted(observed_counts),
                active_wait_minutes=visits[0]['active_wait_minutes'], gain=repeated['gain']))
            repeated_rows.append(dict(spec=row['spec'], case_id=case_id, gain=repeated['gain']))
        rows.append(row)
    require({spec_key(r['spec']) for r in rows} == expected and len(worlds) == 32, 'duplicate/missing design cases')

    spatial_pairs = 0
    day_pairs = 0
    for layout, day, total, share in product((610, 611), (0, 1), (100, 150), (.2, .6)):
        uniform, dual = [worlds[(layout, day, total, share, p)] for p in ('uniform', 'dual')]
        require(uniform['times_weights_hash'] == dual['times_weights_hash'], 'spatial comparison changed times/weights')
        require(uniform['network'] == dual['network'], 'spatial comparison changed the network')
        require(uniform['seeds']['releases'] == dual['seeds']['releases'] and
                uniform['seeds']['weights'] == dual['seeds']['weights'], 'spatial comparison changed common RNG streams')
        for band in range(3):
            rates = [sum(rows_[band]['rate'] for rows_ in w['input']['public_information']['demand_model']['rates_by_city'].values())
                     for w in (uniform, dual)]
            close(*rates, 'spatial comparison network-wide arrival rate')
        spatial_pairs += 1
    for layout, total, share, pattern in product((610, 611), (100, 150), (.2, .6), ('uniform', 'dual')):
        a, b = [worlds[(layout, day, total, share, pattern)] for day in (0, 1)]
        for stream in ('releases', 'marks', 'weights'):
            require(a['seeds'][stream] != b['seeds'][stream], 'different operating days reused '+stream+' seed')
        require(a['input_hash'] != b['input_hash'] and a['times_weights_hash'] != b['times_weights_hash'],
                'different days reused the realized input')
        day_pairs += 1
    for layout, pattern in product((610, 611), ('uniform', 'dual')):
        laws = {w['public_hash'] for key, w in worlds.items() if key[0] == layout and key[4] == pattern}
        require(len(laws) == 1, 'public forecast was conditioned on evaluated N/d/day')

    close(summary['baseline_mean_net'], mean(r['baseline']['net'] for r in rows), 'summary baseline mean')
    close(summary['baseline_mean_accepted'], mean(r['baseline']['accepted'] for r in rows), 'summary acceptance mean')
    require(summary['days_with_pair'] == len(system_rows), 'summary sampled-day count mismatch')
    require(summary['decision_epochs'] == sum(interval_histogram.values()), 'summary decision count mismatch')
    require(summary['interval_histogram'] == {str(k):v for k,v in interval_histogram.items()}, 'summary h histogram mismatch')
    require(summary['single_intervention_system']['n'] == len(system_rows), 'summary system denominator mismatch')
    if system_rows:
        close(summary['single_intervention_system']['mean_gain'], mean(x['gain'] for x in system_rows), 'summary system mean')

    comparisons = {}
    for name, predicate in [('fixed_'+str(h), lambda p, h=h: p['requested_h'] == h)
                            for h in protocol['fixed_intervals']] + [('temporal_only', lambda p: p['is_temporal_interval'])]:
        common = []
        excluded = []
        for row in rows:
            system = [p for p in row['pairs'] if p.get('feasible') and p.get('is_system_interval')]
            other = [p for p in row['pairs'] if p.get('feasible') and predicate(p)]
            if not system or not other:
                excluded.append(dict(spec=row['spec'], reason='no feasible system or comparator pair at sampled state'))
                continue
            require(len(system) == len(other) == 1, 'ambiguous paired comparator')
            common.append(dict(spec=row['spec'], case_id=digest(row['spec'])[:20],
                system_gain=system[0]['gain'], comparator_gain=other[0]['gain'],
                gain_difference=system[0]['gain']-other[0]['gain'],
                system_h=system[0]['h'], comparator_h=other[0]['h']))
        grouped = groups(common, 'gain_difference')
        by_spatial_pair = defaultdict(dict)
        for item in common:
            by_spatial_pair[spec_key(item['spec'])[:4]][item['spec']['pattern']] = item['gain_difference']
        spatial_contrasts = [dict(layout_id=k[0], day_id=k[1], total=k[2], share=k[3],
            dual_minus_uniform=v['dual']-v['uniform']) for k,v in by_spatial_pair.items() if set(v) == {'uniform','dual'}]
        comparisons[name] = dict(statistics=grouped,
            system_mean_on_common=mean(x['system_gain'] for x in common) if common else None,
            comparator_mean_on_common=mean(x['comparator_gain'] for x in common) if common else None,
            rows=common, excluded=excluded,
            spatial_gain_interaction=stats(x['dual_minus_uniform'] for x in spatial_contrasts),
            spatial_gain_interaction_rows=spatial_contrasts)

    system_pairs = [(r,p) for r in rows for p in r['pairs'] if p.get('feasible') and p.get('is_system_interval')]
    extremes = {}
    if system_pairs:
        extremes['maximum_gain'] = inspect_extreme(out, *max(system_pairs, key=lambda rp: rp[1]['gain']))
        extremes['minimum_gain'] = inspect_extreme(out, *min(system_pairs, key=lambda rp: rp[1]['gain']))
    mechanisms = {}
    for component in ('revenue',) + COSTS:
        effects = [p['ledger_delta'][component]*(1 if component == 'revenue' else -1) for _,p in system_pairs]
        mechanisms[component] = dict(chinese_name=CHINESE_COSTS[component], **stats(effects))
    analysis = dict(status='complete_verified', experiment='system_wait_pilot_20261125',
        protocol_sha256=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit=dict(design_days=32, independent_layouts=2, spatial_pairs_checked=spatial_pairs,
            independent_day_pairs_checked=day_pairs, source_manifest_matches=True,
            frozen_model_matches=True, world_input_and_i0_bindings_match=True,
            public_forecast_independent_of_evaluated_N_d_day=True, feasible_pairs_checked=pairs_checked,
            outcomes_checked=outcomes_checked, public_cadences_recomputed=h_checks,
            distinct_cadences_recomputed=len(cadence_cache), max_numeric_identity_error=ledger_error,
            note='I0 hashes are bound to saved worlds; this analysis does not regenerate initialization or rerun physical simulations.'),
        design=dict(total_days=32, independent_layouts=2, days_with_sample=len(system_rows),
            rule=protocol['rule'], fixed_intervals=protocol['fixed_intervals'],
            snapshot_time_histogram=dict(Counter(str(r['sampled_state']['target_time']) for r in rows if r['sampled_state'])),
            snapshot_fallback_days=sum(not r['sampled_state']['target_reached'] for r in rows if r['sampled_state']),
            interval_histogram={str(k):v for k,v in sorted(interval_histogram.items())}),
        single_intervention_system=groups(system_rows), common_case_comparisons=comparisons,
        repeated_diagnostic=groups(repeated_rows),
        recurrence=dict(days=len(recurrence), days_with_two_actual_reviews=sum(x['actual_reviews']>=2 for x in recurrence),
            days_with_three_actual_reviews=sum(x['actual_reviews']>=3 for x in recurrence),
            days_observing_second_wait_state=sum(any(n>=1 for n in x['observed_prior_review_counts']) for x in recurrence),
            days_observing_third_wait_state=sum(any(n>=2 for n in x['observed_prior_review_counts']) for x in recurrence), rows=recurrence),
        mean_profit_components=mechanisms, extremes=extremes,
        limitations=[
            'Single intervention per sampled operating day, followed by the same frozen dispatcher; not a complete learned waiting policy.',
            'Sampled opportunities are conditioned on feasibility of the system-rule wait; this can favor that interval relative to others and excludes unsampled station states.',
            'Fixed and temporal comparisons use only common feasible cases; each denominator is shown separately.',
            'The repeated diagnostic forces at most three reviews on one visit; it is not a performance claim for a deployed policy.',
            'Only two independent layouts; no confidence intervals, significance claims, or fixed-interval selection from these pilot outcomes.',
            'Each day contributes one sampled state; subgroup differences can also reflect sampled times and feasibility selection, not only N/d/spatial effects.',
            'Public forecasts come from the same independent historical mixture across evaluated N/d cells; changing realized N/d does not disclose or refit that day\'s future intensity.',
            'The spatial readiness signal assumes an idle drone at the nearest eligible station; it is a geometric opportunity proxy, not live pickup/truck feasibility.',
            'Positive residual targets can reflect interrupted waits and avoided forecast costs; they do not by themselves prove positive realized waiting profit.'])
    return analysis


def number(value, digits=3):
    return '无可比样本' if value is None else f'{value:.{digits}f}'


def report_markdown(a):
    main = a['single_intervention_system']['all']
    design, audit = a['design'], a['audit']
    text = ['# 系统统一等待间隔：32 日先导核验与机制分析', '',
        f"32 个预定运营日全部完成，来自 2 个独立布局。共有 {design['days_with_sample']} 日取得可行的系统间隔配对。",
        f"在这些采样状态上，单次系统间隔干预相对立即离站的平均全日收益差为 {number(main['mean'])} 元，"
        f"{main['wins']} 胜、{main['ties']} 平、{main['losses']} 负。", '',
        '这是一日一个状态的反事实干预：先固定六特征调度器选出的同一接拒与服务方案，比较立即离站和等待一次，'
        '然后使用相同的冻结调度器完成全日，后续不再主动等待未知需求。下表不是完整自适应等待策略的整日排名，也没有训练新的等待评分器。', '',
        '## 核验范围', '',
        f"- 32 日设计、当前数值源码、冻结模型以及各案例与保存 world 的 input/I0/public 哈希一致。",
        f"- {audit['spatial_pairs_checked']} 对均匀/双簇算例共用地图、释放时刻和重量；各分布按各自坐标构造可行 I0。",
        f"- {audit['independent_day_pairs_checked']} 对不同运营日采用不同释放、位置和重量随机流。",
        f"- 核对 {audit['feasible_pairs_checked']} 个可行配对和 {audit['outcomes_checked']} 个保存的运营结果，"
        f"账本差、净收益差及残差标签恒等式的最大数值误差为 {audit['max_numeric_identity_error']:.3g}。",
        f"- 从保存的公开需求规律重算 {audit['public_cadences_recomputed']} 条间隔记录，"
        '同一时刻各车候选使用同一系统间隔，截尾仅由公开揭示期末决定。',
        '- 本分析核对保存的 I0 哈希绑定；未重新生成 I0，也未重跑物理仿真。原执行程序逐条进行了终态、履约和独立账本检查。', '',
        '## 统一间隔如何变化', '',
        '系统规则由公开的分时到达强度与独立历史客户位置形成的几何取件就绪时间分布决定。'
        f"在 {design['rule']['min_h']}—{design['rule']['max_h']} 分钟中选择最小整数 h，"
        f"使全网未来新增取件的预计几何就绪数量达到 {design['rule']['target_expected']}；达不到时取上限。"
        '这个 h 控制下次复核，不强迫卡车等待；新请求或货物就绪事件仍可提前打断。', '',
        '| 基线决策中使用的 h（分钟） | 出现次数 |', '| ---: | ---: |']
    text += [f'| {h} | {n} |' for h,n in design['interval_histogram'].items()]
    text += ['', '几何就绪假设最近可服务站有空闲无人机，不包括实时排队、车辆余载和赶上具体回收车的条件。'
        '它仅决定全系统复核节奏。运营日的 N、动态度和实际取件总数不进入该规则；'
        '公开强度来自统一的独立历史混合样本，因此此先导也包含历史预测与当日实际负荷不同的情况。', '',
        '## 单次系统间隔干预的分组结果', '',
        '单位为元；每行 n 是该组可行系统干预日数。负值表示等待一次后全日净收益低于立即离站。', '',
        '| 空间分布 | N | 动态度 | n | 平均收益差 | 胜/平/负 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in a['single_intervention_system']['cells']:
        text.append(f"| {'均匀' if r['pattern']=='uniform' else '双簇'} | {r['total']} | {r['share']} | {r['n']} | "
                    f"{number(r['mean'])} | {r['wins']}/{r['ties']}/{r['losses']} |")
    text += ['', '## 与固定间隔及仅时间强度规则的配对差', '',
        '每行只保留系统规则和该对照均可行的同一运营日、同一采样状态。'
        '“系统−对照”是两次干预全日收益差的直接差值；同一行的均值使用相同分母。'
        '不从这批先导结果选出“最佳固定间隔”。', '',
        '| 对照 | 共同可行 n | 系统干预均值 | 对照干预均值 | 系统−对照 | 胜/平/负 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for key, r in a['common_case_comparisons'].items():
        s = r['statistics']['all']
        name = '仅时间强度' if key == 'temporal_only' else f"固定 {key.split('_')[1]} 分钟"
        text.append(f"| {name} | {s['n']} | {number(r['system_mean_on_common'])} | {number(r['comparator_mean_on_common'])} | "
                    f"{number(s['mean'])} | {s['wins']}/{s['ties']}/{s['losses']} |")
    text += ['', '采样在预定的 0/200/400 分钟目标后寻找第一个“系统间隔等待可行”的机会，'
        f"未达到目标时使用首个机会（本批 {design['snapshot_fallback_days']} 日）。"
        '因此采样条件本身依赖系统规则的可行性，不能据此推断全部站点状态下的无条件策略优劣。'
        '各分组的共同分母、被排除案例和双簇−均匀增益差完整保存在 analysis.json。', '',
        '## 收益由哪些项目改变', '',
        '下表列系统单次干预的平均利润贡献，正值增加利润；成本项目已经取负号。', '',
        '| 账本项目 | 平均利润贡献（元） |', '| --- | ---: |']
    for component, row in a['mean_profit_components'].items():
        text.append(f"| {row['chinese_name']} | {number(row['mean'])} |")
    for label, title in [('maximum_gain', '系统单次干预收益差最大案例'), ('minimum_gain', '系统单次干预收益差最小案例')]:
        if label not in a['extremes']:
            continue
        e = a['extremes'][label]
        s = e['spec']
        text += ['', f'### {title}', '',
            f"布局 {s['layout_id']}、运营日 {s['day_id']}、N={s['total']}、动态度 {s['share']}、"
            f"{'均匀' if s['pattern']=='uniform' else '双簇'}；时刻 {number(e['sample_at'])} 对 {e['truck']} "
            f"施加一次 h={number(e['h'], 0)} 分钟干预，全日收益差 {number(e['gain'])} 元。"
            f"接单数变化 {e['accepted_delta']:+d}，车趟数变化 {e['truck_trips_delta']:+d}，无人机航次数变化 {e['sorties_delta']:+d}。", '',
            '| 变化项目 | 对净收益的贡献（元） |', '| --- | ---: |']
        for k,v in e['profit_effects'].items():
            if abs(v) > TOL:
                text.append(f'| {CHINESE_COSTS[k]} | {number(v)} |')
        text.append('')
        if e['combined_sorties']:
            details = ['、'.join(x['orders'])+f"（{x['actor']}，{number(x['at'])} 分钟起飞）"
                       for x in e['combined_sorties'][:3]]
            text.append('轨迹中出现原本分属不同航次的订单合并：'+'；'.join(details)+'。'
                        '这支持航次合并的解释，但不能仅凭等候地点把所有收益归为本站新取件赶上当前车；下游补货和配送时刻也可能受影响。')
        if e['split_baseline_sorties']:
            details = ['、'.join(x['orders']) for x in e['split_baseline_sorties'][:3]]
            text.append('基线中的部分组合在干预后分属多个航次：'+'；'.join(details)+'。')
        if not e['combined_sorties'] and not e['split_baseline_sorties']:
            text.append('当前轨迹差异未直接显示跨原航次的合并或拆分；上述解释限定于实际账本项目变化，不推断未验证的赶车或资源稀缺机制。')
        text.append(f"已知计划差 ΔJ={number(e['known_plan_delta'])} 元，残差标签 Y={number(e['residual_target'])} 元。"
                    'Y 为正也可能来自等待被提前打断而避免了原计划费用，不能把残差正值当作等待实际盈利。')
    recurrence = a['recurrence']
    rstat = a['repeated_diagnostic']['all']
    text += ['', '## 连续等待是否真正覆盖', '',
        f"{recurrence['days']} 日完成独立连续探索；其中 {recurrence['days_with_two_actual_reviews']} 日实际安装至少 2 次复核，"
        f"{recurrence['days_with_three_actual_reviews']} 日安装 3 次。"
        f"记录到第二次等待决策状态的日数为 {recurrence['days_observing_second_wait_state']}，"
        f"第三次为 {recurrence['days_observing_third_wait_state']}。"
        '核验了同次停靠 visit_id、真实累计主动等待分钟和逐次增加的 review_count。', '',
        f"这条最多 3 次复核的强制探索轨迹平均收益差为 {number(rstat['mean'])} 元，"
        f"{rstat['wins']} 胜、{rstat['ties']} 平、{rstat['losses']} 负。"
        '它用于产生续等训练状态，不作为已学策略的性能结果。', '',
        '## 可据此决定的下一步', '',
        '本先导可以检查候选是否生成、统一间隔是否随公开需求分布变化，以及等待影响哪些实际费用。'
        '后续训练应将首次与续等状态同时纳入，保留 ΔΠ−ΔJ 的增量标签，并按独立布局划分训练、验证和测试。'
        '是否等待由评分器判断；系统 h 的规则、目标及范围应在正式测试前冻结。', '',
        '本批只有 2 个独立布局，不提供确认性置信区间或显著性结论；同一运营日多个 h 的回放也不能视为独立运营样本。'
        '每个运营日仅抽取一个状态，分组差异也可能反映采样时刻与可行性筛选，不能直接解释为规模、动态度或空间分布的单独因果效应。'
        '完整日策略比较、验证集选择固定间隔、正式训练及最终测试仍未由本先导完成。', '',
        '数据入口：[逐日结果](cases/) · [原始汇总](summary.json) · [机器可读核验与共同样本比较](analysis.json) · [冻结方案](protocol.json)', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check-only', action='store_true', help='Require the complete design and audit, but write no artifacts.')
    args = parser.parse_args()
    try:
        result = analyze(args.input)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    if not args.check_only:
        out = args.input.resolve()
        outputs = {'analysis.json': json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                   'REPORT.md': report_markdown(result)}
        for name, content in outputs.items():
            temporary = out/(name+'.tmp')
            temporary.write_text(content, encoding='utf-8')
            temporary.replace(out/name)
    print(json.dumps(dict(status=result['status'], audit=result['audit'],
        system_single_intervention=result['single_intervention_system']['all'],
        outputs_written=not args.check_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
