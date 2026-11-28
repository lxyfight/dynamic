"""Strict, offline aggregation of complete matched first-action wait experiments.

Independent test data are never loaded. Original realized suffix checks remain
separate from the eight conditional suffix draws used to estimate each state.
"""
import argparse
from collections import Counter, defaultdict
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

INTERVALS = (2, 5, 10)
T7_975 = 2.3646242510102993
TOL = 1e-7


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def sample_statistics(values):
    values = list(values)
    if len(values) != 8:
        raise ValueError('This predefined screen requires eight conditional draws per state')
    center, sd = mean(values), stdev(values)
    se = sd / sqrt(len(values))
    return dict(n=len(values), mean=center, sample_sd=sd, standard_error=se,
        exploratory_t7_interval_95=[center-T7_975*se, center+T7_975*se],
        wins=sum(x>TOL for x in values), ties=sum(abs(x)<=TOL for x in values),
        losses=sum(x< -TOL for x in values), minimum=min(values), maximum=max(values),
        values=values)


def state_summary(site, rows, interval):
    key = str(interval)
    data = [r['comparisons'][key] for r in rows]
    gains = [r['net_gain'] for r in data]
    net = sample_statistics(gains)
    delays = [r['actual_departure_delay'] for r in data]
    overlaps = [r['overlap']['counts'][key] for r in rows]
    # Recompute from per-order rows so earlier smoke files remain compatible.
    releases = [sum(site['at']<o['release']<=site['at']+interval
        for o in r['overlap']['rows']) for r in rows]
    idle_ready = [sum(site['at']<o['idle_drone_ready']<=site['at']+interval
        for o in r['overlap']['rows']) for r in rows]
    actual_ready = []
    for r in rows:
        selected = r['branches'][key]
        actual_ready.append(sum(o['accepted'] and o['station']==site['station']
            and o['release']>site['at'] and o['available_at'] is not None
            and site['at']<o['available_at']<=site['at']+interval
            for o in selected['pickup_records']))
    stratum=site['qualification'].get('stratum',
        'post_unload_primary' if site['qualification']['just_unloaded'] else 'other_departure_ready')
    return dict(site_id=site['id'], case=site['case'], spec=site['spec'], stratum=stratum,
        at=site['at'], truck=site['truck'], station=site['station'], interval=interval,
        net_gain=net,
        accepted_delta=dict(mean=mean(r['accepted_delta'] for r in data),
            additional_net_accept_draws=sum(r['accepted_delta']>0 for r in data),
            any_added_accept_draws=sum(bool(r['added_accepts']) for r in data),
            any_lost_accept_draws=sum(bool(r['lost_accepts']) for r in data)),
        trips_saved=dict(mean=mean(r['trips_saved'] for r in data),
            at_least_one_saved_draws=sum(r['trips_saved']>0 for r in data),
            extra_trip_draws=sum(r['trips_saved']<0 for r in data)),
        extra_truck_time_cost=dict(mean=mean(r['extra_time_cost'] for r in data),
            minimum=min(r['extra_time_cost'] for r in data), maximum=max(r['extra_time_cost'] for r in data)),
        ledger_mean_delta={name:mean(r['ledger_delta'][name] for r in data) for name in data[0]['ledger_delta']},
        departure_delay=dict(mean=mean(delays), minimum=min(delays), maximum=max(delays),
            earlier_than_review_draws=sum(x<interval-TOL for x in delays),
            later_than_review_draws=sum(x>interval+TOL for x in delays),
            at_review_draws=sum(abs(x-interval)<=TOL for x in delays)),
        optimistic_idle_drone_loadable_overlap=dict(mean_orders=mean(overlaps),
            positive_draws=sum(x>0 for x in overlaps),total_orders=sum(overlaps)),
        eligible_new_releases_in_window=dict(mean_orders=mean(releases),
            positive_draws=sum(x>0 for x in releases),total_orders=sum(releases)),
        idle_ready_in_window=dict(mean_orders=mean(idle_ready),
            positive_draws=sum(x>0 for x in idle_ready),total_orders=sum(idle_ready)),
        actually_accepted_future_ready_at_target=dict(mean_orders=mean(actual_ready),
            positive_draws=sum(x>0 for x in actual_ready),
            note='Policy-selected observations; cannot alone identify opportunity absence'),
        continuation_search_truncated_draws=sum(any(b['continuation_search']['truncated_epochs']>0
            for b in r['branches'].values()) for r in rows))


def aggregate(units):
    if not units:
        return dict(states=0)
    case_values = defaultdict(list)
    for unit in units:
        case_values[unit['case']].append(unit['net_gain']['mean'])
    return dict(states=len(units), cases=len(case_values), draws_per_state=8,
        conditional_paired_draws=8*len(units),
        mean_gain_equal_state=mean(u['net_gain']['mean'] for u in units),
        mean_gain_equal_case=mean(mean(v) for v in case_values.values()),
        positive_mean_states=sum(u['net_gain']['mean']>TOL for u in units),
        profitable_wait_but_chain_departs=sum(u['profitable_wait_but_chain_departs'] for u in units),
        pointwise_positive_ci_but_chain_departs=sum(u['profitable_wait_but_chain_departs'] and
            u['net_gain']['exploratory_t7_interval_95'][0]>0 for u in units),
        zero_mean_states=sum(abs(u['net_gain']['mean'])<=TOL for u in units),
        negative_mean_states=sum(u['net_gain']['mean']< -TOL for u in units),
        pointwise_ci_lower_above_zero_states=sum(u['net_gain']['exploratory_t7_interval_95'][0]>0 for u in units),
        pointwise_ci_upper_below_zero_states=sum(u['net_gain']['exploratory_t7_interval_95'][1]<0 for u in units),
        wins=sum(u['net_gain']['wins'] for u in units), ties=sum(u['net_gain']['ties'] for u in units),
        losses=sum(u['net_gain']['losses'] for u in units),
        mean_accepted_delta=mean(u['accepted_delta']['mean'] for u in units),
        any_added_accept_draws=sum(u['accepted_delta']['any_added_accept_draws'] for u in units),
        net_additional_accept_draws=sum(u['accepted_delta']['additional_net_accept_draws'] for u in units),
        mean_trips_saved=mean(u['trips_saved']['mean'] for u in units),
        trip_saving_draws=sum(u['trips_saved']['at_least_one_saved_draws'] for u in units),
        extra_trip_draws=sum(u['trips_saved']['extra_trip_draws'] for u in units),
        mean_extra_truck_time_cost=mean(u['extra_truck_time_cost']['mean'] for u in units),
        mean_actual_departure_delay=mean(u['departure_delay']['mean'] for u in units),
        early_departure_draws=sum(u['departure_delay']['earlier_than_review_draws'] for u in units),
        late_departure_draws=sum(u['departure_delay']['later_than_review_draws'] for u in units),
        optimistic_overlap_positive_draws=sum(u['optimistic_idle_drone_loadable_overlap']['positive_draws'] for u in units),
        optimistic_overlap_total_orders=sum(u['optimistic_idle_drone_loadable_overlap']['total_orders'] for u in units),
        eligible_new_release_positive_draws=sum(u['eligible_new_releases_in_window']['positive_draws'] for u in units),
        eligible_new_release_total_orders=sum(u['eligible_new_releases_in_window']['total_orders'] for u in units),
        idle_ready_positive_draws=sum(u['idle_ready_in_window']['positive_draws'] for u in units),
        idle_ready_total_orders=sum(u['idle_ready_in_window']['total_orders'] for u in units),
        ledger_mean_delta={k:mean(u['ledger_mean_delta'][k] for u in units) for k in units[0]['ledger_mean_delta']},
        uncertainty_note='No iid-state confidence interval: states share original scenario prefixes and layouts.')


def build_summary(root):
    root = Path(root)
    protocol = read(root/'protocol.json')
    if protocol['samples']!=8:
        raise ValueError('Protocol differs from predefined eight-draw analysis')
    if protocol.get('independent_test_opened'):
        raise ValueError('Expected development-only campaign')
    sites = read(root/'sites.json')
    if not sites or len({s['id'] for s in sites})!=len(sites):
        raise ValueError('Missing or duplicate captured sites')
    captures = {r['case']:r for r in (read(p) for p in (root/'capture').glob('*.json'))}
    missing=[]; banks={}; original=[]; checks=0
    expected_names=set()
    for site in sites:
        assert site['qualification']['eligible']
        assert not site['qualification']['known_cargo_reservation']
        assert captures[site['case']]['reference_reproduced']
        bank=[];visible_hashes=set()
        for replica in [-1]+list(range(8)):
            path=root/'paired'/(site['id']+f'_r{replica:02}.json')
            expected_names.add(path.name)
            if not path.exists():
                missing.append(path.name);continue
            row=read(path)
            visible_hashes.add(row['online_known_hash'])
            assert row['replica']==replica
            # Classification metadata can be enriched after the initial smoke
            # run. Frozen physical identity, prefix and all four plans cannot.
            for field in ['id','case','spec','at','truck','station','prefix_hash','base_plan_hash']:
                assert row['site'][field]==site[field], (site['id'],field)
            for h in ['0','2','5','10']:
                assert row['site']['interventions'][h]['branch_hash']==site['interventions'][h]['branch_hash']
            assert set(row['branches'])=={'0','2','5','10'}
            assert set(row['comparisons'])=={'2','5','10'}
            assert all(r['suffix_hash']==row['suffix_hash'] for r in row['branches'].values())
            assert all(r['terminal'] and r['ledger_audited'] for r in row['branches'].values())
            base=row['branches']['0']
            assert abs(base['actual_first_departure_delay'])<TOL
            for h in INTERVALS:
                alt=row['branches'][str(h)];delta=row['comparisons'][str(h)]
                assert abs(alt['net']-base['net']-delta['net_gain'])<TOL
                assert alt['accepted']-base['accepted']==delta['accepted_delta']
                assert base['trips']-alt['trips']==delta['trips_saved']
                assert abs(alt['ledger']['truck_time']-base['ledger']['truck_time']-delta['extra_time_cost'])<TOL
            if replica==-1:
                ref=captures[site['case']]['reference']
                assert base['net']==ref['net'] and base['trace']==ref['trace']
                original.append(row)
            else: bank.append(row)
            checks+=1
        if visible_hashes:
            assert len(visible_hashes)==1, 'future resampling changed the online-visible state'
        banks[site['id']]=bank
    actual_names={p.name for p in (root/'paired').glob('*.json')}
    unexpected=actual_names-expected_names
    if missing or unexpected:
        raise ValueError(json.dumps(dict(missing_count=len(missing),missing=missing,
            unexpected=sorted(unexpected)),ensure_ascii=False))
    units=[state_summary(s,banks[s['id']],h) for s in sites for h in INTERVALS]
    model_scores=read(root/'chain_scores.json')
    assert not model_scores['future_suffix_used'] and not model_scores['new_candidate_search']
    score_by_site={r['site_id']:r for r in model_scores['sites']}
    assert set(score_by_site)=={s['id'] for s in sites}
    for site in sites:
        row=score_by_site[site['id']]
        for field in ['case','at','truck','station']:
            assert row[field]==site[field]
        for h in ['0','2','5','10']:
            assert row['branches'][h]['canonical_plan_hash']==site['interventions'][h]['branch_hash']
    for unit in units:
        row=score_by_site[unit['site_id']]
        unit['frozen_model_preferences']={name:m['recommended_h'] for name,m in row['models'].items()}
        unit['model_predicted_wait_minus_depart']={name:m['scores'][str(unit['interval'])]['delta_vs_depart']
            for name,m in row['models'].items()}
        unit['profitable_wait_but_chain_departs']=(unit['net_gain']['mean']>TOL and
            unit['frozen_model_preferences']['chain']==0)
    overall={str(h):aggregate([u for u in units if u['interval']==h]) for h in INTERVALS}
    strata={}
    for name,get in [('state_type',lambda u:u['stratum']),('pattern',lambda u:u['spec']['pattern']),
                     ('nominal_orders',lambda u:u['spec']['total']),('station',lambda u:u['station'])]:
        values=sorted({get(u) for u in units})
        strata[name]={str(v):{str(h):aggregate([u for u in units if get(u)==v and u['interval']==h])
                            for h in INTERVALS} for v in values}
    original_summary={}
    for h in INTERVALS:
        rows=[r['comparisons'][str(h)] for r in original]
        original_summary[str(h)]=dict(states=len(rows),mean_gain=mean(r['net_gain'] for r in rows),
            wins=sum(r['net_gain']>TOL for r in rows),ties=sum(abs(r['net_gain'])<=TOL for r in rows),
            losses=sum(r['net_gain']< -TOL for r in rows),
            mean_accepted_delta=mean(r['accepted_delta'] for r in rows),
            mean_trips_saved=mean(r['trips_saved'] for r in rows),
            mean_extra_time_cost=mean(r['extra_time_cost'] for r in rows))
    rankings=[]
    for site in sites:
        options=[u for u in units if u['site_id']==site['id']]
        best=max(options,key=lambda u:(u['net_gain']['mean'],-u['interval']))
        rankings.append(dict(site_id=site['id'],case=site['case'],at=site['at'],station=site['station'],
            best_sample_mean_interval=best['interval'] if best['net_gain']['mean']>TOL else 0,
            best_sample_mean_gain=max(0.,best['net_gain']['mean'])))
    exclusions=Counter()
    event_status=Counter()
    for capture in captures.values():
        for event in capture['offer_audit']:
            selected=False
            for truck in event['trucks']:
                if truck['qualification']['eligible'] and not truck.get('branch_failure'):
                    selected=True
                else:
                    exclusions.update(truck['qualification']['reasons'])
                    if truck.get('branch_failure'): exclusions['branch_failure']+=1
            event_status['eligible_event' if selected else 'excluded_event']+=1
    capture_audit=dict(candidate_offer_events=sum(len(r['offer_audit']) for r in captures.values()),
        event_status=dict(event_status),truck_exclusion_reason_counts=dict(exclusions),
        note='Exclusion reasons can overlap. Post-unload states are the primary layer; other departure-ready states are reported separately.',
        states_by_type=dict(Counter(u['stratum'] for u in units if u['interval']==2)),
        states_by_case={case:len(r['sites']) for case,r in sorted(captures.items())},
        no_eligible_state_cases=[case for case,r in captures.items() if not r['sites']],
        intervention_timing=dict(strict_non_target_action_times_unchanged=sum(
            s['interventions']['10']['strict_non_target_actions_unchanged'] for s in sites),
            required_drone_time_propagation_states=sum(
            not s['interventions']['10']['strict_non_target_actions_unchanged'] for s in sites)))
    return dict(protocol=protocol,integrity=dict(complete=True,sites=len(sites),source_cases=len(captures),
        eligible_cases=len({s['case'] for s in sites}),
        original_paired_draws=len(sites),conditional_paired_draws=8*len(sites),
        audited_branches=checks*4,verified_paired_files=checks,independent_test_opened=False),
        capture_audit=capture_audit,
        overall=overall,
        primary_post_unload={str(h):aggregate([u for u in units
            if u['interval']==h and u['stratum']=='post_unload_primary']) for h in INTERVALS},
        supplementary_departure_ready={str(h):aggregate([u for u in units
            if u['interval']==h and u['stratum']!='post_unload_primary']) for h in INTERVALS},
        strict_fixed_drone_timing={str(h):aggregate([u for u in units if u['interval']==h and
            next(s for s in sites if s['id']==u['site_id'])['interventions'][str(h)]['strict_non_target_actions_unchanged']]) for h in INTERVALS},
        strata=strata,states=units,original_realized_suffix=original_summary,
        frozen_model_choice_summary=model_scores['summary'],
        exploratory_state_best_actions=rankings,
        uncertainty=dict(per_state='Eight independent suffix draws conditioned on one observed state; t7 intervals are exploratory.',
            multiple_comparisons='No correction for testing many states or three intervals.',
            rare_events='Eight draws may miss rare gains; all-negative draws do not prove waiting is never useful.',
            selection='Choosing the best interval/state by these same sample means is optimistic; not an independent test.',
            shared_prefixes='State-level means share original day/layout prefixes; no iid pooling across states.',
            time_cost='Extra truck_time is the difference in whole-fleet whole-day time cost, not merely the first dwell cost.',
            overlap='Idle-drone ready/loadable overlap is an optimistic time bound, not a certificate of jointly feasible service or a necessary condition for positive whole-day gain: waiting can shift later station visits and consolidation.'))


def text_report(result):
    lines=['配对主动等待回放：完整性检查通过；独立测试未打开。',
        json.dumps(result['integrity'],ensure_ascii=False),'',
        '候选事件筛选：'+json.dumps(result['capture_audit'],ensure_ascii=False),'',
        '条件未来样本：每状态8个；原实现后缀单独核验，不混入均值。',
        'h / 等状态平均差 / 等场景平均差 / 正均值状态数 / 胜平负样本 / 接单均差 / 省车趟均值 / 额外时间费均值 / 提前离站样本 / 乐观重叠样本']
    for label,group in [('主层：刚卸完货',result['primary_post_unload']),
                        ('补充层：其他可离站状态',result['supplementary_departure_ready']),
                        ('敏感性：无人机计划时刻也严格不变',result['strict_fixed_drone_timing']),
                        ('全部状态（混合层，仅作补充）',result['overall'])]:
        lines.append(label)
        for h,r in group.items():
            if not r['states']:
                lines.append(h+' / no states');continue
            lines.append(f"{h} / {r['mean_gain_equal_state']:.6f} / {r['mean_gain_equal_case']:.6f} / "
                f"{r['positive_mean_states']}/{r['states']} / {r['wins']},{r['ties']},{r['losses']} / "
                f"{r['mean_accepted_delta']:.6f} / {r['mean_trips_saved']:.6f} / {r['mean_extra_truck_time_cost']:.6f} / "
                f"{r['early_departure_draws']} / {r['optimistic_overlap_positive_draws']}")
            lines.append(f"  条件平均收益>0但chain选走的状态={r['profitable_wait_but_chain_departs']}；"
                f"点态区间下界>0且chain选走={r['pointwise_positive_ci_but_chain_departs']}；"
                f"窗口内新单/空闲无人机就绪/可装车重叠样本数="
                f"{r['eligible_new_release_positive_draws']}/{r['idle_ready_positive_draws']}/{r['optimistic_overlap_positive_draws']}")
    lines+=['','逐状态：state / at / station / h / 平均利润差 / SD / SE / 探索性95%区间 / 胜平负 / 接单差 / 省车趟 / 时间费']
    for u in result['states']:
        r=u['net_gain'];lo,hi=r['exploratory_t7_interval_95']
        lines.append(f"{u['site_id']} / {u['at']:.6f} / {u['station']} / {u['interval']} / {r['mean']:.6f} / "
            f"{r['sample_sd']:.6f} / {r['standard_error']:.6f} / [{lo:.6f},{hi:.6f}] / "
            f"{r['wins']},{r['ties']},{r['losses']} / {u['accepted_delta']['mean']:.6f} / "
            f"{u['trips_saved']['mean']:.6f} / {u['extra_truck_time_cost']['mean']:.6f}")
    lines+=['','原实现未来后缀（每状态一次）：']
    lines += [h+' / '+json.dumps(r,ensure_ascii=False) for h,r in result['original_realized_suffix'].items()]
    lines+=['','限制：每状态仅8个未来样本，区间是未做多重比较校正的探索性t区间；',
        '同一场景内状态相关，不能当作独立布局；从同一批样本选最佳时长或最佳状态有选择偏差。',
        '没有观测到盈利等待不能证明所有未来下等待均无益。就绪时间重叠采用空闲无人机的乐观下界。',
        '目标站窗口内可装车重叠不是全天收益为正的必要条件；等待也会改变下游经过时刻及合单机会。',
        '额外时间费是全日全车队费用差；预约已知货物等待未混入首次主动等待干预。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('../../results/paired_wait_20261128'))
    args=parser.parse_args()
    result=build_summary(args.output)
    write(args.output/'summary.json',result)
    report=text_report(result)
    (args.output/'RESULT_TABLES.txt').write_text(report,encoding='utf8')
    print(report,flush=True)


if __name__=='__main__': main()
