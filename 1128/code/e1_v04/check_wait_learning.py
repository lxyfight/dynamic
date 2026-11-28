"""Reconcile paired node labels, the feature ranking and independent full-day accounts."""
from collections import defaultdict
import csv
from itertools import combinations
from math import isclose
from statistics import mean

from e1_sim.initializer import digest
from run_multiscale import read, write
from run_wait_increment_learning import EPS, OUT, SPLIT, fit


def equal(a,b):
    assert isclose(a,b,abs_tol=1e-7,rel_tol=1e-11), (a,b)


def net(ledger):
    return ledger['revenue']-sum(v for k,v in ledger.items() if k not in ('revenue','net'))


def estimate(model, x):
    return (x['known_plan_delta']+model['intercept']
            +sum((x[key]-m)/s*c for key,m,s,c in zip(model['columns'],model['mean'],
                                                       model['scale'],model['coefficients'])))


def block_mean(rows, value):
    by_case = defaultdict(list)
    for row in rows:
        by_case[tuple(row['case'])].append(value(row))
    by_layout = defaultdict(list)
    for case, values in by_case.items():
        by_layout[case[0]].append(mean(values))
    return mean(mean(v) for v in by_layout.values())


def loss(rows,model):
    return block_mean(rows,lambda r:max(0,r['y'])-(r['y'] if model and estimate(model,r['x'])>0 else 0))


def read_bank(stage):
    rows = read(OUT/(stage+'.json'))
    assert len({(tuple(r['case']), r['time']) for r in rows})==len(rows)
    for r in rows:
        case=r['case']
        assert case[0] in SPLIT[stage]
        common='_'.join(map(str,case))
        base=read(OUT/'runs'/(common+'_depart.json'))
        target=next(t for t in base['selected'] if t['time']==r['time'])
        w=read(OUT/'runs'/(common+'_wait_'+digest(target)[:16]+'.json'))
        assert w['target']==target and (r['prefix'],r['prefix_decisions'])==(target['prefix'],target['prefix_decisions'])
        assert w['input_hash']==base['input_hash']==r['input_hash']
        assert w['i0_hash']==base['i0_hash']==r['i0_hash']
        assert r['x']==base['x_by_time'][r['time']]==w['x_by_time'][r['time']]
        equal(net(w['ledger'])-net(base['ledger']),r['y'])
        equal(net(r['delta_ledger']),r['y'])
    return rows


def expanded_rows():
    rows = read(OUT/'expanded_train.json')
    assert {r['case'][0] for r in rows} == set(range(440,444))|set(SPLIT['train'])
    assert len({(tuple(r['case']),r['time']) for r in rows})==len(rows)
    for r in rows:
        case=r['case']
        common='_'.join(map(str,case))
        src=read(OUT/'expanded_sources'/(common+'.json'))
        assert src['x'][r['time']]==r['x']
        site=next(s for s in src['sites'] if s['time']==r['time'])
        paths=OUT.parent/'wait_counterfactual_v1' if case[0]<500 else OUT
        base=read(paths/'runs'/(common+'_depart.json'))
        other=read(paths/'runs'/(common+'_wait_'+digest(site)[:16]+'.json'))
        assert (r['prefix'],r['depart_plan'],r['wait_plan'])==(site['prefix'],site['depart'],site['wait'])
        assert base['input_hash']==other['input_hash']==r['input_hash']
        assert base['i0_hash']==other['i0_hash']==r['i0_hash']
        equal(net(other['ledger'])-net(base['ledger']),r['y'])
    return rows


def verify(root,train,val,test):
    frozen=read(root/'wait_model_freeze.json')
    report=read(root/'wait_policy_results.json')
    entries=frozen['table']
    assert {tuple(e['groups']) for e in entries}=={
        c for n in range(4) for c in combinations(['arrival','drone','truck'],n)}
    stable=[]
    for e in entries:
        equal(loss(val,e['model']),e['validation']['regret'])
    selected=min(entries,key=lambda e:(round(loss(val,e['model']),10),len(e['groups']),
                                        e['validation']['mae'],e['groups']))
    assert selected==frozen['learned']
    model=selected['model']
    assert ('learned' if loss(val,model)<loss(val,None)-EPS else 'depart')==frozen['selected']
    assert digest(train)==frozen['train_sha'] and digest(val)==frozen['validation_sha']
    # Sensitivity only: refit the already shortlisted columns; no test data or reselection.
    for layout in sorted({r['case'][0] for r in train}):
        reduced=fit([r for r in train if r['case'][0]!=layout],model['columns'])
        stable.append(dict(removed=layout,validation_waits=sum(estimate(reduced,r['x'])>0 for r in val),
                           validation_regret=loss(val,reduced)))
    cases=report['learned_online']['by_case']
    gains=[]
    all_triggers=0
    for r in cases:
        common='_'.join(map(str,r['case']))
        base=read(OUT/'runs'/(common+'_depart.json'))
        online=read(root/'online'/(common+'.json'))
        equal(net(online['ledger'])-net(base['ledger']),r['gain'])
        assert sum(probe['wait'] for probe in online['probes'])==r['waiting']
        for probe in online['probes']:
            equal(estimate(model,probe['x']),probe['score'])
            assert (probe['score']>0)==probe['wait']
        if r['waiting']==0:
            assert online['trace']==base['trace']
        all_triggers+=r['waiting']
        gains.append(r['gain'])
    assert {tuple(r['case']) for r in cases}=={(i,100 if i%2==0 else 150,s)
        for i in SPLIT['test'] for s in ('uniform','dual')}
    equal(mean(gains),report['learned_online']['mean_gain'])
    equal(loss(test,model),report['heldout_one_action']['regret'])
    splits={stage:dict(nodes=len(rows),cases=len({tuple(r['case']) for r in rows}),
                       layouts=len({r['case'][0] for r in rows}),wait_better=sum(r['y']>EPS for r in rows))
            for stage,rows in zip(SPLIT,[train,val,test])}
    result=dict(splits=splits,confirmed_model=model['columns'],deployment=frozen['selected'],
                waits_online=all_triggers,mean_daily_gain=mean(gains),
                score_perturbation_by_training_layout=stable,
                hidden_suffix_checks=read(OUT/'information_check.json'))
    write(root/'node_and_day_audit.json',result)
    print(splits,model['columns'],frozen['selected'],all_triggers,mean(gains),stable)


def export_selection_table():
    prior=OUT.parent/'high_demand_screen_v1'
    bank=read(prior/'all_feature_groups.json')
    core=read(prior/'freeze.json')
    groups={'recovery':'回收机会','clock_demand':'时间和预计需求','commitments':'已排定的任务',
            'known_pickup_connection':'已知取件衔接','station_opportunity':'站内回收机会',
            'arrival':'五分钟到达概率','drone':'无人机可用程度折减后的到达概率',
            'truck':'卡车容量与期限折减后的到达概率'}
    def group_names(keys):
        return '、'.join(groups[k] for k in keys) or '仅截距'
    table=[]
    for row in bank['scores']:
        k=next((name for name,m in core['shortlisted'].items() if m and m['columns']==row['columns']),None)
        status='最终保留' if k==core['chosen'] else '入围' if k else ''
        daily=core['validation'][k]['all']['net'] if k else ''
        table.append(['全日调度',group_names(row['groups']),bank['validation_samples'],row['r2'],
                      row['mae'],'','',daily,status])
    for label,folder in [('等待初筛',OUT),('等待扩充',OUT/'expanded')]:
        f=read(folder/'wait_model_freeze.json')
        for row in f['table']:
            v=row['validation']
            status=''
            if row==f['learned']:
                status='验证选用' if f['selected']=='learned' else '学习组合排名第一，验证保留立即离站'
            table.append([label,group_names(row['groups']),v['n'],v['r2'],v['mae'],v['regret'],v['wait'],'',status])
    assert len(table)==48
    with (OUT/'特征组合筛选明细.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['筛选任务','候选组（含截距）','预测验证节点数','验证R²','验证平均绝对误差（元）',
                         '验证平均选错动作损失（元/节点）','验证节点选择等待次数','另组策略验证平均净收益（元/日）','筛选结果'])
        writer.writerows(table)


if __name__=='__main__':
    train,val,test=[read_bank(s) for s in SPLIT]
    verify(OUT,train,val,test)
    verify(OUT/'expanded',expanded_rows(),val,test)
    export_selection_table()
