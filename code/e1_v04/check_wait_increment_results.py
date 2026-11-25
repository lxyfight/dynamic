"""Independent ledger and score checks for the frozen waiting candidate."""
from itertools import combinations
from collections import Counter
from statistics import mean
import csv
import numpy as np

from run_multiscale import digest,read,write
from run_wait_increment_learning import OUT, GROUPS, load_split


def check():
    frozen=read(OUT/'wait_model_freeze.json')
    summary=read(OUT/'wait_policy_results.json')
    assert summary['selected']==frozen['selected']
    train=load_split('train')
    xnames=sum(GROUPS.values(),[])
    train_x=np.array([[r['x'][k] for k in xnames] for r in train])
    spread={name:dict(min=float(col.min()),max=float(col.max()),unique=len(np.unique(col)))
            for name,col in zip(xnames,train_x.T)}
    correlations={}
    for i,j in combinations(range(len(xnames)),2):
        if np.ptp(train_x[:,i])>1e-12 and np.ptp(train_x[:,j])>1e-12:
            correlations[' / '.join((xnames[i],xnames[j]))]=float(np.corrcoef(train_x[:,i],train_x[:,j])[0,1])
    # Recompute pointwise held-out policies independently from the raw logs.
    model=frozen['learned']['model']
    ns={}
    for stage in ('train','validation','test'):
        rows=load_split(stage)
        costs=Counter()
        for r in rows:
            cols=model['columns']
            scaled=(np.array([r['x'][k] for k in cols])-model['mean'])/model['scale']
            prediction=r['known_plan_delta']+model['intercept']+float(scaled@model['coefficients'])
            observed=r['wait_net']-r['depart_net']
            assert abs(observed-r['y'])<1e-7
            costs[('positive' if observed>1e-7 else 'negative' if observed< -1e-7 else 'tie')]+=1
            if prediction>0:
                costs['predicted_wait']+=1
        ns[stage]=dict(costs)
    assert ns['test'].get('predicted_wait',0)==summary['heldout_one_action']['wait']
    totals=[]
    for ab in summary['learned_online']['by_case']:
        case=ab['case']
        tag='_'.join(map(str,case))
        baseline=read(OUT/'runs'/(tag+'_depart.json'))
        observed=read(OUT/'online'/(tag+'.json'))
        world=read(OUT/'worlds'/(digest(case)+'.json'))
        assert digest(world['input'])==baseline['input_hash']==observed['input_hash']
        assert world['i0_hash']==baseline['i0_hash']==observed['i0_hash']
        for r in (baseline,observed):
            account=r['ledger']
            net=account['revenue']-sum(v for k,v in account.items() if k not in ('revenue','net'))
            assert abs(r['net']-net)<1e-7
        for probe in observed['probes']:
            z=[(probe['x'][c]-m)/s for c,m,s in zip(model['columns'],model['mean'],model['scale'])]
            score=probe['x']['known_plan_delta']+model['intercept']+np.dot(z,model['coefficients'])
            assert abs(score-probe['score'])<1e-7 and probe['wait']==(score>0)
        totals.append(ab['gain'])
        assert abs((observed['net']-baseline['net'])-ab['gain'])<1e-7
    assert len(totals)==8 and abs(mean(totals)-summary['learned_online']['mean_gain'])<1e-7
    details=dict(sample_labels=ns,train_feature_ranges=spread,train_feature_correlations=correlations,
                 online_audited_cases=len(totals),online_mean_gain=mean(totals))
    write(OUT/'wait_model_checks.json',details)
    print(details)


def export_selection():
    labels={'arrival':'到达概率','drone':'无人机机会','truck':'卡车机会'}
    with (OUT/'等待候选组合筛选.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        out=csv.writer(stream)
        out.writerow(['训练方案','特征组合（已知收益差均固定纳入）','训练节点数','验证节点数',
                      '验证R²','验证平均绝对误差（元）','验证后验动作损失（元）',
                      '验证选择等待次数','验证平均配对增益（元）','用于独立诊断回放'])
        for path,name in ((OUT,'首个及后半日首个节点'),(OUT/'expanded','开发布局全部可严格配对节点')):
            f=read(path/'wait_model_freeze.json')
            for r in f['table']:
                v=r['validation']
                out.writerow([name,'＋'.join(labels[k] for k in r['groups']) or '仅截距',
                              r['training']['n'],v['n'],v['r2'],v['mae'],v['regret'],v['wait'],
                              v['paired_gain'],'是' if r['groups']==f['learned']['groups'] else '否'])


if __name__=='__main__':
    check()
    if (OUT/'expanded/wait_model_freeze.json').exists():
        export_selection()
