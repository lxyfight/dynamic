"""F1 ablation and genuinely on-policy iterative MC refitting; development scale."""
from run_vfa_development import *
from e1_sim.formal_features import names
from e1_sim.formal_learning import FormalPolicy

def fit_equal_day(days,ridge=1.):
    samples=[(x,y,1/len(day)) for day in days if day for x,y in day]
    X=np.array([x for x,y,w in samples]);y=np.array([y for x,y,w in samples]);w=np.array([w for x,y,w in samples])
    theta=np.linalg.solve(X.T@(w[:,None]*X)+ridge*np.eye(X.shape[1]),X.T@(w*y))
    mse=float(np.sum(w*(X@theta-y)**2)/sum(w))
    return theta.tolist(),mse,len(samples)

def main():
    out=ROOT/'实验/F1特征消融与迭代训练_20261122';out.mkdir(exist_ok=True)
    require(not (out/'results.json').exists(),'completed output exists')
    gate=json.loads((ROOT/'实验/PI_WTQ修复核验_20261122/WTQ全日验收.json').read_text())
    require(gate['runs']==96 and gate['all_passed'],'WTQ gate not passed')
    e1=json.loads((ROOT/'实验/PI_WTQ修复核验_20261122/summary.json').read_text())
    require(e1['counts']['PASS']==59,'E1 gate not passed')
    protocol=dict(scope='development validation of F1 specification; 1D+6P; not the full manuscript scale grid',
        groups=['base','coordination','full'],seeds=[17,29,43],iterations=3,days_per_iteration=12,
        train=list(range(8000,8036)),validation=list(range(9000,9008)),test=list(range(10000,10016)),
        exploration=[.2,.1,.05],ridge=1.,batch='BM',equal_weight='each nonempty training day sums to one',
        selection='checkpoint by validation mean; group by mean of selected checkpoints across seeds; all frozen before test',
        features={g:names(g) for g in ['base','coordination','full']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    models=[];curves=[]
    for group in protocol['groups']:
        for seed in protocol['seeds']:
            weights=None;days=[];history=[]
            for iteration in range(3):
                training=[]
                for env_seed in protocol['train'][iteration*12:(iteration+1)*12]:
                    row,ss=episode(env_seed,FormalPolicy(weights,group=group,seed=seed*100000+env_seed,explore=protocol['exploration'][iteration],candidate_budget=16))
                    days.append(ss);training.append(row)
                weights,mse,count=fit_equal_day(days)
                validation=[episode(s,FormalPolicy(weights,group=group,candidate_budget=16))[0] for s in protocol['validation']]
                checkpoint=dict(group=group,seed=seed,iteration=iteration+1,training_days=len(days),samples=count,weights=weights,
                    training_mse=mse,validation_mean=float(np.mean([r['net'] for r in validation])),validation=validation,training=training)
                curves.append(checkpoint);history.append(checkpoint)
                print(group,seed,iteration+1,'validation',checkpoint['validation_mean'],flush=True)
            best=min(history,key=lambda x:(-x['validation_mean'],x['iteration']));models.append(best)
            (out/f'train_{group}_{seed}.json').write_text(json.dumps(dict(history=history,day_samples=days),indent=2),encoding='utf8')
    selected=max(protocol['groups'],key=lambda g:np.mean([m['validation_mean'] for m in models if m['group']==g]))
    (out/'frozen_models.json').write_text(json.dumps(dict(selected_group=selected,models=models),indent=2),encoding='utf8')
    rows=[]
    for s in protocol['test']:
        base,_=episode(s,FormalPolicy(candidate_budget=16),save=out/f'{s}_myopic.json')
        variants=[]
        for m in models:
            row,_=episode(s,FormalPolicy(m['weights'],group=m['group'],candidate_budget=16),save=out/f'{s}_{m["group"]}_{m["seed"]}.json')
            variants.append(dict(group=m['group'],training_seed=m['seed'],**row))
        rows.append(dict(seed=s,myopic=base,variants=variants));print('test',s,flush=True)
    (out/'results.json').write_text(json.dumps(dict(protocol=protocol,selected_group=selected,models=models,curves=curves,rows=rows),indent=2),encoding='utf8')

if __name__ == '__main__':
    raise RuntimeError('历史入口已封存；v04请运行verify_unknown_demand.py，不覆盖旧实验')
