"""Controlled V3 feature ablation and iterative MC training; formal test stays closed."""
from run_multiscale import *
from e1_sim.formal_features import names

DEST=ROOT/'实验/实验记录_20261123_VFA迭代训练'
GROUPS=['base','coordination','full','waiting']
SEEDS=[17,29,43]
EPS=[.20,.18,.15,.12,.10,.08,.06,.05]


class EqualDayRLS:
    """Cumulative weighted ridge (lambda=1), recursive update, no forgetting."""
    def __init__(self,dimension,ridge=1.):
        if dimension<1 or ridge<=0:raise ValueError('positive dimension and ridge required')
        self.theta=np.zeros(dimension);self.P=np.eye(dimension)/ridge
        self.A=np.eye(dimension)*ridge;self.b=np.zeros(dimension);self.days=[]
    def add(self,day):
        if not day:return
        X=np.asarray([x for x,y in day],float);y=np.asarray([y for x,y in day],float)
        if X.ndim!=2 or X.shape[1]!=len(self.theta) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('invalid training samples')
        w=1./len(day)
        for x,target in zip(X,y):
            px=self.P@x;gain=px/(1./w+x@px)
            self.theta+=gain*(target-x@self.theta)
            self.P-=np.outer(gain,px);self.P=(self.P+self.P.T)/2
        self.A+=X.T@(w*X);self.b+=X.T@(w*y);self.days.append(day)
    def checkpoint(self):
        exact=np.linalg.solve(self.A,self.b)
        err=float(np.max(np.abs(exact-self.theta)))
        require(np.allclose(exact,self.theta,rtol=1e-7,atol=1e-7),'RLS/batch ridge disagreement')
        residual=[np.mean([(np.dot(x,self.theta)-y)**2 for x,y in day]) for day in self.days]
        return dict(weights=self.theta.tolist(),mse=float(np.mean(residual)),samples=sum(map(len,self.days)),
                    rls_batch_max_difference=err,condition_number=float(np.linalg.cond(self.A)))


def specs_v3(split,seed=0):
    rows=[]
    cells=[(c,p) for c in ([2,4,6] if split=='confirmation' else [2,4]) for p in [.2,.4]]
    rounds=24 if split=='train' else 4
    for d in range(rounds):
        layout=d%3 if split=='train' else d//2
        for c,p in cells:
            family=f'v3learn_{split}_s{seed}' if split=='train' else f'v3learn_{split}'
            rows.append(dict(cities=c,n=10,p=p,layout_seed=seedof(family,'layout',c,layout),
                day_seed=seedof(family,'day',c,layout,d),identity=f'{family}_c{c}_p{int(p*100)}_l{layout}_d{d}',
                layout_id=f'{family}_c{c}_l{layout}',training_seed=seed))
    return rows


def model(group,seed,weights=None):
    return dict(name=f'{group}_{seed}',kind='VFA',group=group,weights=weights,seed=seed,enhance=True)


def fingerprints():
    h=source_hashes();h['train_revision.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return h


def specification():
    return dict(version='v03-iterative-features-1',groups=GROUPS,seeds=SEEDS,iterations=8,days_per_iteration=12,
      total_training_days_per_model=96,epsilon=EPS,features={g:names(g) for g in GROUPS},
      train=[s for seed in SEEDS for s in specs_v3('train',seed)],validation=specs_v3('validation'),
      confirmation=specs_v3('confirmation'),formal_test=False,
      fit='Executed MC residual: final day net minus selected known-only plan J. Weighted RLS, prior I, lambda=1, no forgetting. Each nonempty day total weight=1; update policy every 12 days; cumulative samples.',
      selection='Validation only: per group/seed highest mean, within 0.01 yuan use earliest checkpoint including zero. Group: mean of all 3 selected seed means, within 0.01 yuan use smallest dimension. Never select best seed. Lock before generating confirmation instances.',
      evaluation='All groups and Myopic use exactly shared improved I0, candidate search, waiting and enhancement budgets. Cache only exact identical public states. No hidden future revealed to policy. Full independent ledger audit for every episode.',
      search=dict(initial=32,base=16,wait=16,enhancement=16,delta=5,beam_width=4,max_joint=3),
      scope='Development feature selection and independent confirmation; 2/4 cities train/validation, 2/4/6 confirmation, 10 orders per city, pickup proportions .2/.4. Not full formal scale grid.',
      failure='Any infeasible construction, nonterminal run, ledger mismatch or missing accepted obligation fails campaign; no silently excluded days.',
      source_hashes=fingerprints())


def work(task):
    sp,models,phase,eps=task
    target=DEST/'runs'/phase/(sp['identity']+'.json')
    signature=digest(dict(spec=sp,models=models,phase=phase,epsilon=eps))
    if target.exists():
        result=read(target);require(result['signature']==signature,'resume task mismatch');return result
    ip=DEST/'instances'/(sp['identity']+'.json')
    if ip.exists():x=read(ip)
    else:
        x=make_instance(CFG,**{k:v for k,v in sp.items() if k not in ('layout_id','training_seed')});write(ip,x)
    cache={};rows=[]
    for m in models:
        s,detail,samples=episode(x,m,explore=eps,explore_seed=seedof('v3learn','explore',sp['training_seed'],sp['identity']),cache=cache)
        s.update(group=m.get('group'),training_seed=m.get('seed'),layout_id=sp['layout_id'],samples=samples)
        tracepath=DEST/'traces'/phase/(sp['identity']+'__'+m['name']+'.json.gz')
        tracepath.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(tracepath,'wt',encoding='utf8') as f:json.dump(detail,f,ensure_ascii=False)
        rows.append(s)
    result=dict(signature=signature,id=sp['identity'],rows=rows);write(target,result);return result


def execute(tasks,workers,phase):
    result=[];start=perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for f in as_completed(futures):
            result.append(f.result())
            progress=dict(phase=phase,done=len(result),total=len(tasks),seconds=perf_counter()-start)
            write(DEST/'progress.json',progress)
            if len(result)%4==0 or len(result)==len(tasks):print(json.dumps(progress),flush=True)
    return sorted(result,key=lambda r:r['id'])


def select(curves):
    chosen=[]
    for g in GROUPS:
        for seed in SEEDS:
            options=[r for r in curves if r['group']==g and r['seed']==seed]
            top=max(r['validation_mean'] for r in options)
            chosen.append(min((r for r in options if r['validation_mean']>=top-.01),key=lambda r:r['training_days']))
    means={g:float(np.mean([r['validation_mean'] for r in chosen if r['group']==g])) for g in GROUPS}
    top=max(means.values());group=min((g for g in GROUPS if means[g]>=top-.01),key=lambda g:len(names(g)))
    return dict(selected_group=group,models=chosen,group_mean=means,formal_test_opened=False,
                selected_before_confirmation=True,all_seeds_retained=True)


def main(workers):
    DEST.mkdir(parents=True,exist_ok=True);proto=specification()
    if (DEST/'protocol.json').exists():require(read(DEST/'protocol.json')==proto,'protocol/source changed; do not mix campaigns')
    else:write(DEST/'protocol.json',proto)
    myopic=dict(name='Myopic',kind='VFA',enhance=True,group='base')
    baselines=execute([(s,[myopic],'validation_baseline',0) for s in proto['validation']],workers,'validation baseline')
    base={d['id']:d['rows'][0]['net'] for d in baselines};curves=[]
    fits={(g,s):EqualDayRLS(len(names(g))) for g in GROUPS for s in SEEDS}
    for g in GROUPS:
        for s in SEEDS:
            curves.append(dict(group=g,seed=s,iteration=0,training_days=0,weights=[0.]*len(names(g)),mse=None,samples=0,
                validation_mean=float(np.mean(list(base.values()))),paired_improvement=0.,validation=[d['rows'][0] for d in baselines]))
    weights={(g,s):None for g in GROUPS for s in SEEDS}
    for it,eps in enumerate(EPS,1):
        tasks=[]
        for seed in SEEDS:
            ms=[model(g,seed,weights[g,seed]) for g in GROUPS]
            tasks += [(sp,ms,f'train_{it}',eps) for sp in specs_v3('train',seed)[(it-1)*12:it*12]]
        training=execute(tasks,workers,f'train iteration {it}/8')
        for d in training:
            for r in d['rows']:fits[r['group'],r['training_seed']].add(r['samples'])
        models=[];stats={}
        for g in GROUPS:
            for seed in SEEDS:
                stats[g,seed]=fits[g,seed].checkpoint();weights[g,seed]=stats[g,seed]['weights'];models.append(model(g,seed,weights[g,seed]))
        validation=execute([(sp,models,f'validation_{it}',0) for sp in proto['validation']],workers,f'validate iteration {it}/8')
        for g in GROUPS:
            for seed in SEEDS:
                rows=[next(r for r in d['rows'] if r['model']==f'{g}_{seed}') for d in validation]
                curves.append(dict(group=g,seed=seed,iteration=it,training_days=it*12,**stats[g,seed],
                    validation_mean=float(np.mean([r['net'] for r in rows])),
                    paired_improvement=float(np.mean([r['net']-base[r['id']] for r in rows])),validation=rows))
        write(DEST/'training_curves.json',curves)
        print('checkpoint',it,[(g,round(float(np.mean([r['paired_improvement'] for r in curves if r['group']==g and r['iteration']==it])),3)) for g in GROUPS],flush=True)
    frozen=select(curves);write(DEST/'selected_development_models.json',frozen)
    models=[myopic]+[model(r['group'],r['seed'],r['weights']) for r in frozen['models']]
    confirmation=execute([(sp,models,'confirmation',0) for sp in proto['confirmation']],workers,'independent confirmation')
    require(fingerprints()==proto['source_hashes'],'source drift during training')
    write(DEST/'results.json',dict(protocol=proto,curves=curves,selection=frozen,confirmation=confirmation,
          completed=True,formal_test_opened=False,source_hashes_verified=True))
    print('COMPLETE: training, validation selection, independent confirmation',flush=True)


if __name__ == '__main__':
    raise RuntimeError('历史入口已封存；v04请运行verify_unknown_demand.py，不覆盖旧实验')
