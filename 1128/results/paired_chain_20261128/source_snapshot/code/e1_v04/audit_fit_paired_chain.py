"""Independent arithmetic audit of frozen paired-chain fits and raw returns.

Does not import the fitter, features, policy, simulator or summary functions.
Only runs after both frozen dev reports exist. Never opens reserved test data.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean,stdev

import numpy as np

ACTIONS=(0,2,5,10)
ALPHAS=(.001,.01,.1,1.,10.,100.)
T7=2.3646242510102993
TOL=1e-9
RESERVED={112891,112892,112893,112894}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def near(a,b):
    return abs(float(a)-float(b)) <= 1e-8+1e-11*max(abs(float(a)),abs(float(b)))


def stats(values):
    values=list(map(float,values));n=len(values)
    sd=stdev(values) if n>1 else None
    return dict(n=n,mean=mean(values),sample_sd=sd,se=sd/n**.5 if sd is not None else None,
        win=sum(v>TOL for v in values),tie=sum(abs(v)<=TOL for v in values),loss=sum(v < -TOL for v in values))


def inference(values):
    values=list(values)
    assert len(values)==8
    out=stats(values)
    out['ci95']=[out['mean']-T7*out['se'],out['mean']+T7*out['se']]
    flipped=[mean(v*s for v,s in zip(values,signs)) for signs in product((-1,1),repeat=8)]
    out.update(t_df=7,signflip_assignments=256,
        signflip_one_sided_p=sum(v>=out['mean']-TOL for v in flipped)/256,
        signflip_two_sided_p=sum(abs(v)>=abs(out['mean'])-TOL for v in flipped)/256)
    return out


def choice(feature,model):
    scores=[]
    for j,x in zip(feature['J'],feature['features']):
        value=0. if feature['closed'] else sum(float(w)*float(x[c])/float(s)
            for c,w,s in zip(model['columns'],model['weights'],model['scales']))
        scores.append(float(j)+value)
    best=max(scores)
    selected=next(i for i,score in enumerate(scores) if best-score<=TOL)
    return selected,scores


def run(output):
    output=Path(output).resolve();source=Path(__file__).resolve().parent
    reports={stage:read(output/stage/'results.json') for stage in ('base','augmented')}
    if any('dev' not in r.get('reports',{}) or not r.get('dev_evaluated') for r in reports.values()):
        raise ValueError('both finalized dev reports are required; do not open raw dev labels before global freeze')
    checks=[];details={};source_receipts={}
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),**({'detail':detail} if detail is not None else {})))
    def compare_dict(name,observed,expected):
        good=True
        for key,value in expected.items():
            actual=observed.get(key)
            if value is None:good &= actual is None
            elif isinstance(value,list):good &= actual is not None and len(actual)==len(value) and all(near(a,b) for a,b in zip(actual,value))
            else:good &= actual is not None and near(actual,value)
        check(name,good)
    features={};captures={};protocols={};raw={};receipts={};original={}
    for stage in ('base','augmented'):
        features[stage]={}
        for split in ('train','tune','dev'):
            path=output/stage/'known_features'/(split+'.json')
            content=read(path);source_receipts[str(path.relative_to(output))]=sha(path)
            features[stage][split]={r['site_id']:r for r in content['states']}
            for name,expected in content['input_receipts'].items():
                path=(source/name[7:]) if name.startswith('source/') else output/split/name
                check('known receipt '+stage+'/'+split+'/'+name,sha(path)==expected)
    for split in ('train','tune','dev'):
        protocols[split]=read(output/split/'protocol.json')
        protocol=protocols[split]
        captures[split]={r['case']:r for r in (read(p) for p in (output/split/'capture').glob('*.json'))}
        sites={r['id']:r for r in read(output/split/'sites.json')}
        base=features['base'][split];aug=features['augmented'][split]
        check('stage state sets '+split,set(sites)==set(base)==set(aug))
        check('capture case coverage '+split,len(captures[split])==len(protocol['cases']) and
            {digest(c['spec']) for c in captures[split].values()}=={digest(s) for s in protocol['cases']})
        check('only primary '+split,all(s['qualification']['eligible'] and s['qualification']['just_unloaded']
            and s['qualification']['stratum']=='post_unload_primary' for s in sites.values()))
        for sid,row in base.items():
            extra=aug[sid]
            check('stage known J features plans identical '+sid,row['J']==extra['J']
                and row['features']==[x[:7] for x in extra['features']]
                and row['known_state_hash']==extra['known_state_hash'] and row['plan_hashes']==extra['plan_hashes'])
        expected={sid+f'_r{k:02}.json' for sid in sites for k in [-1]+list(range(protocol['samples']))}
        check('all pairs present '+split,expected=={p.name for p in (output/split/'paired').glob('*.json')})
        raw[split]={};receipts[split]={};original[split]={}
        for sid in sites:
            arrays={k:[] for k in ('G','accepted','trips','truck_time','departure_delay')}
            for replica in [-1]+list(range(protocol['samples'])):
                name=sid+f'_r{replica:02}.json';path=output/split/'paired'/name
                pair=read(path);receipts[split][name]=sha(path)
                check('pair identity '+name,pair['site']==sites[sid] and pair['replica']==replica and set(pair['branches'])==set(map(str,ACTIONS)))
                check('online isolation '+name,pair['online_known_hash']==base[sid]['known_state_hash'])
                branches=[pair['branches'][str(h)] for h in ACTIONS]
                check('terminal ledger shared suffix '+name,all(b['terminal'] and b['ledger_audited'] and b['suffix_hash']==pair['suffix_hash'] for b in branches))
                if replica<0:
                    reference=captures[split][sites[sid]['case']]['reference']
                    check('original reproduction '+sid,branches[0]['net']==reference['net'] and branches[0]['trace']==reference['trace'])
                    original[split][sid]=[float(b['net']) for b in branches]
                    continue
                arrays['G'].append([float(b['net']) for b in branches])
                arrays['accepted'].append([b['accepted'] for b in branches])
                arrays['trips'].append([b['trips'] for b in branches])
                arrays['truck_time'].append([float(b['ledger']['truck_time']) for b in branches])
                arrays['departure_delay'].append([float(b['actual_first_departure_delay']) for b in branches])
            raw[split][sid]={k:np.asarray(v) for k,v in arrays.items()}
        for stage in ('base','augmented'):
            saved=read(output/stage/'input_audits'/(split+'.json'))
            check('stage labels identical raw '+stage+'/'+split,saved['file_receipts']==receipts[split])
    frozen={stage:read(output/stage/'frozen_selection.json') for stage in ('base','augmented')}
    models={stage:{p.stem:read(p) for p in (output/stage/'models').glob('*.json') if not p.stem.endswith('_candidates')}
            for stage in ('base','augmented')}
    check('shared fitted chain unchanged',models['base']['paired_chain']==models['augmented']['paired_chain'])
    check('shared legacy chain unchanged',models['base']['old_chain']==models['augmented']['old_chain'])
    check('shared level control unchanged',models['base']['level_chain']==models['augmented']['level_chain'])
    for stage in ('base','augmented'):
        check('freeze excludes dev '+stage,not frozen[stage]['dev_outcomes_used_for_selection'])
        for split in ('train','tune'):
            check('frozen label receipts '+stage+'/'+split,frozen[stage]['label_receipt_hashes'][split]==digest(receipts[split]))
        for alias,model in models[stage].items():
            if alias=='old_chain':continue
            check('model equals freeze '+stage+'/'+alias,model==frozen[stage]['selected_models'][alias])
    # Independently reconstruct weighted normal equations for each trained
    # family/control and select alpha solely on the raw tune action returns.
    refits={}
    for stage,alias,kind,columns in (('base','paired_chain','paired',[0,1]),
            ('base','level_chain','level',[0,1]),('augmented','paired_augmented','paired',[0,1,7])):
        train=features[stage]['train'];counts=Counter(r['case'] for r in train.values())
        x=[];y=[];weight=[]
        indices=(1,2,3) if kind=='paired' else (0,1,2,3)
        for sid,row in train.items():
            j=np.asarray(row['J']);g=raw['train'][sid]['G'].mean(axis=0)
            phi=np.asarray(row['features'])[:,columns]
            for a in indices:
                x.append(phi[a]-phi[0] if kind=='paired' else phi[a])
                y.append(g[a]-g[0]-j[a]+j[0] if kind=='paired' else g[a]-j[a])
                weight.append(1/(len(counts)*counts[row['case']]*len(indices)))
        x=np.asarray(x);y=np.asarray(y);weight=np.asarray(weight)
        scale=np.sqrt((weight[:,None]*x*x).sum(axis=0));scale[scale<1e-8]=1.
        z=x/scale
        choices=[]
        for alpha in ALPHAS:
            normal=sum(w*np.outer(v,v) for w,v in zip(weight,z))+alpha*np.eye(len(columns))
            rhs=sum(w*v*target for w,v,target in zip(weight,z,y))
            beta=np.linalg.solve(normal,rhs)
            candidate=dict(columns=columns,weights=beta.tolist(),scales=scale.tolist())
            by_case=defaultdict(list);counts_h=Counter()
            for sid,row in features[stage]['tune'].items():
                index,_=choice(row,candidate);g=raw['tune'][sid]['G']
                by_case[row['case']].append(float((g[:,index]-g[:,0]).mean()));counts_h[ACTIONS[index]]+=1
            gain=mean(mean(values) for values in by_case.values())
            choices.append(dict(alpha=alpha,gain=gain,weights=beta.tolist(),scales=scale.tolist(),choices=dict(counts_h)))
        best=max(c['gain'] for c in choices)
        selected=max((c for c in choices if best-c['gain']<=TOL),key=lambda c:c['alpha'])
        saved=models[stage][alias]
        check('independent weighted ridge '+alias,all(near(a,b) for a,b in zip(selected['weights'],saved['weights']))
            and all(near(a,b) for a,b in zip(selected['scales'],saved['scales'])) and selected['alpha']==saved['alpha'])
        check('independent tune selection '+alias,near(selected['gain'],saved['selection']['tune_equal_case_gain']))
        refits[alias]=selected
    family=max(('paired_chain','paired_augmented'),key=lambda name:(refits[name]['gain'],name=='paired_chain'))
    if abs(refits['paired_chain']['gain']-refits['paired_augmented']['gain'])<=TOL:family='paired_chain'
    check('primary chosen only by tune',family==frozen['augmented']['primary_family']==reports['augmented']['primary_family'])
    details['independent_fits_and_tune']=refits
    details['dev']={}
    for stage in ('base','augmented'):
        saved=reports[stage]['reports']['dev']
        reported_states={r['site_id']:r for r in saved['states']}
        summaries={name:defaultdict(list) for name in list(models[stage])+['depart','fixed10']}
        state_counts={name:Counter() for name in summaries}
        auxiliary={name:defaultdict(list) for name in models[stage]}
        auxiliary_cases={name:defaultdict(list) for name in models[stage]}
        selected_by_state={}
        for sid,row in features[stage]['dev'].items():
            observed=reported_states[sid];g=raw['dev'][sid]['G']
            selected={};scores={}
            for name,model in models[stage].items():selected[name],scores[name]=choice(row,model)
            selected.update(depart=0,fixed10=3)
            selected_by_state[sid]={name:ACTIONS[a] for name,a in selected.items()}
            for name,index in selected.items():
                state_counts[name][ACTIONS[index]]+=1
                gains={ref:stats(g[:,index]-g[:,selected[ref]]) for ref in ('depart','fixed10','old_chain')}
                summaries[name][row['case']].append({ref:s['mean'] for ref,s in gains.items()})
                if name in models[stage]:
                    item=observed['models'][name]
                    check('dev choice '+stage+'/'+sid+'/'+name,item['chosen_h']==ACTIONS[index])
                    for a,h in enumerate(ACTIONS):
                        compare_dict('dev score '+stage+'/'+sid+'/'+name+'/'+str(h),item['scores'][str(h)],
                            dict(J=row['J'][a],W=scores[name][a]-row['J'][a],total=scores[name][a],delta_vs_depart=scores[name][a]-scores[name][0]))
                    for ref,s in gains.items():compare_dict('dev paired metrics '+stage+'/'+sid+'/'+name+'/'+ref,item['comparisons'][ref],s)
                    other=dict(accepted_delta_vs_depart=float((raw['dev'][sid]['accepted'][:,index]-raw['dev'][sid]['accepted'][:,0]).mean()),
                        trips_saved_vs_depart=float((raw['dev'][sid]['trips'][:,0]-raw['dev'][sid]['trips'][:,index]).mean()),
                        extra_truck_time_cost_vs_depart=float((raw['dev'][sid]['truck_time'][:,index]-raw['dev'][sid]['truck_time'][:,0]).mean()),
                        actual_departure_delay=float(raw['dev'][sid]['departure_delay'][:,index].mean()))
                    compare_dict('dev accepted trips time '+stage+'/'+sid+'/'+name,item,other)
                    for key,value in other.items():auxiliary[name][key].append(value)
                    auxiliary_cases[name][row['case']].append(other)
        details['dev'][stage]={}
        for name,cases in summaries.items():
            case_means={case:{ref:mean(v[ref] for v in values) for ref in ('depart','fixed10','old_chain')}
                        for case,values in cases.items()}
            layouts=defaultdict(list)
            for case,values in case_means.items():layouts[captures['dev'][case]['spec']['layout_id']].append(values)
            layout_means={str(l):{ref:mean(v[ref] for v in values) for ref in ('depart','fixed10','old_chain')}
                          for l,values in layouts.items()}
            summary=saved['models'][name]
            check('dev state choice counts '+stage+'/'+name,summary['choice_counts']=={str(h):n for h,n in state_counts[name].items()})
            for case,values in case_means.items():compare_dict('dev case means '+stage+'/'+name+'/'+case,summary['cases'][case]['gains'],values)
            for layout,values in layout_means.items():compare_dict('dev layout means '+stage+'/'+name+'/'+layout,summary['layout_means'][layout],values)
            compare_dict('dev equal case mean '+stage+'/'+name,summary['equal_case_gains'],
                {ref:mean(v[ref] for v in case_means.values()) for ref in ('depart','fixed10','old_chain')})
            independent={ref:inference([values[ref] for values in layout_means.values()]) for ref in ('depart','fixed10','old_chain')}
            for ref,values in independent.items():compare_dict('dev t7 signflip '+stage+'/'+name+'/'+ref,summary['layout_inference'][ref],values)
            aux_cases={case:{key:mean(v[key] for v in values) for key in values[0]}
                for case,values in auxiliary_cases.get(name,{}).items()}
            aux_by_layout=defaultdict(list)
            for case,values in aux_cases.items():aux_by_layout[captures['dev'][case]['spec']['layout_id']].append(values)
            aux_layouts={str(l):{key:mean(v[key] for v in values) for key in values[0]}
                         for l,values in aux_by_layout.items()}
            details['dev'][stage][name]=dict(choice_counts=dict(state_counts[name]),case_count=len(case_means),layout_count=len(layout_means),
                case_means=case_means,layout_means=layout_means,layout_inference=independent,
                equal_state_auxiliary={k:mean(v) for k,v in auxiliary.get(name,{}).items()},
                auxiliary_case_means=aux_cases,auxiliary_layout_means=aux_layouts,
                equal_case_auxiliary={key:mean(v[key] for v in aux_cases.values()) for key in next(iter(aux_cases.values()),{})},
                equal_layout_auxiliary={key:mean(v[key] for v in aux_layouts.values()) for key in next(iter(aux_layouts.values()),{})})
        if stage=='augmented':
            check('primary all26 exactly fixed10',len(selected_by_state)==26 and all(v[family]==10 for v in selected_by_state.values()))
            check('primary raw paired fixed10 differences exactly zero',all(
                np.array_equal(raw['dev'][sid]['G'][:,3]-raw['dev'][sid]['G'][:,3],np.zeros(protocols['dev']['samples'])) for sid in selected_by_state))
            check('gate false without model gain beyond fixed10',not reports[stage]['proceed_to_full_day_model_comparison']
                and not reports[stage]['candidate_gates'][family]['pass_both_lower95']
                and independent is not None
                and details['dev'][stage][family]['layout_inference']['fixed10']['ci95']==[0.,0.])
    layouts={split:{s['layout_id'] for s in protocol['cases']} for split,protocol in protocols.items()}
    check('split layouts disjoint',len(set.union(*layouts.values()))==sum(map(len,layouts.values())))
    check('reserved test never used',not set.union(*layouts.values())&RESERVED
        and all(not p['independent_test_opened'] for p in protocols.values())
        and all(not r['independent_test_opened'] for r in reports.values()))
    check('no full-day policy experiment outputs',not (output/'full_day').exists())
    known_audit=read(output/'audit/full_day_selector_known_only.json')
    check('only known case selector validation',not known_audit['full_day_run'] and not known_audit['offline_future_labels_read']
        and known_audit['all_four_feature_vectors_match'] and known_audit['max_score_error']==0.)
    for stage in ('base','augmented'):
        for name in ('results.json','frozen_selection.json'):
            path=output/stage/name;source_receipts[stage+'/'+name]=sha(path)
        for path in (output/stage/'models').glob('*.json'):source_receipts[str(path.relative_to(output))]=sha(path)
    source_receipts['audit_code']=sha(__file__)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),independent_implementation=True,
        imported_fitter_or_simulator=False,all_passed=all(c['passed'] for c in checks),checks=checks,
        counts=dict(train_states=len(raw['train']),tune_states=len(raw['tune']),dev_states=len(raw['dev']),
            paired_files=sum(len(v) for v in receipts.values()),audited_full_day_branches=4*sum(len(v) for v in receipts.values()),
            dev_eligible_cases=len({r['case'] for r in features['base']['dev'].values()}),planned_dev_cases=16,dev_layouts=len(layouts['dev'])),
        source_receipts=source_receipts,raw_receipt_hashes={split:digest(v) for split,v in receipts.items()},
        details=details,full_day_policy_experiment_run=False,
        full_day_selector_validation='Six unit tests plus one known-state frozen-score consistency check only; no full-day policy rollout.',
        independent_test_opened=False)
    destination=output/'audit/fit_final_integrity.json';destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(all_passed=result['all_passed'],checks=len(checks),failed=[c for c in checks if not c['passed']],
        counts=result['counts'],output=str(destination)),ensure_ascii=False))
    if not result['all_passed']:raise AssertionError('independent final fit audit failed')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
