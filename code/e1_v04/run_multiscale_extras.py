"""PI relaxation study, isolated timing and complete artifact audit."""
from e1_sim.pi_adapter import PI_PYTHON
from run_multiscale import *
from e1_sim.pi_adapter import decode_commands


def pi(workers):
    frozen=read(OUT/'frozen_models.json');models=[dict(name=n,kind=n) for n in ['Myopic','Greedy','RH']]+frozen['models']
    cases=[]
    for c in [2,3]:
        for total in [10,15,20]:
            for j in range(10):
                p=.2 if j<5 else .4
                cases.append(dict(cities=c,n=0,p=p,small_total=total,layout_seed=seedof('pi-layout',c,total,j),day_seed=seedof('pi-day',c,total,j),identity=f'pi_c{c}_N{total}_d{j}',layout_id=f'pi_c{c}_N{total}_l{j}'))
    results=parallel([(sp,models,'pi_online',0,0) for sp in cases],workers,'PI feasible lower bounds')
    rows=[]
    for i,result in enumerate(results,1):
        ip=OUT/'instances'/(result['id']+'.json');bp=OUT/'pi_bounds'/(result['id']+'.json');bp.parent.mkdir(exist_ok=True)
        if not bp.exists():subprocess.run([str(PI_PYTHON),'-m','e1_sim.pi_route_bound',str(ip),str(bp),'60'],check=True,capture_output=True)
        bound=read(bp);lower=max(r['net'] for r in result['rows']);require(bound['upper_bound']+1e-5>=lower,'PI upper bound below executable incumbent')
        row=dict(id=result['id'],cities=read(ip)['cities'],N=len(read(ip)['input']['orders']),p=read(ip)['p'],lower=lower,upper=bound['upper_bound'],absolute_bracket=bound['upper_bound']-lower,routes=bound['routes'],solver_status=bound['solver_status'],solve_seconds=bound['solve_seconds'])
        rows.append(row)
        print('PI',i,len(results),'width',round(row['absolute_bracket'],3),flush=True)
    write(OUT/'pi_results.json',rows)

def timing():
    models=[dict(name=n,kind=n) for n in ['Myopic','Greedy','RH']]+read(OUT/'frozen_models.json')['models']
    rows=[]
    for c in [2,4,6,8,10]:
        sp=next(s for s in specs('test') if s['cities']==c and s['n']==20 and s['p']==.4 and s['identity'].endswith('_l0_d0'))
        instance=read(OUT/'instances'/(sp['identity']+'.json'))
        for m in models:
            row,detail,_=episode(instance,m,cache=None)
            saved=read(OUT/'test'/(sp['identity']+'.json'))
            original=next(r for r in saved['rows'] if r['model']==m['name'])
            require(row['net']==original['net'] and row['accepted']==original['accepted'],'uncached replay changed results')
            rows.append(row);write(OUT/'isolated_timing.json',rows)
            print('isolated timing',c,m['name'],row['seconds'],flush=True)

def audit():
    require(read(OUT/'source_hashes.json')==source_hashes(),'campaign source changed')
    split_layouts={k:set() for k in ['train','validation','test']}
    for s in SEEDS:split_layouts['train'].update(x['layout_seed'] for x in specs('train',s))
    for k in ['validation','test']:split_layouts[k].update(x['layout_seed'] for x in specs(k))
    require(not(split_layouts['train']&split_layouts['test'] or split_layouts['train']&split_layouts['validation'] or split_layouts['validation']&split_layouts['test']),'layout leakage')
    counts={};seen=0
    for phase,want in [('test',600),('rollout',200),('pi_online',60)]:
        files=list((OUT/phase).glob('*.json'));require(len(files)==want,'missing '+phase)
        runs=0
        for file in files:
            day=read(file);rows=day['rows'];require(len({r['input_hash'] for r in rows})==len({r['i0_hash'] for r in rows})==1,'unpaired inputs/I0')
            for row in rows:
                require(row['terminal'] and row['accepted']==row['completed'],'unfinished obligations')
                with gzip.open(OUT/'traces'/phase/(day['id']+'__'+row['model']+'.json.gz'),'rt',encoding='utf8') as f:detail=json.load(f)
                require(not detail['audit']['errors'],'audit error')
                require(all(abs(dec(v)-dec(detail['audit']['ledger'][k]))<dec('1e-7') for k,v in row['ledger'].items()),'ledger discrepancy')
                runs+=1
        counts[phase]=dict(days=len(files),runs=runs)
    for phase,n in [('train_r1',300),('train_r2',600),('train_r3',900),('validation_100',120),('validation_300',120),('validation_600',120)]:
        files=list((OUT/phase).glob('*.json'));require(len(files)==n,'missing '+phase)
        for file in files:
            for row in read(file)['rows']:require(row['terminal'] and row['accepted']==row['completed'],'training/validation failure')
        counts[phase]=dict(days=len(files))
    pirows=read(OUT/'pi_results.json');require(len(pirows)==60 and all(r['upper']+1e-5>=r['lower'] for r in pirows),'invalid PI brackets')
    timingrows=read(OUT/'isolated_timing.json');require(len(timingrows)==30,'isolated timing incomplete')
    report=dict(passed=True,counts=counts,source_hashes_match=True,layout_family_disjoint=True,
        unique_layout_counts={k:len(v) for k,v in split_layouts.items()},all_accepted_completed=True,all_test_independent_ledgers_match=True,
        PI_brackets=60,uncached_reproduced_episodes=30,limitations=['PI relaxation bound, not exact PI optimum','finite candidate search','fixed count main budgets, no hard 1-second end-to-end claim','only three test layouts per city','B1 formal retraining and E4-E8 mechanisms not included'])
    write(OUT/'最终核验.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    raise RuntimeError('历史入口已封存；v04请运行verify_unknown_demand.py，不覆盖旧实验')
