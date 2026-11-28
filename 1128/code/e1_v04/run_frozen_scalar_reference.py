"""Run the earlier, fixed single-feature VFA on the new comparison worlds."""
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

from e1_sim.initializer import digest
from high_demand_features import run_case
from run_frozen_reopt_comparison import CASES, OUT as COMPARISON
from run_multiscale import read, write


OUT = COMPARISON/'scalar_reference'
THETA = 28


def evaluate(case):
    w = read(COMPARISON/'worlds'/(digest(case)+'.json'))
    path = OUT/'runs'/('_'.join(map(str, case))+'.json')
    if not path.exists():
        r = run_case(w, teacher=THETA)
        r.update(case=case, theta=THETA)
        write(path,r)
    r = read(path)
    assert r['theta']==THETA and tuple(r['case'])==case
    assert r['input_hash']==digest(w['input']) and r['i0_hash']==w['i0_hash']
    return r


def report(rows):
    paired = []
    for ref in rows:
        case = ref['case']
        v = read(COMPARISON/'runs'/('_'.join(map(str,case))+'_VFA.json'))
        assert (v['input_hash'],v['i0_hash'])==(ref['input_hash'],ref['i0_hash'])
        ledger = ref['ledger']
        assert abs(ledger['revenue']-sum(v for k,v in ledger.items()
                   if k not in ('net','revenue'))-ref['net'])<1e-7
        paired.append(dict(case=case, scalar=ref['net'], frozen=v['net'],
                           gain=v['net']-ref['net'], accepted_scalar=ref['accepted'],
                           accepted_frozen=v['accepted']))
    summaries = {}
    for n in (100,150,'all'):
        group = [r for r in paired if n=='all' or r['case'][1]==n]
        ds = [r['gain'] for r in group]
        summaries[n] = dict(cases=len(ds),mean_scalar=mean(r['scalar'] for r in group),
                            gain=mean(ds),wins=sum(d>1e-7 for d in ds),
                            losses=sum(d< -1e-7 for d in ds),ties=sum(abs(d)<=1e-7 for d in ds))
    write(OUT/'comparison.json',dict(groups=summaries,paired=paired))
    print(summaries)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--report',action='store_true',help='Compare completed scalar and frozen runs.')
    args = parser.parse_args()
    protocol = dict(cases=CASES, theta=THETA, model='Previously fixed extra recovery exposure coefficient',
                    original_frozen=read(COMPARISON/'protocol.json')['freeze_hash'])
    if (OUT/'protocol.json').exists():
        assert digest(read(OUT/'protocol.json')) == digest(protocol)
    write(OUT/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(evaluate,CASES))
    if args.report:
        report(rows)
    else:
        print('Completed', len(rows), 'scalar reference replays.')
