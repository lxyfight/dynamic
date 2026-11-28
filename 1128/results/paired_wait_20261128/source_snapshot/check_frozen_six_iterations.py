"""Recompute fixed-basis checkpoints and paired ledgers from saved raw episodes."""
import csv
import math
import numpy as np

from run_frozen_six_iterations import (PARENT, OUT, BATCHES, SHAPES, VALIDATION, TRANSFER,
                                      cases, digest, paired, read, refit)


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-7),(a,b)


def main():
    result=read(OUT/'results.json')
    p=result['protocol']
    assert p==read(OUT/'protocol.json')
    model=read(PARENT/'freeze.json')['model']
    seen={}; days={}
    rows=[read(path) for path in (OUT/'runs').glob('*.json')]
    for r in rows:
        key=(r['stage'],r['checkpoint'],tuple(r['case']))
        assert key not in seen
        seen[key]=r
        spec=tuple(r['case'])
        if spec not in days:
            days[spec]=read(OUT/'worlds'/(digest(spec)+'.json'))
        world=days[spec]
        assert r['input_hash']==world['input_hash']==digest(world['input'])
        assert r['i0_hash']==world['i0_hash']
        ledger=r['ledger']
        close(r['net'],float(ledger['revenue'])-sum(float(v) for k,v in ledger.items() if k not in ('revenue','net')))
        close(r['net'],ledger['net'])
    assert len(rows)==12+16+8 and len(days)==12+4+4
    baseline=[seen['validation',0,s] for s in cases(VALIDATION)]
    updated=[seen['validation',12,s] for s in cases(VALIDATION)]
    expected=paired(updated,baseline)
    # Fresh workers return tuple keys; cached and restored JSON rows contain lists.
    for i in range(16):
        mixed=[dict(r,case=tuple(r['case'])) if i&(1<<j) else r for j,r in enumerate(updated)]
        reverse=[r if i&(1<<j) else dict(r,case=tuple(r['case'])) for j,r in enumerate(baseline)]
        close(paired(mixed,reverse)['gain'],expected['gain'])
    try:
        paired([dict(updated[0],case=[-1,*updated[0]['case'][1:]]),*updated[1:]],baseline)
    except AssertionError:
        pass
    else:
        raise ValueError('different cases must fail pairing')
    previous=[read(f) for f in (PARENT/'runs').glob('*.json')]
    bank=sorted([r for r in previous if r['policy']=='teacher' and r['layout'] in (410,411,412,413)],key=lambda r:r['case'])
    curve=result['validation_curve']
    expected_models=[model]
    for i, layouts in enumerate(BATCHES):
        new=[seen['training',4*i,sp] for sp in cases(layouts)]
        assert all(r['model_hash']==digest(expected_models[i]) for r in new)
        for r in new:
            world=days[tuple(r['case'])]
            arrivals=[(float(o['release']),o['weight']*world['configuration']['rejection_per_kg'])
                      for o in world['input']['orders'] if o['kind']=='P']
            for s in r['samples']:
                # This conditional label is computed only after the full simulated episode.
                close(s['y'],r['net']-s['j']+sum(v for t,v in arrivals if t>s['t']))
        bank.extend(new)
        fitted,count=refit(bank,model)
        saved=read(OUT/f'model_ep{4*(i+1)}.json')
        assert np.allclose(fitted['weights'],saved['weights']) and count==curve[i+1]['training_samples']
        close(fitted['intercept'],saved['intercept'])
        expected_models.append(saved)
    outputs=[]
    for stage, checkpoints, layouts in [('validation',range(4),VALIDATION),('transfer',(3,),TRANSFER)]:
        for i in checkpoints:
            group=curve[i]['validation'] if stage=='validation' else result['transfer_final_vs_frozen']
            expected={}
            for spec in cases(layouts):
                r=seen[stage,4*i,spec]
                b=seen[stage,0,spec]
                assert r['model_hash']==digest(expected_models[i]) and not r['samples']
                assert b['model_hash']==digest(model) and b['input_hash']==r['input_hash'] and b['i0_hash']==r['i0_hash']
                expected[spec]=r['net']-b['net']
                outputs.append([stage,8+4*i,4*i,*spec,r['net'],b['net'],expected[spec],r['accepted'],b['accepted']])
            for pair in group['pairs']:
                close(pair['gain'],expected[tuple(pair['case'])])
            close(group['gain'],sum(expected.values())/len(expected))
    path=OUT/'checkpoint_paired.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.writer(f)
        out.writerow(['用途','累计完成训练episode','本轮新增训练episode','布局','总订单数','空间分布',
                      '检查点净收益_元','原冻结策略净收益_元','配对净增益_元','检查点取件数','原冻结策略取件数'])
        out.writerows(outputs)
    print('VERIFIED',len(rows),'raw episodes;',len(outputs),'paired checkpoint rows')


if __name__=='__main__':
    main()
