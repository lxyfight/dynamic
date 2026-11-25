"""Exact finite station subproblem for a feature screen, not a full-system oracle.

One free slot remains on a waiting truck. Five-minute reviews. A slot with
one or more feasible offers supplies a single opportunity to occupy that slot.
The stylized net marginal gain per offer already includes subsequent service.
No other known pickups are pending in this restricted state. Leaving now is
the zero reference. Waiting pays depot-away time even if no offer arrives.
"""
from math import expm1
from random import Random
import numpy as np
from run_multiscale import ROOT, CFG, write


def states(seed,count):
    rng=Random(seed)
    result=[]
    for _ in range(count):
        slots=rng.randint(1,8)
        base=rng.uniform(.003,.06)
        rates=[base*rng.choice((.5,1.,2.)) for _ in range(slots)]
        margin=rng.uniform(2,55)
        prob=[-expm1(-5*r) for r in rates]
        cost=5*CFG['truck_time_cost']
        best=0.
        for p in reversed(prob):
            wait=-cost+p*max(margin,best)+(1-p)*best
            best=max(0.,wait)
        # Q(wait)-J_known(wait), where J_known(wait)=-cost.
        feature=[1.,slots,prob[0],sum(prob),prob[0]*margin,
                 -expm1(-5*sum(rates))*margin]
        result.append((feature,wait+cost))
    return tuple(np.array(v) for v in zip(*result))


def main():
    train,y=states(2026112401,800)
    evaluation,target=states(2026112402,500)
    cost=5*CFG['truck_time_cost']
    actual_wait=target-cost
    result={}
    for name,cols in {'clock':[0,1], 'arrival':[0,1,2,3], 'opportunity_margin':[0,1,2,3,4,5]}.items():
        weights=np.linalg.lstsq(train[:,cols],y,rcond=None)[0]
        pred=evaluation[:,cols]@weights
        loss=np.maximum(0,actual_wait)-np.where(pred>cost,actual_wait,0)
        result[name]=dict(r2=1-float(np.sum((pred-target)**2)/np.sum((target-target.mean())**2)),
                          value_mae=float(np.mean(abs(pred-target))),one_decision_regret=float(loss.mean()),
                          wait_depart_accuracy=float(np.mean((pred>cost)==(actual_wait>0))))
    write(ROOT/'tmp/global_wait_dp_screen_v1.json',dict(train_states=800,held_states=500,metrics=result,
        assumption='five-minute decision intervals, one available slot, first available opportunity per interval, marginal gain public, exact restricted nonanticipative DP; not a full day'))
    print(result)


if __name__=='__main__':
    main()
