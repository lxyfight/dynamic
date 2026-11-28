"""Explicit adapted Greedy/RH and conditional full-day rollout comparators."""
from copy import deepcopy
from collections import defaultdict
from random import Random
from time import perf_counter
from math import floor
import heapq
from .formal_learning import FormalPolicy
from .learning import wait_pool
from .online import evaluate_plan,select_best,install_plan,_planning_copy,OnlineCoordinator
from .quantities import dec
from .constraints import eligibility


class Comparator(FormalPolicy):
    def __init__(self,method,*,rollout_h=10,rollout_seconds=1.,**kw):
        super().__init__(**kw);self.method=method;self.rollout_h=rollout_h;self.rollout_seconds=rollout_seconds;self.rollout_log=[]
    def choose(self,known,evaluated):
        pool=self.final_pool(known,evaluated)
        if self.method=='Greedy':
            base=next(c for c in pool if c['id']=='original')
            positive=[c for c in pool if c['score']>base['score'] and any(c['plan']['decisions'].values())]
            # Simultaneous admission follows the public request-ID order in the
            # shared proposal pool; compare earliest new pickup depot finish.
            chosen=min(positive,key=lambda c:(tuple(float(c['state'].orders[o]['completed_at']) for o,v in sorted(c['plan']['decisions'].items()) if v),c['id'])) if positive else base
        elif self.method=='RH':
            # A second bounded neighborhood pass distinguishes adapted RH
            # from the one-pass common enhanced Myopic policy.
            from .enhancement import enhance_pool
            extra,_=enhance_pool(known,sorted(pool,key=lambda c:(-c['score'],c['id']))[:4],8,self.waiting)
            chosen=select_best(pool+extra)
        elif self.method=='Rollout':
            chosen=self.rollout(known,pool)
        else:raise ValueError(self.method)
        self.selection_log.append(dict(at=float(known.now),method=self.method,selected=chosen['id'],J=float(chosen['score']),pool_size=len(pool)))
        return chosen

    def rollout(self,known,pool):
        baseline=select_best(pool)
        from .information import opportunities_open
        if not opportunities_open(known.public,known.now):return baseline
        # Four predeclared candidate finalists; shared sampled worlds each round.
        finalists=sorted(pool,key=lambda c:(-c['score'],c['id']))[:4]
        totals=[0.]*len(finalists);rounds=0;deadline=perf_counter()+self.rollout_seconds
        seed=int(float(known.now)*1000003)+len(known.orders)*7919
        for h in range(self.rollout_h):
            scenario=sample_future(known,Random(seed+h*104729));values=[]
            try:
                for candidate in finalists:
                    if perf_counter()>deadline:raise TimeoutError('rollout round incomplete')
                    world=_planning_copy(known);world._hidden=deepcopy(scenario)
                    # Reveal events are synthetic only; no actual hidden stream
                    # is available in known_clone or this sampler.
                    by=defaultdict(list)
                    for oid,o in scenario.items():by[o['release']].append(oid)
                    for at,ids in by.items():world._schedule(dec(at),'request_reveal',dict(orders=ids),priority=1)
                    world._schedule(dec(known.public['revelation_end']),'revelation_end',{},priority=1)
                    world._compute_deadline=deadline;world.full_day=True;world.stop=dec(10000);world._has_run=False
                    install_plan(world,candidate['plan']);driver=OnlineCoordinator(FormalPolicy(candidate_budget=self.candidate_budget),extend_horizon=True)
                    world.run(decision_hook=driver.on_event,stop_at_terminal=True)
                    from .kernel import COSTS
                    values.append(float(world.ledger['revenue']-sum(world.ledger[k] for k in COSTS)))
            except TimeoutError:break
            for k,v in enumerate(values):totals[k]+=v
            rounds+=1
        self.rollout_log.append(dict(at=float(known.now),target=self.rollout_h,complete_common_rounds=rounds,candidates=len(finalists),fallback=rounds==0,seconds_budget=self.rollout_seconds))
        return baseline if not rounds else finalists[min(range(len(finalists)),key=lambda k:(-totals[k]/rounds,finalists[k]['id']))]


def sample_future(known,rng):
    from .information import unknown
    if unknown(known.public):
        from .unknown_demand import sample_orders
        return sample_orders(known.public,known.now,rng,known.network,known.cfg)
    """City-conditioned counts, full coordinates/weights/releases/deadlines."""
    result={};low=floor(float(known.now))+1;end=int(known.public['revelation_end'])
    totals=known.public['city_pickup_totals']
    for city,total in sorted(totals.items()):
        remaining=total-sum(o['kind']=='P' and o['city']==city for o in known.orders.values())
        assert remaining>=0
        ix=int(city[1:]);stations=[f'L{ix}+',f'L{ix}-'];x=known.network['stations'][stations[0]][0]
        for j in range(remaining):
            assert low<=end
            weight=rng.randint(6,19);release=rng.randint(low,end);xy=[round(x+rng.uniform(-10,10),6),round(rng.choice([-1,1])*rng.uniform(.5,10),6)]
            oid=f'SIM_{city}_{j}';eligible=[s for s in stations if eligibility(dict(xy=xy,weight=weight),known.network['stations'][s],known.cfg)['feasible']]
            assert eligible
            result[oid]=dict(id=oid,kind='P',weight=weight,release=dec(release),deadline=dec(release+360),xy=xy,city=city,eligible_stations=eligible,accepted=None,completed_at=None,available_at=None,collected_at=None,station=None,assigned_drone=None)
    assert len(result)==known.public['future_count']
    return result
