"""Finite waiting candidates and Monte Carlo residual linear VFA.

Training labels use ONLY the executed candidate and its actual episode return.
All policies share the same physical candidate oracle. No PI labels are used.
"""
from copy import deepcopy
from random import Random
from math import floor
from time import perf_counter
from .online import MyopicPolicy, evaluate_plan, select_best
from .quantities import dec


def sample_release_times(now, count, end, rng):
    """Public integer-uniform release law conditioned on history through now."""
    if type(count) is not int or count < 0:
        raise ValueError('nonnegative integer remaining count required')
    if count == 0:
        return []
    low = floor(now)+1
    if low > end:
        raise ValueError('positive remaining count after release support')
    return sorted(rng.randint(low,end) for _ in range(count))


def wait_pool(known, evaluated, review_interval=5, budget=16):
    """Delay an unstarted station departure; no real inventory is created."""
    from .kernel import require
    fixed_delta = None if callable(review_interval) else dec(review_interval)
    require(fixed_delta is None or fixed_delta > 0,'positive waiting review interval required')
    from .information import opportunities_open
    if not opportunities_open(known.public,known.now):
        return evaluated
    pool=list(evaluated)
    attempts=0
    # Deterministic bounded extension, identical for Myopic and learned VFA.
    for row in evaluated[:8]:
        for a in row['plan']['actions']:
            if a['kind'] not in ('truck_continue_circuit','truck_load') or dec(a['at'])!=known.now:
                continue
            t=known.trucks.get(a['actor'])
            if not t or t['location'] not in known.network['stations'] or t['active_operation'] is not None:
                continue
            if attempts>=budget:return pool
            delta = fixed_delta if fixed_delta is not None else dec(review_interval(known, t['location']))
            require(delta > 0, 'positive station review interval required')
            end=dec(known.public.get('revelation_end',known.now+delta))
            minutes=min(delta,end-known.now)
            if minutes<=0:continue
            for minutes in (minutes,):
                attempts+=1
                plan=deepcopy(row['plan'])
                plan['id']=row['id']+f"_wait_{a['actor']}_{minutes}"
                for cmd in plan['actions']:
                    if cmd.get('actor')==a['actor'] and dec(cmd['at'])>=known.now:
                        cmd['at']=dec(cmd['at'])+minutes
                plan.setdefault('wait_reviews',{})[a['actor']]=known.now+minutes
                candidate=evaluate_plan(known,plan,'WK')
                if not candidate['feasible']:
                    # Conservative fallback: shift remaining planned operations
                    # together. Already-started operations live in the physical
                    # event queue, not these commands, and are never shifted.
                    # Full replay still rejects deadlines, commitment conflicts,
                    # dispatch-cutoff violations and unavailable onboard cargo.
                    plan=deepcopy(row['plan'])
                    plan['id']=row['id']+f"_wait_{a['actor']}_{minutes}_synchronized"
                    for cmd in plan['actions']:
                        if dec(cmd['at'])>=known.now:
                            cmd['at']=dec(cmd['at'])+minutes
                    plan.setdefault('wait_reviews',{})[a['actor']]=known.now+minutes
                    candidate=evaluate_plan(known,plan,'WK')
                if candidate['feasible'] and candidate['allowed']:
                    pool.append(candidate)
    return pool


def vector(known, candidate):
    from .information import unknown
    if unknown(known.public):raise ValueError('legacy VFA cannot score unknown-total demand; use resource features')
    n=known.public.get('future_count') or 0
    if n==0:return [0.0]*10
    actions=candidate['plan']['actions']
    departures=[max(0,float(dec(a['at'])-known.now)) for a in actions if a['kind']=='truck_continue_circuit']
    horizon=max(1, float(known.public.get('revelation_end',100))-float(known.now))
    hold=min(1,sum(departures)/horizon)
    accepted=sum(candidate['plan']['decisions'].values())
    active=sum(t['departed_at'] is not None and t['return_time'] is None for t in known.trucks.values())
    free=sum(u['active_operation'] is None for u in known.drones.values())
    # Fixed dimensions; request quantities remain integers in the executor.
    return [1,n/6,float(known.now)/100,hold,hold*n/6,hold*free/4,
            accepted/3,len(candidate['plan']['new_trucks'])/3,active/4,
            float(candidate['score'])/500]


class LearningPolicy(MyopicPolicy):
    def __init__(self, weights=None, *, seed=0, explore=0, scale=1, **kwargs):
        super().__init__('WK',**kwargs)
        self.weights=weights
        self.scale=scale
        self.rng=Random(seed)
        self.explore=explore
        self.samples=[]
        self.selection_log=[]
        self.timings=[]

    def decide(self,known):
        start=perf_counter()
        result=super().decide(known)
        self.timings.append(perf_counter()-start)
        return result

    def choose(self,known,evaluated):
        pool=wait_pool(known,evaluated)
        scored=[]
        for c in pool:
            x=vector(known,c)
            w=sum(a*b for a,b in zip(x,self.weights))*self.scale if self.weights is not None else 0
            scored.append((float(c['score'])+w,c,x,w))
        if self.explore and self.rng.random()<self.explore:
            row=self.rng.choice(scored)
        else:
            row=min(scored,key=lambda r:(-r[0],r[1]['id']))
        _,chosen,x,w=row
        self.samples.append((x,float(chosen['score'])))
        self.selection_log.append(dict(at=float(known.now),selected=chosen['id'],J=float(chosen['score']),W=w,
                                      pool=[dict(id=c['id'],J=float(c['score']),W=v,score=s) for s,c,_,v in scored]))
        return chosen
