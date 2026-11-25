"""UNACCEPTED diagnostic of time-window candidate filtering.

Physical full-day feasibility is checked, but this is NOT the WTQ controller:
it can retain delayed reservations after closure. See the saved counterexample.
Do not use this class as a formal WTQ baseline before repairing that semantic gap.
"""
from copy import deepcopy
from .learning import LearningPolicy
from .online import evaluate_plan, select_best
from .quantities import dec

def time_threshold_candidate(known, candidate, threshold):
    plan=deepcopy(candidate['plan'])
    for _ in range(100):
        row=evaluate_plan(known,plan,'WK')
        if not row['feasible']:return None
        trace=known.trace+row['trace'];arrivals={};changed=False
        for e in trace:
            key=e.get('actor');kind=e['kind']
            if kind=='truck_arrived':arrivals[key]=(e['destination'],dec(e['at']))
            if kind=='truck_load_started' and dec(e['at'])>=known.now:
                station,at=arrivals.get(key,(e['station'],known.now))
                close=at+dec(threshold)
                if dec(e['at'])>=close and any(dec(row['state'].orders[o]['available_at'])>close for o in e['orders']):
                    return None
            if kind!='road_started' or dec(e['at'])<known.now or key not in arrivals:continue
            station,at=arrivals[key]
            if station!=e['source'] or station not in known.network['stations']:continue
            # Only deliberate stops have an explicit departure command.
            departure=next((a for a in plan['actions'] if a.get('actor')==key and a['kind']=='truck_continue_circuit' and dec(a['at'])==dec(e['at'])),None)
            if departure is None:continue
            delta=at+dec(threshold)-dec(e['at'])
            if delta>0:
                for a in plan['actions']:
                    if a.get('actor')==key and dec(a['at'])>=dec(e['at']):a['at']=dec(a['at'])+delta
                changed=True;break
        if not changed:return row
    return None

class TimeThresholdPolicy(LearningPolicy):
    def __init__(self,threshold,**kwargs):
        super().__init__(**kwargs);self.threshold=threshold

    def choose(self,known,evaluated):
        pool=[r for c in evaluated if (r:=time_threshold_candidate(known,c,self.threshold)) is not None]
        # Incumbents are already checked against this same threshold.
        chosen=select_best(pool)
        self.selection_log.append(dict(at=float(known.now),selected=chosen['id'],J=float(chosen['score']),W=0,
                                      threshold=self.threshold,pool=[dict(id=c['id'],J=float(c['score'])) for c in pool]))
        return chosen
