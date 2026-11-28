from .learning import LearningPolicy
from .online import select_best
from .wtq_compiler import compile_candidate

class WTQPolicy(LearningPolicy):
    def __init__(self,T=15,Q=30,**kwargs):
        super().__init__(**kwargs);self.T=T;self.Q=Q
    def choose(self,known,evaluated):
        checked=[compile_candidate(known,c,self.T,self.Q) for c in evaluated if c['feasible'] and c['allowed']]
        chosen=select_best(checked)
        self.selection_log.append(dict(at=float(known.now),selected=chosen['id'],J=float(chosen['score']),W=0,
                                      T=self.T,Q=self.Q,pool=[dict(id=c['id'],feasible=c['feasible'],J=float(c['score']) if c['feasible'] else None,reason=c.get('reason')) for c in checked]))
        return chosen
