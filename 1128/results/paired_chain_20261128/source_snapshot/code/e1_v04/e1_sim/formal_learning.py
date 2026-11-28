from .learning import LearningPolicy,wait_pool
from .formal_features import vector
from .kernel import require
from .initializer import digest

class FormalPolicy(LearningPolicy):
    def __init__(self,weights=None,group=None,*,enhance=True,enhancement_budget=16,review_interval=5,wait_budget=16,**kwargs):
        super().__init__(weights,**kwargs);self.group=group
        require(enhancement_budget>=0 and wait_budget>=0 and review_interval>0,'invalid revision search settings')
        self.enhance=enhance;self.enhancement_budget=enhancement_budget
        self.review_interval=review_interval;self.wait_budget=wait_budget
        self.pool_log=[]
    def final_pool(self,known,evaluated):
        pool=wait_pool(known,evaluated,self.review_interval,self.wait_budget) if self.waiting=='WK' else list(evaluated)
        stats=dict(attempted=0,accepted=0,by_move={})
        if self.enhance:
            from .enhancement import enhance_pool
            pool,stats=enhance_pool(known,pool,self.enhancement_budget,self.waiting)
        self.pool_log.append(dict(at=float(known.now),base=len(evaluated),final=len(pool),enhancement=stats))
        return pool
    def choose(self,known,evaluated):
        from .information import unknown
        group=self.group or ('resource' if unknown(known.public) else 'full')
        pool=self.final_pool(known,evaluated);scored=[]
        for c in pool:
            x=vector(known,c,group)
            require(self.weights is None or len(self.weights)==len(x),'VFA weight/feature dimension mismatch')
            w=sum(a*b for a,b in zip(x,self.weights)) if self.weights is not None else 0.
            scored.append((float(c['score'])+w,c,x,w))
        row=self.rng.choice(scored) if self.explore and self.rng.random()<self.explore else min(scored,key=lambda r:(-r[0],r[1]['id']))
        _,chosen,x,w=row;self.samples.append((x,float(chosen['score'])))
        self.selection_log.append(dict(at=float(known.now),selected=chosen['id'],plan_hash=digest(chosen['plan']),features=x,J=float(chosen['score']),W=w,group=group,
             pool=[dict(id=c['id'],J=float(c['score']),W=v,score=z) for z,c,xx,v in scored]))
        return chosen
