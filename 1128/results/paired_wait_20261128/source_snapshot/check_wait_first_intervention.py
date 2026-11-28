"""Retrospective single-scenario explanation; never used to train or select a policy."""
from e1_sim.online import prepare_online
from run_wait_increment_learning import BASE, OUT, WithWait, read, write, digest
from run_wait_counterfactual import pure_pairs


class FoundFirst(Exception):
    pass


class InspectFirst(WithWait):
    def pick(self,known,ranked,theta):
        chosen,record=super().pick(known,ranked,theta)
        if self.probes and self.probes[-1]['t']==float(known.now) and self.probes[-1]['wait']:
            dep,_=super(WithWait,self).pick(known,ranked,theta)
            candidate=next(c for _,_,c in ranked if c['plan'] is dep)
            _,pairs=pure_pairs(known,ranked,candidate)
            truck,station,wait=min(pairs,key=lambda t:(t[0],t[2]['id']))
            assert wait['plan'] is chosen and truck==self.probes[-1]['truck'] and station==self.probes[-1]['station']
            self.match=dict(wait=digest(chosen),depart=digest(dep),time=str(known.now),
                            physical_prefix=digest(known.trace),decision_prefix=digest(self.records))
            raise FoundFirst
        return chosen,record


def main():
    case=[526,100,'dual']
    world=read(OUT/'worlds'/(digest(case)+'.json'))
    model=read(OUT/'expanded/wait_model_freeze.json')
    checker=InspectFirst(read(BASE)['model'],model['learned']['model'])
    s,driver=prepare_online(world['input'],world['configuration'],world['network'],full_day=True)
    s._record_snapshots=False
    driver.policy=checker
    try:
        s.run(decision_hook=driver.on_event,stop_at_terminal=True)
    except FoundFirst:
        pass
    hit=checker.match
    base=read(OUT/'runs/526_100_dual_depart.json')
    all_waits=read(OUT/'expanded/online/526_100_dual.json')
    prefix=all_waits['probes'][:len(checker.probes)]
    assert prefix==checker.probes and sum(p['wait'] for p in prefix)==1
    rows=[read(p) for p in (OUT/'runs').glob('526_100_dual_wait_*.json')]
    single,=[r for r in rows if r['target']['time']==hit['time'] and r['target']['wait']==hit['wait']]
    target=single['target']
    assert target['depart']==hit['depart'] and target['prefix']==hit['physical_prefix']
    assert target['prefix_decisions']==hit['decision_prefix']
    assert all(r['i0_hash']==world['i0_hash'] and r['input_hash']==world['input_hash']
               for r in (base,all_waits,single))
    assert single['operation_metrics']['new_wait_chosen']==1
    result=dict(case=case,method='Retrospective explanation, same realized requests and initial plan.',
                first_match=hit,score=prefix[-1]['score'],input_hash=world['input_hash'],
                full_day={label:dict(net=r['net'],accepted=r['accepted'],gain=r['net']-base['net'],
                                     waits=r['operation_metrics']['new_wait_chosen'])
                          for label,r in [('frozen',base),('first_only',single),('repeated',all_waits)]})
    write(OUT/'expanded/first_intervention_526.json',result)
    print(result['full_day'])


if __name__=='__main__':
    main()
