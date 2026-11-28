"""Trace changed non-target commands in ambiguous waiting offers."""
from run_all_wait_counterfactual import DEST, FeaturePolicy, read, write, new_reviews, dec
from run_wait_counterfactual import cases, world_for, PRIOR, run


class Inspected(Exception):
    pass


class Inspector(FeaturePolicy):
    def __init__(self,model,times):
        super().__init__(model)
        self.times=set(times)
        self.result=[]

    def pick(self,known,ranked,theta):
        plan,record=super().pick(known,ranked,theta)
        t=str(known.now)
        if t not in self.times:return plan,record
        rows=[]
        for _,_,candidate in ranked:
            for truck,review in new_reviews(known,candidate['plan']):
                p=candidate['plan']
                parent_id=candidate['id'].split('_wait_'+truck+'_',1)[0]
                parent=next((c for _,_,c in ranked if c['id']==parent_id),None)
                shifted=[]
                if parent:
                    old=parent['plan']['actions']
                    new=p['actions']
                    if len(new)==len(old):
                        for a,b in zip(old,new):
                            if a['actor']!=truck and a['actor']==b['actor'] and a['kind']==b['kind'] and dec(a['at'])!=dec(b['at']):
                                shifted.append(a['actor'])
                rows.append(dict(id=candidate['id'],parent=parent_id,parent_in_pool=parent is not None,
                                 station_truck=truck,shifted_other_actors=sorted(set(shifted))))
        assert rows
        self.result.append(dict(time=t,wait_offers=rows))
        self.times.remove(t)
        if not self.times:raise Inspected
        return plan,record


def main():
    all_details=[]
    for spec in cases('confirmation'):
        prefix='_'.join(map(str,spec))
        audit=read(DEST/'audit'/(prefix+'.json'))
        times=[r['time'] for r in audit if r['status']=='other_unmatched_or_nonisolated']
        if not times:continue
        p=Inspector(read(PRIOR/'freeze.json')['model'],times)
        try:
            run(world_for(spec),28,policy_override=p)
        except Inspected:
            assert not p.times
        assert len(p.result)==len(times)
        all_details.extend(dict(case=spec,**r) for r in p.result)
    write(DEST/'unmatched_details.json',all_details)
    print([(r['case'],r['time'],[(v['parent_in_pool'],len(v['shifted_other_actors'])) for v in r['wait_offers']]) for r in all_details])


if __name__=='__main__':
    main()
