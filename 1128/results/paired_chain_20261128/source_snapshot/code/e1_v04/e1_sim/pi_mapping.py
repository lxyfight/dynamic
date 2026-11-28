"""Independent non-service waypoint dominance witness; not a PI optimizer."""
from .constraints import distance
from .quantities import dec,quantity
from .kernel import require

def compression_mapping(configuration, data):
    p=data['original_path'];o=data['original_schedule'];c=data['compressed_schedule']
    q=data['orders'][0];station=data['local_station']['xy']
    weight=quantity(q['weight'],positive=True)
    require(weight<=configuration['drone_capacity'],'payload capacity')
    require(all(u['takeoff_at'] is None and u['active_operation'] is None for u in data['drones']),
            'dominance witness cannot revise an already airborne commitment')
    require(p[0]==station and p[-1]==station and p[-2]==q['xy'],'path endpoints mismatch')
    require(q['kind']=='D' and q['accepted'] is True,'witness requires mandatory delivery')
    original=sum(distance(a,b) for a,b in zip(p,p[1:]))
    direct=2*distance(station,q['xy']);speed=dec(configuration['drone_speed_km_min'])
    service=dec(configuration['customer_service_per_request'])
    require(dec(o['takeoff'])==dec(c['takeoff']),'takeoff changed')
    require(sum(distance(a,b) for a,b in zip(p[:-1],p[1:-1]))/speed+dec(o['takeoff'])==dec(o['service_start']),'original arrival mismatch')
    require(dec(o['service_end'])==dec(o['service_start'])+service,'service mismatch')
    require(dec(c['arrival'])==dec(c['takeoff'])+direct/2/speed,'compressed arrival mismatch')
    require(dec(c['wait_until'])>=dec(c['arrival']),'negative waiting')
    require(dec(c['service_end'])==dec(c['wait_until'])+service,'compressed service mismatch')
    require(dec(c['return'])==dec(c['service_end'])+direct/2/speed,'compressed return mismatch')
    require(dec(o['return'])==dec(o['service_end'])+distance(p[-2],p[-1])/speed,'original return mismatch')
    require(dec(c['service_end'])<=dec(q['deadline']),'delivery deadline')
    require(dec(o['return'])-dec(o['takeoff'])<=dec(configuration['drone_endurance']),'endurance')
    require(dec(c['return'])-dec(c['takeoff'])<=dec(configuration['drone_endurance']),'compressed endurance')
    gain=(original-direct)*dec(configuration['drone_distance_cost'])
    return dict(mapping=dict(original_distance=original,compressed_distance=direct,
        service_complete_preserved=dec(c['service_end'])==dec(o['service_end']),
        return_preserved=dec(c['return'])==dec(o['return']),profit_improvement=gain,
        profit_nondecreasing=gain>=0),ledger={})
