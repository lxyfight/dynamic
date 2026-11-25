"""Information contract, independent of the physical simulator."""
from math import isfinite, floor
from .quantities import InvalidAction

MODE='unknown_daily_total'
FORBIDDEN={'future_count','city_pickup_totals','total_by_city','remaining_by_city','realized_total','actual_remaining'}

def unknown(public):return public.get('information_mode')==MODE

def validate_public(public):
    if not unknown(public):return
    if FORBIDDEN.intersection(public):raise InvalidAction('unknown-total mode forbids realized demand counts')
    allowed={'information_mode','revelation_end','revelation_closed','demand_model'}
    if set(public)-allowed:raise InvalidAction('undeclared unknown-total public information')
    end=public['revelation_end']
    if isinstance(end,bool) or int(end)!=end or end<1:raise InvalidAction('integer public revelation horizon required')
    model=public['demand_model']
    if set(model)!={'kind','rates_by_city','attributes'} or model['kind']!='independent_poisson_per_minute':
        raise InvalidAction('unsupported public demand law')
    attr=model['attributes']
    if set(attr)!={'weight_min','weight_max','x_radius','y_min','y_max','pickup_deadline_lag'}:
        raise InvalidAction('invalid public mark distribution')
    if not (type(attr['weight_min']) is int and type(attr['weight_max']) is int and 0<attr['weight_min']<=attr['weight_max']):
        raise InvalidAction('integer positive mark weights required')
    if not (0<=attr['x_radius'] and 0<=attr['y_min']<=attr['y_max'] and attr['pickup_deadline_lag']>0):
        raise InvalidAction('invalid mark support')
    for segments in model['rates_by_city'].values():
        last=0
        for row in segments:
            if set(row)!={'start','end','rate'}:raise InvalidAction('invalid arrival segment fields')
            a,b,r=row['start'],row['end'],row['rate']
            if not (type(a) is int and type(b) is int and last<a<=b<=end and isfinite(r) and r>=0):
                raise InvalidAction('invalid/overlapping arrival segments')
            last=b

def closed(public):
    if unknown(public):
        validate_public(public)
        return public.get('revelation_closed') is True
    return public.get('revelation_closed') is True or public.get('future_count')==0

def opportunities_open(public,now):
    if closed(public):return False
    end=public.get('revelation_end')
    return end is None or float(now)<float(end)

def expected_arrivals(public,city,now,end=None):
    """Expected requests in integer release slots (now,end], not actual remaining count."""
    validate_public(public)
    if closed(public):return 0.
    hi=min(float(public['revelation_end']),float(end) if end is not None else float(public['revelation_end']))
    lo=floor(float(now))+1;hi=floor(hi)
    return sum(max(0,min(hi,r['end'])-max(lo,r['start'])+1)*r['rate'] for r in public['demand_model']['rates_by_city'][city])

def sample_release_rows(public,now,rng):
    """Independent Poisson increments; past observed counts are never subtracted."""
    validate_public(public)
    if closed(public):return []
    rows=[]
    for city,segments in sorted(public['demand_model']['rates_by_city'].items()):
        for r in segments:
            lo=max(floor(float(now))+1,r['start']);hi=r['end'];rate=r['rate']
            if lo>hi or rate==0:continue
            position=lo-1+rng.expovariate(rate)
            while position<hi:
                rows.append(dict(city=city,release=floor(position)+1))
                position+=rng.expovariate(rate)
    return sorted(rows,key=lambda r:(r['release'],r['city']))
