"""Information contract, independent of the physical simulator."""
from math import isfinite, floor
from .quantities import InvalidAction
from . import p2_arrivals

MODE='unknown_daily_total'
FORBIDDEN={'future_count','city_pickup_totals','total_by_city','remaining_by_city','realized_total','actual_remaining'}

def unknown(public):return public.get('information_mode')==MODE


def pickup_deadline_lags(attributes):
    """Equiprobable return allowances in minutes, including older constant laws."""
    if 'pickup_deadline_lags' in attributes:
        values = attributes['pickup_deadline_lags']
        if (not isinstance(values, list) or not values or 'pickup_deadline_lag' in attributes or
                any(type(v) is not int or v <= 0 for v in values) or len(set(values)) != len(values)):
            raise InvalidAction('distinct positive integer pickup deadline allowances required')
        return tuple(values)
    value = attributes.get('pickup_deadline_lag')
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise InvalidAction('positive finite pickup deadline allowance required')
    return (value,)


def validate_public(public):
    if not unknown(public):return
    if FORBIDDEN.intersection(public):raise InvalidAction('unknown-total mode forbids realized demand counts')
    allowed={'information_mode','revelation_end','revelation_closed','demand_model'}
    if set(public)-allowed:raise InvalidAction('undeclared unknown-total public information')
    end=public['revelation_end']
    if isinstance(end,bool) or int(end)!=end or end<1:raise InvalidAction('integer public revelation horizon required')
    model=public['demand_model']
    required={'kind','rates_by_city','attributes'}
    optional={'spatial_service_lags_by_city'}
    is_p2 = model.get('kind') == p2_arrivals.KIND
    if is_p2:
        required.update({'space', 'count_by_slot'})
    if (not required<=set(model) or set(model)-required-optional or
            model['kind'] not in (p2_arrivals.KIND, 'independent_poisson_per_minute')):
        raise InvalidAction('unsupported public demand law')
    attr=model['attributes']
    lag_field = 'pickup_deadline_lags' if 'pickup_deadline_lags' in attr else 'pickup_deadline_lag'
    fields = {'weight_min', 'weight_max', lag_field}
    if not is_p2:
        fields.update({'x_radius', 'y_min', 'y_max'})
    if set(attr) != fields:
        raise InvalidAction('invalid public mark distribution')
    if not (type(attr['weight_min']) is int and type(attr['weight_max']) is int and 0<attr['weight_min']<=attr['weight_max']):
        raise InvalidAction('integer positive mark weights required')
    pickup_deadline_lags(attr)
    if not is_p2 and not (0<=attr['x_radius'] and 0<=attr['y_min']<=attr['y_max']):
        raise InvalidAction('invalid mark support')
    for segments in model['rates_by_city'].values():
        last=0
        for row in segments:
            if set(row)!={'start','end','rate'}:raise InvalidAction('invalid arrival segment fields')
            a,b,r=row['start'],row['end'],row['rate']
            if not (type(a) is int and type(b) is int and last<a<=b<=end and isfinite(r) and r>=0):
                raise InvalidAction('invalid/overlapping arrival segments')
            last=b
    if is_p2:
        p2_arrivals.validate_law(model, end)
    if 'spatial_service_lags_by_city' in model:
        lags=model['spatial_service_lags_by_city']
        if not isinstance(lags,dict) or set(lags)!=set(model['rates_by_city']):
            raise InvalidAction('spatial service lag cities must match public rate cities')
        for city,values in lags.items():
            if not isinstance(values,list) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or v<0 for v in values):
                raise InvalidAction('finite nonnegative historical service lags required')
            if values!=sorted(values):raise InvalidAction('historical service lags must be sorted')
            if not values and any(r['rate']>0 for r in model['rates_by_city'][city]):
                raise InvalidAction('positive-rate city requires historical service lags')

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
    """Draw future time/city pairs conditional on the declared public law."""
    validate_public(public)
    if closed(public):return []
    model = public['demand_model']
    if model['kind'] == p2_arrivals.KIND:
        return [dict(city=r['city'], release=r['release'])
                for r in p2_arrivals.sample_marked_rows(model, now, rng)]
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
