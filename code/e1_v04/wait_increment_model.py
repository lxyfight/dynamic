"""Observable station-level predictors of the effect of one five-minute review."""
from math import dist, expm1
from statistics import median

from e1_sim.information import expected_arrivals


GROUPS = {'arrival': ['offer_probability'], 'drone': ['drone_opportunity'],
          'truck': ['truck_opportunity']}


def wait_features(known, frozen, wait, truck, station):
    """Candidate traces simulate disclosed tasks only. No completed-day suffix enters."""
    assert not known._hidden
    now = float(known.now)
    city = 'C'+station[1:-1]
    marks = [o for o in known.orders.values() if o['city']==city]
    eligible = [o for o in marks if station in o['eligible_stations']]
    arrivals = expected_arrivals(known.public, city, now, now+5)
    share = len(eligible)/len(marks) if marks else 0.
    probability = -expm1(-arrivals*share)
    xy = known.network['stations'][station]
    speed = float(known.cfg['drone_speed_km_min'])
    lag = median(2*dist(xy,o['xy'])/speed+float(known.cfg['customer_service_per_request'])
                 +float(known._handling('drone',o['weight'])) for o in eligible) if eligible else 0.
    tail = {uid:max(now,float((u['active_operation'] or {}).get('end',now)))
            for uid,u in known.drones.items() if u['home']==station}
    for e in frozen['trace']:
        actor = e.get('actor')
        if actor in tail:
            tail[actor] = max(tail[actor],float(e.get('end',e['at'])))
    drone_delay = min(tail.values())-now if tail else None
    drone_factor = 5/(5+drone_delay+lag) if drone_delay is not None else 0.
    room = float(known.cfg['truck_capacity'])-float(known.trucks[truck]['onboard_weight'])
    fit = sum(o['weight']<=room for o in eligible)/len(eligible) if eligible else 0.
    # New pickup deadlines are defined relative to disclosure. Also include
    # slack on disclosed commitments actually delayed by this paired shift.
    returned = float(frozen['state'].trucks[truck]['return_time'])
    slack = [now+float(known.public['demand_model']['attributes']['pickup_deadline_lag'])-returned]
    for oid,o in frozen['state'].orders.items():
        end, shifted = o['completed_at'],wait['state'].orders[oid]['completed_at']
        if end is not None and shifted is not None and shifted>end:
            slack.append(float(o['deadline']-end))
    minutes = max(0.,min(slack))
    return dict(offer_probability=probability,drone_opportunity=probability*drone_factor,
                truck_opportunity=probability*fit*minutes/(5+minutes),
                known_plan_delta=float(wait['score']-frozen['score']),
                city_five_minute_arrivals=arrivals,eligible_share=share,
                standalone_pickup_lag=lag,drone_delay=drone_delay if drone_delay is not None else -1.,
                free_kg=room,weight_fit=fit,deadline_slack=minutes)
