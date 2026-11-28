"""Common full-day policy and observed-trajectory bank for grouped VFA screens."""
from collections import defaultdict
from math import isfinite
from random import Random

from e1_sim.information import closed
from e1_sim.initializer import digest
from nn_refinement import inputs as common_inputs
from recovery_opportunity import run
from run_global_demand_pilot import ReviewPolicy, operation_metrics


NAMES = [
    'extra_recovery_exposure',
    'time_fraction', 'expected_remaining_per_city',
    'new_trips_per_city', 'remaining_truck_road_time', 'remaining_sorties_per_city',
    'mean_recovery_lag', 'pending_kg_without_alternative_per_city',
    'mean_positive_alternative_return_slack',
    'station_capacity_time', 'station_matched_ready_opportunity', 'station_arrival_exposure',
]
GROUPS = {
    'recovery': [0],
    'clock_demand': [1, 2],
    'commitments': [3, 4, 5],
    'known_pickup_connection': [6, 7, 8],
    'station_opportunity': [9, 10, 11],
}


def alternative_recovery(known, candidate):
    """Measure robustness of a known pickup's assigned truck in the candidate.

    A substitute is a distinct truck passing the same station at/after the
    assigned loading start, with free capacity, handling time and an on-time
    depot return. It is a local opportunity proxy, not a committed reroute.
    """
    final = candidate['state']
    capacity = float(known.cfg['truck_capacity'])
    payload = {actor: float(truck['onboard_weight']) for actor, truck in known.trucks.items()}
    for command in candidate['plan']['actions']:
        if command['kind'] == 'truck_dispatch':
            allocation = command.get('quantities')
            if allocation is None:
                allocation = {oid: known.orders[oid]['weight'] for oid in command.get('delivery_orders', [])}
            payload[command['actor']] = sum(allocation.values())
    passes = defaultdict(list)
    scheduled = []
    for e in candidate['trace']:
        kind, actor = e['kind'], e.get('actor')
        if kind == 'truck_load_started':
            scheduled.extend((oid, actor, e['station'], float(e['at'])) for oid in e['orders'])
        elif kind == 'truck_unload_done':
            payload[actor] -= sum(e['quantities'].values())
        elif kind == 'truck_load_done':
            payload[actor] += sum(known.orders[oid]['weight'] for oid in e['orders'])
        elif kind == 'road_started' and e['source'] in known.network['stations']:
            passes[e['source']].append((actor, float(e['at']), capacity - payload[actor],
                                        float(final.trucks[actor]['return_time'])))
    no_backup_kg = total_slack = 0.
    for oid, assigned, station, load_time in scheduled:
        order = final.orders[oid]
        ready = float(order['available_at'])
        deadline = float(order['deadline'])
        handling = float(known._handling('truck', order['weight']))
        slack = [deadline - returned for actor, depart, spare, returned in passes[station]
                 if actor != assigned and spare >= order['weight'] and depart >= load_time
                 and ready + handling <= depart and returned <= deadline]
        if slack:
            total_slack += max(slack)
        else:
            no_backup_kg += order['weight']
    cities = len(known.network['city_x'])
    return (no_backup_kg / cities,
            total_slack / (max(1, len(scheduled)) * float(known.public['revelation_end'])))


def features(known, candidate):
    if closed(known.public):
        return [0.] * len(NAMES)
    base = common_inputs(known, candidate)
    kg, slack = alternative_recovery(known, candidate)
    return [float(base[i]) for i in (0, 1, 2, 4, 5, 6, 14)] + [kg, slack] + [
        float(base[i]) for i in (16, 18, 19)]


def prediction(model, x):
    cols = model['columns']
    return model.get('base_theta', 0.) * x[0] + model['intercept'] + sum(
        (x[col] - mu) / sigma * coeff for col, mu, sigma, coeff in
        zip(cols, model['mean'], model['std'], model['weights']))


class FeaturePolicy(ReviewPolicy):
    def __init__(self, model=None, teacher=28, collect=False, explore_seed=None):
        super().__init__(teacher, 'fixed')
        self.model, self.collect = model, collect
        self.samples = []
        self.exploration = None if explore_seed is None else Random(explore_seed)
        if model is not None:
            cols = model['columns']
            assert all(0 <= i < len(NAMES) for i in cols)
            assert all(len(model[k]) == len(cols) for k in ('mean','std','weights'))
            assert all(v > 0 and isfinite(v) for v in model['std'])

    def pick(self, known, ranked, theta):
        terminal = closed(known.public)
        xx = None
        if self.model is None:
            scores = [j + theta * phi for j, phi, _ in ranked]
        elif terminal:
            scores = [j for j, _, _ in ranked]
        else:
            xx = [features(known, c) for _, _, c in ranked]
            scores = [j + prediction(self.model, x) for (j, _, _), x in zip(ranked, xx)]
        greedy = min(range(len(ranked)), key=lambda i: (-scores[i], ranked[i][2]['id']))
        chosen = greedy
        if self.exploration is not None and not terminal and self.exploration.random() < .10:
            chosen = self.exploration.randrange(len(ranked))
        j, phi, c = ranked[chosen]
        base = min(range(len(ranked)), key=lambda i: (-ranked[i][0], ranked[i][2]['id']))
        record = dict(t=float(known.now), j=j, x=phi, plan=digest(c['plan']),
                      decision=c['id'], differs=chosen != base)
        if self.collect:
            x = features(known, c) if xx is None else xx[chosen]
            self.samples.append(dict(t=record['t'], x=x, j=j, closed=terminal,
                plan=record['plan'], candidates=len(ranked), explored=chosen != greedy))
        return c['plan'], record


def run_case(world, model=None, teacher=28, collect=False, explore_seed=None):
    policy = FeaturePolicy(model, teacher, collect, explore_seed)
    row = run(world, teacher, policy_override=policy, metrics=operation_metrics)
    if collect:
        # Completed-world suffix is used exclusively to form offline labels.
        rejection = world['configuration']['rejection_per_kg']
        released = [(float(o['release']), o['weight'] * rejection)
                    for o in world['input']['orders'] if o['kind'] == 'P']
        for sample in policy.samples:
            reject_all = sum(cost for t, cost in released if t > sample['t'])
            sample['y'] = row['net'] - sample['j'] + reject_all
    row.update(samples=policy.samples, input_hash=world['input_hash'], i0_hash=world['i0_hash'])
    return row
