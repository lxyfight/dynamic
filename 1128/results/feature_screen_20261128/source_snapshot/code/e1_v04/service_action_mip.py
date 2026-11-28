"""Online pickup admission by recombining verified per-resource schedules.

The binary master offers new combinations of represented drone/vehicle schedules.
Gross pickup earnings prioritize exploration. Only complete, retimed simulator
replays enter the joint J_known + VFA comparison. No surrogate gap certifies it.
"""
from collections import defaultdict
from copy import deepcopy
from math import prod

from e1_sim.initializer import digest
from e1_sim.online import evaluate_plan
from e1_sim.quantities import InvalidAction, dec
from e1_sim.retiming import retime_plan


def _patterns(bases):
    plans = [row['plan'] for row in bases]
    actors = sorted({a['actor'] for p in plans for a in p['actions']} |
                    {v['id'] for p in plans for v in p.get('new_trucks', [])} |
                    {tid for p in plans for tid in p.get('wait_reviews', {})})
    columns = {actor: {} for actor in actors}
    originals = set()
    for row in bases:
        plan = row['plan']
        # Some legacy commands omit the truck's station. Keep the station at
        # which this pattern actually loaded in its verified source replay.
        load_station = {oid: e['station'] for e in row['trace']
                        if e['kind'] == 'truck_load_started' for oid in e['orders']}
        commands = defaultdict(list)
        for a in plan['actions']:
            if a['kind'] == 'truck_load' and 'station' not in a:
                a = dict(a, station=load_station[a['orders'][0]])
            commands[a['actor']].append(a)
        trucks = {v['id']: v for v in plan.get('new_trucks', [])}
        source = []
        for actor in actors:
            pattern = dict(actions=commands[actor], truck=trucks.get(actor),
                           timer=plan.get('wait_reviews', {}).get(actor))
            # Retiming may replace an integer start by an equal Decimal. Do
            # not spend a search slot exploring the same physical schedule.
            numeric_actions = [dict(a, at=dec(a['at']).as_integer_ratio()) for a in commands[actor]]
            timer = pattern['timer']
            key = digest(dict(pattern, actions=numeric_actions,
                              timer=None if timer is None else dec(timer).as_integer_ratio()))
            source.append(key)
            columns[actor][key] = pattern
        originals.add(tuple(source))
    return actors, columns, originals


def _add_pickup_links(model, gp, known, pending, columns, select, accept):
    # Existing commitments may already be in flight, unloaded or on a truck.
    # Only a newly revealed pickup needs *all* three future operations. The
    # physical replay validates the remaining chain from each actual state.
    links = defaultdict(list)
    for actor, patterns in columns.items():
        for key, pattern in patterns.items():
            for action in pattern['actions']:
                kind = action['kind']
                if kind not in ('drone_sortie', 'drone_unload', 'truck_load'):
                    continue
                station = (action['station'] if kind == 'truck_load'
                           else known.drones[actor]['home'])
                for oid in action.get('visit_order', action.get('orders', ())):
                    if oid in pending:
                        links[oid, station, kind].append(select[actor, key])
    for oid in pending:
        visited = []
        for station in known.network['stations']:
            visits = gp.quicksum(links[oid, station, 'drone_sortie'])
            model.addConstr(visits == gp.quicksum(links[oid, station, 'drone_unload']))
            model.addConstr(visits == gp.quicksum(links[oid, station, 'truck_load']))
            visited.append(visits)
        model.addConstr(gp.quicksum(visited) == accept[oid])


def service_action_pool(known, bases, *, pattern_limit):
    """Return all bases and additional replay-certified service plans.

    Each actor has exactly one selected suffix including its future commands,
    optional fleet activation and existing wait timer. All source assignments
    are explicitly excluded from the MIP: they are already evaluated. The cap
    counts only unseen assignments replayed. ``None`` exhausts the restricted
    space; ``truncated`` is true only if an unexplored MIP solution exists.
    """
    if pattern_limit is not None and (type(pattern_limit) is not int or pattern_limit < 1):
        raise ValueError('service pattern limit must be positive or None')
    if known._hidden or any(e[3] in ('request_reveal', 'revelation_end') for e in known._queue):
        raise ValueError('service master requires a sanitized public state')
    pool = list(bases)
    pending = {oid for oid, order in known.orders.items()
               if order['kind'] == 'P' and order['accepted'] is None}
    stats = dict(input_bases=len(pool), pending=sorted(pending), attempted=0, accepted=0,
                 truncated=False, status='no_new_combinations', priority='gross_admission_margin',
                 pattern_limit=pattern_limit, replay_rejections={})
    if not pending or len(pool) < 2:
        return pool, stats
    # Outside this model's domain, preserve complete supplied plans verbatim.
    # Independent wakeups are global rather than associated with one actor.
    local_keys = {'id', 'actions', 'decisions', 'new_trucks', 'wait_reviews'}
    extras = [{k: v for k, v in row['plan'].items() if k not in local_keys} for row in pool]
    if (any(extra != extras[0] for extra in extras) or
            any('actor' not in a for row in pool for a in row['plan']['actions'])):
        stats.update(status='unrepresented_global_commands', truncated=True)
        return pool, stats
    actors, columns, originals = _patterns(pool)
    if prod(len(values) for values in columns.values()) <= len(originals):
        return pool, stats

    import gurobipy as gp
    from gurobipy import GRB
    with gp.Model('pickup_service_patterns') as master:
        master.Params.OutputFlag = 0
        master.Params.Threads = 1
        select = {(a, k): master.addVar(vtype=GRB.BINARY)
                  for a in actors for k in columns[a]}
        accept = {oid: master.addVar(vtype=GRB.BINARY) for oid in sorted(pending)}
        for actor in actors:
            master.addConstr(gp.quicksum(select[actor, k] for k in columns[actor]) == 1)
        _add_pickup_links(master, gp, known, sorted(pending), columns, select, accept)
        rate = float(known.cfg['revenue_per_kg']) + float(known.cfg['rejection_per_kg'])
        master.setObjective(gp.quicksum(float(known.orders[oid]['weight']) * rate * var
                                        for oid, var in accept.items()), GRB.MAXIMIZE)

        def exclude(keys):
            master.addConstr(gp.quicksum(select[actor, key] for actor, key in zip(actors, keys))
                             <= len(actors) - 1)

        for source in sorted(originals):
            exclude(source)
        while True:
            master.optimize()
            if master.Status == GRB.INFEASIBLE:
                stats['status'] = 'restricted_service_patterns_exhausted'
                break
            if master.SolCount == 0 or (pattern_limit is not None and stats['attempted'] >= pattern_limit):
                stats.update(status='no_incumbent' if master.SolCount == 0 else 'pattern_limit',
                             truncated=True)
                break
            keys = [next(k for k in columns[a] if select[a, k].X > .5) for a in actors]
            admissions = {oid: var.X > .5 for oid, var in accept.items()}
            exclude(keys)
            identity = 'service_' + digest([actors, keys, admissions])
            plan = dict(deepcopy(extras[0]), id=identity, actions=[], new_trucks=[],
                        decisions=admissions, wait_reviews={})
            for actor, key in zip(actors, keys):
                pattern = columns[actor][key]
                plan['actions'].extend(pattern['actions'])
                if pattern['truck'] is not None:
                    plan['new_trucks'].append(pattern['truck'])
                if pattern['timer'] is not None:
                    plan['wait_reviews'][actor] = pattern['timer']
            stats['attempted'] += 1
            try:
                checked = evaluate_plan(known, retime_plan(known, plan), 'WK')
            except (InvalidAction, KeyError) as exc:
                checked = dict(feasible=False, reason=str(exc))
            if checked['feasible'] and checked['allowed']:
                pool.append(checked)
                stats['accepted'] += 1
            else:
                reason = checked['reason']
                stats['replay_rejections'][reason] = stats['replay_rejections'].get(reason, 0) + 1
    return pool, stats
