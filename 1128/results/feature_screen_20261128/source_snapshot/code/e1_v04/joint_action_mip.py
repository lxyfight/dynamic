"""Restricted online action search for fixed or truck-specific review intervals.

A binary master picks a retained service plan and the trucks whose local
wait/depart choice changes. Physical retiming and the nonadditive city-level
VFA require exact replay for each inspected pattern. Feasibility failures cut
only that pattern. Singleton price differences order the *search*, never bound
unseen payoffs. A second restricted master optimizes exact J + W over all
priced, feasible patterns, including every unchanged base (the incumbent).
"""
from math import isfinite, prod

from e1_sim.information import opportunities_open


def _priced_winner(gp, pool, prices):
    """Optimize verified column payoffs, breaking equal-score ties by plan ID."""
    if len(prices) == 1:
        return 0
    with gp.Model('verified_joint_actions') as m:
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.MIPGap = 0.
        m.Params.MIPGapAbs = 0.
        columns = m.addVars(len(pool), vtype=gp.GRB.BINARY, name='feasible_pattern')
        m.addConstr(columns.sum() == 1)
        # Subtract the common historical ledger component to improve scaling.
        m.setObjective(gp.quicksum((value - prices[0]) * columns[i]
                                   for i, value in enumerate(prices)), gp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f'priced Gurobi action master status {m.Status}')
        winner = next(i for i, variable in columns.items() if variable.X > .5)
        value = prices[winner]
        if any(price > value for price in prices):
            # Floating-point MILP tolerances can merge extremely close payoffs.
            # Ordinal prices preserve exactly the Python floating-point ordering.
            rank = {p: k for k, p in enumerate(sorted(set(prices)))}
            m.setObjective(gp.quicksum(rank[p] * columns[i] for i, p in enumerate(prices)),
                           gp.GRB.MAXIMIZE)
            m.optimize()
            if m.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f'tie refinement Gurobi status {m.Status}')
            winner = next(i for i, variable in columns.items() if variable.X > .5)
            value = prices[winner]
        return min((i for i, price in enumerate(prices) if price == value),
                   key=lambda i: (pool[i]['id'], i))


def _search_patterns(gp, known, bases, h, options, groups, pattern_limit, prices, pool, stats,
                     score, replay):
    active = [i for i, local in enumerate(options) if local]
    keys = [(i, j) for i in active for j in range(len(options[i]))]
    total = stats['possible_patterns']
    with gp.Model('unpriced_joint_actions') as m:
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.MIPGap = 0.
        m.Params.MIPGapAbs = 0.
        m.ModelSense = gp.GRB.MAXIMIZE
        base = m.addVars(active, vtype=gp.GRB.BINARY, name='service_plan')
        change = m.addVars(keys, vtype=gp.GRB.BINARY, name='truck_change')
        m.addConstr(base.sum() == 1)
        for i in active:
            # All unchanged bases have already been priced. Mutually exclusive
            # service plans cannot contribute truck choices to another plan.
            m.addConstr(change.sum(i, '*') >= base[i])
        for i in active:
            for truck_options in groups[i]:
                # A truck either retains this base's action, or takes exactly
                # one alternative (e.g. wait 2, 5 or 10 minutes).
                m.addConstr(gp.quicksum(change[i, j] for j in truck_options) <= base[i])
        priority = gp.quicksum((prices[i] - prices[0]) * base[i] for i in active)
        m.setObjectiveN(priority, 0, priority=2, abstol=0., reltol=0., name='estimated_payoff')
        m.setObjectiveN(-change.sum(), 1, priority=1, abstol=0., reltol=0., name='fewest_changes')
        attempts = total if pattern_limit is None else min(pattern_limit, total)
        for _ in range(attempts):
            m.optimize()
            if m.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f'Gurobi joint-pattern search status {m.Status}')
            i = next(i for i in active if base[i].X > .5)
            selected = tuple(j for j in range(len(options[i])) if change[i, j].X > .5)
            selected_set = set(selected)
            # Exactly the probed configuration is removed, including when
            # retiming or rechecking cargo, capacities or time limits failed.
            different = gp.quicksum((1 - change[i, j]) if j in selected_set else change[i, j]
                                    for j in range(len(options[i])))
            m.addConstr(different >= base[i])
            candidate, info = replay(known, bases[i], h, options[i], selected)
            stats['attempts'].append(info)
            stats['attempted'] += 1
            stats['by_base'][i]['attempted'] += 1
            if candidate['feasible'] and candidate['allowed']:
                price = float(score(candidate))
                if not isfinite(price):
                    raise ValueError('nonfinite verified action price')
                pool.append(candidate)
                prices.append(price)
                stats['accepted'] += 1
                if len(selected) == 1:
                    priority += (price - prices[i]) * change[i, selected[0]]
                    m.setObjectiveN(priority, 0, priority=2, abstol=0., reltol=0.,
                                    name='estimated_payoff')


def search_joint_actions(known, bases, h, *, pattern_limit, score, review_intervals=None):
    """Return priced, feasible actions and a truthful scope/search audit.

    ``score(row)`` is the existing full-precision J + W for a replayed row.
    ``pattern_limit`` caps additional replays *per event*, across all bases;
    None inspects the complete finite wait/depart neighborhood. A menu in
    ``review_intervals`` lets each eligible truck choose its own duration. The service
    (admission, flight and recovery) plan generator's own limits still apply.
    """
    if not bases:
        raise ValueError('Gurobi action search requires a feasible base plan')
    if pattern_limit is not None and (type(pattern_limit) is not int or pattern_limit < 1):
        raise ValueError('pattern_limit must be a positive integer or None')
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError('Gurobi action search requires gurobipy and a usable Gurobi license') from exc
    from unified_vfa_policy import _option_groups, _truck_choices, replay_wait_combination

    pool = list(bases)
    prices = [float(score(row)) for row in pool]
    if not all(isfinite(v) for v in prices):
        raise ValueError('nonfinite verified base price')
    is_open = opportunities_open(known.public, known.now)
    options = [_truck_choices(known, row['plan'], review_intervals=review_intervals)
               if is_open else [] for row in pool]
    groups = [_option_groups(local) for local in options]
    by_base = [dict(base_id=row['id'], trucks=[local[indices[0]][0] for indices in truck_groups],
                    combinations=prod(1 + len(indices) for indices in truck_groups) - 1, attempted=0)
               for row, local, truck_groups in zip(pool, options, groups)]
    total = sum(base['combinations'] for base in by_base)
    stats = dict(backend='gurobi', pattern_limit=pattern_limit, base_count=len(pool),
                 by_base=by_base, possible_patterns=total, attempted=0, accepted=0, attempts=[])
    if total:
        _search_patterns(gp, known, bases, h, options, groups, pattern_limit, prices, pool,
                         stats, score, replay_wait_combination)
    for base in by_base:
        base['truncated'] = base['attempted'] < base['combinations']
    stats['truncated'] = stats['attempted'] < total
    stats['infeasible_patterns'] = stats['attempted'] - stats['accepted']
    stats['priced_feasible_patterns'] = len(pool)
    stats['complete_given_bases'] = not stats['truncated']
    stats['winner_index'] = _priced_winner(gp, pool, prices)
    stats['winner_id'] = pool[stats['winner_index']]['id']
    stats['verified_objective'] = prices[stats['winner_index']]
    return pool, stats
