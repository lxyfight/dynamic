"""Truck-local fixed or adaptive review intervals and a shared postdecision value.

This is a bounded candidate policy, not an exhaustive continuous-action solver.
Every verified base plan survives expansion. A model is optional: ``None`` is
the zero-residual Myopic control, not a fitted or transferred waiting model.
The historical SystemIntervalPolicy and its experiments remain independent.
"""
from argparse import ArgumentTypeError
from copy import deepcopy
from decimal import Decimal
from itertools import combinations, islice, product
from math import isfinite, prod
from random import Random
from time import perf_counter

from e1_sim.information import closed, opportunities_open, pickup_deadline_lags
from e1_sim.initializer import digest
from e1_sim.online import MyopicPolicy, evaluate_plan
from e1_sim.quantities import InvalidAction, dec, plain
from e1_sim.retiming import retime_plan
from unified_vfa_features import FEATURE_NAMES, features


MODEL_VERSION = 'unified-vfa-development-v3'


def canonical_plan_digest(plan):
    """Compare plans without their logging label or numeric representation.

    Actor/order identities, action sequence, admissions and review times remain
    significant. Equal hashes certify equal represented decisions; different
    hashes alone do not establish different eventual executed trajectories.
    """
    def canonical(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, Decimal)):
            return {'__number__': list(dec(value).as_integer_ratio())}
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value
    return digest(canonical({key: value for key, value in plan.items() if key != 'id'}))


def operating_rules(config, deadline_lags):
    """Physical and demand rules that changed the meaning of the fitted values."""
    return dict(pickup_deadline_law='independent_discrete_uniform',
                pickup_deadline_lags=list(deadline_lags),
                **{name: config[name] for name in (
                    'truck_handling_setup', 'drone_handling_setup_per_request',
                    'truck_handling_per_kg', 'drone_handling_per_kg',
                    'customer_service_per_request')})


def validate_saved_version(model):
    if not isinstance(model, dict) or model.get('version') != MODEL_VERSION or not model.get('operating_rules'):
        raise ValueError('Incompatible saved VFA model. Retrain with the current '
                         'pickup deadline and handling rules.')


def parse_pattern_limit(value):
    """CLI count per search phase, with uncapped enumeration for tiny instances."""
    if value == 'all':
        return None
    try:
        limit = int(value)
    except ValueError:
        limit = 0
    if limit < 1:
        raise ArgumentTypeError("use a positive integer or 'all'")
    return limit


def _positive_h(h):
    if isinstance(h, bool):
        raise ValueError('h must be a finite positive daily constant')
    value = dec(h)
    if not value.is_finite() or value <= 0:
        raise ValueError('h must be a finite positive daily constant')
    return value


def normalized_review_intervals(values):
    """None denotes fixed review. Sort the adaptive menu for stable model identity."""
    if values is None:
        return None
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError('adaptive review_intervals must be a nonempty sequence')
    intervals = tuple(sorted(_positive_h(h) for h in values))
    if len(set(intervals)) != len(intervals):
        raise ValueError('adaptive review intervals must be distinct')
    return intervals


def validate_model(model, h, review_intervals=None, feature_horizon=None):
    """Reject old scorers and mismatched interval/feature schemas explicitly."""
    if model is None:
        return None
    validate_saved_version(model)
    required = {'feature_names', 'weights', 'scales', 'h'}
    if not required.issubset(model):
        raise ValueError('unified VFA requires feature_names, weights, scales and h')
    if tuple(model['feature_names']) != tuple(FEATURE_NAMES):
        raise ValueError('unified VFA feature schema mismatch')
    if _positive_h(model['h']) != h:
        raise ValueError('model h does not match policy h')
    if _positive_h(model.get('feature_horizon', model['h'])) != (feature_horizon or h):
        raise ValueError('model feature horizon does not match policy feature horizon')
    wait_mode = 'fixed' if review_intervals is None else 'adaptive'
    if model.get('wait_mode', 'fixed') != wait_mode:
        raise ValueError('model wait mode does not match policy')
    if normalized_review_intervals(model.get('review_intervals')) != review_intervals:
        raise ValueError('model review intervals do not match the available actions')
    if any(len(model[k]) != len(FEATURE_NAMES) for k in ('weights', 'scales')):
        raise ValueError('unified VFA coefficient dimension mismatch')
    if not all(isfinite(float(v)) for v in model['weights']):
        raise ValueError('unified VFA weights must be finite')
    if not all(isfinite(float(v)) and float(v) > 0 for v in model['scales']):
        raise ValueError('unified VFA scales must be finite and positive')
    return deepcopy(model)


def _truck_choices(known, plan, review_intervals=None):
    """List alternatives for each idle truck. A live timer is never restarted."""
    choices = []
    reviews = getattr(known, '_wait_reviews', {})
    for tid, truck in sorted(known.trucks.items()):
        if truck['location'] not in known.network['stations'] or truck['active_operation'] is not None:
            continue
        own = sorted(((i, a) for i, a in enumerate(plan['actions']) if a.get('actor') == tid),
                     key=lambda row: (dec(row[1]['at']), row[0]))
        if not own:
            continue
        index, action = own[0]
        previous = dec(reviews[tid]) if tid in reviews else known.now
        if previous > known.now:
            # At an early real event, retain this deadline in the base plan or
            # leave now. Choosing to stay cannot reset it to now+h.
            if action['kind'] == 'truck_continue_circuit' and dec(action['at']) > known.now:
                choices.append((tid, index, 'depart_now'))
        elif dec(action['at']) == known.now and action['kind'] in ('truck_continue_circuit', 'truck_load'):
            if review_intervals is None:
                choices.append((tid, index, 'wait'))
            else:
                choices.extend((tid, index, 'wait', delta) for delta in review_intervals)
    return choices


def _option_groups(options):
    """Group indices of mutually exclusive alternatives for the same truck."""
    groups = {}
    for i, option in enumerate(options):
        groups.setdefault(option[0], []).append(i)
    return tuple(tuple(values) for values in groups.values())


def _subsets(n):
    """Offer the joint alternative first, then singles and smaller coalitions."""
    if not n:
        return
    yield tuple(range(n))
    for size in range(1, n):
        yield from combinations(range(n), size)


def _adaptive_combinations(groups):
    """Visit joint and single-truck alternatives before the residual product."""
    if not groups:
        return
    seen = set()
    for offset in range(max(map(len, groups))):
        # Seed common timers (fixed-policy alternatives), mixed timers and a
        # rotating single truck before traversing the rest of the full product.
        same = tuple(group[offset % len(group)] for group in groups)
        mixed = tuple(group[(offset + i) % len(group)] for i, group in enumerate(groups))
        single = groups[offset % len(groups)]
        round_choices = (same, mixed, (single[offset % len(single)],))
        for selected in round_choices:
            if selected not in seen:
                seen.add(selected)
                yield selected
    for size in range(1, len(groups) + 1):
        for coalition in combinations(groups, size):
            for selected in product(*coalition):
                if selected not in seen:
                    yield selected


def replay_wait_combination(known, row, h, options, selected):
    """Apply selected truck changes to one base and physically verify the joint plan."""
    plan = deepcopy(row['plan'])
    changes = []
    if len({options[i][0] for i in selected}) != len(selected):
        raise ValueError('multiple changes for the same truck')
    adaptive = any(len(option) == 4 for option in options)
    for choice in selected:
        tid, index, kind, *durations = options[choice]
        action = plan['actions'][index]
        change = dict(truck=tid, action=kind)
        if kind == 'wait':
            duration = durations[0] if durations else h
            action['at'] = known.now + duration
            plan.setdefault('wait_reviews', {})[tid] = known.now + duration
            if adaptive:
                change.update(interval=float(duration), review_at=float(known.now + duration))
        else:
            shift = dec(action['at']) - known.now
            for command in plan['actions']:
                if command.get('actor') == tid:
                    command['at'] = dec(command['at']) - shift
            plan.get('wait_reviews', {}).pop(tid, None)
        changes.append(change)
    identity = dict(base=digest(row['plan']), at=str(known.now), h=str(h), changes=changes)
    plan['id'] = row['id'] + '~joint_' + digest(identity)[:16]
    info = dict(base_id=row['id'], id=plan['id'], changes=changes)
    if not adaptive and any(c['action'] == 'wait' for c in changes):
        info.update(h=float(h), review_at=float(known.now+h))
    try:
        plan = retime_plan(known, plan, order_reference=row['plan'])
        checked = evaluate_plan(known, plan, 'WK')
    except (InvalidAction, KeyError) as exc:
        checked = dict(feasible=False, allowed=False, reason=str(exc))
    info.update(feasible=checked['feasible'], allowed=checked['allowed'],
                reason=checked.get('reason'))
    if checked['feasible'] and checked['allowed']:
        checked['joint_wait'] = info
    return checked, info


def joint_wait_pool(known, evaluated, h, *, budget_per_base=8, review_intervals=None):
    """Replay a bounded joint wait/depart neighborhood for *each* base plan.

    Each new timer uses the fixed interval or one of the adaptive intervals.
    Known-cargo reservations without a discretionary timer are not abandoned.
    Dependent tasks are retimed; active execution queues are never modified.
    """
    h = _positive_h(h)
    review_intervals = normalized_review_intervals(review_intervals)
    if type(budget_per_base) is not int or budget_per_base < 0:
        raise ValueError('nonnegative integer wait budget per base required')
    pool = list(evaluated)
    stats = dict(base_count=len(pool), attempted=0, accepted=0, truncated=False,
                 budget_per_base=budget_per_base, by_base=[], attempts=[])
    if not opportunities_open(known.public, known.now):
        return pool, stats
    for row in evaluated:
        options = _truck_choices(known, row['plan'], review_intervals)
        groups = _option_groups(options)
        available = prod(len(group) + 1 for group in groups) - 1
        attempted = min(available, budget_per_base)
        stats['by_base'].append(dict(base_id=row['id'], trucks=[options[g[0]][0] for g in groups],
                                    combinations=available, attempted=attempted,
                                    truncated=available > attempted))
        stats['truncated'] |= available > attempted
        candidates = _subsets(len(options)) if review_intervals is None else _adaptive_combinations(groups)
        for selected in islice(candidates, budget_per_base):
            checked, info = replay_wait_combination(known, row, h, options, selected)
            stats['attempted'] += 1
            stats['attempts'].append(info)
            if checked['feasible'] and checked['allowed']:
                pool.append(checked)
                stats['accepted'] += 1
    return pool, stats


def _enhance_bases(known, rows, budget):
    """Bounded round-robin operation neighbors before joint waiting expansion."""
    from e1_sim.enhancement import variants
    streams = [(row['id'], iter((kind, plan) for kind, plan in variants(known, row)
                               if kind != 'depart_now')) for row in rows]
    pool = list(rows)
    stats = dict(attempted=0, accepted=0, truncated=False, by_move={})
    while streams:
        next_round = []
        for parent, stream in streams:
            item = next(stream, None)
            if item is None:
                continue
            if stats['attempted'] >= budget:
                stats['truncated'] = True
                return pool, stats
            kind, plan = item
            next_round.append((parent, stream))
            stats['attempted'] += 1
            stats['by_move'][kind] = stats['by_move'].get(kind, 0) + 1
            plan['id'] = parent + '~operation_' + str(stats['attempted'])
            checked = evaluate_plan(known, plan, 'WK')
            if checked['feasible'] and checked['allowed']:
                pool.append(checked)
                stats['accepted'] += 1
        streams = next_round
    return pool, stats


class UnifiedVFAPolicy(MyopicPolicy):
    """One scorer for admissions, service/recovery plans and truck wait vectors.

    Usage: ``driver.policy = UnifiedVFAPolicy(h=5, model=None)``. A fitted model
    must declare the exact feature schema and h. This module does not fit one.
    """
    def __init__(self, h=5, model=None, *, enhancement_budget=16,
                 wait_budget_per_base=8, candidate_budget=16, explore=0., seed=0,
                 action_solver='bounded', mip_pattern_limit=64, review_intervals=None,
                 feature_horizon=None, **search):
        super().__init__('WK', candidate_budget=candidate_budget, **search)
        self._h = _positive_h(h)
        self.review_intervals = normalized_review_intervals(review_intervals)
        self.wait_mode = 'fixed' if self.review_intervals is None else 'adaptive'
        default_horizon = self.h if self.wait_mode == 'fixed' else 5
        self.feature_horizon = _positive_h(default_horizon if feature_horizon is None else feature_horizon)
        self.model = validate_model(model, self.h, self.review_intervals, self.feature_horizon)
        for name, value in [('enhancement_budget', enhancement_budget),
                            ('wait_budget_per_base', wait_budget_per_base)]:
            if type(value) is not int or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        if action_solver not in ('bounded', 'gurobi'):
            raise ValueError("action_solver must be 'bounded' or 'gurobi'")
        if mip_pattern_limit is not None and (type(mip_pattern_limit) is not int or mip_pattern_limit < 1):
            raise ValueError('mip_pattern_limit must be a positive integer or None')
        self.action_solver = action_solver
        self.mip_pattern_limit = mip_pattern_limit
        self.enhancement_budget = enhancement_budget
        self.wait_budget_per_base = wait_budget_per_base
        if not isfinite(float(explore)) or not 0 <= explore <= 1:
            raise ValueError('explore must be a probability')
        self.explore, self.rng = float(explore), Random(seed)
        self.enhance = enhancement_budget > 0
        self.selection_log, self.records, self.samples = [], [], []

    @property
    def h(self):
        return self._h

    def choose(self, known, evaluated):
        if known._hidden or any(e[3] in ('request_reveal', 'revelation_end') for e in known._queue):
            raise ValueError('unified VFA requires the sanitized known-state clone')
        if self.model is not None:
            lags = pickup_deadline_lags(known.public['demand_model']['attributes'])
            if self.model['operating_rules'] != operating_rules(known.cfg, lags):
                raise ValueError('VFA model and world use different pickup deadline or handling rules; retrain.')
        bases = [c for c in evaluated if c['feasible'] and c['allowed']]
        if not bases:
            raise ValueError('no verified permissible base plan')
        bases, enhancement = _enhance_bases(known, bases, self.enhancement_budget)
        cache = {}

        def assess(candidate):
            # IDs in plans are logging labels. Different rows can share a label.
            key = id(candidate)
            if key not in cache:
                x = tuple(features(known, candidate, self.feature_horizon))
                if len(x) != len(FEATURE_NAMES) or not all(isfinite(float(v)) for v in x):
                    raise ValueError('invalid unified postdecision feature vector')
                future = 0. if self.model is None or closed(known.public) else sum(
                    float(w)*float(v)/float(s) for w, v, s in zip(self.model['weights'], x, self.model['scales']))
                total = float(candidate['score']) + future
                if not isfinite(total):
                    raise ValueError('nonfinite unified candidate value')
                cache[key] = (total, candidate, x, future)
            return cache[key]

        service_search = None
        if self.action_solver == 'gurobi':
            from service_action_mip import service_action_pool
            from joint_action_mip import search_joint_actions
            bases, service_search = service_action_pool(known, bases, pattern_limit=self.mip_pattern_limit)
            pool, waiting = search_joint_actions(
                known, bases, self.h, pattern_limit=self.mip_pattern_limit,
                score=lambda candidate: assess(candidate)[0], review_intervals=self.review_intervals)
        else:
            pool, waiting = joint_wait_pool(known, bases, self.h,
                                           budget_per_base=self.wait_budget_per_base,
                                           review_intervals=self.review_intervals)
        # Both modes explore only the feasible, physically replayed actions returned.
        scores = [assess(candidate) for candidate in pool]
        if self.action_solver == 'gurobi':
            greedy = scores[waiting['winner_index']]
        else:
            greedy = min(scores, key=lambda row: (-row[0], row[1]['id']))
        selected = (self.rng.choice(scores) if self.explore and not closed(known.public)
                    and self.rng.random() < self.explore else greedy)
        _, chosen, x, future = selected
        prior_reviews = getattr(known, '_wait_reviews', {})
        fresh_waits = {tid: float(dec(at) - known.now)
                       for tid, at in chosen['plan'].get('wait_reviews', {}).items()
                       if dec(prior_reviews.get(tid, known.now)) <= known.now < dec(at)}
        feasible_waits = {}
        for candidate in pool:
            for tid, at in candidate['plan'].get('wait_reviews', {}).items():
                if dec(prior_reviews.get(tid, known.now)) <= known.now < dec(at):
                    duration = format(float(dec(at) - known.now), 'g')
                    feasible_waits[duration] = feasible_waits.get(duration, 0) + 1
        entry = dict(at=float(known.now), h=float(self.h), selected=chosen['id'],
                     wait_mode=self.wait_mode, feature_horizon=float(self.feature_horizon),
                     chosen_wait_intervals=fresh_waits,
                     feasible_wait_interval_counts=feasible_waits,
                     action_solver=self.action_solver,
                     model_mode='zero_residual' if self.model is None else 'supplied_model',
                     J=float(chosen['score']), W=future, features=x,
                     explored=chosen is not greedy[1],
                     feature_names=list(FEATURE_NAMES), plan_hash=digest(chosen['plan']),
                     canonical_plan_hash=canonical_plan_digest(chosen['plan']),
                     admissions=deepcopy(chosen['plan']['decisions']),
                     enhancement=enhancement, waiting=waiting,
                     truncated=enhancement['truncated'] or waiting['truncated'],
                     pool=[dict(id=c['id'], J=float(c['score']), W=w, score=s,
                                features=v, plan_hash=digest(c['plan']),
                                decisions=deepcopy(c['plan']['decisions']),
                                wait_reviews=plain(c['plan'].get('wait_reviews', {})),
                                joint_wait=c.get('joint_wait')) for s,c,v,w in scores])
        if service_search is not None:
            entry['service_search'] = service_search
            entry['truncated'] |= service_search['truncated']
        self.selection_log.append(entry)
        self.records.append(dict(t=float(known.now), j=float(chosen['score']), x=x,
                                 plan=entry['plan_hash'], decision=chosen['id']))
        # Observations only. No realized suffix or training target enters policy.
        self.samples.append(dict(t=float(known.now), x=x, j=float(chosen['score']),
                                 h=float(self.h), plan=entry['plan_hash'], closed=closed(known.public),
                                 explored=entry['explored']))
        return chosen

    def search_metrics(self):
        """Per-day search effort, including truncated upstream admission searches."""
        rows = self.selection_log
        durations = {}
        feasible_counts, feasible_epochs = {}, {}
        for row in rows:
            for duration in row['chosen_wait_intervals'].values():
                key = format(duration, 'g')
                durations[key] = durations.get(key, 0) + 1
            for key, count in row.get('feasible_wait_interval_counts', {}).items():
                feasible_counts[key] = feasible_counts.get(key, 0) + count
                feasible_epochs[key] = feasible_epochs.get(key, 0) + 1
        feature_diagnostics = {}
        for i, name in enumerate(FEATURE_NAMES):
            values = [candidate['features'][i] for row in rows for candidate in row['pool']]
            feature_diagnostics[name] = dict(
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                nonzero_candidates=sum(abs(value) > 1e-12 for value in values),
                action_varying_epochs=sum(
                    max(candidate['features'][i] for candidate in row['pool']) -
                    min(candidate['features'][i] for candidate in row['pool']) > 1e-12
                    for row in rows if row['pool']))
        return dict(
            solver=self.action_solver, epochs=len(rows), wait_mode=self.wait_mode,
            chosen_wait_interval_counts=durations,
            feasible_wait_epochs=sum(bool(r.get('feasible_wait_interval_counts')) for r in rows),
            feasible_wait_interval_counts=feasible_counts,
            feasible_wait_interval_epochs=feasible_epochs,
            feature_diagnostics=feature_diagnostics,
            decision_seconds=sum(r.get('decision_seconds', 0.) for r in rows),
            truncated_epochs=sum(r.get('search_truncated', r['truncated']) for r in rows),
            base_search_truncated_epochs=sum(r.get('base_search_truncated', False) for r in rows),
            enhancement_truncated_epochs=sum(r['enhancement']['truncated'] for r in rows),
            forced_reject_requests=sum(len(r.get('admission_ids_forced_reject', ())) for r in rows),
            service_patterns_attempted=sum(r.get('service_search', {}).get('attempted', 0) for r in rows),
            service_patterns_accepted=sum(r.get('service_search', {}).get('accepted', 0) for r in rows),
            wait_patterns_attempted=sum(r['waiting']['attempted'] for r in rows),
            wait_patterns_accepted=sum(r['waiting']['accepted'] for r in rows),
            wait_search_truncated_epochs=sum(r['waiting']['truncated'] for r in rows),
            service_search_truncated_epochs=sum(r.get('service_search', {}).get('truncated', False) for r in rows),
        )

    def decide(self, known):
        start = perf_counter()
        plan, log = super().decide(known)
        joint = self.selection_log[-1]
        pending = sorted(oid for oid, order in known.orders.items()
                         if order['kind'] == 'P' and order['accepted'] is None)
        log['search_limits'] = dict(candidate_budget=self.candidate_budget,
            beam_width=self.beam_width, max_joint_requests=self.max_joint_requests,
            enhancement_budget=self.enhancement_budget,
            wait_budget_per_base=self.wait_budget_per_base,
            admission_ids_considered=pending[:self.max_joint_requests],
            admission_ids_forced_reject=pending[self.max_joint_requests:])
        if self.action_solver == 'gurobi':
            log['search_limits'].pop('wait_budget_per_base')
            log['search_limits']['mip_pattern_limit'] = self.mip_pattern_limit
        log['base_candidates'] = log['candidates']
        joint['base_search_truncated'] = log['search_truncated']
        joint['admission_ids_forced_reject'] = log['search_limits']['admission_ids_forced_reject']
        log['candidates'] = joint['pool']
        log['joint_candidate_audit'] = joint
        log['search_truncated'] |= joint['truncated']
        joint['search_truncated'] = log['search_truncated']
        joint['decision_seconds'] = perf_counter() - start
        return plan, log
