"""Fixed daily review interval and one postdecision value for the joint action.

This is a bounded candidate policy, not an exhaustive continuous-action solver.
Every verified base plan survives expansion. A model is optional: ``None`` is
the zero-residual Myopic control, not a fitted or transferred waiting model.
The historical SystemIntervalPolicy and its experiments remain independent.
"""
from copy import deepcopy
from itertools import combinations
from math import isfinite
from random import Random

from e1_sim.information import closed, opportunities_open
from e1_sim.initializer import digest
from e1_sim.online import MyopicPolicy, evaluate_plan
from e1_sim.quantities import InvalidAction, dec, plain
from e1_sim.retiming import retime_plan
from unified_vfa_features import FEATURE_NAMES, features


def _positive_h(h):
    if isinstance(h, bool):
        raise ValueError('h must be a finite positive daily constant')
    value = dec(h)
    if not value.is_finite() or value <= 0:
        raise ValueError('h must be a finite positive daily constant')
    return value


def validate_model(model, h):
    """Reject old scorers and mismatched interval/feature schemas explicitly."""
    if model is None:
        return None
    if not isinstance(model, dict):
        raise ValueError('unified VFA model must be a mapping')
    required = {'feature_names', 'weights', 'scales', 'h'}
    if not required.issubset(model):
        raise ValueError('unified VFA requires feature_names, weights, scales and h')
    if tuple(model['feature_names']) != tuple(FEATURE_NAMES):
        raise ValueError('unified VFA feature schema mismatch')
    if _positive_h(model['h']) != h:
        raise ValueError('model h does not match the fixed daily interval')
    if any(len(model[k]) != len(FEATURE_NAMES) for k in ('weights', 'scales')):
        raise ValueError('unified VFA coefficient dimension mismatch')
    if not all(isfinite(float(v)) for v in model['weights']):
        raise ValueError('unified VFA weights must be finite')
    if not all(isfinite(float(v)) and float(v) > 0 for v in model['scales']):
        raise ValueError('unified VFA scales must be finite and positive')
    return deepcopy(model)


def _truck_choices(known, plan):
    """One alternate per currently idle truck; a live timer is never restarted."""
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
            choices.append((tid, index, 'wait'))
    return choices


def _subsets(n):
    """Offer the joint alternative first, then singles and smaller coalitions."""
    if not n:
        return
    yield tuple(range(n))
    for size in range(1, n):
        yield from combinations(range(n), size)


def joint_wait_pool(known, evaluated, h, *, budget_per_base=8):
    """Replay a bounded joint wait/depart neighborhood for *each* base plan.

    A fresh choice schedules exactly now+h, including near revelation closure.
    Known-cargo reservations without a discretionary timer are not abandoned.
    Dependent tasks are retimed; active execution queues are never modified.
    """
    h = _positive_h(h)
    if type(budget_per_base) is not int or budget_per_base < 0:
        raise ValueError('nonnegative integer wait budget per base required')
    pool = list(evaluated)
    stats = dict(base_count=len(pool), attempted=0, accepted=0, truncated=False,
                 budget_per_base=budget_per_base, by_base=[], attempts=[])
    if not opportunities_open(known.public, known.now):
        return pool, stats
    for row in evaluated:
        options = _truck_choices(known, row['plan'])
        available = (1 << len(options)) - 1
        attempted = min(available, budget_per_base)
        stats['by_base'].append(dict(base_id=row['id'], trucks=[o[0] for o in options],
                                    combinations=available, attempted=attempted,
                                    truncated=available > attempted))
        stats['truncated'] |= available > attempted
        for serial, selected in enumerate(_subsets(len(options))):
            if serial >= budget_per_base:
                break
            plan = deepcopy(row['plan'])
            changes = []
            for choice in selected:
                tid, index, kind = options[choice]
                action = plan['actions'][index]
                if kind == 'wait':
                    action['at'] = known.now + h
                    plan.setdefault('wait_reviews', {})[tid] = known.now + h
                else:
                    shift = dec(action['at']) - known.now
                    for command in plan['actions']:
                        if command.get('actor') == tid:
                            command['at'] = dec(command['at']) - shift
                    plan.get('wait_reviews', {}).pop(tid, None)
                changes.append(dict(truck=tid, action=kind))
            identity = dict(base=digest(row['plan']), at=str(known.now), h=str(h), changes=changes)
            plan['id'] = row['id'] + '~joint_' + digest(identity)[:16]
            info = dict(base_id=row['id'], id=plan['id'], changes=changes,
                        h=float(h), review_at=float(known.now+h))
            stats['attempted'] += 1
            try:
                plan = retime_plan(known, plan, order_reference=row['plan'])
                checked = evaluate_plan(known, plan, 'WK')
            except (InvalidAction, KeyError) as exc:
                checked = dict(feasible=False, allowed=False, reason=str(exc))
            info.update(feasible=checked['feasible'], allowed=checked['allowed'],
                        reason=checked.get('reason'))
            stats['attempts'].append(info)
            if checked['feasible'] and checked['allowed']:
                checked['joint_wait'] = info
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
                 wait_budget_per_base=8, candidate_budget=16, explore=0., seed=0, **search):
        super().__init__('WK', candidate_budget=candidate_budget, **search)
        self._h = _positive_h(h)
        self.model = validate_model(model, self.h)
        for name, value in [('enhancement_budget', enhancement_budget),
                            ('wait_budget_per_base', wait_budget_per_base)]:
            if type(value) is not int or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')
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
        bases = [c for c in evaluated if c['feasible'] and c['allowed']]
        if not bases:
            raise ValueError('no verified permissible base plan')
        bases, enhancement = _enhance_bases(known, bases, self.enhancement_budget)
        pool, waiting = joint_wait_pool(known, bases, self.h,
                                       budget_per_base=self.wait_budget_per_base)
        scores = []
        for candidate in pool:
            x = tuple(features(known, candidate, self.h))
            if len(x) != len(FEATURE_NAMES) or not all(isfinite(float(v)) for v in x):
                raise ValueError('invalid unified postdecision feature vector')
            future = 0. if self.model is None or closed(known.public) else sum(
                float(w)*float(v)/float(s) for w, v, s in zip(self.model['weights'], x, self.model['scales']))
            if not isfinite(future) or not isfinite(float(candidate['score'])+future):
                raise ValueError('nonfinite unified candidate value')
            scores.append((float(candidate['score'])+future, candidate, x, future))
        greedy = min(scores, key=lambda row: (-row[0], row[1]['id']))
        selected = (self.rng.choice(scores) if self.explore and not closed(known.public)
                    and self.rng.random() < self.explore else greedy)
        _, chosen, x, future = selected
        entry = dict(at=float(known.now), h=float(self.h), selected=chosen['id'],
                     model_mode='zero_residual' if self.model is None else 'supplied_model',
                     J=float(chosen['score']), W=future, features=x,
                     explored=chosen is not greedy[1],
                     feature_names=list(FEATURE_NAMES), plan_hash=digest(chosen['plan']),
                     enhancement=enhancement, waiting=waiting,
                     truncated=enhancement['truncated'] or waiting['truncated'],
                     pool=[dict(id=c['id'], J=float(c['score']), W=w, score=s,
                                features=v, plan_hash=digest(c['plan']),
                                decisions=deepcopy(c['plan']['decisions']),
                                wait_reviews=plain(c['plan'].get('wait_reviews', {})),
                                joint_wait=c.get('joint_wait')) for s,c,v,w in scores])
        self.selection_log.append(entry)
        self.records.append(dict(t=float(known.now), j=float(chosen['score']), x=x,
                                 plan=entry['plan_hash'], decision=chosen['id']))
        # Observations only. No realized suffix or training target enters policy.
        self.samples.append(dict(t=float(known.now), x=x, j=float(chosen['score']),
                                 h=float(self.h), plan=entry['plan_hash'], closed=closed(known.public),
                                 explored=entry['explored']))
        return chosen

    def decide(self, known):
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
        log['base_candidates'] = log['candidates']
        log['candidates'] = joint['pool']
        log['joint_candidate_audit'] = joint
        log['search_truncated'] |= joint['truncated']
        return plan, log
