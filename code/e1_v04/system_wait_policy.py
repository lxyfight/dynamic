"""Frozen admission/dispatch policy with a network-wide review cadence.

The interval rule reads public demand only. Vehicle-specific information decides
whether to wait, never which interval to use. Counterfactual alternatives are
collected by the offline harness and cannot select the online cadence.
"""
from copy import deepcopy

from e1_sim.information import closed, opportunities_open
from e1_sim.initializer import digest
from e1_sim.learning import wait_pool
from e1_sim.quantities import dec, plain
from high_demand_features import FeaturePolicy, features, prediction
from recovery_opportunity import coverage


def departure_actors(known, candidate):
    return sorted({a['actor'] for a in candidate['plan']['actions']
        if a['kind'] == 'truck_continue_circuit' and dec(a['at']) == known.now
        and a['actor'] in known.trucks
        and known.trucks[a['actor']]['location'] in known.network['stations']
        and known.trucks[a['actor']]['active_operation'] is None})


def value(model, known, candidate):
    return float(candidate['score']) + (0. if closed(known.public) else
                                       prediction(model, features(known, candidate)))


def paired_candidates(known, base, interval, *, truck=None):
    """Verified timing-only interventions of one common base plan.

    The retimer preserves admissions, cargo assignments and routes. Necessary
    dependent start times may move; independent devices are not shifted.
    """
    diagnostics = []
    actors = departure_actors(known, base)
    rows = wait_pool(known, [base], review_interval=interval,
                     budget=max(16, len(actors)), departure_only=True,
                     diagnostics=diagnostics)
    matches = []
    for row in rows[1:]:
        fresh = [(tid, dec(at)) for tid, at in row['plan'].get('wait_reviews', {}).items()
                 if dec(at) > known.now
                 and dec(at) != getattr(known, '_wait_reviews', {}).get(tid)]
        if len(fresh) != 1:
            raise AssertionError('paired intervention must change one truck review')
        tid, review = fresh[0]
        if tid not in actors or (truck is not None and tid != truck):
            continue
        if row['plan']['decisions'] != base['plan']['decisions']:
            raise AssertionError('waiting changed current admissions')
        memory = getattr(known, '_station_wait_memory', {}).get(tid, {})
        identity = dict(public_prefix=digest(known.trace), at=str(known.now), truck=tid,
                        visit_id=memory.get('visit_id'), base_plan=digest(base['plan']),
                        h=str(review-known.now))
        matches.append(dict(id=digest(identity), identity=identity, truck=tid,
                            station=known.trucks[tid]['location'],
                            h=float(review-known.now), requested_h=float(interval),
                            known_plan_delta=float(row['score']-base['score']),
                            memory=plain(deepcopy(memory)), candidate=row))
    return matches, diagnostics


class SystemIntervalPolicy(FeaturePolicy):
    """Use the frozen six-feature scorer; learn no new weights in the pilot.

    mode='depart' prohibits new speculative reviews but retains disclosed-cargo
    reservations. mode='scored' evaluates all current trucks at the same h and
    changes at most one truck's wait per decision. Ties retain the base plan.
    mode='explore' is explicitly diagnostic: force up to repeat_limit reviews
    on one previously selected visit. It is never a reported learned policy.
    """
    def __init__(self, model, rule, *, mode='depart', repeat_visit=None,
                 repeat_limit=0, observe=True):
        super().__init__(model)
        if mode not in ('depart', 'scored', 'explore'):
            raise ValueError('unsupported system waiting mode')
        self.rule, self.mode = rule, mode
        self.wait_budget = 0
        self.observe = observe
        self.repeat_visit, self.repeat_limit = repeat_visit, repeat_limit
        self.forced_reviews = 0
        self.wait_decisions = []
        self.snapshot_callback = None

    def choose(self, known, evaluated):
        # No speculative waits enter this common admission/dispatch ranking.
        ranked = self.rank(known, evaluated)
        base_plan, record = FeaturePolicy.pick(self, known, ranked, self.theta)
        base = next(c for _, _, c in ranked if c['plan'] is base_plan)
        chosen = base
        if not opportunities_open(known.public, known.now):
            self.records.append(record)
            return chosen
        cadence = self.rule.decision(known.public, known.now)
        pairs, diagnostics = (paired_candidates(known, base, cadence['h'])
                              if self.observe or self.mode != 'depart' else ([], []))
        base_value = value(self.model, known, base)
        for pair in pairs:
            pair['frozen_score_delta'] = value(self.model, known, pair['candidate'])-base_value
        selected = None
        if self.mode == 'scored' and pairs:
            best = min(pairs, key=lambda p: (-p['frozen_score_delta'], p['id']))
            if best['frozen_score_delta'] > 1e-7:
                selected = best
        elif self.mode == 'explore' and self.forced_reviews < self.repeat_limit:
            eligible = [p for p in pairs if p['identity']['visit_id'] == self.repeat_visit]
            if eligible:
                selected = min(eligible, key=lambda p: p['id'])
                self.forced_reviews += 1
        entry = dict(at=float(known.now), cadence=cadence,
                     departure_actors=departure_actors(known, base),
                     diagnostics=plain(diagnostics),
                     pairs=[{k:v for k,v in p.items() if k != 'candidate'} for p in pairs],
                     selected=None if selected is None else selected['id'])
        self.wait_decisions.append(entry)
        # The callback belongs to the offline experiment, not feature generation.
        if self.snapshot_callback is not None and pairs:
            self.snapshot_callback(known, base, pairs, entry)
        if selected is not None:
            chosen = selected['candidate']
            record = dict(record, plan=digest(chosen['plan']), decision=chosen['id'],
                          j=float(chosen['score']))
        self.records.append(record)
        return chosen
