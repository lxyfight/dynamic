"""Replay a shared prefix after deleting every unrevealed pickup from the environment."""
from copy import deepcopy

from e1_sim.initializer import digest
from e1_sim.online import prepare_online
from high_demand_features import FeaturePolicy, features, prediction
from run_frozen_reopt_comparison import RollingReoptimization
from run_high_demand_screen import OUT, read, write, ROOT


class EndOfPrefix(Exception):
    pass


def assert_observable(known):
    assert not known._hidden
    assert all(kind != 'request_reveal' for _, _, _, kind, _ in known._queue)
    assert all(o['kind'] == 'D' or o['release'] <= known.now for o in known.orders.values())


class InspectCandidates(FeaturePolicy):
    def __init__(self, model):
        super().__init__(model=model)
        self.prefix = []

    def pick(self, known, ranked, theta):
        assert_observable(known)
        candidate_rows = []
        for j, _, candidate in ranked:
            x = features(known, candidate)
            candidate_rows.append((digest(candidate['plan']), j, [x[i] for i in self.model['columns']],
                                   prediction(self.model, x)))
        plan, record = super().pick(known, ranked, theta)
        self.prefix.append(dict(at=float(known.now), pool=len(candidate_rows),
                                inputs_and_scores=digest(candidate_rows), chosen=digest(plan)))
        return plan, record


class InspectRH(RollingReoptimization):
    def __init__(self):
        super().__init__()
        self.prefix = []

    def pick(self, known, ranked, theta):
        assert_observable(known)
        rows = [(digest(c['plan']), j) for j, _, c in ranked]
        chosen, record = super().pick(known, ranked, theta)
        self.prefix.append(dict(at=float(known.now), pool=len(rows),
                                inputs_and_scores=digest(rows), chosen=digest(chosen)))
        return chosen, record


def prefix(data, world, cutoff, inspector):
    state, driver = prepare_online(data, world['configuration'], world['network'], full_day=True)
    driver.policy = inspector
    state._record_snapshots = False

    def through_cutoff(environment):
        if environment.now > cutoff:
            raise EndOfPrefix
        driver.on_event(environment)

    try:
        state.run(decision_hook=through_cutoff, stop_at_terminal=True)
    except EndOfPrefix:
        pass
    assert inspector.prefix and inspector.prefix[-1]['at'] == cutoff
    return inspector.prefix


def main():
    model = read(OUT/'freeze.json')['model']
    results = []
    for case in [(440, 100, 'uniform'), (441, 150, 'dual')]:
        world = read(OUT/'worlds'/(digest(case) + '.json'))
        original = world['input']
        arrivals = sorted({o['release'] for o in original['orders'] if o['kind'] == 'P'})
        cutoff = arrivals[2]
        modified = deepcopy(original)
        modified['orders'] = [o for o in modified['orders'] if o['kind'] == 'D' or o['release'] <= cutoff]
        modified['external_events'] = [e for e in modified['external_events']
                                       if e['kind'] != 'request_reveal' or e['at'] <= cutoff]
        removed = len(original['orders']) - len(modified['orders'])
        assert removed > 0 and original['public_information'] == modified['public_information']
        for name, factory in [('VFA', lambda: InspectCandidates(model)), ('RH', InspectRH)]:
            first = prefix(original, world, cutoff, factory())
            other = prefix(modified, world, cutoff, factory())
            assert first == other
            results.append(dict(case=case, policy=name, cutoff=cutoff, removed_hidden=removed,
                                decisions=len(first), candidates=sum(r['pool'] for r in first),
                                common_prefix=first))
    path = ROOT/'tmp/frozen_followup_validation/information.json'
    write(path, dict(frozen_hash=digest(model), cases=results))
    print([(r['case'], r['policy'], r['removed_hidden'], r['decisions'], r['candidates']) for r in results])


if __name__ == '__main__':
    main()
