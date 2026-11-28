"""All 16 feature-group combinations on an exactly solvable station-waiting model.

The truck has one free pickup slot. At each five-minute review it can leave
with value zero or wait through one interval. Any usable offer in the interval
can fill the slot for a public marginal contribution; it may also be declined.
The exact one-step and continuation values use no realized future requests.
This is a local diagnostic and does not supply labels for the full-day solver.
"""
from itertools import combinations
from math import expm1

import numpy as np

from run_multiscale import CFG, ROOT, write


GROUPS = {
    'horizon': (0,),
    'exposure': (1, 2),
    'margin': (3,),
    'interaction': (4, 5),
}
RATES = {'low': (.005, .025), 'high': (.035, .085)}
INTERVAL = 5.
COST = INTERVAL * CFG['truck_time_cost']


def draw_states(seed, per_regime):
    rng = np.random.default_rng(seed)
    features, labels, regimes = [], [], []
    for regime, bounds in RATES.items():
        for _ in range(per_regime):
            slots = rng.integers(1, 9)
            base = rng.uniform(*bounds)
            rates = base * rng.choice([.5, 1., 2.], size=slots)
            margin = rng.uniform(2., 55.)
            probabilities = -np.expm1(-INTERVAL * rates)
            # A waiting interval is paid even when no offer arrives; after
            # the last interval leaving pays zero. At most one deal is taken.
            best = 0.
            for probability in reversed(probabilities):
                wait = -COST + probability * max(margin, best) + (1.-probability) * best
                best = max(0., wait)
            probability_any = -expm1(-INTERVAL * sum(rates))
            features.append([slots, probabilities[0], sum(probabilities), margin,
                             probabilities[0] * margin, probability_any * margin])
            # The known cost -COST is already included in the action score.
            labels.append(wait + COST)
            regimes.append(regime)
    return np.array(features), np.array(labels), np.array(regimes)


def predict_model(x, columns, model):
    if not columns:
        return np.full(len(x), model['intercept'])
    return model['intercept'] + ((x[:, columns] - model['mean']) / model['sd']) @ model['coef']


def fit(x, y, columns):
    if not columns:
        return {'intercept': y.mean()}
    part = x[:, columns]
    means = part.mean(axis=0)
    sd = np.maximum(part.std(axis=0), 1e-8)
    z = (part - means) / sd
    beta = np.linalg.solve(z.T @ z + 1e-6 * np.eye(len(columns)), z.T @ (y-y.mean()))
    return dict(intercept=y.mean(), mean=means, sd=sd, coef=beta)


def evaluate(data, columns, model):
    x, y, regimes = data
    fitted = predict_model(x, columns, model)
    metrics = {}
    for regime in ('both', *RATES):
        keep = np.ones(len(y), dtype=bool) if regime=='both' else regimes==regime
        target, pred = y[keep], fitted[keep]
        error = target-pred
        wait_gain = target-COST
        regret = np.maximum(0., wait_gain) - np.where(pred>COST, wait_gain, 0.)
        metrics[regime] = dict(n=len(target), r2=float(1.-(error@error)/np.sum((target-target.mean())**2)),
            value_mae=float(np.mean(abs(error))), decision_regret=float(regret.mean()),
            correct_wait_or_depart=float(np.mean((wait_gain>0)==(pred>COST))))
    return metrics


def main():
    train = draw_states(37501, 1000)
    validation = draw_states(37502, 500)
    rows = {}
    models = {}
    for size in range(len(GROUPS)+1):
        for groups in combinations(GROUPS, size):
            name = '+'.join(groups) or 'intercept'
            columns = [index for group in groups for index in GROUPS[group]]
            model = fit(*train[:2], columns)
            models[name] = (columns, model)
            scores = evaluate(validation, columns, model)
            rows[name] = dict(groups=groups, n_predictors=len(columns), validation=scores)
    # Prefer a compact basis whose validation decision loss is practically
    # equivalent to the best, then use absolute value error to break ties.
    cutoff = min(r['validation']['both']['decision_regret'] for r in rows.values()) + .01
    eligible = [k for k,r in rows.items() if r['validation']['both']['decision_regret'] <= cutoff]
    selected = min(eligible,key=lambda k:(rows[k]['n_predictors'],rows[k]['validation']['both']['value_mae'],k))
    # Exactly one frozen selection is evaluated against a constant reference.
    fresh = draw_states(37503, 1000)
    held = {k:evaluate(fresh,*models[k]) for k in {selected,'intercept'}}
    assert len(rows)==16
    assert all(q['decision_regret']>=0 for metrics in held.values() for q in metrics.values())
    out = dict(selected=selected, candidate_count=len(rows), validation=rows, independent_confirmation=held,
        groups=GROUPS, base_rate_per_minute=RATES, rate_multipliers=[.5,1,2], train_n=2000, validation_n=1000,
        confirmation_n=2000, interval_minutes=INTERVAL, waiting_cost=COST,
        selection='smallest number of scalar predictors within 0.01 CNY of best hold-out action regret; then smaller MAE',
        target='Q*(wait) minus known immediate cost; DP allows decline with optimal continuation. Restricted station proxy, not whole-day PI or whole-day truth.')
    write(ROOT/'tmp/high_wait_dp_batch_v1.json', out)
    print(selected, held)


if __name__=='__main__':
    main()
