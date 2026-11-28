"""Screen predeclared basis groups on common full-episode Monte Carlo labels."""
from collections import Counter
from itertools import combinations
import math

import numpy as np


def _layout(day):
    if 'layout' in day:
        return day['layout']
    if 'layout_seed' in day:
        return day['layout_seed']
    spec = day.get('spec')
    if isinstance(spec, dict):
        return spec['layout_seed']
    if isinstance(spec, (list, tuple)) and spec:
        return spec[0]
    raise ValueError('a preassigned layout identifier is required for each day')


def _matrix(days):
    """First give each layout equal weight, then each day within each layout."""
    if not days:
        raise ValueError('empty episode split')
    layout_counts = Counter(_layout(day) for day in days)
    xs, ys, ws = [], [], []
    identifiers = set()
    for day in days:
        key = day.get('input_hash')
        if key is not None:
            if key in identifiers:
                raise ValueError('repeated input in episode split')
            identifiers.add(key)
        included = [sample for sample in day['samples'] if not sample['closed']]
        if not included:
            raise ValueError('episode contains no open decision samples')
        day_weight = 1. / (len(layout_counts)*layout_counts[_layout(day)])
        for sample in included:
            xs.append(sample['x'])
            ys.append(sample['y'])
            ws.append(day_weight / len(included))
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    w = np.asarray(ws, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('nonfinite or invalid Monte Carlo samples')
    return x, y, w/w.sum()


def screen(train_days, validation_days, groups, *, ridge=.01):
    """Fit all group subsets once and shortlist two for full policy execution.

    Model coefficients return CNY. A Monte Carlo target can contain sampling
    noise and reflects the data-generating continuation policy.
    """
    train_days, validation_days = list(train_days), list(validation_days)
    if not groups or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('require predeclared groups and positive ridge penalty')
    train_layouts = {_layout(d) for d in train_days}
    validation_layouts = {_layout(d) for d in validation_days}
    if train_layouts & validation_layouts:
        raise ValueError('training and feature validation layouts overlap')
    hashes = lambda days: {d['input_hash'] for d in days if d.get('input_hash') is not None}
    if hashes(train_days) & hashes(validation_days):
        raise ValueError('training and validation scenarios overlap')
    train, target, weights = _matrix(train_days)
    validation, truth, val_weights = _matrix(validation_days)
    if train.shape[1] != validation.shape[1]:
        raise ValueError('inconsistent validation feature dimension')
    all_columns = []
    for name, cols in groups.items():
        if not cols or any(type(i) is not int or not 0 <= i < train.shape[1] for i in cols):
            raise ValueError(f'invalid columns for group {name}')
        all_columns.extend(cols)
    if len(set(all_columns)) != len(all_columns):
        raise ValueError('feature groups must be disjoint')

    center = weights @ train
    centered = train - center
    std = np.sqrt(weights @ np.square(centered))
    # Dropping the influence of a train-constant feature avoids arbitrary
    # behavior when its value changes on validation days.
    constant = std < 1e-10
    std[constant] = 1.
    z = centered / std
    z[:, constant] = 0.
    intercept = float(weights @ target)
    gram = (z.T * weights) @ z
    rhs = z.T @ (weights * (target-intercept))
    z_val = (validation-center)/std
    val_var = float(val_weights @ np.square(truth - val_weights @ truth))
    scores, models = [], {}
    keys = tuple(groups)
    for n in range(len(keys)+1):
        for subset in combinations(keys, n):
            cols = sorted(i for key in subset for i in groups[key])
            beta = (np.linalg.solve(gram[np.ix_(cols,cols)] + ridge*np.eye(len(cols)), rhs[cols])
                    if cols else np.empty(0))
            pred = intercept + z_val[:, cols] @ beta
            residual = pred-truth
            mse = float(val_weights @ np.square(residual))
            r2 = float(1-mse/val_var) if val_var > 1e-12 else None
            mae = float(val_weights @ np.abs(residual))
            model = dict(groups=list(subset), columns=cols, weights=beta.tolist(),
                         mean=center[cols].tolist(), std=std[cols].tolist(), intercept=intercept)
            scores.append(dict(groups=list(subset), columns=cols, r2=r2, mae=mae,
                               rmse=math.sqrt(mse), mean_signed_error=float(val_weights @ residual)))
            models[tuple(cols)] = model

    def by_mae(s):
        return s['mae'], len(s['columns']), tuple(s['groups'])

    def by_r2(s):
        # Sorting by MSE also covers a constant held-out target (undefined R²).
        return s['rmse'], len(s['columns']), tuple(s['groups'])

    first = min(scores, key=by_mae)
    second = min((s for s in scores if s['columns'] != first['columns']), key=by_r2)
    return dict(
        scores=sorted(scores,key=by_mae),
        finalists=[models[tuple(s['columns'])] for s in (first,second)],
        train_days=len(train_days), validation_days=len(validation_days),
        train_layouts=len(train_layouts), validation_layouts=len(validation_layouts),
        train_samples=len(train), validation_samples=len(validation), ridge=ridge,
        target='Realized terminal net minus known-plan net plus offline future rejection loss (CNY)',
        weighting='Each layout, each episode within layout, each open decision within episode',
        shortlist='Minimum validation MAE, then maximum validation R2 among other subsets. Ties: fewer columns, alphabetical names.',
    )
