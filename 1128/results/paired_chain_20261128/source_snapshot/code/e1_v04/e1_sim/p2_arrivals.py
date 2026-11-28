"""Discrete independent slot counts and i.i.d. order marks (P2-style scenario law).

The binomial probabilities and geographic distribution are this experiment's
choices. The paper supplies the count-then-type factorization.
"""
from math import ceil, erfc, exp, floor, isclose, isfinite, lgamma, log, log1p, sqrt

from .quantities import InvalidAction

KIND = 'independent_binomial_slots_iid_marks'


def point(rng, net, pattern):
    xs = net['city_x']
    left, right = xs[0]-8, xs[-1]+8
    if pattern == 'uniform':
        x, y = rng.uniform(left, right), rng.uniform(-9, 9)
    elif pattern == 'dual':
        # Draw the mixture component once, then truncate each component.
        cx, cy = rng.choice(((xs[0], 4.), (xs[-1], -4.)))
        while True:
            x, y = rng.gauss(cx, 3), rng.gauss(cy, 2)
            if left <= x <= right and abs(y) <= 9:
                break
    else:
        raise ValueError(pattern)
    xy = [round(x, 6), round(y, 6)]
    city = min(range(len(xs)), key=lambda j: abs(xs[j]-xy[0])) + 1
    return f'C{city}', xy


def _normal_mass(a, b, center):
    # Stable differences of Gaussian tails for the truncated N(center, 3²) x law.
    lo, hi = (a-center)/(3*sqrt(2)), (b-center)/(3*sqrt(2))
    if lo >= 0:
        return (erfc(lo)-erfc(hi))/2
    return (erfc(-hi)-erfc(-lo))/2


def city_probabilities(xs, pattern):
    """Integrate each rounded-coordinate city's strip under the simulator mark law."""
    left, right = xs[0]-8, xs[-1]+8
    edges = [left]
    for x1, x2 in zip(xs, xs[1:]):
        mid = floor((x1+x2)*0.5*1e6)
        # point() assigns the rounded grid midpoint to the closest city, first on ties.
        while abs(mid/1e6-x1) > abs(mid/1e6-x2):
            mid -= 1
        while abs((mid+1)/1e6-x1) <= abs((mid+1)/1e6-x2):
            mid += 1
        edges.append((mid+0.5)/1e6)
    edges.append(right)
    result = {}
    for i, (a, b) in enumerate(zip(edges, edges[1:]), 1):
        if pattern == 'uniform':
            mass = (b-a)/(right-left)
        elif pattern == 'dual':
            mass = sum(_normal_mass(a, b, cx)/_normal_mass(left, right, cx)
                       for cx in (xs[0], xs[-1]))/2
        else:
            raise ValueError(pattern)
        result[f'C{i}'] = mass
    return result


def count_law(bands, expected_count):
    normalizer = sum((hi-lo+1)*weight for lo, hi, weight in bands)
    max_mean = expected_count * max(weight for _, _, weight in bands) / normalizer
    trials = max(2, ceil(2*max_mean))
    return [dict(start=lo, end=hi, trials=trials,
                 probability=expected_count*weight/(normalizer*trials)) for lo, hi, weight in bands]


def slot_pmf(segment):
    """Probabilities of observing 0, 1, …, trials pickups network-wide in one slot."""
    n, p = segment['trials'], segment['probability']
    if p == 0:
        return (1.,) + (0.,)*n
    if p == 1:
        return (0.,)*n + (1.,)
    log_factorial = lgamma(n+1)
    log_p, log_q = log(p), log1p(-p)
    return tuple(exp(log_factorial-lgamma(k+1)-lgamma(n-k+1)+k*log_p+(n-k)*log_q)
                 for k in range(n+1))


def sample_slots(segments, now, rng):
    result = []
    for row in segments:
        for minute in range(max(row['start'], floor(now)+1), row['end']+1):
            count = sum(rng.random() < row['probability'] for _ in range(row['trials']))
            result.extend([minute]*count)
    return result


def sample_marked_rows(model, now, rng):
    """Sample times, positions and weights from the declared future law."""
    space = model['space']
    attrs = model['attributes']
    for release in sample_slots(model['count_by_slot'], float(now), rng):
        city, xy = point(rng, space, space['pattern'])
        yield dict(release=release, city=city, xy=xy,
                   weight=rng.randint(attrs['weight_min'], attrs['weight_max']))


def validate_law(model, end):
    space = model['space']
    if (not isinstance(space, dict) or set(space) != {'pattern', 'city_x'} or
            space['pattern'] not in ('uniform', 'dual')):
        raise InvalidAction('invalid P2 spatial mark law')
    xs = space['city_x']
    if (not isinstance(xs, list) or not xs or
            any(type(x) not in (float, int) or not isfinite(x) for x in xs)
            or any(a >= b for a, b in zip(xs, xs[1:]))):
        raise InvalidAction('P2 city coordinates must be strictly increasing')
    segments = model['count_by_slot']
    if not isinstance(segments, list) or not segments:
        raise InvalidAction('missing P2 count law')
    last = 0
    for row in segments:
        if (set(row) != {'start', 'end', 'trials', 'probability'} or
                type(row['start']) is not int or type(row['end']) is not int or
                row['start'] != last+1 or not last < row['end'] <= end or
                type(row['trials']) is not int or row['trials'] < 1 or
                type(row['probability']) not in (int, float) or not 0 <= row['probability'] <= 1):
            raise InvalidAction('invalid binomial slot parameters')
        last = row['end']
    probabilities = city_probabilities(xs, space['pattern'])
    if last != end or probabilities.keys() != model['rates_by_city'].keys():
        raise InvalidAction('P2 count/mark coverage mismatch')
    for city, share in probabilities.items():
        rates = model['rates_by_city'][city]
        if len(rates) != len(segments) or any(
                (r['start'], r['end']) != (s['start'], s['end']) or
                not isclose(r['rate'], share*s['trials']*s['probability'], rel_tol=1e-12, abs_tol=1e-15)
                for r, s in zip(rates, segments)):
            raise InvalidAction('P2 forecast rates must be the marginals of the public count/mark law')
