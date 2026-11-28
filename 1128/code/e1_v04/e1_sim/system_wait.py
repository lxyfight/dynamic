"""One review interval for the whole network, from public demand forecasts only.

The interval controls when an optional wait is reviewed; it does not authorize
waiting, impose a minimum dwell, or replace earlier request/cargo-ready events.
No truck, station, drone, realized daily total, or hidden order stream is read.

Spatial pilot: use SystemWaitRule(signal='ready', max_h=30). An independent
10,000-position diagnostic on layout 610 gave mean geometric readiness lags
10.72 minutes (uniform) and 8.69 (dual). At low/peak/late demand rates, B(t,10)
was .0339/.1001/.0686 (uniform) and .0603/.1781/.1220 (dual). Thus a target of
.5 with a 10-minute ceiling would always clip; keeping the predeclared .5 target
and raising the ceiling to 30 yielded h=24/16/18 and 22/14/16 respectively.
These were geometry-only diagnostics, not profit-based parameter selection.
The production history uses 160 independent days compressed into 128 equally
weighted midpoint quantiles per city, so its exact numbers can differ slightly.
"""
from dataclasses import dataclass
from bisect import bisect_right
from math import floor, isfinite

from .information import expected_arrivals, opportunities_open, unknown, validate_public


def network_expected_arrivals(public, now, end=None):
    """Forecast arrivals in integer release slots (now, end], over all cities."""
    validate_public(public)
    if not unknown(public):
        raise ValueError('system waiting requires the unknown-total public demand law')
    return sum(expected_arrivals(public, city, now, end)
               for city in sorted(public['demand_model']['rates_by_city']))


def network_expected_ready(public, now, h, *, daily_average=False):
    """Expected geometric pickup readiness by now+h from *future* arrivals.

    B(t,h) = sum_city sum_(integer u in (t,t+h]) lambda_city(u)
             * F_city(h-(u-t)). Historical F includes nearest eligible station
    round-trip flight, customer service and drone unloading, assuming an idle
    drone. It is a spatial opportunity proxy, not a live feasibility certificate:
    queues, live aircraft/trucks, capacity and truck loading are not modeled.
    """
    validate_public(public)
    if not unknown(public):
        raise ValueError('system waiting requires the unknown-total public demand law')
    model = public['demand_model']
    if 'spatial_service_lags_by_city' not in model:
        raise ValueError('ready signal requires independent historical spatial service lags')
    if not opportunities_open(public, now):
        return 0.
    total = 0.
    horizon = public['revelation_end']
    # The daily-only alternative uses a stationary forecast and t=0, so the
    # selected cadence is independent of the current time and realized totals.
    t = 0. if daily_average else float(now)
    for city, segments in model['rates_by_city'].items():
        lags = model['spatial_service_lags_by_city'][city]
        if not lags:
            continue
        average = sum((r['end']-r['start']+1)*r['rate'] for r in segments)/horizon
        for u in range(floor(t)+1, min(horizon, floor(t+float(h)))+1):
            rate = average if daily_average else next(
                (r['rate'] for r in segments if r['start'] <= u <= r['end']), 0.)
            total += rate * bisect_right(lags, float(h)-(u-t)) / len(lags)
    return total


@dataclass(frozen=True)
class SystemWaitRule:
    """A common cadence, with two explicitly different forecast-based modes.

    adaptive: smallest integer h in [min_h, max_h] whose forecast arrivals in
        (now, now+h] reach target_expected; use max_h when none do.
    daily: use the full-day *forecast* average arrival rate and choose h once
        by the same threshold rule. This is constant during an operating day.
    fixed: use fixed_h, including the one-minute fine-grid comparator.

    signal='ready' uses historical spatial service lags together with temporal
    rates; signal='arrivals' is the temporal-only ablation. The default target
    of 0.5 expected requests is a declared pilot setting,
    not a fitted or validated optimum. Fixed mode is bounded to 1..max_h;
    adaptive/daily modes use min_h..max_h. Threshold ties use 1e-12 tolerance.
    """
    mode: str = 'adaptive'
    target_expected: float = .5
    min_h: int = 2
    max_h: int = 10
    fixed_h: int = 5
    signal: str = 'arrivals'

    def __post_init__(self):
        if self.mode not in ('adaptive', 'daily', 'fixed'):
            raise ValueError('mode must be adaptive, daily, or fixed')
        if self.signal not in ('arrivals', 'ready'):
            raise ValueError('signal must be arrivals or ready')
        if (type(self.min_h) is not int or type(self.max_h) is not int or
                not 1 <= self.min_h <= self.max_h):
            raise ValueError('positive integer interval bounds required')
        if (isinstance(self.target_expected, bool) or
                not isfinite(self.target_expected) or self.target_expected <= 0):
            raise ValueError('target_expected must be finite and positive')
        if (type(self.fixed_h) is not int or self.fixed_h < 1 or
                (self.mode == 'fixed' and self.fixed_h > self.max_h)):
            raise ValueError('fixed_h must be an integer in 1..max_h')

    def decision(self, public, now):
        """Return the common interval and a serializable explanation."""
        if isinstance(now, bool) or not isfinite(float(now)) or float(now) < 0:
            raise ValueError('now must be finite and nonnegative')
        validate_public(public)
        if not unknown(public):
            raise ValueError('system waiting requires the unknown-total public demand law')
        horizon = public['revelation_end']
        rates = public['demand_model']['rates_by_city']
        forecast_daily = sum((r['end'] - r['start'] + 1) * r['rate']
                             for city in sorted(rates) for r in rates[city])
        counts = {h: (network_expected_ready(public, now, h, daily_average=self.mode=='daily')
                      if self.signal == 'ready' else
                      forecast_daily / horizon * h if self.mode == 'daily' else
                      network_expected_arrivals(public, now, float(now) + h))
                  for h in range(self.min_h, self.max_h + 1)}
        if self.mode == 'fixed':
            h, reason = self.fixed_h, 'fixed_interval'
        else:
            reached = [h for h, count in counts.items()
                       if count + 1e-12 >= self.target_expected]
            h = reached[0] if reached else self.max_h
            reason = 'threshold_reached' if reached else 'upper_bound_below_target'
        return dict(h=h, mode=self.mode, signal=self.signal, at=float(now), reason=reason,
                    target_expected=self.target_expected,
                    expected_by_h={str(k):v for k, v in counts.items()},
                    forecast_daily_total=forecast_daily,
                    opportunities_open=opportunities_open(public, now))

    def interval(self, public, now):
        return self.decision(public, now)['h']

    def __call__(self, public, now):
        return self.interval(public, now)
