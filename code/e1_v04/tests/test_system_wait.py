import unittest
from copy import deepcopy

from e1_sim.quantities import InvalidAction
from e1_sim.information import validate_public
from e1_sim.system_wait import SystemWaitRule, network_expected_arrivals, network_expected_ready
from e1_sim.unknown_demand import public_law


def piecewise_law():
    law = public_law(2, 0, 100)
    law['demand_model']['rates_by_city'] = {
        'C1': [dict(start=1, end=50, rate=.01), dict(start=51, end=100, rate=.2)],
        'C2': [dict(start=1, end=50, rate=.01), dict(start=51, end=100, rate=.3)],
    }
    return law


class SystemWaitTests(unittest.TestCase):
    def test_spatial_signal_changes_global_h_with_identical_arrival_rates(self):
        near, far = public_law(1,10,100), public_law(1,10,100)
        near['demand_model']['spatial_service_lags_by_city'] = {'C1':[2.,2.]}
        far['demand_model']['spatial_service_lags_by_city'] = {'C1':[6.,6.]}
        ready = SystemWaitRule(signal='ready', max_h=30)
        self.assertEqual(ready.interval(near,0), 7)
        self.assertEqual(ready.interval(far,0), 11)
        temporal = SystemWaitRule(signal='arrivals', max_h=30)
        self.assertEqual(temporal.interval(near,0), temporal.interval(far,0))

    def test_ready_signal_integrates_release_times_and_lag_cdf(self):
        law = public_law(1,10,100)
        law['demand_model']['spatial_service_lags_by_city'] = {'C1':[2.,4.]}
        # By t=5: u=1 reaches both lags, u=2 and u=3 only the 2min lag.
        self.assertAlmostEqual(network_expected_ready(law,0,5), .2)
        self.assertAlmostEqual(network_expected_ready(law,.5,5), .2)
        self.assertEqual(network_expected_ready(law,0,2), 0)

    def test_ready_signal_uses_time_varying_rate_with_fixed_spatial_profile(self):
        law = piecewise_law()
        law['demand_model']['spatial_service_lags_by_city'] = {'C1':[4.], 'C2':[4.]}
        rule = SystemWaitRule(signal='ready', max_h=30)
        self.assertEqual(rule.interval(law,0), 29)
        self.assertEqual(rule.interval(law,50), 5)

    def test_spatial_schema_is_optional_validated_and_not_silently_imputed(self):
        law = public_law(1,10,100)
        validate_public(law)
        with self.assertRaises(ValueError):
            SystemWaitRule(signal='ready', max_h=30).interval(law,0)
        for profile in ({'C2':[1.]}, {'C1':[-1.]}, {'C1':[float('nan')]},
                        {'C1':[4.,2.]}, {'C1':[]}, {'C1':[True]}):
            bad = deepcopy(law)
            bad['demand_model']['spatial_service_lags_by_city'] = profile
            with self.subTest(profile=profile), self.assertRaises(InvalidAction):
                validate_public(bad)

    def test_global_rate_changes_common_interval(self):
        law = piecewise_law()
        rule = SystemWaitRule(target_expected=.5)
        self.assertEqual(rule.interval(law, 0), 10)
        self.assertEqual(rule.interval(law, 50), 2)
        self.assertEqual(rule.decision(law, 0)['reason'], 'upper_bound_below_target')
        self.assertEqual(rule.decision(law, 50)['reason'], 'threshold_reached')

    def test_city_redistribution_does_not_change_network_interval(self):
        a, b = piecewise_law(), piecewise_law()
        b['demand_model']['rates_by_city'] = {
            'C1': [dict(start=1, end=50, rate=.02), dict(start=51, end=100, rate=.5)],
            'C2': [dict(start=1, end=100, rate=0)],
        }
        rule = SystemWaitRule()
        for now in (0, 30, 49.5, 50, 80, 99):
            self.assertEqual(rule.interval(a, now), rule.interval(b, now))

    def test_integrates_through_band_boundary_and_counts_both_cities(self):
        law = piecewise_law()
        self.assertAlmostEqual(network_expected_arrivals(law, 49.5, 51.5), .52)
        self.assertEqual(SystemWaitRule().interval(law, 49.5), 2)

    def test_daily_mode_constant_and_uses_public_full_day_average(self):
        rule, law = SystemWaitRule(mode='daily'), piecewise_law()
        # 26 expected requests / 100 minutes: 2 minutes reach 0.5.
        self.assertEqual({rule.interval(law, t) for t in (0, 15, 50, 99)}, {2})
        self.assertEqual(SystemWaitRule().interval(law, 0), 10)

    def test_fixed_modes_ignore_arrival_intensity(self):
        for h in (1, 2, 5, 10):
            rule = SystemWaitRule(mode='fixed', fixed_h=h)
            self.assertEqual(rule.interval(piecewise_law(), 0), h)
            self.assertEqual(rule.interval(piecewise_law(), 50), h)

    def test_horizon_caps_forecast_and_does_not_authorize_waiting(self):
        law = piecewise_law()
        self.assertAlmostEqual(network_expected_arrivals(law, 99, 120), .5)
        self.assertEqual(network_expected_arrivals(law, 100, 120), 0)
        decision = SystemWaitRule().decision(law, 100)
        self.assertFalse(decision['opportunities_open'])
        law['revelation_closed'] = True
        self.assertEqual(network_expected_arrivals(law, 10, 20), 0)
        self.assertFalse(SystemWaitRule(mode='daily').decision(law, 10)['opportunities_open'])

    def test_no_mutation_and_forbidden_future_fields_rejected(self):
        law = piecewise_law()
        before = deepcopy(law)
        for mode in ('adaptive', 'daily', 'fixed'):
            SystemWaitRule(mode=mode).decision(law, 45)
        self.assertEqual(law, before)
        law['realized_total'] = 15
        with self.assertRaises(InvalidAction):
            SystemWaitRule().interval(law, 0)

    def test_bad_configuration_and_times_rejected(self):
        for kwargs in (dict(mode='station'), dict(target_expected=0),
                       dict(target_expected=float('nan')), dict(min_h=0),
                       dict(min_h=4, max_h=3), dict(mode='fixed', fixed_h=11)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SystemWaitRule(**kwargs)
        for t in (-1, float('inf'), float('nan'), True):
            with self.subTest(now=t), self.assertRaises(ValueError):
                SystemWaitRule().interval(piecewise_law(), t)


if __name__ == '__main__':
    unittest.main()
