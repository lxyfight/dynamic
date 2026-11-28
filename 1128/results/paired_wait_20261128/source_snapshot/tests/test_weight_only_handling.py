import json
import unittest
from decimal import Decimal

from e1_sim.runner import DEFAULT_SPEC, build_replay
from uniform_wait_worlds import historical_law, make_world


class WeightOnlyHandlingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = make_world(710, 0, 6, .5, 'uniform', cities=1,
                               arrival_mode='p2', history_days=2, improvement_budget=0)

    def test_truck_and_drone_handling_and_customer_service(self):
        package = json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
        package['configuration'] = self.world['configuration']
        # The same 10 kg delivery and 10 kg pickup pass through all four operations.
        case = next(case for case in package['cases'] if case['id'] == 'E02')
        trace = build_replay(package, case).run().trace
        kinds = {'truck_unload_started', 'truck_load_started', 'drone_handling_started'}
        handling = [(e['kind'], e.get('loading'), e['end'] - e['at'])
                    for e in trace if e['kind'] in kinds]
        quarter = Decimal('0.25')
        self.assertCountEqual(handling, [('truck_unload_started', None, quarter),
                                          ('truck_load_started', None, quarter),
                                          ('drone_handling_started', True, quarter),
                                          ('drone_handling_started', False, quarter)])
        self.assertEqual([e['end'] - e['at'] for e in trace if e['kind'] == 'service_started'],
                         [1, 1])

    def test_historical_forecast_uses_same_drone_unloading_time(self):
        # Reuse the same historical marks: changing unloading time moves every
        # idle-drone round-trip + service + unloading quantile by exactly a minute.
        current = self.world['input']['public_information']['demand_model']
        one_minute_setup = dict(self.world['configuration'], drone_handling_setup_per_request=1)
        other = historical_law(self.world['network'], 'uniform', cfg=one_minute_setup,
                               history_days=2)['demand_model']
        current_lags = current['spatial_service_lags_by_city']['C1']
        other_lags = other['spatial_service_lags_by_city']['C1']
        self.assertGreater(len(current_lags), 0)
        self.assertEqual(len(current_lags), len(other_lags))
        for revised, previous in zip(current_lags, other_lags):
            self.assertAlmostEqual(previous - revised, 1)
