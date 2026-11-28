"""Forecasts integrate over the publicly known pickup deadline distribution."""
from types import SimpleNamespace
import unittest

from e1_sim.unknown_demand import public_law
from unified_vfa_features import features
from wait_increment_model import wait_features


class DeadlineExposureTests(unittest.TestCase):
    def test_positive_lateness_is_averaged_after_applying_each_deadline(self):
        # One expected pickup at t=1. The chain can serve it with certainty,
        # and it returns to the depot 330 minutes after disclosure.
        law = public_law(1, 1, end=1)
        model = law['demand_model']
        model['spatial_service_lags_by_city'] = {'C1': [2.]}
        attr = model['attributes']
        attr.pop('pickup_deadline_lag')
        attr['pickup_deadline_lags'] = [240, 300, 360]
        known = SimpleNamespace(now=0, public=law, network={'stations': {'L1+': [0, 1]}},
            cfg=dict(truck_capacity=100, truck_handling_setup=0, truck_handling_per_kg=0),
            trucks={'T': {'onboard_weight': 0}}, drones={'U': {'home': 'L1+'}}, orders={})
        candidate = dict(plan=dict(actions=[]),
            state=SimpleNamespace(trucks={'T': dict(return_time=331)}, orders={}),
            trace=[dict(kind='road_started', actor='T', source='L1+', at=4)])
        self.assertAlmostEqual(features(known, candidate, 5)[5], (90+30+0)/3)

        # A legacy deterministic law whose lag equals this distribution's mean
        # gives a different answer despite identical assets and return times.
        attr.pop('pickup_deadline_lags')
        attr['pickup_deadline_lag'] = 300
        self.assertAlmostEqual(features(known, candidate, 5)[5], 30.)

    def test_wait_probe_averages_slack_over_future_lags(self):
        law = public_law(1, 1, end=1)
        attr = law['demand_model']['attributes']
        attr.pop('pickup_deadline_lag')
        attr['pickup_deadline_lags'] = [240, 300, 360]
        station = 'L1+'
        known = SimpleNamespace(now=0, public=law, _hidden={}, _handling=lambda *_: 0,
            network={'stations': {station: [0, 1]}},
            cfg=dict(truck_capacity=100, drone_speed_km_min=1, customer_service_per_request=1),
            trucks={'T': {'onboard_weight': 0}}, drones={'U': {'home': station, 'active_operation': None}},
            orders={'P': dict(city='C1', xy=[0, 2], weight=10, eligible_stations=[station])})
        plan = dict(trace=[], score=0, state=SimpleNamespace(trucks={'T': dict(return_time=300)}, orders={}))
        x = wait_features(known, plan, plan, 'T', station)
        self.assertAlmostEqual(x['deadline_slack'], 20.)
        self.assertAlmostEqual(x['truck_opportunity'], x['offer_probability'] * (60/65) / 3)
