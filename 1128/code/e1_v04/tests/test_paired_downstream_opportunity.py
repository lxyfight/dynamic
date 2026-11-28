from copy import deepcopy
from types import SimpleNamespace
import unittest

from e1_sim.p2_arrivals import KIND
from paired_downstream_opportunity import opportunity_from_candidate, downstream_opportunity
from uniform_wait_worlds import CFG


def fixture():
    public=dict(information_mode='unknown_daily_total',revelation_end=30,revelation_closed=False,
        demand_model=dict(kind=KIND,space=dict(pattern='uniform',city_x=[0.]),
            count_by_slot=[dict(start=1,end=30,trials=2,probability=.1)],
            rates_by_city={'C1':[dict(start=1,end=30,rate=.2)]},
            attributes=dict(weight_min=10,weight_max=10,pickup_deadline_lags=[240,300,360])))
    known=SimpleNamespace(now=0.,public=public,cfg=deepcopy(CFG),orders={},
        network=dict(stations={'L1+':[0.,.25],'L1-':[0.,-.25]}),
        trucks={'T':dict(onboard_weight=0)},
        drones={'U+':dict(home='L1+',active_operation=None),'U-':dict(home='L1-',active_operation=None)},
        _hidden={},_queue=[])
    candidate=dict(feasible=True,allowed=True,plan=dict(actions=[],decisions={},new_trucks=[]),
        trace=[dict(kind='road_started',actor='T',source='L1+',at=5.)],
        state=SimpleNamespace(orders={},trucks={'T':dict(return_time=50.)}))
    return known,candidate


class DownstreamOpportunityTests(unittest.TestCase):
    def test_late_readiness_can_be_reserved_after_reveal(self):
        known,candidate=fixture()
        value,audit=opportunity_from_candidate(known,candidate,'T',spatial_cells=2)
        # All four midpoint customers need >8 min flight+service. Releases
        # 1..5 cannot be ready by minute 5, but profitable reservations exist.
        self.assertGreater(value,0.)
        self.assertGreater(audit['expected_late_ready_reservations'],0.)
        self.assertAlmostEqual(audit['expected_profitable_requests'],
                               audit['expected_late_ready_reservations'])
        self.assertLess(audit['max_profitable_extra_delay'],(10*4.8-25)/.4)
        self.assertEqual(audit['units'],'expected_requests')
        self.assertEqual(value,audit['expected_profitable_requests'])
        self.assertGreater(audit['expected_positive_margin_CNY'],value)

    def test_negative_margins_are_not_counted(self):
        known,candidate=fixture()
        known.cfg['revenue_per_kg']=known.cfg['rejection_per_kg']=0.
        value,audit=opportunity_from_candidate(known,candidate,'T',spatial_cells=2)
        self.assertEqual(value,0.)
        self.assertEqual(audit['expected_profitable_requests'],0.)

    def test_downstream_passage_changes_feature_without_current_wait_credit(self):
        known,candidate=fixture()
        candidate['trace'][0]['at']=0.
        candidate['trace'].append(dict(kind='road_started',actor='T',source='L1-',at=10.))
        before,a=opportunity_from_candidate(known,candidate,'T',spatial_cells=4)
        later=deepcopy(candidate);later['trace'][1]['at']=20.
        later['state'].trucks['T']['return_time']+=10
        after,b=opportunity_from_candidate(known,later,'T',spatial_cells=4)
        self.assertGreater(after,before)
        self.assertEqual(a['station_profitable_requests']['L1+'],0.)
        self.assertEqual(b['station_profitable_requests']['L1+'],0.)

    def test_delivery_slack_bounds_reservation_and_later_capacity_binds(self):
        known,candidate=fixture()
        known.orders['D']=dict(kind='D',weight=10,completed_at=None,deadline=10.)
        candidate['state'].orders['D']=dict(kind='D',completed_at=10.)
        value,audit=opportunity_from_candidate(known,candidate,'T',spatial_cells=2)
        self.assertEqual(value,0.)
        self.assertEqual(audit['windows'][0]['delivery_delay_budget'],0.)
        known,candidate=fixture()
        known.orders['P']=dict(kind='P',weight=1300,completed_at=None,deadline=400.)
        candidate['state'].orders['P']=dict(kind='P',accepted=True,completed_at=50.)
        candidate['trace'] += [dict(kind='truck_load_done',actor='T',orders=['P'],at=10.),
                               dict(kind='road_started',actor='T',source='L1-',at=11.)]
        self.assertEqual(opportunity_from_candidate(known,candidate,'T',spatial_cells=2)[0],0.)

    def test_known_and_future_lateness_costs_reduce_opportunities(self):
        known,candidate=fixture()
        value,_=opportunity_from_candidate(known,candidate,'T',spatial_cells=2)
        late=deepcopy(candidate);late['state'].trucks['T']['return_time']=300.
        self.assertLess(opportunity_from_candidate(known,late,'T',spatial_cells=2)[0],value)
        known.orders['P']=dict(kind='P',weight=10,completed_at=None,deadline=1.)
        candidate['state'].orders['P']=dict(kind='P',accepted=True,completed_at=50.)
        known.cfg['pickup_lateness_per_request_min']=100.
        self.assertEqual(opportunity_from_candidate(known,candidate,'T',spatial_cells=2)[0],0.)

    def test_hidden_future_is_not_read_and_raw_environment_is_rejected(self):
        known,candidate=fixture()
        baseline=opportunity_from_candidate(known,candidate,'T',spatial_cells=2)[0]
        class ForbiddenFuture:
            def __getattribute__(self,name): raise AssertionError('future read')
            def __iter__(self): raise AssertionError('future read')
        candidate['state']._hidden=ForbiddenFuture()
        self.assertEqual(opportunity_from_candidate(known,candidate,'T',spatial_cells=2)[0],baseline)
        known._hidden=ForbiddenFuture()
        with self.assertRaisesRegex(ValueError,'known_clone'):
            opportunity_from_candidate(known,candidate,'T',spatial_cells=2)
        known._hidden={'secret':dict(weight=1)}
        with self.assertRaisesRegex(ValueError,'known_clone'):
            downstream_opportunity(known,candidate['plan'],'T')

    def test_duplicate_station_opportunities_are_not_double_counted(self):
        known,candidate=fixture()
        baseline=opportunity_from_candidate(known,candidate,'T',spatial_cells=2)[0]
        known.network['stations']['L1-']=known.network['stations']['L1+'][:]
        candidate['trace'].append(dict(kind='road_started',actor='T',source='L1-',at=5.))
        self.assertEqual(opportunity_from_candidate(known,candidate,'T',spatial_cells=2)[0],baseline)


if __name__=='__main__': unittest.main()
