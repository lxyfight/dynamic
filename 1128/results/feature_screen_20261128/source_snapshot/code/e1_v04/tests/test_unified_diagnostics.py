"""Diagnostics must distinguish decisions without treating labels as actions."""
from copy import deepcopy
from decimal import Decimal
import unittest

from unified_vfa_policy import canonical_plan_digest


class CanonicalDecisionTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(id='first_label', decisions={'P1': True}, new_trucks=[],
                         actions=[dict(kind='truck_load', actor='T1', at=5,
                                       station='L1+', orders=['P1'])],
                         wait_reviews={'T1': 10})

    def test_label_and_equivalent_number_types_do_not_change_decisions(self):
        other = deepcopy(self.plan)
        other['id'] = 'different_label'
        other['actions'][0]['at'] = Decimal('5.00')
        other['wait_reviews']['T1'] = 10.0
        self.assertEqual(canonical_plan_digest(self.plan), canonical_plan_digest(other))

    def test_admissions_action_times_actors_and_waits_remain_significant(self):
        variants = []
        for target, value in [('admission', False), ('action', 6), ('actor', 'T2'), ('wait', 12)]:
            other = deepcopy(self.plan)
            if target == 'admission':
                other['decisions']['P1'] = value
            elif target == 'action':
                other['actions'][0]['at'] = value
            elif target == 'actor':
                other['actions'][0]['actor'] = value
            else:
                other['wait_reviews']['T1'] = value
            variants.append(other)
        for other in variants:
            self.assertNotEqual(canonical_plan_digest(self.plan), canonical_plan_digest(other))


if __name__ == '__main__':
    unittest.main()
