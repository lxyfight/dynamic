"""Mixed representations of replay times must identify the same RH plan."""
from decimal import Decimal
import unittest

from run_frozen_reopt_comparison import physical_key


def row(starts, review, wakes):
    return {'plan':dict(actions=[{'kind':'truck_unload','actor':'T1','at':t,'test_id':i}
                                for i,t in starts],
                        new_trucks=[],decisions={},wait_reviews={'T1':review},wakeups=wakes)}


class ReoptPhysicalSignature(unittest.TestCase):
    def test_mixed_numeric_representations_and_chronology(self):
        first = row([(1,'21.0'),(2,Decimal('40.00'))],Decimal('46.000'),['46.0',63])
        reordered = row([(2,40),(1,Decimal('21'))],'46.0',[Decimal('63.0'),46])
        self.assertEqual(physical_key(first),physical_key(reordered))
        shifted = row([(2,40),(1,'21.00000000000000000000000000001')],46,[46,63])
        self.assertNotEqual(physical_key(first),physical_key(shifted))

    def test_same_time_execution_order_is_retained(self):
        self.assertNotEqual(physical_key(row([(1,'21.0'),(2,21)],46,[46])),
                            physical_key(row([(2,21),(1,Decimal('21.00'))],46,[46])))
