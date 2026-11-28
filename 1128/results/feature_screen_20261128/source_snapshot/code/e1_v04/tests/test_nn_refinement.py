import unittest
import numpy as np
from nn_refinement import predict,execute
from nn_comparison import execute as original_execute
import test_nn_comparison as previous_tests

class NNRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_tests.NNComparisonTests.setUpClass();cls.world=previous_tests.NNComparisonTests.world
    def test_zero_correction_is_exact_scalar(self):
        x=np.arange(60).reshape(3,20)/10
        model=dict(kind='residual',theta=30,alpha=0,correction=dict(kind='zero'))
        np.testing.assert_array_equal(predict(model,x),predict(dict(kind='scalar',theta=30),x))
    def test_centering_is_offline_only_and_preserves_actions(self):
        model=dict(kind='scalar',theta=30)
        a=execute(self.world,model);b=original_execute(self.world,model)
        self.assertEqual(a['plans'],b['plans']);self.assertEqual(a['net'],b['net'])
        for r in a['rows']:
            future=sum(o['weight']*self.world['configuration']['rejection_per_kg'] for o in self.world['input']['orders'] if o['kind']=='P' and o['release']>r['t'])
            self.assertAlmostEqual(r['y'],a['net']-r['j']+future)
    def test_fresh_confirmation_layouts(self):
        from run_nn_comparison import specs as prior
        from run_nn_refinement import specs
        old={r['layout_seed'] for split in ('train','valid','confirm') for s in (0,17,29,43) for r in prior(split,s)}
        new={r['layout_seed'] for r in specs('confirm')}
        self.assertFalse(old&new)
