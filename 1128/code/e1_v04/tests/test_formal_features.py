import unittest
from copy import deepcopy
import numpy as np
from run_vfa_development import world,PACKAGE
from e1_sim.online import prepare_online,known_clone,remaining_plan,evaluate_plan
from e1_sim.formal_features import vector,names
from run_feature_iterations import fit_equal_day

class FormalFeatureTests(unittest.TestCase):
    def setUp(self):
        raw,_=world(8000);s,c=prepare_online(raw,PACKAGE['configuration'],PACKAGE['networks']['fixed_two_city'])
        self.s=known_clone(s);self.c=evaluate_plan(self.s,remaining_plan(self.s))
    def test_dimensions_finite(self):
        for group,size in [('base',6),('coordination',15),('full',18)]:
            x=vector(self.s,self.c,group);self.assertEqual(len(x),size);self.assertEqual(len(names(group)),size);self.assertTrue(np.isfinite(x).all())
    def test_unknown_count_is_not_silently_zero(self):
        self.s.public['future_count']=None
        with self.assertRaises(ValueError):vector(self.s,self.c)
    def test_terminal_future_residual_zero(self):
        self.s.public['future_count']=0;self.assertEqual(vector(self.s,self.c),[0.]*18)
    def test_candidate_changes_features(self):
        c=deepcopy(self.c);c['plan']['new_trucks'].append({'id':'diagnostic'})
        self.assertNotEqual(vector(self.s,self.c),vector(self.s,c))
    def test_hidden_future_changes_do_not_change_features(self):
        raw,_=world(8001);s,c=prepare_online(raw,PACKAGE['configuration'],PACKAGE['networks']['fixed_two_city'])
        k=known_clone(s);candidate=evaluate_plan(k,remaining_plan(k))
        self.assertEqual(vector(self.s,self.c),vector(k,candidate))
    def test_day_repetition_does_not_change_day_weight(self):
        a=[([1.,0.],4.),([1.,1.],8.)];b=[([1.,2.],1.)]
        x,_,_=fit_equal_day([a,b]);y,_,_=fit_equal_day([a*7,b])
        np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)

if __name__=='__main__':unittest.main()
