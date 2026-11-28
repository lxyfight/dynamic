import unittest
import numpy as np
from train_revision import EqualDayRLS, specs_v3, select, GROUPS, SEEDS
from e1_sim.formal_features import names

class TrainingRevisionTests(unittest.TestCase):
    def test_rls_equals_weighted_ridge(self):
        rng=np.random.default_rng(117);rls=EqualDayRLS(19);days=[]
        for n in [1,30,5,12]:
            X=rng.normal(size=(n,19));y=rng.normal(size=n)*100
            day=list(zip(X.tolist(),y.tolist()));days.append(day);rls.add(day)
        got=rls.checkpoint()
        A=np.eye(19);b=np.zeros(19)
        for day in days:
            X=np.array([x for x,y in day]);y=np.array([y for x,y in day]);A+=X.T@X/len(day);b+=X.T@y/len(day)
        np.testing.assert_allclose(got['weights'],np.linalg.solve(A,b),atol=1e-9)
    def test_repeat_samples_does_not_overweight_day(self):
        day=[([1.,2.],3.),([1.,4.],6.)];a=EqualDayRLS(2);b=EqualDayRLS(2)
        a.add(day);b.add(day*10);np.testing.assert_allclose(a.theta,b.theta,atol=1e-12)
    def test_disjoint_splits_and_balanced_checkpoints(self):
        splits=[specs_v3('train',s) for s in SEEDS]+[specs_v3('validation'),specs_v3('confirmation')]
        ids=[x['identity'] for rows in splits for x in rows];self.assertEqual(len(ids),len(set(ids)))
        layouts=[set(x['layout_seed'] for x in rows) for rows in splits]
        for i in range(len(layouts)):
            for j in range(i):self.assertFalse(layouts[i]&layouts[j])
        for rows in splits[:3]:
            self.assertEqual(len(rows),96)
            for k in range(8):
                chunk=rows[k*12:(k+1)*12]
                self.assertEqual({(c,p):sum(x['cities']==c and x['p']==p for x in chunk) for c in [2,4] for p in [.2,.4]}, {(c,p):3 for c in [2,4] for p in [.2,.4]})
    def test_selection_keeps_zero_if_no_improvement(self):
        rows=[dict(group=g,seed=s,training_days=k,validation_mean=10.-k) for g in GROUPS for s in SEEDS for k in [0,12,24]]
        chosen=select(rows);self.assertEqual(chosen['selected_group'],'base')
        self.assertEqual(len(chosen['models']),12);self.assertTrue(all(x['training_days']==0 for x in chosen['models']))
    def test_dimension_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):EqualDayRLS(0)
        with self.assertRaises(ValueError):EqualDayRLS(2).add([([1.],0)])
        with self.assertRaises(ValueError):EqualDayRLS(1).add([([float('nan')],0)])
    def test_fixed_feature_dimensions(self):
        self.assertEqual([len(names(g)) for g in GROUPS],[6,15,18,19])

if __name__=='__main__':unittest.main()
