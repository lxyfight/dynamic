"""Difference target, train-only scaling, pairing and selection boundaries."""
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fit_paired_chain as fit


def row(case='a', layout=1, shift=0.):
    return dict(site_id=case+'_s',case=case,spec=dict(layout_id=layout,pattern='uniform'),
        at=10.,station='L1+',closed=False,
        features=[[0.,1.,0,0,0,0,0],[2.,2.,0,0,0,0,0],
                  [5.,3.,0,0,0,0,0],[10.,5.,0,0,0,0,0]],
        J=[100.,98.,95.,90.],G=[[20.+shift,24.+shift,28.+shift,40.+shift],
                               [22.+shift,28.+shift,32.+shift,44.+shift]],
        accepted=[[1,1,2,2],[1,2,2,3]],trips=[[1]*4]*2,
        truck_time=[[0,2,5,10]]*2,departure_delay=[[0,2,5,10]]*2,
        original_G=[-10.,-12.,-15.,-20.])


class PairedChainFitTests(unittest.TestCase):
    def test_target_subtracts_known_score_and_common_state_level(self):
        x,y,w = fit.regression_design([row()],[0,1])
        np.testing.assert_allclose(x,[[2,1],[5,2],[10,4]])
        np.testing.assert_allclose(y,[7,14,31])
        shifted = fit.regression_design([row(shift=100000)],[0,1])
        np.testing.assert_allclose(shifted[1],y)
        np.testing.assert_allclose(w,[1/3]*3)
        _,level,_ = fit.regression_design([row()],[0,1],'level')
        np.testing.assert_allclose(level,[-79,-72,-65,-48])

    def test_weighted_rms_and_case_state_weights_are_explicit(self):
        a,b,c = row('a'),row('b'),row('b')
        c['site_id']='b_second'
        x,y,w = fit.regression_design([a,b,c],[0,1])
        self.assertAlmostEqual(w[:3].sum(),.5)
        self.assertAlmostEqual(w[3:].sum(),.5)
        model=fit.fit_ridge([a,b,c],[0,1],.1)
        np.testing.assert_allclose(model['scales'],np.sqrt(np.sum(w[:,None]*x*x,axis=0)))
        z=x/np.asarray(model['scales'])
        expected=np.linalg.solve(z.T@(w[:,None]*z)+.1*np.eye(2),z.T@(w*y))
        np.testing.assert_allclose(model['weights'],expected)
        self.assertEqual(model['intercept'],0.)
        self.assertAlmostEqual(model['fit']['weight_sum'],1.)

    def test_suffix_replication_does_not_change_fit_or_effective_alpha(self):
        original=row()
        repeated=deepcopy(original)
        repeated['G']*=20
        a=fit.fit_ridge([original],[0,1],1)
        b=fit.fit_ridge([repeated],[0,1],1)
        self.assertEqual(a,b)

    def test_tune_selects_actions_but_scaling_is_train_only(self):
        training=[row()]
        tune=[row('tune')]
        tune[0]['features']=[[v*100 for v in p] for p in tune[0]['features']]
        selected,_=fit.select_model(training,tune,[0,1])
        expected=fit.fit_ridge(training,[0,1],selected['alpha'])
        self.assertEqual(selected['scales'],expected['scales'])
        self.assertNotIn('dev',inspect.signature(fit.select_model).parameters)
        self.assertNotIn('dev',inspect.signature(fit.fit_ridge).parameters)

    def test_ties_prefer_larger_alpha_then_shorter_action_and_simple_family(self):
        train=[row()]
        tune=[row('tune')]
        tune[0]['G']=[[0.]*4,[0.]*4]
        model,_=fit.select_model(train,tune,[0,1])
        self.assertEqual(model['alpha'],100.)
        self.assertEqual(fit.choose_action([0,1e-10,0,0]),0)
        self.assertEqual(fit.choose_action([-1,4,4,4]),2)
        base=dict(model)
        aug=dict(model,columns=[0,1,7])
        self.assertEqual(fit.select_family({'paired_augmented':aug,'paired_chain':base}),'paired_chain')

    def test_layout_interval_uses_eight_clusters_and_signflip_resolution(self):
        stats=fit.cluster_stats([1,2,3,4,5,6,7,8])
        self.assertEqual(stats['n'],8)
        self.assertEqual(stats['t_df'],7)
        self.assertEqual(stats['signflip_assignments'],256)
        self.assertAlmostEqual(stats['signflip_one_sided_p'],1/256)
        self.assertAlmostEqual(stats['signflip_two_sided_p'],2/256)
        self.assertAlmostEqual(stats['ci95'][0],4.5-2.364624251*np.std(range(1,9),ddof=1)/np.sqrt(8))

    def test_case_then_layout_aggregation_does_not_weight_state_counts(self):
        rows=[row('a',1),row('b',1),row('c',2)]
        extra=deepcopy(rows[0]);extra['site_id']='a_second';rows.append(extra)
        captures=[dict(case=c,spec=dict(layout_id=l,pattern=p)) for c,l,p in
                  [('a',1,'uniform'),('b',1,'dual'),('c',2,'uniform')]]
        model=dict(columns=[0,1],scales=[1.,1.],weights=[0.,0.])
        result=fit.report_split(rows,dict(old_chain=model,paired_chain=model),captures)
        inference=result['models']['paired_chain']['layout_inference']['depart']
        self.assertEqual(inference['n'],2)
        self.assertEqual(inference['mean'],0.)
        self.assertEqual(result['models']['paired_chain']['choice_counts'],{0:4})

    def test_run_freezes_before_dev_outcomes_and_dev_cannot_change_alpha(self):
        old=dict(kind='legacy',columns=[0,1],scales=[1.,1.],weights=[0.,0.],source={'sha256':'old'})
        observed=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def known(path,destination,augmented):
                r=row(path.name,1)
                return [r],dict(samples=2,cases=[r['spec']]),[dict(case=r['case'],spec=r['spec'])]
            def labels(path,features,protocol,captures):
                if path.name=='dev':
                    frozen=fit.read(root/'base/frozen_selection.json')
                    observed.append(frozen['selected_models']['paired_chain']['alpha'])
                    self.assertFalse(frozen['dev_outcomes_used_for_selection'])
                    features=deepcopy(features)
                    features[0]['G']=[[1e12,-1e12,1e12,-1e12]]*2
                return features,dict(file_receipts={path.name:'fixed'})
            with patch.object(fit,'known_feature_bank',side_effect=known), \
                 patch.object(fit,'load_labels',side_effect=labels), \
                 patch.object(fit,'old_chain_model',return_value=old):
                result=fit.run(root,'base')
            self.assertEqual(len(observed),1)
            self.assertEqual(observed[0],fit.read(root/'base/models/paired_chain.json')['alpha'])
            self.assertIsNone(result['primary_family'])
            self.assertFalse(result['proceed_to_full_day_model_comparison'])

    def test_original_suffix_never_enters_fit(self):
        a=row();b=deepcopy(a);b['original_G']=[1e10,-1e10,1e10,-1e10]
        self.assertEqual(fit.fit_ridge([a],[0,1],1),fit.fit_ridge([b],[0,1],1))

    def test_defer_dev_does_not_load_dev_features_or_labels(self):
        old=dict(kind='legacy',columns=[0,1],scales=[1.,1.],weights=[0.,0.],source={'sha256':'old'})
        observed=[]
        with tempfile.TemporaryDirectory() as directory:
            def known(path,destination,augmented):
                self.assertNotEqual(path.name,'dev')
                r=row(path.name,1)
                return [r],{},[dict(case=r['case'],spec=r['spec'])]
            def labels(path,features,protocol,captures):
                self.assertNotEqual(path.name,'dev')
                observed.append(path.name)
                return features,dict(file_receipts={path.name:'fixed'})
            with patch.object(fit,'known_feature_bank',side_effect=known), \
                 patch.object(fit,'load_labels',side_effect=labels), \
                 patch.object(fit,'old_chain_model',return_value=old):
                result=fit.run(directory,'base',defer_dev=True)
            self.assertEqual(observed,['train','tune'])
            self.assertEqual(set(result['reports']),{'train','tune'})
            self.assertFalse(result['dev_evaluated'])
            self.assertTrue((Path(directory)/'base/frozen_selection.json').exists())

    def test_prepare_never_opens_offline_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            def known(path,destination,augmented):
                fit.write(destination,{'states':[row()]})
                return [row()],{},[{'case':'a'}]
            with patch.object(fit,'known_feature_bank',side_effect=known), \
                 patch.object(fit,'load_labels',side_effect=AssertionError('offline labels opened')):
                result=fit.prepare(directory,'base')
            self.assertFalse(result['future_labels_read'])
            self.assertEqual(set(result['splits']),{'train','tune','dev'})

    def test_two_empty_cases_still_cover_all_eight_layouts(self):
        captures=[];rows=[]
        for layout in range(8):
            for pattern in ('uniform','dual'):
                case=str(layout)+'_'+pattern
                spec=dict(layout_id=layout,pattern=pattern)
                captures.append(dict(case=case,spec=spec))
                if layout in (2,3) and pattern=='dual':
                    continue
                r=row(case,layout);r['spec']=spec;rows.append(r)
        model=dict(columns=[0,1],scales=[1.,1.],weights=[0.,0.])
        report=fit.report_split(rows,dict(old_chain=model),captures)
        protocol=dict(cases=[c['spec'] for c in captures])
        self.assertTrue(fit.complete_layout_coverage(report,protocol))
        self.assertEqual(report['case_coverage']['eligible'],14)
        self.assertEqual(len(report['case_coverage']['no_qualifying_states']),2)
        self.assertEqual(report['models']['old_chain']['layout_inference']['depart']['n'],8)
        report['states']=[s for s in report['states'] if s['spec']['layout_id']!=7]
        self.assertFalse(fit.complete_layout_coverage(report,protocol))

    def test_freeze_rejects_selection_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frozen.json'
            fit.freeze(path,{'alpha':1})
            fit.freeze(path,{'alpha':1})
            with self.assertRaises(ValueError):fit.freeze(path,{'alpha':2})
            actions=Path(directory)/'actions.json'
            fit.freeze(actions,{'choices':{0:2,10:3}})
            fit.freeze(actions,{'choices':{0:2,10:3}})


if __name__=='__main__':
    unittest.main()
