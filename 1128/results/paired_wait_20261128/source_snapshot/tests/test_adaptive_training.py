import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from run_unified_vfa import audited_episode, fit_model, main, resolve_modes, review_settings


class AdaptiveTrainingTests(unittest.TestCase):
    def test_defaults_and_legacy_model_resolution(self):
        self.assertEqual(resolve_modes(None, None, default_mode='compare'), ('compare', 'p2'))
        self.assertEqual(resolve_modes(None, None, default_mode='adaptive'), ('adaptive', 'p2'))
        self.assertEqual(resolve_modes(None, None, default_mode='adaptive', fixed_interval_supplied=True),
                         ('fixed', 'fixed_count'))
        self.assertEqual(resolve_modes(None, None, default_mode='compare', fixed_interval_supplied=True,
                                       adaptive_menu_supplied=True), ('compare', 'p2'))
        self.assertEqual(resolve_modes('adaptive', 'fixed_count', default_mode='adaptive'),
                         ('adaptive', 'fixed_count'))
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            resolve_modes(None, None, default_mode='adaptive', model={'h': 5})
        with self.assertRaises(ValueError):
            resolve_modes('adaptive', None, default_mode='adaptive', model={'h': 5})

    def test_fixed_grid_and_common_feature_horizon(self):
        settings = review_settings('compare', None, [10, 5, 2])
        self.assertEqual([c['h'] for c in settings if c['review_intervals'] is None], [2, 5, 10])
        self.assertEqual(settings[-1]['review_intervals'], [2, 5, 10])
        self.assertEqual({c['feature_horizon'] for c in settings}, {5})
        self.assertEqual(review_settings('fixed', [10], [2, 5])[0]['feature_horizon'], 10)
        with self.assertRaises(ValueError):
            review_settings('adaptive', None, [0, 2, 5])
        with self.assertRaises(ValueError):
            review_settings('adaptive', None, [5, 5])

    def test_model_law_cannot_be_reused_silently(self):
        old = dict(h=5)
        with self.assertRaisesRegex(ValueError, 'arrival laws'):
            audited_episode({'spec': {'arrival_mode': 'p2'}}, 5, old)
        samples = [[dict(x=[1.] * 7, y=2.)]]
        fitted = fit_model(samples, 5, review_intervals=[2, 5, 10], arrival_mode='p2')
        self.assertEqual(fitted['wait_mode'], 'adaptive')
        self.assertEqual(fitted['review_intervals'], [2, 5, 10])
        with self.assertRaisesRegex(ValueError, 'arrival laws'):
            audited_episode({'spec': {}}, 5, fitted)

    def test_paired_comparison_freezes_both_winners_before_opening_test(self):
        with TemporaryDirectory() as dirname:
            out = Path(dirname)
            def world(layout, day, total, share, pattern, **kw):
                if layout == 3:
                    for stem in ('selected_fixed_model', 'selected_adaptive_model'):
                        self.assertTrue((out / (stem + '.json')).exists())
                return dict(spec={'layout_id': layout, 'total': total, 'arrival_mode': kw['arrival_mode']},
                            input_hash=str(layout), public_hash='public', i0_hash='initial')
            def episode(world, h, model=None, review_intervals=None, feature_horizon=None, **kw):
                self.assertEqual(feature_horizon, 5)
                layout = world['spec']['layout_id']
                if review_intervals is not None:
                    score = 11 if layout == 2 else -2
                else:
                    score = 10 if h == 10 else 0
                return dict(net=score), [dict(x=[1.] * 7, y=2.)]
            argv = ['run', 'train', '--review-mode', 'compare', '--arrival-mode', 'p2',
                    '--feature-sets', 'none', '--train-layouts', '1', '--validation-layouts', '2',
                    '--test-layouts', '3', '--days', '1', '--orders', '4', '--patterns', 'uniform',
                    '--iterations', '1', '--output', dirname]
            with patch('sys.argv', argv), patch('run_unified_vfa.make_world', side_effect=world), \
                    patch('run_unified_vfa.audited_episode', side_effect=episode):
                main()
            fixed = json.loads((out / 'selected_fixed_model.json').read_text())
            adaptive = json.loads((out / 'selected_adaptive_model.json').read_text())
            protocol = json.loads((out / 'protocol.json').read_text())
            for model in (fixed, adaptive):
                self.assertEqual(model['operating_rules'], protocol['operating_rules'])
                self.assertEqual(model['version'], protocol['version'])
            report = json.loads((out / 'heldout_test.json').read_text())
            self.assertEqual(fixed['h'], 10)
            self.assertEqual(adaptive['review_intervals'], [2, 5, 10])
            self.assertEqual(report['mean_adaptive_minus_fixed'], -12)
            self.assertEqual(report['pairs'][0]['fixed']['net'], 10)


if __name__ == '__main__':
    unittest.main()
