from copy import deepcopy
import unittest,json
from random import Random
from e1_sim.learning import LearningPolicy,vector,wait_pool,sample_release_times
from e1_sim.online import known_clone,remaining_plan,evaluate_plan,prepare_online
from e1_sim.online_examples import make_world
from e1_sim.runner import DEFAULT_SPEC
P=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))

class VFATests(unittest.TestCase):
    def state(self,releases=(1,)):
        raw,_=make_world(P,releases)
        return prepare_online(raw,P['configuration'],P['networks']['fixed_two_city'])[0]

    def test_zero_weights_select_exact_myopic(self):
        s=known_clone(self.state())
        a=LearningPolicy();b=LearningPolicy([0]*10)
        self.assertEqual(a.decide(s)[0],b.decide(s)[0])

    def test_terminal_future_value_zero(self):
        s=known_clone(self.state(()))
        c=evaluate_plan(s,remaining_plan(s))
        self.assertEqual(vector(s,c),[0]*10)

    def test_learning_never_changes_physical_world(self):
        s=known_clone(self.state());before=deepcopy(s.__dict__)
        LearningPolicy([1]*10).decide(s)
        self.assertEqual(s.__dict__,before)

    def test_features_and_candidate_selection_ignore_hidden_realization(self):
        a=known_clone(self.state((1,20)));b=known_clone(self.state((2,90)))
        x=LearningPolicy([1]*10);y=LearningPolicy([1]*10)
        self.assertEqual(x.decide(a)[0],y.decide(b)[0])
        self.assertEqual(x.selection_log,y.selection_log)

    def test_same_exploration_seed_same_decision(self):
        s=known_clone(self.state())
        self.assertEqual(LearningPolicy(seed=99,explore=1).decide(s)[0],LearningPolicy(seed=99,explore=1).decide(s)[0])

    def test_conditional_sampler_strict_future(self):
        draws=sample_release_times(300.5,200,590,Random(1))
        self.assertTrue(all(301<=v<=590 and type(v) is int for v in draws))

    def test_zero_count_has_no_sampled_request(self):
        self.assertEqual(sample_release_times(600,0,590,Random(1)),[])

    def test_invalid_positive_count_after_support_rejected(self):
        with self.assertRaises(ValueError):sample_release_times(590,1,590,Random(1))

    def test_all_evaluated_features_fixed_dimension(self):
        s=known_clone(self.state());p=LearningPolicy();p.decide(s)
        self.assertTrue(all(len(x)==10 for x,j in p.samples))
