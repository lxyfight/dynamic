from copy import deepcopy
from decimal import Decimal
import json
import unittest

from e1_sim.constraints import execute_zero_closure, validate_quantities
from e1_sim.initializer import construct_i0, digest, public_delivery_input, verify_plan
from e1_sim.quantities import InvalidAction
from e1_sim.runner import DEFAULT_SPEC, MODULE_IDS, build_replay, execute_case

PACKAGE = json.loads(DEFAULT_SPEC.read_text(encoding="utf8"))
CASES = {c["id"]: c for c in PACKAGE["cases"]}


def problem(ids=1, *, xy=None):
    fixture = deepcopy(CASES["E01"]["input"])
    template = fixture["orders"][0]
    orders = []
    for i in range(ids):
        o = deepcopy(template)
        o["id"] = f"D{i+1}"
        if xy is not None:
            o["xy"] = xy
        orders.append(o)
    return public_delivery_input(PACKAGE["configuration"], PACKAGE["networks"]["fixed_two_city"], orders, fixture["drones"])


class ExecutionLayerTests(unittest.TestCase):
    def test_all_pure_module_cases(self):
        for cid in sorted(MODULE_IDS):
            with self.subTest(case=cid):
                r = execute_case(PACKAGE, deepcopy(CASES[cid]))
                self.assertEqual(r["status"], "PASS", r["assertions"])
                self.assertIsNone(r["independent_audit"])
                self.assertTrue(r["validation_only"])

    def test_external_inventory_injection_forbidden_in_full_day(self):
        c = deepcopy(CASES["E03-BM"])
        c["scope"] = "full_day"
        with self.assertRaisesRegex(InvalidAction, "injection"):
            build_replay(PACKAGE, c)

    def test_duplicate_external_completion_rejected(self):
        c = deepcopy(CASES["E03-BM"])
        c["input"]["external_events"].append(deepcopy(c["input"]["external_events"][0]))
        with self.assertRaisesRegex(InvalidAction, "insufficient"):
            build_replay(PACKAGE, c).run()

    def test_single_batch_permission_blocks_second_batch(self):
        c = deepcopy(CASES["E03-BM"])
        c["input"]["policy"]["batch_permission"] = "B1"
        with self.assertRaisesRegex(InvalidAction, "single-batch"):
            build_replay(PACKAGE, c).run()

    def test_mismatched_inherited_road_event_rejected(self):
        c = deepcopy(CASES["E08-plain"])
        c["input"]["external_events"][0]["at"] = 109
        with self.assertRaisesRegex(InvalidAction, "matching inherited motion"):
            build_replay(PACKAGE, c)

    def test_cycle_does_not_invoke_driver(self):
        calls = []
        with self.assertRaisesRegex(InvalidAction, "CYCLE"):
            execute_zero_closure([dict(id="a", duration=0, depends_on=["a"])], lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_duplicate_closure_identity_rejected(self):
        with self.assertRaisesRegex(InvalidAction, "duplicate"):
            execute_zero_closure([dict(id="a", duration=0, depends_on=[])]*2, lambda: None)

    def test_quantity_capacity_aggregated_per_truck(self):
        p = deepcopy(CASES["E33-split"]["input"]["candidates"][0]["validation_payload"])
        for a in p["allocations"]:
            a.update(truck="T1", residual_capacity=6)
        self.assertEqual(validate_quantities(p)["reason"], "CAPACITY_EXCEEDED")

    def test_quantity_cannot_split_across_service_stations(self):
        p = deepcopy(CASES["E33-split"]["input"]["candidates"][0]["validation_payload"])
        p["allocations"][1]["station"] = "L1-"
        self.assertFalse(validate_quantities(p)["accepted"])

    def test_fee_include_amendment_and_no_warning(self):
        for cid in ("E09", "E21"):
            r = execute_case(PACKAGE, deepcopy(CASES[cid]))
            self.assertEqual(r["warnings"], [])
        before_path = DEFAULT_SPEC.with_name("e1_cases_before_cost_fix.json")
        before = json.loads(before_path.read_text(encoding="utf8"))
        self.assertEqual([c["assertions"] for c in before["cases"]], [c["assertions"] for c in PACKAGE["cases"]])
        # E11's later commitment amendment is independent of fee bookkeeping.
        self.assertEqual([c["input"] for c in before["cases"] if c['group'] != 'E11'],
                         [c["input"] for c in PACKAGE["cases"] if c['group'] != 'E11'])

    def test_locked_flight_end_cannot_be_changed(self):
        c = deepcopy(CASES["E11-exact"])
        c["input"]["candidates"][0]["locked_leg_end"] = 110
        r = execute_case(PACKAGE, c)
        self.assertFalse(r["observations"]["candidate"]["feasible"])


class InitializerTests(unittest.TestCase):
    def test_single_delivery_matches_independent_e01(self):
        r = construct_i0(problem())
        self.assertEqual(r["status"], "VERIFIED", r)
        self.assertEqual(r["validation"]["ledger"]["net"], -211.1)
        self.assertEqual(r["validation"]["completed_deliveries"], {"D1": 21.5})

    def test_empty_delivery_returns_empty_plan(self):
        r = construct_i0(problem(0), max_candidates=0)
        self.assertEqual(r["status"], "VERIFIED", r)
        self.assertEqual(r["plan"]["replay_input"]["actions"], [])
        self.assertEqual(r["validation"]["ledger"]["net"], 0)

    def test_multiple_deliveries_same_drone_serialized_and_first_use_once(self):
        r = construct_i0(problem(3))
        self.assertEqual(r["status"], "VERIFIED", r)
        assignments = r["plan"]["assignments"]
        for a, b in zip(assignments, assignments[1:]):
            self.assertLessEqual(Decimal(a["drone_return"]), Decimal(b["drone_load"]))
        self.assertEqual(r["validation"]["ledger"]["drone_activation"], 45)

    def test_tight_deadlines_use_multiple_entities_in_parallel(self):
        p = problem(2)
        p["drones"].append(dict(id="U2", home="L1+"))
        for o in p["deliveries"]:
            o["deadline"] = 21.5
        r = construct_i0(p)
        self.assertEqual(r["status"], "VERIFIED", r)
        self.assertEqual({a["drone"] for a in r["plan"]["assignments"]}, {"U1", "U2"})

    def test_deadline_failure_never_returns_partial_plan_or_proof(self):
        p = problem(2)
        p["deliveries"][1]["deadline"] = 21.5
        p["deliveries"][0]["deadline"] = 21.5
        r = construct_i0(p)
        self.assertEqual(r["status"], "CONSTRUCTION_FAILED", r)
        self.assertIsNone(r["plan"])
        self.assertFalse(r["infeasibility_proven"])

    def test_noninteger_input_rejected(self):
        p = problem()
        p["deliveries"][0]["weight"] = 10.5
        r = construct_i0(p)
        self.assertEqual(r["status"], "INVALID_INPUT")
        self.assertIsNone(r["plan"])

    def test_deadline_equality_and_just_late(self):
        p = problem()
        p["deliveries"][0]["deadline"] = 21.5
        self.assertEqual(construct_i0(p)["status"], "VERIFIED")
        p["deliveries"][0]["deadline"] = 21.49
        self.assertEqual(construct_i0(p)["status"], "CONSTRUCTION_FAILED")

    def test_geometry_filter_recomputed_not_trusting_eligibility_list(self):
        p = problem(xy=[20, 21.8])
        p["drones"].append(dict(id="U2", home="L1-"))
        r = construct_i0(p)
        self.assertEqual(r["status"], "VERIFIED", r)
        self.assertEqual(r["plan"]["assignments"][0]["station"], "L1+")

    def test_hidden_pickups_do_not_change_public_input_or_plan_hash(self):
        c = deepcopy(CASES["E02"]["input"])
        one = public_delivery_input(PACKAGE["configuration"], PACKAGE["networks"]["fixed_two_city"], c["orders"], c["drones"])
        c["orders"][1].update(weight=19, release=520, xy=[22, 9], deadline=999)
        two = public_delivery_input(PACKAGE["configuration"], PACKAGE["networks"]["fixed_two_city"], c["orders"], c["drones"])
        self.assertEqual(digest(one), digest(two))
        self.assertEqual(construct_i0(one)["plan_hash"], construct_i0(two)["plan_hash"])

    def test_nonpublic_input_fields_refused(self):
        p = problem()
        p["future_orders"] = []
        self.assertEqual(construct_i0(p)["status"], "INVALID_INPUT")

    def test_input_not_mutated(self):
        p = problem(2)
        before = deepcopy(p)
        construct_i0(p)
        self.assertEqual(p, before)

    def test_budget_exhaustion_has_no_partial_incumbent(self):
        r = construct_i0(problem(3), max_candidates=1)
        self.assertEqual(r["status"], "TIME_LIMIT_NO_PLAN")
        self.assertIsNone(r["plan"])
        r = construct_i0(problem(), max_seconds=0)
        self.assertEqual(r["status"], "TIME_LIMIT_NO_PLAN")

    def test_lossless_json_replay_for_irrational_flight_times(self):
        p = problem(3, xy=[21.125, 4.875])
        r = construct_i0(p)
        self.assertEqual(r["status"], "VERIFIED", r)
        loaded = json.loads(json.dumps(r["plan"], ensure_ascii=False))
        self.assertEqual(verify_plan(p, loaded)["ledger"], r["validation"]["ledger"])

    def test_tampered_plan_missing_order_fails_verification(self):
        p = problem()
        r = construct_i0(p)
        plan = deepcopy(r["plan"])
        plan["replay_input"]["actions"] = []
        with self.assertRaises(InvalidAction):
            verify_plan(p, plan)

    def test_broken_network_rejected(self):
        p = problem()
        p["network"]["arcs"][-1]["to"] = "L2+"
        self.assertEqual(construct_i0(p)["status"], "INVALID_INPUT")

    def test_delivery_cannot_be_marked_rejected_to_pass_termination(self):
        c = deepcopy(CASES["E01"])
        c["input"]["orders"][0]["accepted"] = False
        c["input"]["actions"] = []
        with self.assertRaisesRegex(InvalidAction, "mandatory"):
            build_replay(PACKAGE, c)

    def test_tampered_assignment_quantity_fails(self):
        p = problem()
        r = construct_i0(p)
        r["plan"]["assignments"][0]["quantity"] = 9
        with self.assertRaisesRegex(InvalidAction, "assignment quantity"):
            verify_plan(p, r["plan"])

    def test_projection_rejects_already_used_fleet(self):
        c = deepcopy(CASES["E01"]["input"])
        c["drones"][0]["activated"] = True
        with self.assertRaisesRegex(InvalidAction, "unused initial fleet"):
            public_delivery_input(PACKAGE["configuration"], PACKAGE["networks"]["fixed_two_city"], c["orders"], c["drones"])


if __name__ == "__main__":
    unittest.main()
