"""Regression and negative tests beyond the original E1 expected-value fixtures."""
from copy import deepcopy
from decimal import Decimal
import json
import unittest

from e1_sim.audit import audit_ledger
from e1_sim.quantities import InvalidAction, quantity
from e1_sim.runner import DEFAULT_SPEC, REPLAY_IDS, build_replay, execute_case


PACKAGE = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
CASES = {c["id"]: c for c in PACKAGE["cases"]}


def case(cid):
    return deepcopy(CASES[cid])


def truncate(c, stop):
    c["scope"] = "local"
    c["input"]["stop_time"] = stop
    c["input"]["actions"] = [a for a in c["input"]["actions"] if a["at"] <= stop]
    c["input"]["external_events"] = [e for e in c["input"]["external_events"] if e["at"] <= stop]
    return c


def station_pickups(two_trucks=False):
    c = case("E21")
    d = c["input"]
    d.update(stop_time=115, actions=[], drones=[], inventory=[], external_events=[])
    d["orders"] = []
    for i in (1, 2):
        o = deepcopy(CASES["E02"]["input"]["orders"][1])
        o.update(id=f"P{i}", accepted=True)
        d["orders"].append(o)
        d["inventory"].append(dict(order=o["id"], lot=o["id"], weight=10, state="available", holder="L1+"))
    d["trucks"] = d["trucks"] if two_trucks else d["trucks"][:1]
    for t in d["trucks"]:
        t.update(onboard_weight=0, unloading_phase="skipped")
    return c


class KernelTests(unittest.TestCase):
    def test_all_replay_fixtures_and_independent_ledger(self):
        for cid in sorted(REPLAY_IDS):
            with self.subTest(case=cid):
                r = execute_case(PACKAGE, case(cid))
                self.assertEqual("PASS", r["status"], r["assertions"])
                self.assertTrue(r["independent_audit"]["matches_online"])

    def test_expected_values_never_drive_execution(self):
        c = case("E01")
        a = build_replay(PACKAGE, c).run().observations()
        for assertion in c["assertions"]:
            assertion["expected"] = "nonsense"
        b = build_replay(PACKAGE, c).run().observations()
        self.assertEqual(a, b)

    def test_integer_contract_no_rounding_or_bool(self):
        for bad in (0, -1, 6.5, True, float("nan"), float("inf"), "10"):
            with self.subTest(value=bad), self.assertRaises(InvalidAction):
                quantity(bad, positive=True)
        self.assertEqual(quantity(10.0), 10)
        self.assertEqual(quantity(0), 0)

    def test_noninteger_demand_rejected_at_input(self):
        for kind in ("D", "P"):
            c = case("E02")
            next(o for o in c["input"]["orders"] if o["kind"] == kind)["weight"] = 10.5
            with self.subTest(kind=kind), self.assertRaises(InvalidAction):
                build_replay(PACKAGE, c)

    def test_wrong_inventory_balance_rejected(self):
        c = case("E21")
        c["input"]["inventory"][1]["weight"] = 5
        c["input"]["trucks"][1]["onboard_weight"] = 5
        with self.assertRaisesRegex(InvalidAction, "conservation"):
            build_replay(PACKAGE, c)

    def test_unknown_inventory_vehicle_rejected(self):
        c = case("E21")
        c["input"]["inventory"][0]["holder"] = "not-a-truck"
        with self.assertRaisesRegex(InvalidAction, "vehicle"):
            build_replay(PACKAGE, c)

    def test_fractional_unload_is_atomic(self):
        r = build_replay(PACKAGE, case("E21"))
        before = deepcopy(r.__dict__)
        with self.assertRaises(InvalidAction):
            r.apply(dict(kind="truck_unload", actor="T1", station="L1+", quantities={"D1": 5.5}))
        self.assertEqual(before, r.__dict__)

    def test_dispatch_integer_split_across_two_trucks(self):
        c = case("E01")
        d = c["input"]
        second = deepcopy(d["trucks"][0])
        second["id"] = "T2"
        d["trucks"].append(second)
        d["actions"] = []
        r = build_replay(PACKAGE, c)
        for actor, q in (("T1", 6), ("T2", 4)):
            r.apply(dict(kind="truck_dispatch", actor=actor, delivery_orders=["D1"], quantities={"D1": q}))
        self.assertEqual(r.trucks["T1"]["onboard_weight"], 6)
        self.assertEqual(r.trucks["T2"]["onboard_weight"], 4)
        self.assertEqual(sum(l["weight"] for l in r.lots), 10)

    def test_valid_probe_is_pure_and_does_not_activate(self):
        r = build_replay(PACKAGE, case("E09"))
        before = r.public_snapshot()
        result = r.probe(dict(kind="probe", operation="drone_load", actor="U1", orders=["D1"]))
        self.assertTrue(result["accepted"])
        self.assertTrue(result["state_unchanged"])
        self.assertEqual(before, r.public_snapshot())
        self.assertEqual(r.ledger["drone_activation"], 0)
        self.assertEqual(r._queue, [])

    def test_two_truck_partial_delivery_must_wait_for_completion(self):
        r = build_replay(PACKAGE, case("E21")).run([105.15, 107.1])
        self.assertEqual(r.observe("station.L1+.delivery_available_weight", 105.15), 6)
        self.assertFalse(r.probes[0]["accepted"])
        self.assertEqual(r.observe("station.L1+.delivery_available_weight", 107.1, "after_completions"), 10)
        self.assertEqual(r.observe("station.L1+.delivery_available_weight", 107.1), 0)

    def test_drone_overlap_rejected_atomically(self):
        r = build_replay(PACKAGE, case("E09"))
        r.apply(dict(kind="drone_load", actor="U1", orders=["D1"]))
        before = deepcopy(r.__dict__)
        with self.assertRaisesRegex(InvalidAction, "active operation"):
            r.apply(dict(kind="drone_load", actor="U1", orders=["D2"]))
        self.assertEqual(before, r.__dict__)

    def test_delivery_completion_deadline_exact_and_late(self):
        for deadline, valid in ((21.5, True), (21.49, False)):
            c = case("E01")
            c["input"]["orders"][0]["deadline"] = deadline
            r = build_replay(PACKAGE, c)
            if valid:
                r.run()
            else:
                with self.assertRaisesRegex(InvalidAction, "deadline"):
                    r.run()

    def test_endurance_includes_service_but_excludes_ground_loading(self):
        for limit, valid in ((5, True), (4.99, False)):
            c = case("E01")
            c["parameter_overrides"]["drone_endurance"] = limit
            if valid:
                build_replay(PACKAGE, c).run()
            else:
                with self.assertRaisesRegex(InvalidAction, "endurance"):
                    build_replay(PACKAGE, c).run()

    def test_flight_segment_capacity_depends_on_visit_order(self):
        for order, valid in ((["D1", "P1"], True), (["P1", "D1"], False)):
            c = truncate(case("E02"), 18.5)
            c["input"]["orders"][1]["weight"] = 11
            next(a for a in c["input"]["actions"] if a["kind"] == "drone_sortie")["visit_order"] = order
            if valid:
                build_replay(PACKAGE, c).run()
            else:
                with self.assertRaisesRegex(InvalidAction, "capacity"):
                    build_replay(PACKAGE, c).run()

    def test_dynamic_request_hidden_until_reveal(self):
        c = case("E02")
        r = build_replay(PACKAGE, c)
        self.assertNotIn("P1", r.public_snapshot()["orders"])
        before = deepcopy(r.__dict__)
        with self.assertRaisesRegex(InvalidAction, "unrevealed"):
            r.apply(dict(kind="accept", actor=None, orders=["P1"]))
        self.assertEqual(before, r.__dict__)
        r.run([1])
        self.assertIsNone(r.observe("order.P1.accepted", 1, "before_starts"))
        self.assertTrue(r.observe("order.P1.accepted", 1))

    def test_public_projection_does_not_expose_hidden_world(self):
        c1, c2 = case("E02"), case("E02")
        c2["input"]["orders"][1].update(weight=11, xy=[21, 4.0], release=2)
        c2["input"]["external_events"][0]["at"] = 2
        a, b = build_replay(PACKAGE, c1), build_replay(PACKAGE, c2)
        self.assertEqual(a.public_snapshot(), b.public_snapshot())
        self.assertNotIn("actions", a.public_snapshot())

    def test_same_address_has_two_services_and_zero_flight(self):
        r = build_replay(PACKAGE, case("E02")).run()
        services = [e for e in r.trace if e["kind"] == "service_started"]
        self.assertEqual([e["order"] for e in services], ["D1", "P1"])
        self.assertEqual([e["at"] for e in services], [Decimal("20.5"), Decimal("21.5")])
        zero = [e for e in r.trace if e["kind"] == "flight_started" and e["at"] == e["end"]]
        self.assertEqual(len(zero), 1)

    def test_pickup_revenue_only_at_depot(self):
        r = build_replay(PACKAGE, truncate(case("E02"), 25.75)).run()
        self.assertEqual(r.ledger["revenue"], 32)
        self.assertIsNone(r.orders["P1"]["completed_at"])
        self.assertEqual(r.observe("inventory.P1.state"), "loading")

    def test_missing_pickup_recovery_fails_full_day(self):
        c = case("E02")
        c["input"]["actions"] = [a for a in c["input"]["actions"] if a["kind"] != "truck_continue_circuit"]
        with self.assertRaisesRegex(InvalidAction, "not finished"):
            build_replay(PACKAGE, c).run()

    def test_event_partition_does_not_change_accounting(self):
        a = build_replay(PACKAGE, case("E10")).run()
        b = build_replay(PACKAGE, case("E10")).run([100, 480.1, 693.8, 696, 700.1, 720, 721.23])
        self.assertEqual(a.ledger, b.ledger)
        self.assertEqual(a.orders["P1"]["completed_at"], Decimal("723.25"))

    def test_independent_audit_does_not_read_online_postings(self):
        r = build_replay(PACKAGE, case("E02")).run()
        r.ledger["truck_time"] = Decimal(999)
        a = audit_ledger(r.cfg, r.network, r.initial, r.trace)
        self.assertEqual(a["ledger"]["truck_time"], 30.8)
        self.assertNotEqual(float(r.ledger["truck_time"]), a["ledger"]["truck_time"])

    def test_independent_audit_detects_bad_flight_trace(self):
        r = build_replay(PACKAGE, case("E01")).run()
        next(e for e in r.trace if e["kind"] == "flight_started")["end"] += 1
        a = audit_ledger(r.cfg, r.network, r.initial, r.trace)
        self.assertIn("flight duration/distance mismatch", a["errors"])

    def test_partial_flight_window_counts_only_flown_distance(self):
        r = build_replay(PACKAGE, truncate(case("E01"), 20)).run()
        self.assertEqual(r.ledger["flight"], Decimal("0.675"))
        a = audit_ledger(r.cfg, r.network, r.initial, r.trace)
        self.assertEqual(a["ledger"]["flight"], 0.675)

    def test_multibatch_loading_and_locked_batch_cannot_grow(self):
        c = station_pickups()
        c["input"]["actions"] = [
            dict(at=100, kind="truck_load", actor="T1", orders=["P1"]),
            dict(at=103, kind="probe", operation="truck_load", actor="T1", orders=["P2"]),
            dict(at=110, kind="truck_load", actor="T1", orders=["P2"])]
        c["input"]["stop_time"] = 115.25
        r = build_replay(PACKAGE, c).run([103, 105.25])
        self.assertEqual(r.observe("truck.T1.active_batch.orders", 103), ["P1"])
        self.assertFalse(r.probes[0]["accepted"])
        self.assertEqual(r.trucks["T1"]["loading_batches"], 2)
        self.assertEqual(r.trucks["T1"]["onboard_weight"], 20)
        self.assertEqual(r.ledger["truck_time"], Decimal("6.10"))

    def test_loading_count_survives_downstream_station_passes(self):
        r = build_replay(PACKAGE, case("E02")).run()
        self.assertEqual(r.trucks["T1"]["loading_batches"], 1)
        self.assertEqual(r.trucks["T1"]["completed_batches"], 0)

    def test_previous_station_pickup_loading_does_not_block_new_station_unload(self):
        c = station_pickups()
        d = c["input"]
        delivery = deepcopy(CASES["E01"]["input"]["orders"][0])
        d["orders"].append(delivery)
        d["inventory"].append(dict(order="D1", lot="D1", weight=10, state="onboard", holder="T1"))
        d["trucks"][0]["onboard_weight"] = 10
        d["stop_time"] = 144.5
        d["actions"] = [dict(at=100, kind="truck_load", actor="T1", orders=["P1"]),
                        dict(at=105.25, kind="truck_continue_circuit", actor="T1"),
                        dict(at=139.25, kind="truck_unload", actor="T1", station="L1-", quantities={"D1": 10})]
        r = build_replay(PACKAGE, c).run()
        self.assertEqual(r.trucks["T1"]["loading_batches"], 1)
        self.assertEqual(r.trucks["T1"]["unloading_phase"], "completed")
        self.assertEqual(r.observe("station.L1-.delivery_available_weight"), 10)

    def test_pickup_partial_quantity_action_rejected(self):
        r = build_replay(PACKAGE, station_pickups())
        before = deepcopy(r.__dict__)
        with self.assertRaisesRegex(InvalidAction, "whole-order"):
            r.apply(dict(kind="truck_load", actor="T1", orders=["P1"], quantities={"P1": 6}))
        self.assertEqual(before, r.__dict__)

    def test_concurrent_trucks_lock_in_id_order(self):
        c = station_pickups(two_trucks=True)
        c["input"]["stop_time"] = 105.25
        c["input"]["actions"] = [
            dict(at=100, kind="probe", operation="truck_load", actor="T2", station="L1+", orders=["P1"]),
            dict(at=100, kind="truck_load", actor="T1", orders=["P1"])]
        r = build_replay(PACKAGE, c).run([100])
        self.assertFalse(r.probes[0]["accepted"])
        self.assertEqual(r.observe("inventory.P1.holder", 100), "T1")
        self.assertEqual(r.ledger["truck_time"], Decimal("4.20"))

    def test_distinct_trucks_can_load_in_parallel(self):
        c = station_pickups(two_trucks=True)
        c["input"]["stop_time"] = 105.25
        c["input"]["actions"] = [dict(at=100, kind="truck_load", actor=f"T{i}", orders=[f"P{i}"]) for i in (1, 2)]
        r = build_replay(PACKAGE, c).run()
        self.assertEqual([t["loading_end"] for t in r.trucks.values()], [Decimal("105.25")] * 2)

    def test_batch_capacity_exact_and_exceeded(self):
        for capacity, valid in ((20, True), (19, False)):
            c = station_pickups()
            c["parameter_overrides"]["truck_capacity"] = capacity
            r = build_replay(PACKAGE, c)
            action = dict(kind="truck_load", actor="T1", orders=["P1", "P2"])
            if valid:
                r.apply(action)
                self.assertEqual(r.trucks["T1"]["active_batch"]["weight"], 20)
            else:
                before = deepcopy(r.__dict__)
                with self.assertRaisesRegex(InvalidAction, "capacity"):
                    r.apply(action)
                self.assertEqual(before, r.__dict__)

    def test_repeat_unloading_after_loading_forbidden(self):
        r = build_replay(PACKAGE, station_pickups())
        result = r.probe(dict(kind="probe", operation="truck_unload", actor="T1", station="L1+", quantities={"P1": 10}))
        self.assertFalse(result["accepted"])
        self.assertTrue(result["state_unchanged"])

    def test_sequential_drone_unloading_individual_ready_times(self):
        c = station_pickups()
        d = c["input"]
        u = deepcopy(CASES["E21"]["input"]["drones"][0])
        u["onboard_weight"] = 20
        d["drones"] = [u]
        for lot in d["inventory"]:
            lot.update(state="onboard", holder="U1")
        d["stop_time"] = 106.5
        d["actions"] = [dict(at=100, kind="drone_unload", actor="U1", orders=["P1", "P2"]),
                        dict(at=101.25, kind="truck_load", actor="T1", station="L1+", orders=["P1"])]
        r = build_replay(PACKAGE, c).run([101.25, 102.5])
        self.assertEqual(r.orders["P1"]["available_at"], Decimal("101.25"))
        self.assertEqual(r.orders["P2"]["available_at"], Decimal("102.5"))
        self.assertEqual(r.observe("inventory.P1.state", 101.25), "loading")
        self.assertEqual(r.observe("inventory.P2.state", 101.25), "unloading")

    def test_zero_handling_completion_closure(self):
        c = case("E01")
        c["parameter_overrides"].update(truck_handling_setup=0, truck_handling_per_kg=0,
                                        drone_handling_setup_per_request=0, drone_handling_per_kg=0)
        for a in c["input"]["actions"]:
            if a["kind"] in ("drone_load", "truck_continue_circuit", "drone_sortie"):
                a["at"] = 12
        r = build_replay(PACKAGE, c).run([12])
        self.assertEqual(r.orders["D1"]["completed_at"], Decimal(15))
        self.assertEqual(r.drones["U1"]["return_time"], Decimal(17))
        self.assertEqual(r.trucks["T1"]["return_time"], Decimal(58))

    def test_missing_observation_is_reported_not_skipped(self):
        c = case("E01")
        c["assertions"].append(dict(target="unimplemented.observation", expected=True, op="equal", at=None, phase="after_closure"))
        r = execute_case(PACKAGE, c)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["assertions"][-1]["status"], "NOT_IMPLEMENTED")


if __name__ == "__main__":
    unittest.main()
