"""I0 basic deterministic delivery constructor and independent replay handoff.

One replenishment trip + one customer sortie per delivery supplies a transparent
feasible baseline. It is not a complete feasibility algorithm or a local search.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from time import perf_counter

from .audit import audit_ledger
from .cost_model import fee_key, departure_cost
from .constraints import distance, eligibility
from .kernel import require
from .quantities import InvalidAction, dec, plain, quantity
from .runner import build_replay, compare


PHYSICAL_KEYS = (
    "truck_capacity", "drone_capacity", "drone_endurance", "drone_speed_km_min",
    "truck_handling_setup", "truck_handling_per_kg", "drone_handling_setup_per_request",
    "drone_handling_per_kg", "customer_service_per_request", "truck_time_cost",
    "truck_distance_cost", "toll_per_marked_km", "drone_distance_cost",
    "revenue_per_kg", "rejection_per_kg", "pickup_lateness_per_request_min", "truck_dispatch_cutoff")
ORDER_KEYS = ("id", "kind", "weight", "release", "deadline", "city", "xy", "eligible_stations")


def lossless(value):
    """Keep generated nonintegral times as decimal strings across JSON reloads."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: lossless(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [lossless(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(lossless(value), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def public_delivery_input(configuration, network, orders, drones):
    """Whitelist projection: no P attributes, future stream, policies or labels."""
    require(all(not u.get("activated", False) and u.get("onboard_weight", 0) == 0
                and u.get("takeoff_at") is None and u.get("active_operation") is None
                and u.get("location", u["home"]) == u["home"] for u in drones),
            "I0 requires an unused initial fleet, not an inherited online state")
    return deepcopy(dict(configuration={k: configuration[k] for k in PHYSICAL_KEYS + (fee_key(configuration),)},
                         network={"arcs": network["arcs"], "stations": network["stations"]},
                         deliveries=sorted([{k: o[k] for k in ORDER_KEYS} for o in orders if o["kind"] == "D"], key=lambda o: o["id"]),
                         drones=sorted([dict(id=u["id"], home=u["home"]) for u in drones], key=lambda u: u["id"])))


def _roads(network):
    arcs = {a["from"]: a for a in network["arcs"]}
    require(len(arcs) == len(network["arcs"]), "ambiguous fixed circuit")
    node, elapsed, arrivals, seen = "depot", Decimal(0), {}, set()
    while node not in seen:
        seen.add(node)
        require(node in arcs, "broken fixed circuit")
        arc = arcs[node]
        require(dec(arc["min"]) > 0 and dec(arc["km"]) >= 0, "invalid road weight")
        elapsed += dec(arc["min"])
        node = arc["to"]
        if node == "depot":
            require(len(seen) == len(arcs), "disconnected road circuit")
            require(set(arrivals) == set(network["stations"]), "circuit/station mismatch")
            return arrivals, elapsed
        require(node in network["stations"], "unknown road station")
        arrivals[node] = elapsed
    raise InvalidAction("circuit does not return to depot")


def verify_plan(problem, plan, *, record_snapshots=True):
    """Replay the generated D-only baseline; do not alter real future counts."""
    cfg, network = problem["configuration"], problem["network"]
    case = dict(id="I0-generated", scope="full_day", network="generated", parameter_overrides={},
                input=deepcopy(plan["replay_input"]), assertions=[],
                ledger_scope=dict(include=["all"], exclude=["historical_costs"]))
    replay = build_replay(dict(configuration=cfg, networks={"generated": network}), case)
    replay._record_snapshots = record_snapshots
    replay.run()
    obs = replay.observations()
    independent = audit_ledger(cfg, network, replay.initial, replay.trace)
    require(not independent["errors"], "independent ledger trace error")
    require(all(compare(obs["ledger"][k], v, "close") for k, v in independent["ledger"].items()), "ledger disagreement")
    require(compare(obs["ledger"]["net"], plan["predicted_net"], "close"), "construction/replay objective mismatch")
    require(set(replay.orders) == {o["id"] for o in problem["deliveries"]}, "missing/extra generated delivery")
    require(all(replay.orders[o["id"]]["weight"] == quantity(o["weight"], positive=True)
                and replay.orders[o["id"]]["xy"] == o["xy"]
                and replay.orders[o["id"]]["deadline"] == dec(o["deadline"])
                and replay.orders[o["id"]]["eligible_stations"] == o["eligible_stations"]
                for o in problem["deliveries"]), "generated plan changed an input request")
    require({u["id"]: u["home"] for u in replay.drones.values()} == {u["id"]: u["home"] for u in problem["drones"]},
            "generated plan changed the drone fleet")
    require(len(plan["assignments"]) == len(problem["deliveries"]) and
            {a["order"] for a in plan["assignments"]} == set(replay.orders), "assignment table missing/duplicated delivery")
    for a in plan["assignments"]:
        oid = a["order"]
        require(quantity(a["quantity"], positive=True) == replay.orders[oid]["weight"], "assignment quantity mismatch")
        require(replay.orders[oid]["station"] == a["station"], "assignment station mismatch")
        require(any(e["kind"] == "service_done" and e["order"] == oid and e["actor"] == a["drone"] for e in replay.trace), "assignment drone mismatch")
        allocations = a.get('allocations', [dict(truck=a.get('truck'), quantity=a['quantity'], truck_return=a.get('truck_return'))])
        require(sum(quantity(r['quantity'], positive=True) for r in allocations) == a['quantity'], 'assignment allocation imbalance')
        for r in allocations:
            require(any(e["kind"] == "truck_unload_done" and e["actor"] == r["truck"] and e["quantities"].get(oid) == r["quantity"] for e in replay.trace), "assignment truck mismatch")
            require(compare(replay.trucks[r['truck']]['return_time'], r['truck_return'], 'close'), 'assignment truck return mismatch')
        require(compare(replay.orders[oid]["completed_at"], a["service_complete"], "close"), "assignment service time mismatch")
    return dict(terminal=obs["terminal"], ledger=obs["ledger"], independent_ledger=independent["ledger"],
                completed_deliveries={oid: o["completed_at"] for oid, o in obs["order"].items()},
                truck_returns={actor: t["return_time"] for actor, t in obs["truck"].items()},
                trace=plain(replay.trace))


def construct_i0(public_problem, *, max_candidates=100000, max_seconds=None):
    """Return a verified complete baseline or a non-feasibility-proof failure.

    Budget covers construction; mandatory replay verification is timed separately.
    This phase has no improving search, and never returns a partially built plan.
    """
    begin = perf_counter()
    attempted = 0
    problem = deepcopy(public_problem)
    try:
        require(set(problem) == {"configuration", "network", "deliveries", "drones"}, "I0 accepts only its public projection")
        cfg, network, orders, drones = (problem[k] for k in ("configuration", "network", "deliveries", "drones"))
        require(set(cfg) == set(PHYSICAL_KEYS + (fee_key(cfg),)), "unexpected/nonpublic configuration fields")
        require(isinstance(max_candidates, int) and not isinstance(max_candidates, bool) and max_candidates >= 0, "invalid candidate budget")
        require(max_seconds is None or dec(max_seconds) >= 0, "invalid time budget")
        if max_seconds is not None:
            max_seconds = float(max_seconds)
        for key in PHYSICAL_KEYS:
            require(dec(cfg[key]) >= 0, "negative physical parameter")
        require(all(dec(cfg[k]) > 0 for k in ("drone_speed_km_min", "drone_capacity", "truck_capacity", "drone_endurance", "customer_service_per_request")),
                "positive physical parameter required")
        arrivals, circuit_min = _roads(network)
        require(len({o["id"] for o in orders}) == len(orders), "duplicate delivery ID")
        require(len({u["id"] for u in drones}) == len(drones), "duplicate drone ID")
        require(all(u["home"] in network["stations"] for u in drones), "unknown drone home")
        for o in orders:
            require(set(o) == set(ORDER_KEYS), "unexpected/nonpublic delivery fields")
            require(o["kind"] == "D" and dec(o["release"]) == 0, "I0 only accepts initial deliveries")
            quantity(o["weight"], positive=True)
            require(dec(o["deadline"]) >= 0, "negative delivery deadline")
            require(isinstance(o["eligible_stations"], list) and set(o["eligible_stations"]) <= set(network["stations"]), "invalid eligible stations")
            distance(o["xy"], o["xy"])
        input_hash = digest(problem)
        trip_cost = sum(dec(a["km"]) * (dec(cfg["truck_distance_cost"]) +
                        (dec(cfg["toll_per_marked_km"]) if a["tolled"] else 0)) for a in network["arcs"])
        free = {u["id"]: Decimal(0) for u in drones}
        used = set()
        assignments, actions, trucks = [], [], []
        cost = Decimal(0)
        stop = Decimal(0)
        for index, o in enumerate(sorted(orders, key=lambda o: (dec(o["deadline"]), o["id"]))):
            q = quantity(o["weight"], positive=True)
            unload_duration = dec(cfg["truck_handling_setup"]) + dec(cfg["truck_handling_per_kg"])*q
            load_duration = dec(cfg["drone_handling_setup_per_request"]) + dec(cfg["drone_handling_per_kg"])*q
            truck_return = circuit_min + unload_duration
            candidates = []
            for station in sorted(o["eligible_stations"]):
                for u in sorted((u for u in drones if u["home"] == station), key=lambda u: u["id"]):
                    if attempted >= max_candidates or (max_seconds is not None and perf_counter()-begin >= max_seconds):
                        return dict(status="TIME_LIMIT_NO_PLAN", reason="construction budget exhausted before a complete plan", plan=None,
                                    public_input_hash=input_hash, candidates_evaluated=attempted, constructed_orders=len(assignments))
                    attempted += 1
                    if q > dec(cfg["truck_capacity"]) or not eligibility(o, network["stations"][station], cfg)["feasible"]:
                        continue
                    ready = arrivals[station] + unload_duration
                    load = max(ready, free[u["id"]])
                    takeoff = load + load_duration
                    km = distance(network["stations"][station], o["xy"])
                    flight = km / dec(cfg["drone_speed_km_min"])
                    complete = takeoff + flight + dec(cfg["customer_service_per_request"])
                    returned = complete + flight
                    if complete > dec(o["deadline"]):
                        continue
                    increment = trip_cost + truck_return*dec(cfg["truck_time_cost"]) + 2*km*dec(cfg["drone_distance_cost"])
                    increment += departure_cost(cfg, u["id"] in used)
                    candidates.append((increment, complete, station, u["id"], load, takeoff, returned, ready))
            if not candidates:
                return dict(status="CONSTRUCTION_FAILED", reason="no feasible insertion in the basic single-order construction", failed_order=o["id"],
                            plan=None, public_input_hash=input_hash, candidates_evaluated=attempted, constructed_orders=len(assignments),
                            infeasibility_proven=False)
            increment, complete, station, uid, load, takeoff, returned, ready = min(candidates)
            tid = f"I0T{index+1:05d}"
            require(tid not in free, "generated truck ID collides with drone ID")
            cost += increment
            free[uid] = returned
            used.add(uid)
            stop = max(stop, returned, truck_return)
            trucks.append(dict(id=tid, location="depot", onboard_weight=0, departed_at=None,
                               unloading_phase="not_started", completed_batches=0, active_operation=None))
            assignments.append(dict(order=o["id"], station=station, drone=uid, truck=tid, quantity=q,
                                    truck_arrival=arrivals[station], replenishment_ready=ready, drone_load=load,
                                    takeoff=takeoff, service_complete=complete, drone_return=returned, truck_return=truck_return))
            actions.extend([
                dict(at=0, kind="truck_dispatch", actor=tid, delivery_orders=[o["id"]], quantities={o["id"]: q}),
                dict(at=arrivals[station], kind="truck_unload", actor=tid, station=station, quantities={o["id"]: q}),
                dict(at=ready, kind="truck_continue_circuit", actor=tid),
                dict(at=load, kind="drone_load", actor=uid, orders=[o["id"]]),
                dict(at=takeoff, kind="drone_sortie", actor=uid, visit_order=[o["id"]])])
        initial = dict(start_time=0, stop_time=stop, orders=[dict(o, accepted=True) for o in orders], trucks=trucks,
                       drones=[dict(u, location=u["home"], activated=False, onboard_weight=0, takeoff_at=None, active_operation=None) for u in drones],
                       inventory=[dict(order=o["id"], lot=o["id"], weight=quantity(o["weight"], positive=True), state="depot", holder="depot") for o in orders],
                       actions=sorted(actions, key=lambda a: dec(a["at"])), external_events=[], policy=None, candidates=[],
                       public_information=dict(future_count=0, revelation_closed=True))
        revenue = sum(dec(cfg["revenue_per_kg"])*quantity(o["weight"], positive=True) for o in orders)
        plan = lossless(dict(algorithm="I0-basic", assignments=assignments, replay_input=initial, predicted_net=revenue-cost))
        construction_seconds = perf_counter()-begin
        validation_begin = perf_counter()
        validation = verify_plan(problem, plan)
        return dict(status="VERIFIED", plan=plan, plan_hash=digest(plan), public_input_hash=input_hash,
                    candidates_evaluated=attempted, construction_seconds=construction_seconds,
                    validation_seconds=perf_counter()-validation_begin, validation=validation,
                    limitations=["single-order trips and sorties; no merging/local improvement",
                                 "construction failure does not prove infeasibility",
                                 "D-only validation world; keep actual future-count information when adopting the plan"])
    except (InvalidAction, KeyError, TypeError, ValueError) as exc:
        return dict(status="INVALID_INPUT" if attempted == 0 else "VALIDATION_FAILED", reason=str(exc), plan=None,
                    candidates_evaluated=attempted)
