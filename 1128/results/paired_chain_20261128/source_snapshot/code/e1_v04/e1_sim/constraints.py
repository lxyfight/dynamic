"""Pure physical predicates; candidate labels/certificates are never trusted."""
from copy import deepcopy
from decimal import Decimal
from .kernel import Replay, require
from .quantities import InvalidAction, dec, plain, quantity


def distance(a, b):
    require(len(a) == len(b) == 2, "two-dimensional coordinates required")
    return sum((dec(x) - dec(y)) ** 2 for x, y in zip(a, b)).sqrt()


def eligibility(order, station_xy, cfg):
    duration = 2 * distance(order["xy"], station_xy) / dec(cfg["drone_speed_km_min"]) + dec(cfg["customer_service_per_request"])
    q = quantity(order["weight"], positive=True)
    return dict(duration=duration, feasible=q <= dec(cfg["drone_capacity"]) and duration <= dec(cfg["drone_endurance"]))


def validate_quantities(payload):
    """Validate a proposed complete replenishment allocation, without applying it."""
    request, allocations = payload["request"], payload["allocations"]
    try:
        demand = quantity(request["weight"], positive=True)
    except InvalidAction:
        return dict(accepted=False, reason="NONINTEGER_DEMAND")
    try:
        qs = [quantity(a["quantity"]) for a in allocations]
    except InvalidAction:
        return dict(accepted=False, reason="NONINTEGER_ALLOCATION")
    if sum(qs) != demand:
        return dict(accepted=False, reason="QUANTITY_IMBALANCE")
    positive = [(a, q) for a, q in zip(allocations, qs) if q]
    if request["kind"] == "P" and (len(positive) != 1 or positive[0][1] != demand):
        return dict(accepted=False, reason="PICKUP_SPLIT_FORBIDDEN")
    if len({a["station"] for a, _ in positive}) > 1:
        return dict(accepted=False, reason="MULTIPLE_SERVICE_STATIONS")
    by_truck = {}
    for a, q in positive:
        by_truck.setdefault(a["truck"], []).append((q, dec(a["residual_capacity"])))
    if any(sum(q for q, _ in rows) > min(cap for _, cap in rows) for rows in by_truck.values()):
        return dict(accepted=False, reason="CAPACITY_EXCEEDED")
    setup, beta, start = (dec(payload[k]) for k in ("alpha_T", "beta_T", "handling_start"))
    require(setup >= 0 and beta >= 0 and start >= 0, "invalid handling parameters")
    return dict(accepted=True, reason="OK", minimum_positive_allocation=min(q for _, q in positive),
                projected_handling_end=max(start + setup + beta * q for _, q in positive))


def execute_zero_closure(operations, decision):
    """Validate an acyclic submitted batch before invoking its single driver."""
    ids = [op["id"] for op in operations]
    require(len(ids) == len(set(ids)), "duplicate zero-operation identity")
    require(all(dec(o["duration"]) == 0 for o in operations), "closure accepts zero duration only")
    deps = {o["id"]: set(o["depends_on"]) for o in operations}
    require(all(d <= set(ids) for d in deps.values()), "unknown dependency")
    order = []
    while len(order) < len(ids):
        ready = [i for i in ids if i not in order and deps[i] <= set(order)]
        if not ready:
            raise InvalidAction("ZERO_TIME_DEPENDENCY_CYCLE")
        order.extend(ready)
    calls = 0
    decision()
    calls += 1
    completed = []
    for identity in order:
        require(identity not in completed, "duplicate operation execution")
        completed.append(identity)
    return dict(completed_ids=completed, policy_calls=calls)


def continuation(data, cfg):
    """Validate execution of an unchanged, fully committed in-flight suffix."""
    u, candidate = data["drones"][0], data["candidates"][0]
    orders = {o["id"]: o for o in data["orders"]}
    leg = u["active_operation"]
    now, takeoff = dec(data["start_time"]), dec(u["takeoff_at"])
    end, leg_start = dec(leg["end"]), dec(leg["start"])
    speed, service = dec(cfg["drone_speed_km_min"]), dec(cfg["customer_service_per_request"])
    require(leg_start <= now < end and takeoff <= leg_start, "invalid inherited flight")
    require(abs(distance(leg["from"], leg["to"]) / speed - (end-leg_start)) <= dec("1e-6"), "inherited leg timing mismatch")
    require(all(abs(dec(p) - (dec(a) + (dec(b)-dec(a))*(now-leg_start)/(end-leg_start))) <= dec("1e-6")
                for p, a, b in zip(u["location"], leg["from"], leg["to"])), "inherited position mismatch")
    ids = candidate["visit_order"]
    valid = len(ids) == len(set(ids)) and dec(candidate["locked_leg_end"]) == end
    commitment = u["sortie_commitment"]
    valid &= ids == commitment["remaining_visits"]
    valid &= candidate.get("return_station", u["home"]) == commitment["return_station"] == data["local_station"]["id"]
    valid &= all(dec(orders[oid]["release"]) <= takeoff for oid in commitment["remaining_visits"])
    onboard = {l["order"] for l in data["inventory"] if l["holder"] == u["id"] and l["state"] == "onboard"}
    valid &= {oid for oid in onboard if orders[oid]["kind"] == "D"} <= set(ids)
    weight, t, earliest, xy = u["onboard_weight"], end, end, leg["to"]
    km = distance(u["location"], xy)
    for oid in ids:
        o = orders[oid]
        leg_km = distance(xy, o["xy"])
        km += leg_km
        earliest = max(earliest + leg_km/speed, dec(o["release"])) + service
        arrival = t + leg_km/speed
        service_start = dec(candidate["service_start"][oid])
        valid &= service_start >= max(arrival, dec(o["release"])) and o["accepted"] is True and u["home"] in o["eligible_stations"]
        t = service_start + service
        if o["kind"] == "D":
            valid &= oid in onboard and t <= dec(o["deadline"])
            weight -= quantity(o["weight"], positive=True)
        else:
            valid &= oid not in onboard
            weight += quantity(o["weight"], positive=True)
        valid &= 0 <= weight <= cfg["drone_capacity"]
        xy = o["xy"]
    back = distance(xy, data["local_station"]["xy"])
    earliest += back/speed
    km += back
    ret = dec(candidate["return_at"])
    valid &= ret >= t + back/speed and ret - takeoff <= dec(cfg["drone_endurance"])
    return dict(feasible=bool(valid), earliest_return=earliest, flight_km_after_start=km)


def module_observations(package, case):
    """Read only input/configuration. No source expected values enter predicates."""
    data = deepcopy(case["input"])
    cfg = deepcopy(package["configuration"])
    cfg.update(case.get("parameter_overrides", {}))
    before = deepcopy(data)
    result = {"ledger": {k: 0 for k in ("revenue", "truck_trip", "truck_time", "drone_activation", "flight", "lateness", "rejection", "net")}}
    if case["group"] == "E33":
        outcome = validate_quantities(data["candidates"][0]["validation_payload"])
        outcome.update(real_state_unchanged=data == before, ledger_delta=result["ledger"]["net"])
        result["quantity_validation"] = outcome
    elif case["scope"] == "eligibility":
        network = package["networks"][case["network"]]
        result["eligibility"] = {station: eligibility(data["orders"][0], network["stations"][station], cfg)
                                 for station in data["eligibility_candidates"]}
    elif case["group"] == "E11":
        result["candidate"] = continuation(data, cfg)
    elif case["group"] == "E20":
        c = data["candidates"][0]
        o, u = data["orders"][0], data["drones"][0]
        network = package["networks"][case["network"]]
        handling = dec(cfg["drone_handling_setup_per_request"]) + dec(cfg["drone_handling_per_kg"])*o["weight"]
        flight = distance(network["stations"][u["home"]], o["xy"]) / dec(cfg["drone_speed_km_min"])
        complete = dec(c["takeoff"]) + flight + dec(cfg["customer_service_per_request"])
        valid = (dec(c["load_start"]) >= dec(data["start_time"]) and dec(c["takeoff"]) >= dec(c["load_start"])+handling
                 and dec(c["service_start"]) >= dec(c["takeoff"])+flight
                 and dec(c["service_end"]) == dec(c["service_start"])+dec(cfg["customer_service_per_request"])
                 and dec(c["return_at"]) >= dec(c["service_end"])+flight
                 and dec(c["return_at"])-dec(c["takeoff"]) <= dec(cfg["drone_endurance"])
                 and dec(c["service_end"]) <= dec(o["deadline"]))
        # The same execution predicate must also admit the supplied load/sortie.
        clone_data = deepcopy(data); clone_data["candidates"] = []
        clone_data["actions"] = [dict(at=c["load_start"], kind="drone_load", actor=u["id"], orders=c["visit_order"]),
                                 dict(at=c["takeoff"], kind="drone_sortie", actor=u["id"], visit_order=c["visit_order"])]
        replay = Replay(cfg, network, clone_data)
        try:
            replay.run()
        except InvalidAction:
            valid = False
        result["candidate"] = dict(feasible=bool(valid), service_complete=dec(c["service_end"]),
                                   invalid_penalty_substitution_allowed=False, validation_mutated_state=data != before)
    elif case["group"] == "E30":
        clone_data = deepcopy(data); clone_data["candidates"] = []
        replay = Replay(cfg, package["networks"][case["network"]], clone_data)
        result["candidate"] = {}
        for c in data["candidates"]:
            p = replay.probe(dict(operation="drone_sortie", actor=c["actor"], visit_order=c["visit_order"]))
            result["candidate"][c["id"]] = dict(feasible=p["accepted"], state_unchanged=p["state_unchanged"])
    elif case["group"] == "E28":
        result["closure"] = {}
        for graph in data["closure_graphs"]:
            try:
                result["closure"][graph["id"]] = execute_zero_closure(graph["operations"], lambda: None)
            except InvalidAction as exc:
                result["closure"][graph["id"]] = dict(rejected=True, error=str(exc), posted_cost=result["ledger"]["net"])
    else:
        raise InvalidAction("unsupported pure constraint module")
    require(data == before, "pure checker mutated its input")
    return plain(result)
