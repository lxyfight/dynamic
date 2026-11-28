"""Independent accounting from physical event records, never online postings.

Uses whole active intervals / geometric leg lengths rather than the kernel's
per-event integrator. An agreement is accounting evidence, not an optimizer proof.
"""
from decimal import Decimal
from .quantities import dec, plain
from .cost_model import per_sortie


def audit_ledger(configuration, network, initial, trace):
    cfg = configuration
    start, stop = dec(initial["start_time"]), dec(initial["stop_time"])
    orders = {o["id"]: o for o in initial["orders"]}
    cost_names = ("truck_trip", "truck_time", "drone_activation", "flight", "lateness", "rejection")
    result = {k: Decimal(0) for k in ("revenue",) + cost_names}
    errors = []
    completed, admissions, rejected = {}, {}, set()
    departures, returns = {}, {}
    activated = {u["id"] for u in initial["drones"] if u["activated"]}
    for t in initial["trucks"]:
        if t["departed_at"] is not None:
            departures[t["id"]] = max(start, dec(t["departed_at"]))
    for o in orders.values():
        if o["kind"] == "P" and o["accepted"] is True:
            admissions[o["id"]] = start
    for e in trace:
        at, kind = dec(e["at"]), e["kind"]
        if not start <= at <= stop:
            errors.append("trace event outside ledger window")
        if kind == "truck_dispatched":
            actor = e["actor"]
            if actor in departures:
                errors.append("duplicate truck dispatch")
            departures[actor] = at
            distance = sum(dec(a["km"]) for a in network["arcs"])
            toll_distance = sum(dec(a["km"]) for a in network["arcs"] if a["tolled"])
            result["truck_trip"] += distance * dec(cfg["truck_distance_cost"]) + toll_distance * dec(cfg["toll_per_marked_km"])
        elif kind == "truck_arrived" and e["destination"] == "depot":
            if e["actor"] in returns:
                errors.append("duplicate truck return")
            returns[e["actor"]] = at
        elif kind == "drone_activated":
            if e["actor"] in activated:
                errors.append("duplicate/historical drone activation charge")
            activated.add(e["actor"])
            if not per_sortie(cfg):
                result["drone_activation"] += dec(cfg["drone_first_use_cost"])
        elif kind == "drone_takeoff" and per_sortie(cfg):
            result["drone_activation"] += dec(cfg["drone_sortie_cost"])
        elif kind == "flight_started":
            end = dec(e["end"])
            distance = sum((dec(x) - dec(y)) ** 2 for x, y in zip(e["source"], e["destination"])).sqrt()
            if abs((end - at) * dec(cfg["drone_speed_km_min"]) - distance) > Decimal("1e-20"):
                errors.append("flight duration/distance mismatch")
            fraction = min(Decimal(1), (stop - at) / (end - at)) if end > at else Decimal(0)
            result["flight"] += distance * fraction * dec(cfg["drone_distance_cost"])
        elif kind == "order_completed":
            oid = e["order"]
            if oid in completed:
                errors.append("duplicate order revenue")
            completed[oid] = at
            result["revenue"] += dec(orders[oid]["weight"]) * dec(cfg["revenue_per_kg"])
        elif kind == "accept":
            if e["order"] in admissions or e["order"] in rejected:
                errors.append("duplicate admission")
            admissions[e["order"]] = at
        elif kind == "reject":
            oid = e["order"]
            if oid in admissions or oid in rejected:
                errors.append("duplicate rejection")
            rejected.add(oid)
            result["rejection"] += dec(orders[oid]["weight"]) * dec(cfg["rejection_per_kg"])
    for actor, begin in departures.items():
        result["truck_time"] += (returns.get(actor, stop) - begin) * dec(cfg["truck_time_cost"])
    for oid, accepted in admissions.items():
        end = completed.get(oid, stop)
        overdue = max(Decimal(0), end - max(start, accepted, dec(orders[oid]["deadline"])))
        result["lateness"] += overdue * dec(cfg["pickup_lateness_per_request_min"])
    result["net"] = result["revenue"] - sum(result[k] for k in cost_names)
    return {"ledger": plain(result), "errors": errors}
