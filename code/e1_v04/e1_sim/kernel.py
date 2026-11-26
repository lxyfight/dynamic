"""Event/inventory replay, deliberately separated from E1 expected assertions.

The environment owns hidden requests. public_snapshot() exposes only revealed
state. Scripted commands are a test driver, NOT an online admission policy.
Each command is prepared on a clone, so rejection/probing is transactional.
"""
from copy import deepcopy
from decimal import Decimal
import heapq

from .cost_model import fee_key, per_sortie
from .quantities import InvalidAction, UnsupportedCase, dec, plain, quantity

ZERO = Decimal(0)
COSTS = ("truck_trip", "truck_time", "drone_activation", "flight", "lateness", "rejection")


def require(condition, reason):
    if not condition:
        raise InvalidAction(reason)


class Replay:
    def __init__(self, configuration, network, initial, *, full_day=False, planned_stops=None):
        self.cfg = deepcopy(configuration)
        self.network = deepcopy(network)
        for key in ("truck_capacity", "drone_capacity", "drone_endurance", "drone_speed_km_min", "customer_service_per_request"):
            require(dec(self.cfg[key]) > 0, "positive physical parameter required: " + key)
        for key in ("truck_handling_setup", "truck_handling_per_kg", "drone_handling_setup_per_request",
                    "drone_handling_per_kg", "truck_time_cost", "truck_distance_cost", "toll_per_marked_km",
                    fee_key(self.cfg), "drone_distance_cost", "revenue_per_kg", "rejection_per_kg",
                    "pickup_lateness_per_request_min", "truck_dispatch_cutoff"):
            require(dec(self.cfg[key]) >= 0, "negative physical/cost parameter: " + key)
        require(all(dec(a["min"]) > 0 and dec(a["km"]) >= 0 for a in network["arcs"]),
                "minimum kernel requires positive road travel durations")
        self.initial = deepcopy(initial)
        self.full_day = full_day
        self.start = self.now = dec(initial["start_time"])
        self.stop = dec(initial["stop_time"])
        require(ZERO <= self.start <= self.stop, "invalid replay interval")
        self.orders = {}
        self._hidden = {}
        self.trucks = {}
        self.drones = {}
        self.lots = []
        self.trace = []
        self.probes = []
        self.ledger = {k: ZERO for k in ("revenue",) + COSTS}
        self._queue = []
        self._serial = 0
        self._lot_serial = 0
        self._stops = deepcopy(planned_stops or {})
        self._snapshots = {}
        self._actions = {}
        self._driver_calls = {}
        self._completed_event_ids = []
        # Controller memory is public history, copied with the physical state.
        # A reservation for already-known cargo has no active-wait interval.
        self._station_wait_memory = {}
        self._station_wait_history = []
        self.absorption_time = None
        self.batch_permission = (initial.get("policy") or {}).get("batch_permission", "BM")
        require(self.batch_permission in ("B1", "BM"), "unknown batch permission")
        self.public = deepcopy(initial.get("public_information", {}))
        from .information import validate_public
        validate_public(self.public)
        self._arcs = {a["from"]: a for a in network["arcs"]}
        require(len(self._arcs) == len(network["arcs"]), "ambiguous fixed road circuit")
        for row in initial["orders"]:
            o = deepcopy(row)
            o["weight"] = quantity(o["weight"], positive=True)
            o["release"], o["deadline"] = dec(o["release"]), dec(o["deadline"])
            require(o["kind"] in ("D", "P"), "unknown request type")
            require(o["kind"] != "D" or o["accepted"] is True, "mandatory delivery cannot be rejected")
            require(o["id"] not in self.orders and o["id"] not in self._hidden, "duplicate order")
            require(o["weight"] <= self.cfg["drone_capacity"], "request exceeds drone capacity")
            require(bool(o["eligible_stations"]), "no eligible station")
            o.update(completed_at=None, available_at=None, collected_at=None, station=None, assigned_drone=None)
            if o["kind"] == "D" or o["accepted"] is not None:
                require(o["release"] <= self.start, "unrevealed inherited request")
                self.orders[o["id"]] = o
            else:
                self._hidden[o["id"]] = o
        for row in initial["trucks"]:
            t = deepcopy(row)
            if t["active_operation"] is not None:
                raise UnsupportedCase("inherited active truck operation")
            require(t["id"] not in self.trucks, "duplicate truck")
            t.update(return_time=None, departure_time=None, loading_end=None,
                     loading_batches=t.get("completed_batches", 0), active_batch=None)
            t["departed_at"] = None if t["departed_at"] is None else dec(t["departed_at"])
            require(t["departed_at"] is None or t["departed_at"] <= self.start,
                    "future inherited departure")
            self.trucks[t["id"]] = t
            if t["location"] in self.network["stations"]:
                self._begin_station_visit(t, arrival_observed=False)
        for row in initial["drones"]:
            u = deepcopy(row)
            op = u["active_operation"]
            if op is not None:
                if full_day or op["kind"] != "customer_service":
                    raise UnsupportedCase("inherited drone operation other than local customer service")
                require(dec(op["start"]) <= self.start <= dec(op["end"]), "invalid inherited service interval")
                u["active_operation"] = dict(kind="service", end=dec(op["end"]))
            u["takeoff_at"] = None if u["takeoff_at"] is None else dec(u["takeoff_at"])
            require(u["id"] not in self.drones, "duplicate drone")
            u.update(return_time=None, return_times=[], activation_count=int(u["activated"]))
            self.drones[u["id"]] = u
        for row in initial["inventory"]:
            lot = deepcopy(row)
            lot["weight"] = quantity(lot["weight"], positive=True)
            require(lot["order"] in self.orders, "inventory references hidden/unknown request")
            if lot["state"] not in ("depot", "onboard", "available", "customer", "delivered", "completed", "external_pipeline"):
                raise UnsupportedCase("inherited inventory state: " + lot["state"])
            self.lots.append(lot)
        require(len({x["lot"] for x in self.lots}) == len(self.lots), "duplicate lot identity")
        for o in self.orders.values():
            if o["kind"] == "P" and not self._lots(o["id"]):
                self._new_lot(o["id"], o["weight"], "customer", o["id"])
            for lot in self._lots(o["id"]):
                if lot["state"] == "available":
                    require(o["station"] in (None, lot["holder"]), "delivery stocked at multiple stations")
                    o["station"] = lot["holder"]
        for a in initial.get("actions", []):
            at = dec(a["at"])
            require(self.start <= at <= self.stop, "script action outside replay window")
            self._actions.setdefault(at, []).append(deepcopy(a))
        for event in initial.get("external_events", []):
            kind = event["kind"]
            if kind in ("pickup_unload_complete", "truck_return", "customer_service_complete"):
                require(not full_day, "external completion injection forbidden in full-day replay")
                if kind == "truck_return":
                    actor = event["truck"]
                    motion = initial.get("inherited_motion", {}).get(actor)
                    require(motion is not None and dec(motion["arrival_at"]) == dec(event["at"])
                            and motion["destination"] == "depot", "return needs matching inherited motion")
                    t = self.trucks[actor]
                    require(t["departed_at"] is not None and t["location"] == "inbound_road", "invalid inherited road state")
                    t["active_operation"] = dict(kind="road", end=dec(event["at"]))
                elif kind == "customer_service_complete":
                    u = self.drones[event["drone"]]
                    raw = next(v for v in initial["drones"] if v["id"] == u["id"])["active_operation"]
                    require(raw is not None and raw["order"] == event["order"]
                            and dec(raw["end"]) == dec(event["at"]), "service completion lacks inherited commitment")
                self._schedule(dec(event["at"]), kind, event)
                continue
            if kind not in ("request_reveal", "revelation_end", "observation", "observation_only"):
                raise UnsupportedCase("external event: " + event["kind"])
            self._schedule(dec(event["at"]), kind, event, priority=1)
        if full_day:
            require(self.start == 0, "full day must start at zero")
            require(all(t["location"] == "depot" and t["departed_at"] is None for t in self.trucks.values()),
                    "full day cannot inherit dispatched trucks")
            require(all(not u["activated"] and u["location"] == u["home"] for u in self.drones.values()),
                    "full day cannot inherit drone activity")
            require(all(l["state"] == "depot" and self.orders[l["order"]]["kind"] == "D"
                        for l in self.lots), "full day cannot inject inventory")
            if self.public.get("future_count") is not None:
                require(self.public["future_count"] == len(self._hidden), "full-day public count inconsistent with supplied world")
            require(not self.public.get("revelation_closed") or not self._hidden,
                    "closed full-day revelation with hidden requests")
        self.check_invariants()

    def _lots(self, order=None, state=None, holder=None):
        return [l for l in self.lots if (order is None or l["order"] == order)
                and (state is None or l["state"] == state)
                and (holder is None or l["holder"] == holder)]

    def _new_lot(self, order, weight, state, holder):
        self._lot_serial += 1
        name = f"generated:{self._lot_serial}"
        while any(l["lot"] == name for l in self.lots):
            self._lot_serial += 1
            name = f"generated:{self._lot_serial}"
        lot = dict(order=order, lot=name, weight=weight, state=state, holder=holder)
        self.lots.append(lot)
        return lot

    def _transfer(self, oid, q, state, holder, new_state, new_holder):
        q = quantity(q, positive=True)
        candidates = self._lots(oid, state, holder)
        require(sum(l["weight"] for l in candidates) >= q, "insufficient available goods")
        left = q
        for lot in candidates:
            if left == 0:
                break
            take = min(left, lot["weight"])
            if take == lot["weight"]:
                lot.update(state=new_state, holder=new_holder)
            else:
                lot["weight"] -= take
                self._new_lot(oid, take, new_state, new_holder)
            left -= take

    def _emit(self, kind, **data):
        self.trace.append(dict(at=self.now, kind=kind, **deepcopy(data)))

    def _schedule(self, at, kind, data, priority=0):
        require(at >= self.now, "event scheduled in the past")
        self._serial += 1
        heapq.heappush(self._queue, (at, priority, self._serial, kind, deepcopy(data)))

    def _idle(self, vehicle):
        require(vehicle["active_operation"] is None, "vehicle has a locked active operation")

    def _begin_station_visit(self, truck, *, arrival_observed=True):
        self._station_wait_memory[truck['id']] = dict(
            visit_id=f"{truck['id']}:{truck['location']}:{self.now}",
            station=truck['location'], arrived_at=self.now,
            arrival_observed=arrival_observed, active_wait_minutes=ZERO,
            review_count=0, active_wait_until=None, departed_at=None)

    def _end_active_wait(self, truck):
        memory = self._station_wait_memory.get(truck['id'])
        if memory is not None:
            memory['active_wait_until'] = None
        getattr(self, '_wait_reviews', {}).pop(truck['id'], None)

    def _known(self, oid, kind=None):
        require(oid in self.orders, "request is not revealed")
        o = self.orders[oid]
        require(kind is None or o["kind"] == kind, "wrong request type")
        require(o["accepted"] is True, "request not accepted")
        require(o["completed_at"] is None, "request already completed")
        return o

    def _order_list(self, action, key="orders", allow_empty=False):
        ids = action.get(key, [])
        require(isinstance(ids, list) and (allow_empty or bool(ids)), "empty or invalid batch")
        require(len(ids) == len(set(ids)), "duplicate order in operation")
        return ids

    def _weight(self, actor, state="onboard"):
        return sum(l["weight"] for l in self._lots(state=state, holder=actor))

    def _handling(self, kind, q):
        require(q > 0, "empty handling is not a positive operation")
        if kind == "truck":
            return dec(self.cfg["truck_handling_setup"]) + dec(self.cfg["truck_handling_per_kg"]) * q
        return dec(self.cfg["drone_handling_setup_per_request"]) + dec(self.cfg["drone_handling_per_kg"]) * q

    def _activate(self, u):
        if not u["activated"]:
            u["activated"] = True
            u["activation_count"] += 1
            if not per_sortie(self.cfg):
                self.ledger["drone_activation"] += dec(self.cfg["drone_first_use_cost"])
            self._emit("drone_activated", actor=u["id"])

    def _distance(self, a, b):
        return sum((dec(x) - dec(y)) ** 2 for x, y in zip(a, b)).sqrt()

    def apply(self, action):
        """Validate and commit atomically; failed actions never change this object."""
        if getattr(self,'_disposable_oracle',False):
            # Only evaluate_plan enables this on an isolated trial. Any failure
            # discards the entire trial, so per-command rollback is redundant.
            self._apply(deepcopy(action))
            self.check_invariants()
            return
        # Event records and snapshots are append-only. Share their existing
        # entries during this private transaction; clone every live state map.
        # This is not a change to public deepcopy or failed-action atomicity.
        memo={id(self.trace):list(self.trace),id(self._snapshots):dict(self._snapshots),
              id(self.initial):self.initial,id(self.cfg):self.cfg,id(self.network):self.network,
              id(self._arcs):self._arcs}
        candidate = deepcopy(self,memo)
        candidate._apply(deepcopy(action))
        candidate.check_invariants()
        self.__dict__.update(candidate.__dict__)

    def probe(self, action):
        before = deepcopy(self.__dict__)
        attempt = {k: v for k, v in action.items() if k not in ("kind", "operation")}
        attempt["kind"] = action["operation"]
        candidate = deepcopy(self)
        try:
            candidate.apply(attempt)
            accepted, reason = True, None
        except InvalidAction as exc:
            accepted, reason = False, str(exc)
        result = dict(accepted=accepted, reason=reason, state_unchanged=self.__dict__ == before)
        self.probes.append(result)
        return result

    def _apply(self, a):
        kind, actor = a["kind"], a.get("actor")
        require(dec(a.get("at", self.now)) == self.now, "action time is not current time")
        if kind == "append_active_batch":
            require(actor in self.trucks, "unknown truck")
            require(self.trucks[actor]["active_batch"] is None, "active batch contents are immutable")
            raise InvalidAction("start a new nonempty batch instead of appending")
        if kind == "truck_depart":
            kind = "truck_continue_circuit"
        if kind in ("accept", "reject"):
            ids = self._order_list(a)
            for oid in ids:
                require(oid in self.orders, "cannot decide an unrevealed request")
                o = self.orders[oid]
                require(o["kind"] == "P" and o["accepted"] is None, "admission already fixed")
            for oid in ids:
                o = self.orders[oid]
                o["accepted"] = kind == "accept"
                if kind == "reject":
                    self.ledger["rejection"] += dec(self.cfg["rejection_per_kg"]) * o["weight"]
                self._emit(kind, order=oid)
            return
        if kind == "truck_dispatch":
            require(self.now <= dec(self.cfg["truck_dispatch_cutoff"]), "dispatch cutoff exceeded")
            require(actor in self.trucks, "truck is not declared")
            t = self.trucks[actor]
            self._idle(t)
            require(t["location"] == "depot" and t["departed_at"] is None, "trip already dispatched")
            ids = self._order_list(a, "delivery_orders", allow_empty=True)
            allocation = a.get("quantities", {oid: self._known(oid, "D")["weight"] for oid in ids})
            require(set(allocation) == set(ids), "dispatch allocation/order mismatch")
            total = 0
            for oid, value in allocation.items():
                self._known(oid, "D")
                q = quantity(value)
                total += q
                if q:
                    self._transfer(oid, q, "depot", "depot", "onboard", actor)
            require(total <= self.cfg["truck_capacity"], "truck capacity exceeded")
            t["onboard_weight"] = total
            t["departed_at"] = self.now
            trip = sum(dec(arc["km"]) * (dec(self.cfg["truck_distance_cost"]) +
                       (dec(self.cfg["toll_per_marked_km"]) if arc["tolled"] else ZERO))
                       for arc in self.network["arcs"])
            self.ledger["truck_trip"] += trip
            self._emit("truck_dispatched", actor=actor)
            self._truck_leg(t)
            return
        if kind in ("truck_unload", "truck_load", "truck_continue_circuit", "skip_empty_unload"):
            require(actor in self.trucks, "unknown truck")
            t = self.trucks[actor]
            self._idle(t)
            require(t["location"] in self.network["stations"], "truck not at a station")
            station = t["location"]
            if "station" in a:
                require(a["station"] == station, "truck at a different station")
            if kind == "truck_continue_circuit":
                require(t["unloading_phase"] in ("completed", "skipped"), "unload phase unresolved")
                t["departure_time"] = self.now
                self._emit("truck_left_station", actor=actor, station=station)
                self._truck_leg(t)
                return
            if kind == "skip_empty_unload":
                require(t["unloading_phase"] == "not_started", "unload already closed")
                require(not a.get("quantities"), "positive unload cannot be skipped")
                t["unloading_phase"] = "skipped"
                self._emit("unload_skipped", actor=actor, station=station)
                return
            self._end_active_wait(t)
            if kind == "truck_unload":
                require(t["unloading_phase"] == "not_started" and t["completed_batches"] == 0,
                        "one unload before any loading only")
                allocation = a.get("quantities", {})
                require(bool(allocation), "empty unload must be skipped")
                total = 0
                for oid, value in allocation.items():
                    o = self._known(oid, "D")
                    q = quantity(value)
                    require(station in o["eligible_stations"], "ineligible unloading station")
                    require(o["station"] in (None, station), "delivery split across service stations")
                    if q:
                        o["station"] = station
                        self._transfer(oid, q, "onboard", actor, "unloading", actor)
                    total += q
                require(total > 0, "zero unloading must be skipped")
                t["unloading_phase"] = "active"
                end = self.now + self._handling("truck", total)
                t["active_operation"] = dict(kind=kind, end=end)
                self._schedule(end, "truck_unload_done", dict(actor=actor, station=station, quantities=allocation))
                self._emit("truck_unload_started", actor=actor, station=station, quantities=allocation, end=end)
            else:
                require("quantities" not in a, "pickup loading is whole-order only")
                require(t["unloading_phase"] in ("completed", "skipped"), "unload must precede loading")
                require(self.batch_permission != "B1" or t["completed_batches"] == 0, "single-batch permission exhausted at this stop")
                ids = self._order_list(a)
                total = sum(self._known(oid, "P")["weight"] for oid in ids)
                require(t["onboard_weight"] + total <= self.cfg["truck_capacity"], "truck capacity exceeded")
                for oid in ids:
                    o = self.orders[oid]
                    require(o["station"] == station, "pickup at another station")
                    self._transfer(oid, o["weight"], "available", station, "loading", actor)
                end = self.now + self._handling("truck", total)
                t["active_operation"] = dict(kind=kind, end=end)
                t["active_batch"] = dict(orders=ids, weight=total, start=self.now, end=end)
                t["loading_end"] = end
                self._schedule(end, "truck_load_done", dict(actor=actor, orders=ids))
                self._emit("truck_load_started", actor=actor, station=station, orders=ids, end=end)
            return
        if kind in ("zero_distance_leg", "customer_service"):
            require(actor in self.drones, "unknown drone")
            u = self.drones[actor]
            self._idle(u)
            require(u["takeoff_at"] is not None, "customer operation requires an active sortie")
            oid = a.get("to_order") if kind == "zero_distance_leg" else a["order"]
            o = self._known(oid)
            xy = self.orders[u["location"]]["xy"] if isinstance(u["location"], str) and u["location"] in self.orders else u["location"]
            require(isinstance(xy, (list, tuple)) and self._distance(xy, o["xy"]) == 0, "customer at a different location")
            if kind == "zero_distance_leg":
                u["location"] = oid
                self._emit("zero_distance_leg", actor=actor, order=oid)
            else:
                end = self.now + dec(self.cfg["customer_service_per_request"])
                require(self.now >= o["release"], "service before release")
                require(u["home"] in o["eligible_stations"], "ineligible service station")
                if o["kind"] == "D":
                    require(self._weight(actor) >= o["weight"] and end <= o["deadline"], "delivery unavailable or late")
                    require(sum(l["weight"] for l in self._lots(oid, "onboard", actor)) == o["weight"], "delivery is not aboard this drone")
                else:
                    require(sum(l["weight"] for l in self._lots(oid, "customer", oid)) == o["weight"], "pickup already collected")
                    require(o["assigned_drone"] in (None, actor), "pickup assigned to another drone")
                    require(u["onboard_weight"] + o["weight"] <= self.cfg["drone_capacity"], "service would exceed capacity")
                earliest_return = end + self._distance(o["xy"], self.network["stations"][u["home"]]) / dec(self.cfg["drone_speed_km_min"])
                require(earliest_return - u["takeoff_at"] <= dec(self.cfg["drone_endurance"]), "service leaves no feasible return")
                u["active_operation"] = dict(kind="service", end=end)
                self._emit("service_started", actor=actor, order=oid, end=end)
                self._schedule(end, "manual_service_done", dict(actor=actor, order=oid))
            return
        if kind in ("drone_load", "drone_unload", "drone_sortie"):
            require(actor in self.drones, "unknown drone")
            u = self.drones[actor]
            self._idle(u)
            require(u["location"] == u["home"], "drone not at home")
            if kind == "drone_sortie":
                self._sortie(u, self._order_list(a, "visit_order"))
            else:
                require("quantities" not in a, "drone handles whole requests only")
                ids = self._order_list(a)
                loading = kind == "drone_load"
                require(not loading or not any(self.orders[l["order"]]["kind"] == "P"
                            for l in self._lots(state="onboard", holder=actor)), "unload pickups before reuse")
                total = sum(self._known(oid, "D" if loading else "P")["weight"] for oid in ids)
                require(not loading or u["onboard_weight"] + total <= self.cfg["drone_capacity"],
                        "drone capacity exceeded")
                for oid in ids:
                    o = self.orders[oid]
                    require(u["home"] in o["eligible_stations"], "ineligible drone station")
                    source_state, source_holder = ("available", u["home"]) if loading else ("onboard", actor)
                    require(sum(l["weight"] for l in self._lots(oid, source_state, source_holder)) == o["weight"],
                            "whole order not available for drone handling")
                if loading and not per_sortie(self.cfg):
                    self._activate(u)
                self._drone_handling_next(u, ids, loading)
            return
        raise UnsupportedCase("action not implemented: " + kind)

    def _truck_leg(self, t):
        arc = self._arcs[t["location"]]
        memory = self._station_wait_memory.get(t['id'])
        if memory is not None and memory['station'] == t['location'] and memory['departed_at'] is None:
            self._end_active_wait(t)
            memory['departed_at'] = self.now
            self._station_wait_history.append(dict(truck=t['id'], **deepcopy(memory)))
        end = self.now + dec(arc["min"])
        t["active_operation"] = dict(kind="road", end=end)
        t["location"] = "road"
        self._emit("road_started", actor=t["id"], source=arc["from"], destination=arc["to"], end=end)
        self._schedule(end, "truck_arrived", dict(actor=t["id"], destination=arc["to"]))

    def _drone_handling_next(self, u, ids, loading):
        oid = ids[0]
        o = self._known(oid, "D" if loading else "P")
        if loading:
            self._transfer(oid, o["weight"], "available", u["home"], "loading", u["id"])
            o["station"] = u["home"]
        # During unloading the goods still occupy the drone's capacity.
        else:
            self._transfer(oid, o["weight"], "onboard", u["id"], "unloading", u["id"])
        end = self.now + self._handling("drone", o["weight"])
        u["active_operation"] = dict(kind="drone_load" if loading else "drone_unload", end=end)
        self._emit("drone_handling_started", actor=u["id"], order=oid, loading=loading, end=end)
        self._schedule(end, "drone_handling_done", dict(actor=u["id"], orders=ids, loading=loading))

    def _sortie(self, u, ids):
        # Validate the whole supplied sortie before starting any leg.
        t, xy = self.now, self.network["stations"][u["home"]]
        weight = u["onboard_weight"]
        onboard_d = {l["order"] for l in self._lots(state="onboard", holder=u["id"])
                     if self.orders[l["order"]]["kind"] == "D"}
        require(onboard_d == {oid for oid in ids if self._known(oid)["kind"] == "D"},
                "all loaded deliveries must be served in the same sortie")
        require(not any(self.orders[l["order"]]["kind"] == "P" for l in self._lots(state="onboard", holder=u["id"])),
                "previous pickups must be unloaded before takeoff")
        for oid in ids:
            o = self._known(oid)
            require(u["home"] in o["eligible_stations"], "ineligible station")
            t += self._distance(xy, o["xy"]) / dec(self.cfg["drone_speed_km_min"])
            require(t >= o["release"], "service precedes release")
            t += dec(self.cfg["customer_service_per_request"])
            if o["kind"] == "D":
                require(t <= o["deadline"], "delivery service completion exceeds deadline")
                weight -= o["weight"]
            else:
                require(o["collected_at"] is None and o["assigned_drone"] is None and
                        sum(l["weight"] for l in self._lots(oid, "customer", oid)) == o["weight"],
                        "pickup already committed or collected")
                weight += o["weight"]
            require(0 <= weight <= self.cfg["drone_capacity"], "flight segment capacity exceeded")
            xy = o["xy"]
        t += self._distance(xy, self.network["stations"][u["home"]]) / dec(self.cfg["drone_speed_km_min"])
        require(t - self.now <= dec(self.cfg["drone_endurance"]), "sortie endurance exceeded")
        for oid in ids:
            o = self.orders[oid]
            if o["kind"] == "P":
                # Takeoff commits the entire visit sequence and return station.
                o["assigned_drone"] = u["id"]
                o["station"] = u["home"]
        self._activate(u)
        u["takeoff_at"] = self.now
        u["sortie_commitment"] = dict(visit_order=list(ids), return_station=u["home"])
        if per_sortie(self.cfg):
            self.ledger["drone_activation"] += dec(self.cfg["drone_sortie_cost"])
        self._emit("drone_takeoff", actor=u["id"], orders=ids)
        self._flight(u, self.network["stations"][u["home"]], ids)

    def _flight(self, u, xy, remaining):
        target = self.orders[remaining[0]]["xy"] if remaining else self.network["stations"][u["home"]]
        distance = self._distance(xy, target)
        end = self.now + distance / dec(self.cfg["drone_speed_km_min"])
        u["active_operation"] = dict(kind="flight", start=self.now, end=end, distance=distance)
        u["location"] = "flight"
        self._emit("flight_started", actor=u["id"], source=xy, destination=target, end=end)
        self._schedule(end, "drone_arrived", dict(actor=u["id"], remaining=remaining))

    def _settle(self, kind, data):
        if kind == "pickup_unload_complete":
            for oid in data["orders"]:
                o = self._known(oid, "P")
                require(data["station"] in o["eligible_stations"], "external completion at ineligible station")
                self._transfer(oid, o["weight"], "external_pipeline", "external", "available", data["station"])
                o.update(station=data["station"], available_at=self.now)
            self._emit(kind, **{k: v for k, v in data.items() if k not in ("at", "kind")})
            return
        if kind == "truck_return":
            require(self.trucks[data["truck"]]["active_operation"] is not None, "duplicate external return")
            return self._settle("truck_arrived", dict(actor=data["truck"], destination="depot"))
        if kind in ("customer_service_complete", "manual_service_done"):
            actor = data.get("drone", data.get("actor"))
            u = self.drones[actor]
            o = self._known(data["order"])
            require(u["active_operation"] is not None and u["active_operation"]["end"] == self.now, "uncommitted service completion")
            if o["kind"] == "D":
                require(self.now <= o["deadline"], "delivery completion late")
                self._transfer(o["id"], o["weight"], "onboard", actor, "delivered", o["id"])
                self._complete(o)
            else:
                self._transfer(o["id"], o["weight"], "customer", o["id"], "onboard", actor)
                o.update(collected_at=self.now, station=u["home"], assigned_drone=actor)
            u.update(onboard_weight=self._weight(actor), active_operation=None, location=o["id"])
            self._emit("service_done", actor=actor, order=o["id"])
            return
        if kind == "truck_arrived":
            t = self.trucks[data["actor"]]
            t.update(location=data["destination"], active_operation=None)
            self._emit("truck_arrived", actor=t["id"], destination=t["location"])
            if t["location"] == "depot":
                require(not any(self.orders[l["order"]]["kind"] == "D" for l in self._lots(holder=t["id"])),
                        "truck returned with undelivered replenishment")
                t["return_time"] = self.now
                for lot in list(self._lots(state="onboard", holder=t["id"])):
                    o = self.orders[lot["order"]]
                    require(o["kind"] == "P" and lot["weight"] == o["weight"], "pickup split at return")
                    lot.update(state="completed", holder="depot")
                    self._complete(o)
                t["onboard_weight"] = 0
            else:
                self._begin_station_visit(t)
                positive_unload = any(a["kind"] == "truck_unload" and a.get("actor") == t["id"]
                                      and a.get("station") == t["location"]
                                      for at, acts in self._actions.items() if at >= self.now for a in acts)
                t.update(unloading_phase="not_started" if positive_unload else "skipped",
                         completed_batches=0, active_batch=None)
                if t["location"] not in self._stops.get(t["id"], set()):
                    if getattr(self, '_online_station_decisions', False):
                        # Finish all same-time completions and disclosures before
                        # committing another road leg. The default remains leave.
                        due = self._actions.setdefault(self.now, [])
                        if not any(a.get('actor') == t['id'] and a['kind'] == 'truck_continue_circuit'
                                   for a in due):
                            due.append(dict(at=self.now, kind='truck_continue_circuit',
                                            actor=t['id'], station=t['location']))
                    else:
                        self._truck_leg(t)
        elif kind == "truck_unload_done":
            t = self.trucks[data["actor"]]
            for oid, q in data["quantities"].items():
                if q:
                    self._transfer(oid, q, "unloading", t["id"], "available", data["station"])
            t.update(active_operation=None, unloading_phase="completed", onboard_weight=self._weight(t["id"]))
            self._emit(kind, **data)
        elif kind == "truck_load_done":
            t = self.trucks[data["actor"]]
            for oid in data["orders"]:
                self._transfer(oid, self.orders[oid]["weight"], "loading", t["id"], "onboard", t["id"])
            t["loading_batches"] += 1
            t["completed_batches"] += 1
            t.update(active_operation=None, active_batch=None, onboard_weight=self._weight(t["id"]))
            self._emit(kind, **data)
        elif kind == "drone_handling_done":
            u = self.drones[data["actor"]]
            oid, loading = data["orders"][0], data["loading"]
            o = self.orders[oid]
            self._transfer(oid, o["weight"], "loading" if loading else "unloading", u["id"],
                           "onboard" if loading else "available", u["id"] if loading else u["home"])
            u["onboard_weight"] = self._weight(u["id"])
            u["active_operation"] = None
            if not loading:
                o["available_at"] = self.now
                o["station"] = u["home"]
            self._emit(kind, actor=u["id"], order=oid, loading=loading)
            if len(data["orders"]) > 1:
                self._drone_handling_next(u, data["orders"][1:], loading)
        elif kind == "drone_arrived":
            u = self.drones[data["actor"]]
            if not data["remaining"]:
                u.update(active_operation=None, location=u["home"], return_time=self.now, takeoff_at=None,
                         sortie_commitment=None)
                u["return_times"].append(self.now)
                self._emit("drone_returned", actor=u["id"])
            else:
                oid = data["remaining"][0]
                end = self.now + dec(self.cfg["customer_service_per_request"])
                u.update(location=oid, active_operation=dict(kind="service", end=end))
                self._emit("service_started", actor=u["id"], order=oid, end=end)
                self._schedule(end, "service_done", data)
        elif kind == "service_done":
            u = self.drones[data["actor"]]
            oid = data["remaining"][0]
            o = self.orders[oid]
            if o["kind"] == "D":
                self._transfer(oid, o["weight"], "onboard", u["id"], "delivered", oid)
                self._complete(o)
            else:
                self._transfer(oid, o["weight"], "customer", oid, "onboard", u["id"])
                o["collected_at"] = self.now
            u["onboard_weight"] = self._weight(u["id"])
            self._emit("service_done", actor=u["id"], order=oid)
            self._flight(u, o["xy"], data["remaining"][1:])
        else:
            raise UnsupportedCase("completion: " + kind)

    def _complete(self, o):
        require(o["completed_at"] is None, "duplicate revenue settlement")
        o["completed_at"] = self.now
        self.ledger["revenue"] += dec(self.cfg["revenue_per_kg"]) * o["weight"]
        self._emit("order_completed", order=o["id"])

    def _advance(self, target):
        require(target >= self.now, "clock moved backwards")
        dt = target - self.now
        for tid, memory in self._station_wait_memory.items():
            truck = self.trucks[tid]
            until = memory['active_wait_until']
            if until is not None and truck['location'] == memory['station'] and truck['active_operation'] is None:
                memory['active_wait_minutes'] += max(ZERO, min(target, until) - self.now)
        outside = sum(t["departed_at"] is not None and t["return_time"] is None for t in self.trucks.values())
        self.ledger["truck_time"] += dt * outside * dec(self.cfg["truck_time_cost"])
        for o in self.orders.values():
            if o["kind"] == "P" and o["accepted"] is True and o["completed_at"] is None:
                late = max(ZERO, target - max(self.now, o["deadline"]))
                self.ledger["lateness"] += late * dec(self.cfg["pickup_lateness_per_request_min"])
        for u in self.drones.values():
            op = u["active_operation"]
            if op and op["kind"] == "flight" and op["end"] > op["start"]:
                flown = op["distance"] * dt / (op["end"] - op["start"])
                self.ledger["flight"] += flown * dec(self.cfg["drone_distance_cost"])
        self.now = target

    def _close_completions(self):
        n = 0
        while self._queue and self._queue[0][0] == self.now and self._queue[0][1] == 0:
            _, _, serial, kind, data = heapq.heappop(self._queue)
            require(serial not in self._completed_event_ids, "duplicate completion event")
            self._settle(kind, data)
            self._completed_event_ids.append(serial)
            self.check_invariants()
            n += 1
            require(n <= 100000, "nonfinite same-time completion closure")

    def _capture(self, phase):
        if getattr(self, "_record_snapshots", True):
            self._snapshots[(self.now, phase)] = self.observations()

    def run(self, checkpoints=(), decision_hook=None, stop_at_terminal=False):
        if getattr(self, "_has_run", False):
            raise InvalidAction("replay can only be run once")
        self._has_run = True
        # A planning replay deliberately has no hook; it must not inherit the
        # online deferral flag from the live state it was cloned from.
        self._online_station_decisions = decision_hook is not None
        ticks = {self.start, self.stop} | {dec(t) for t in checkpoints}
        require(all(self.start <= t <= self.stop for t in ticks), "checkpoint outside replay window")
        processed = None
        while True:
            if hasattr(self,'_compute_deadline'):
                from time import perf_counter
                if perf_counter()>self._compute_deadline:
                    raise TimeoutError('planning simulation compute deadline')
            options = [t for t in ticks if t <= self.stop]
            options += [t for t in self._actions if self.start <= t <= self.stop and (processed is None or t > processed)]
            if self._queue and self._queue[0][0] <= self.stop:
                options.append(self._queue[0][0])
            if not options:
                break
            target = min(options)
            ticks.discard(target)
            self._advance(target)
            self._close_completions()
            self._capture("after_completions")
            while self._queue and self._queue[0][0] == self.now and self._queue[0][1] == 1:
                _, _, _, kind, data = heapq.heappop(self._queue)
                if kind == "revelation_end":
                    require(self.public.get("revelation_end") is not None and dec(self.public["revelation_end"]) == self.now,
                            "undeclared revelation boundary")
                    self.public["revelation_closed"] = True
                    self._emit("revelation_closed")
                if kind == "request_reveal":
                    self._emit("reveal_batch", orders=data["orders"])
                    for oid in data["orders"]:
                        require(oid in self._hidden, "request revealed twice or unknown")
                        o = self._hidden.pop(oid)
                        require(o["release"] == self.now, "reveal timestamp mismatch")
                        self.orders[oid] = o
                        self._new_lot(oid, o["weight"], "customer", oid)
                        count = self.public.get("future_count")
                        if count is not None:
                            require(count > 0, "public future count inconsistent")
                            self.public["future_count"] = count - 1
                        self._emit("request_revealed", order=oid)
            self._capture("before_starts")
            if decision_hook is not None:
                # The event adapter decides whether a policy call is needed.
                # Updating physical state is not itself a learning decision.
                decision_hook(self)
            actions = list(enumerate(self._actions.get(self.now, [])))
            # One batch invocation of the scripted decision driver. This is
            # measured execution, not a claim of an implemented online policy.
            if actions and decision_hook is None:
                self._driver_calls[self.now] = self._driver_calls.get(self.now, 0) + 1
            def rank(pair):
                i, a = pair
                if a["kind"] in ("accept", "reject"):
                    return (0, "", i)
                if a.get("actor") in self.trucks:
                    return (1, a["actor"], i)
                return (2, "", i)
            for _, action in sorted(actions, key=rank):
                if action["kind"] == "probe":
                    self.probe(action)
                else:
                    self.apply(action)
                self._close_completions()
            self.check_invariants()
            if self.absorption_time is None and self.is_terminal():
                self.absorption_time = self.now
            self._capture("after_closure")
            processed = self.now
            if stop_at_terminal and self.is_terminal():
                self.stop = self.now
                break
        if self.full_day:
            require(self.is_terminal(), "full-day obligations/resources not finished at stop")
        return self

    def check_invariants(self, *, force=False):
        # Disposable completion trials retain every operation's local physical
        # checks. Global conservation is checked once at the trial boundary;
        # live execution still checks it after every event and command.
        if getattr(self,'_defer_global_invariants',False) and not force:
            return
        by_order={oid:[] for oid in self.orders}
        carried={};reserved_weight={}
        for lot in self.lots:
            require(lot['order'] in by_order,'unknown inventory order')
            by_order[lot['order']].append(lot)
            if lot['state'] in ('onboard','unloading'):
                carried[lot['holder']]=carried.get(lot['holder'],0)+lot['weight']
            elif lot['state']=='loading':
                reserved_weight[lot['holder']]=reserved_weight.get(lot['holder'],0)+lot['weight']
        for o in self.orders.values():
            lots = by_order[o["id"]]
            require(sum(l["weight"] for l in lots) == o["weight"], "quantity conservation failed")
            if o["kind"] == "P":
                require(len(lots) == 1 and lots[0]["weight"] == o["weight"], "pickup split is forbidden")
        for l in self.lots:
            quantity(l["weight"], positive=True)
            require(l["order"] in self.orders, "unknown inventory order")
            o = self.orders[l["order"]]
            if l["state"] in ("onboard", "loading", "unloading"):
                require(l["holder"] in self.trucks or l["holder"] in self.drones, "unknown inventory vehicle")
            elif l["state"] == "available":
                require(l["holder"] in o["eligible_stations"], "inventory at ineligible station")
            elif l["state"] == "depot":
                require(o["kind"] == "D" and l["holder"] == "depot", "invalid depot inventory")
            elif l["state"] in ("customer", "delivered"):
                require(l["holder"] == o["id"], "invalid customer inventory")
        for vehicles, capacity in ((self.trucks, self.cfg["truck_capacity"]), (self.drones, self.cfg["drone_capacity"])):
            for v in vehicles.values():
                onboard = carried.get(v['id'],0)
                reserved = reserved_weight.get(v['id'],0)
                require(v["onboard_weight"] == onboard, "vehicle/inventory weight mismatch")
                require(onboard + reserved <= capacity, "reserved plus onboard exceeds capacity")

    def is_terminal(self):
        from .information import closed
        public_done = closed(self.public)
        return public_done and all(o["accepted"] is not None and
                    (o["accepted"] is False or o["completed_at"] is not None) for o in self.orders.values()) and all(
                    t["departed_at"] is None or t["return_time"] is not None for t in self.trucks.values()) and all(
                    u["active_operation"] is None and u["location"] == u["home"] and u["onboard_weight"] == 0
                    for u in self.drones.values())

    def public_snapshot(self):
        """No hidden stream, next release, script, event queue or expected values."""
        return plain(deepcopy(dict(time=self.now, orders=self.orders, trucks=self.trucks,
                                  drones=self.drones, inventory=self.lots, public_information=self.public,
                                  ledger=self.ledger)))

    def observations(self):
        result = dict(truck=deepcopy(self.trucks), drone=deepcopy(self.drones),
                      order=deepcopy(self.orders), probes=deepcopy(self.probes),
                      ledger=deepcopy(self.ledger), station={}, inventory={})
        result["ledger"]["net"] = self.ledger["revenue"] - sum(self.ledger[k] for k in COSTS)
        for station in self.network["stations"]:
            result["station"][station] = {
                "delivery_available_weight": sum(l["weight"] for l in self._lots(state="available", holder=station)
                                                  if self.orders[l["order"]]["kind"] == "D")}
        for oid, o in result["order"].items():
            o["lateness"] = max(ZERO, (o["completed_at"] or self.now) - o["deadline"]) if o["kind"] == "P" and o["accepted"] else ZERO
            lots = self._lots(oid)
            if len(lots) == 1:
                result["inventory"][oid] = deepcopy(lots[0])
        result["terminal"] = self.is_terminal()
        result["termination"] = dict(absorbed=self.is_terminal(), absorption_time=self.absorption_time)
        result["events"] = dict(max_policy_calls_per_timestamp=max(self._driver_calls.values(), default=0),
                                duplicate_operation_ids=[x for x in set(self._completed_event_ids) if self._completed_event_ids.count(x) > 1],
                                rejection_postings=sum(e["kind"] == "reject" for e in self.trace))
        for o in result["order"].values():
            o["rejected"] = o["accepted"] is False
        return plain(result)

    def observe(self, target, at=None, phase="after_closure"):
        current = self.observations() if at is None else self._snapshots[(dec(at), phase)]
        for key in target.split("."):
            if isinstance(current, list) and key in ("all_rejected", "all_state_unchanged"):
                current = all(not p["accepted"] for p in current) if key == "all_rejected" else all(p["state_unchanged"] for p in current)
            else:
                current = current[int(key)] if isinstance(current, list) else current[key]
        return current
