"""Public-state Myopic planning, finite completion oracle and one-shot tail.

All candidates are executable remaining plans. Search is deliberately bounded:
append to unstarted sorties or add a sortie, append a recovery batch or new trip.
Active legs/services/batches remain unchanged; no learned value is used.
"""
from copy import deepcopy
from decimal import Decimal
import heapq
from itertools import combinations

from .constraints import distance
from .initializer import digest
from .kernel import COSTS, require
from .quantities import InvalidAction, dec, plain


def _planning_copy(state):
    """Private engine copy: immutable past records/static inputs are read-only.

    All live orders, cargo, operations, queues, plans and accounts are copied.
    Neither engine nor policy edits existing trace records or configuration.
    """
    memo={id(state.trace):list(state.trace),id(state._snapshots):dict(state._snapshots),
          id(state.cfg):state.cfg,id(state.network):state.network,
          id(state._arcs):state._arcs,id(state.initial):state.initial}
    return deepcopy(state,memo)


def remaining_plan(state, identity="original"):
    return dict(id=identity, actions=[deepcopy(a) for t, rows in sorted(state._actions.items()) if t >= state.now for a in rows],
                new_trucks=[], decisions={}, wait_reviews={k:v for k,v in getattr(state,'_wait_reviews',{}).items() if dec(v)>=state.now})


def known_clone(environment):
    """The only environment -> planner boundary. No future reveal data survives."""
    state = _planning_copy(environment)
    from .information import validate_public
    validate_public(state.public)
    state._hidden = {}
    state._queue = [e for e in state._queue if e[3] not in ("request_reveal", "revelation_end")]
    from .information import unknown
    if unknown(state.public):
        # Environment event IDs depend on how many hidden reveals were queued.
        # Preserve relative physical-event order but remove that side channel.
        state._queue=[(at,priority,i,kind,data) for i,(at,priority,_,kind,data) in enumerate(sorted(state._queue),1)]
        state._serial=len(state._queue)
        state._completed_event_ids=[]
    heapq.heapify(state._queue)
    state.start = state.now
    state.stop = state.now  # Do not expose a hidden-stream-dependent simulation end.
    state.initial = dict(start_time=state.now, stop_time=state.now,
                         orders=deepcopy(list(state.orders.values())), trucks=deepcopy(list(state.trucks.values())),
                         drones=deepcopy(list(state.drones.values())), inventory=deepcopy(state.lots),
                         external_events=[], actions=[], policy=None, candidates=[], public_information=deepcopy(state.public))
    state._actions = {t: deepcopy(rows) for t, rows in state._actions.items() if t >= state.now}
    state._snapshots = {}
    state._record_snapshots = False
    state._has_run = False
    state.full_day = False
    return state


def commit_decisions(state, decisions):
    """One atomic partition of all currently undecided revealed requests."""
    pending = {oid for oid, o in state.orders.items() if o["kind"] == "P" and o["accepted"] is None}
    require(set(decisions) == pending, "admission must partition the whole revealed batch")
    require(all(type(v) is bool for v in decisions.values()), "admission decisions must be boolean")
    candidate = _planning_copy(state)
    for accepted in (True, False):
        ids = sorted(oid for oid, value in decisions.items() if value is accepted)
        if ids:
            candidate.apply(dict(kind="accept" if accepted else "reject", orders=ids, at=candidate.now))
    state.__dict__.update(candidate.__dict__)


def install_plan(state, plan, *, admit=True):
    """Replace only unstarted commands. Active operation queues are untouched."""
    if admit:
        commit_decisions(state, plan.get("decisions", {}))
    for t in plan.get("new_trucks", []):
        require(t["id"] not in state.trucks and t["id"] not in state.drones, "new truck ID collision")
        state.trucks[t["id"]] = deepcopy(t)
    actions = deepcopy(plan["actions"])
    require(all(a["kind"] not in ("accept", "reject") and dec(a["at"]) >= state.now for a in actions),
            "plan contains past actions or nontransactional admissions")
    state._actions = {}
    state._stops = {}
    # Stops are derived from the known plan, never from unrevealed requests.
    station_by_order = {oid: o["station"] for oid, o in state.orders.items() if o["station"]}
    for _, _, _, kind, data in state._queue:
        if kind == "pickup_unload_complete":
            station_by_order.update({oid: data["station"] for oid in data["orders"]})
    for a in actions:
        if a["kind"] == "drone_sortie":
            for oid in a["visit_order"]:
                station_by_order[oid] = state.drones[a["actor"]]["home"]
    for a in actions:
        if a["kind"] == "truck_load" and "station" not in a:
            stations = {station_by_order.get(oid) for oid in a["orders"]}
            require(len(stations) == 1 and None not in stations, "ambiguous pickup recovery station")
            a["station"] = next(iter(stations))
        if a["kind"] in ("truck_load", "truck_unload"):
            state._stops.setdefault(a["actor"], set()).add(a["station"])
        state._actions.setdefault(dec(a["at"]), []).append(a)
    for at in plan.get('wakeups',[]):
        require(dec(at)>=state.now,'past scheduled wakeup')
        state._actions.setdefault(dec(at),[])
    previous_reviews = getattr(state, '_wait_reviews', {})
    reviews = {k:dec(v) for k,v in plan.get('wait_reviews',{}).items() if dec(v)>state.now}
    for tid, memory in state._station_wait_memory.items():
        until = reviews.get(tid)
        if until is not None and until != previous_reviews.get(tid):
            truck = state.trucks[tid]
            require(truck['location'] == memory['station'] and truck['active_operation'] is None,
                    'new active wait requires an idle station truck')
            memory['review_count'] += 1
        memory['active_wait_until'] = until
    state._wait_reviews = reviews
    for at in state._wait_reviews.values():
        state._actions.setdefault(at,[])
    state.check_invariants()


def _bound(state):
    latest = max([state.now] + list(state._actions) + [e[0] for e in state._queue])
    # A verification horizon, not a physical return deadline. All scheduled
    # starts are included, plus sufficient road/flight/handling completion time.
    road = sum(dec(a["min"]) for a in state.network["arcs"])
    handling = sum(state._handling("drone", o["weight"]) for o in state.orders.values())
    return latest + road + dec(state.cfg["drone_endurance"]) + handling + 1


def _w0_allowed(initial, completed, future_trace):
    """No new idle truck dwell; infer from physical intervals, not labels."""
    available = {tid: (t["location"], initial.now) for tid, t in initial.trucks.items()
                 if t["location"] in initial.network["stations"] and t["active_operation"] is None}
    for e in future_trace:
        kind, actor, at = e["kind"], e.get("actor"), dec(e["at"])
        if kind == "truck_arrived" and e["destination"] != "depot":
            available[actor] = (e["destination"], at)
        elif kind in ("truck_unload_done", "truck_load_done"):
            station = e.get("station", available.get(actor, (None,))[0])
            available[actor] = (station, at)
        elif kind in ("truck_load_started", "truck_unload_started", "road_started"):
            if actor in available:
                if at > available[actor][1]:
                    return False
                if kind == "road_started":
                    available.pop(actor)
    return True


def evaluate_plan(known, plan, waiting="WK", *, require_complete=True):
    """J is the exact historical + finite known-task completion net, W=0."""
    trial = _planning_copy(known)
    trial._disposable_oracle = True
    trial._defer_global_invariants = True
    trial._record_snapshots = False
    trial._snapshots = {}
    trial._has_run = False
    trial.start = trial.now
    trial.full_day = False
    offset = len(trial.trace)
    try:
        install_plan(trial, plan)
        trial.public = dict(future_count=0, revelation_closed=True)
        trial.stop = _bound(trial)
        trial.run(stop_at_terminal=True)
        trial.check_invariants(force=True)
        if require_complete:
            require(trial.is_terminal(), "candidate has no complete finite recovery of all known obligations")
        trace = trial.trace[offset:]
        allowed = waiting == "WK" or (waiting == "W0" and _w0_allowed(known, trial, trace))
        require(waiting in ("WK", "W0"), "unsupported Myopic waiting permission")
        score = trial.ledger["revenue"] - sum(trial.ledger[k] for k in COSTS)
        return dict(id=plan["id"], feasible=True, allowed=allowed, score=score,
                    completion=trial.absorption_time, trace=trace, state=trial, plan=deepcopy(plan))
    except (InvalidAction, KeyError) as exc:
        return dict(id=plan["id"], feasible=False, allowed=False, score=None, reason=str(exc), plan=deepcopy(plan))


def select_best(evaluated):
    allowed = [r for r in evaluated if r["feasible"] and r["allowed"]]
    require(bool(allowed), "no verified permissible incumbent")
    return min(allowed, key=lambda r: (-r["score"], r["id"]))


class TailManager:
    def __init__(self):
        self.activated = False
        self.replans_after_activation = 0
        self.invalid_repeated_reviews = 0
        self.selected_plan = "original"
        self.cancelled_accepted_orders = []

    def select(self, known, incumbent, candidates=(), *, waiting="WK", search_status="candidate"):
        if self.activated:
            return incumbent
        from .information import closed
        public_closed = closed(known.public)
        if not public_closed:
            return incumbent
        self.activated = True
        self.replans_after_activation += 1
        base = evaluate_plan(known, incumbent, waiting)
        require(base["feasible"] and base["allowed"], "adopted tail plan is not a verified permissible fallback")
        pool = [base]
        if search_status == "candidate":
            pool += [evaluate_plan(known, p, waiting) for p in candidates]
        chosen = select_best(pool)
        self.selected_plan = chosen["id"]
        return chosen["plan"]

    def observations(self):
        return dict(activated=self.activated, replans_after_activation=self.replans_after_activation,
                    invalid_repeated_reviews=self.invalid_repeated_reviews, selected_plan=self.selected_plan,
                    cancelled_accepted_orders=self.cancelled_accepted_orders)


def _new_truck(identity):
    return dict(id=identity, location="depot", onboard_weight=0, departed_at=None, unloading_phase="not_started",
                completed_batches=0, active_operation=None, return_time=None, departure_time=None,
                loading_end=None, loading_batches=0, active_batch=None)


def _sortie_times(state, station, start, visits):
    t, xy = dec(start), state.network["stations"][station]
    for oid in visits:
        o = state.orders[oid]
        t += distance(xy, o["xy"]) / dec(state.cfg["drone_speed_km_min"])
        t += dec(state.cfg["customer_service_per_request"])
        xy = o["xy"]
    t += distance(xy, state.network["stations"][station]) / dec(state.cfg["drone_speed_km_min"])
    ready = t
    pickup_ids = [oid for oid in visits if state.orders[oid]["kind"] == "P"]
    for oid in pickup_ids:
        ready += state._handling("drone", state.orders[oid]["weight"])
    return t, ready, pickup_ids


def _road_prefix(state, station):
    node, elapsed = "depot", Decimal(0)
    while node != station:
        arc = state._arcs[node]
        elapsed += dec(arc["min"])
        node = arc["to"]
        require(node != "depot", "station absent from circuit")
    return elapsed


def _recovery_variants(state, plan, projection, oid, station, ready):
    # Baseline road-start times identify the end of each station's committed
    # work; appending a batch preserves earlier loads/unloads and active queues.
    seen = set()
    for e in projection["trace"]:
        if e["kind"] != "road_started" or e["source"] != station:
            continue
        actor, depart = e["actor"], dec(e["at"])
        if (actor, depart) in seen:
            continue
        seen.add((actor, depart))
        if actor in state.trucks and state.trucks[actor]["active_operation"] is not None and state.trucks[actor]["location"] == "road":
            # Future station visits are still valid, but no already-started arc
            # is retimed: only future actions in the suffix are shifted.
            pass
        start = max(ready, depart)
        end = start + state._handling("truck", state.orders[oid]["weight"])
        delay = end - depart
        variant = deepcopy(plan)
        retained = []
        for a in variant["actions"]:
            if a.get("actor") == actor and a["kind"] in ("truck_continue_circuit", "truck_depart") and dec(a["at"]) == depart:
                continue
            if a.get("actor") == actor and dec(a["at"]) > depart:
                a["at"] = dec(a["at"]) + delay
            retained.append(a)
        retained += [dict(at=start, kind="truck_load", actor=actor, station=station, orders=[oid]),
                     dict(at=end, kind="truck_continue_circuit", actor=actor, station=station)]
        variant["actions"] = retained
        yield variant
    if state.now <= dec(state.cfg["truck_dispatch_cutoff"]):
        variant = deepcopy(plan)
        occupied = set(state.trucks) | set(state.drones) | {t["id"] for t in variant["new_trucks"]}
        index = 1
        while f"OT{index:05d}" in occupied:
            index += 1
        actor = f"OT{index:05d}"
        depart = max(state.now, min(dec(state.cfg["truck_dispatch_cutoff"]), ready - _road_prefix(state, station)))
        arrival = depart + _road_prefix(state, station)
        start = max(arrival, ready)
        end = start + state._handling("truck", state.orders[oid]["weight"])
        variant["new_trucks"].append(_new_truck(actor))
        variant["actions"] += [dict(at=depart, kind="truck_dispatch", actor=actor, delivery_orders=[]),
                               dict(at=start, kind="truck_load", actor=actor, station=station, orders=[oid]),
                               dict(at=end, kind="truck_continue_circuit", actor=actor, station=station)]
        yield variant


def pickup_variants(known, plan, oid):
    # Projection is for constructing physical timings only. Incomplete branches
    # are never scored as J or committed to the operating state.
    baseline = evaluate_plan(known, plan, "WK", require_complete=False)
    if not baseline["feasible"]:
        return
    o = known.orders[oid]
    for uid, u in sorted(known.drones.items()):
        station = u["home"]
        if station not in o["eligible_stations"]:
            continue
        choices = []
        for i, a in enumerate(plan["actions"]):
            if a.get("actor") == uid and a["kind"] == "drone_sortie" and oid not in a["visit_order"]:
                # Only future takeoff commands live in this list. In-flight
                # sequences are internal completion events and cannot be edited.
                # Enumerate insertion positions and adjacent old-visit swaps,
                # preserving every accepted obligation and loaded delivery.
                old = a["visit_order"]
                routes = [old] + [old[:j] + [old[j+1], old[j]] + old[j+2:] for j in range(len(old)-1)]
                seen = set()
                for route in routes:
                    for pos in range(len(route), -1, -1):
                        visits = route[:pos] + [oid] + route[pos:]
                        if tuple(visits) not in seen:
                            seen.add(tuple(visits))
                            choices.append((i, dec(a["at"]), visits))
        final_u = baseline["state"].drones[uid]
        if final_u["active_operation"] is None and final_u["onboard_weight"] == 0 and final_u["location"] == station:
            free = max([known.now] + [dec(e.get("end", e["at"])) for e in baseline["trace"] if e.get("actor") == uid])
            choices.append((None, free, [oid]))
        for replace, takeoff, visits in choices:
            variant = deepcopy(plan)
            returned, ready, pickups = _sortie_times(known, station, takeoff, visits)
            if replace is None:
                variant["actions"].append(dict(at=takeoff, kind="drone_sortie", actor=uid, visit_order=visits))
                variant["actions"].append(dict(at=returned, kind="drone_unload", actor=uid, orders=pickups))
            else:
                old = variant["actions"][replace]["visit_order"]
                old_return, _, old_pickups = _sortie_times(known, station, takeoff, old)
                unload = next((a for a in variant["actions"] if a.get("actor") == uid and a["kind"] == "drone_unload"
                               and dec(a["at"]) >= old_return and set(a["orders"]) == set(old_pickups)), None)
                if old_pickups and unload is None:
                    continue
                boundary = dec(unload["at"]) if unload else old_return
                variant["actions"][replace]["visit_order"] = visits
                if unload is None:
                    # Keep the former return time as a lower bound so the new
                    # unload precedes this drone's later scheduled operations.
                    unload = dict(at=boundary, kind="drone_unload", actor=uid, orders=pickups)
                    variant["actions"].append(unload)
                else:
                    unload.update(orders=pickups)
                unload_index = next(i for i, a in enumerate(variant['actions']) if a is unload)
                from .retiming import retime_plan
                try:
                    variant = retime_plan(known, variant)
                except (InvalidAction, KeyError):
                    continue
                ready = dec(variant['actions'][unload_index]['at'])
                for pickup in pickups:
                    ready += known._handling("drone", known.orders[pickup]["weight"])
            projection = evaluate_plan(known, variant, "WK", require_complete=False)
            if not projection["feasible"]:
                continue
            for recovered in _recovery_variants(known, variant, projection, oid, station, ready):
                # A recovery batch can delay this truck's later replenishment,
                # and hence dependent sorties, but not unrelated devices.
                from .retiming import retime_plan
                try:
                    yield retime_plan(known, recovered)
                except (InvalidAction, KeyError):
                    continue


class MyopicPolicy:
    def choose(self, known, evaluated):
        return select_best(evaluated)

    def __init__(self, waiting="WK", *, max_joint_requests=3, beam_width=4, candidate_budget=64):
        require(waiting in ("WK", "W0"), "Myopic waiting mode not implemented")
        require(max_joint_requests > 0 and beam_width > 0 and candidate_budget > 0, "positive search budgets required")
        self.waiting = waiting
        self.max_joint_requests = max_joint_requests
        self.beam_width = beam_width
        self.candidate_budget = candidate_budget

    def propose(self, known):
        require(not known._hidden and all(e[3] not in ("request_reveal", "revelation_end") for e in known._queue), "planner received hidden future")
        pending = sorted(oid for oid, o in known.orders.items() if o["kind"] == "P" and o["accepted"] is None)
        base = remaining_plan(known)
        base["decisions"] = {oid: False for oid in pending}
        evaluated = [evaluate_plan(known, base, self.waiting)]
        require(evaluated[0]["feasible"] and evaluated[0]["allowed"], "incumbent with rejection fallback is not feasible")
        budget = 0
        for size in range(1, min(len(pending), self.max_joint_requests)+1):
            for subset in combinations(pending[:self.max_joint_requests], size):
                branches = [deepcopy(base)]
                branches[0]["decisions"] = {oid: oid in subset for oid in pending}
                for position, oid in enumerate(subset):
                    next_branches = []
                    for branch in branches:
                        for variant in pickup_variants(known, branch, oid):
                            if budget >= self.candidate_budget:
                                break
                            budget += 1
                            variant["id"] = f"C{budget:05d}"
                            checked = evaluate_plan(known, variant, self.waiting, require_complete=position == len(subset)-1)
                            if checked["feasible"] and checked["allowed"]:
                                if position == len(subset)-1:
                                    evaluated.append(checked)
                                else:
                                    next_branches.append(variant)
                        if budget >= self.candidate_budget:
                            break
                    branches = next_branches[:self.beam_width]
                    if not branches or budget >= self.candidate_budget:
                        break
                if budget >= self.candidate_budget:
                    break
            if budget >= self.candidate_budget:
                break
        return evaluated,budget,pending

    def decide(self,known):
        cache=getattr(self,'proposal_cache',None)
        key=None
        if cache is not None:
            import pickle,hashlib
            key=hashlib.sha256(pickle.dumps((known.__dict__,self.candidate_budget,self.beam_width,self.max_joint_requests,self.waiting),protocol=5)).hexdigest()
        if cache is not None and key in cache:
            evaluated,budget,pending=cache[key]
        else:
            evaluated,budget,pending=self.propose(known)
            if cache is not None and len(cache)<80:cache[key]=(evaluated,budget,pending)
        chosen = self.choose(known, evaluated)
        log = dict(at=known.now, selected=chosen["id"], selected_J=chosen["score"], decisions=chosen["plan"]["decisions"],
                   candidates=[dict(id=r["id"], J=r["score"], allowed=r["allowed"], feasible=r["feasible"],
                                    plan_hash=digest(r["plan"])) for r in evaluated],
                   value_features=features(known), public_state=known.public_snapshot(),
                   candidates_tested=budget, search_truncated=budget >= self.candidate_budget or len(pending) > self.max_joint_requests)
        return chosen["plan"], log


def features(known):
    return plain(dict(time=known.now, pending=sum(o["accepted"] is None for o in known.orders.values()),
                      accepted_unfinished=sum(o["accepted"] is True and o["completed_at"] is None for o in known.orders.values()),
                      station_ready_weight=sum(l["weight"] for l in known.lots if l["state"] == "available"),
                      active_trucks=sum(t["departed_at"] is not None and t["return_time"] is None for t in known.trucks.values()),
                      future_count=known.public.get("future_count")))


class OnlineCoordinator:
    """Environment adapter; policy only receives a newly sanitized known clone."""
    def __init__(self, policy, extend_horizon=False):
        self.policy = policy
        self.tail = TailManager()
        self.decisions = []
        self.extend_horizon = extend_horizon
        self.event_checks = []

    def decision_reason(self, state):
        if any(o["kind"] == "P" and o["accepted"] is None for o in state.orders.values()):
            return "request_batch"
        from .information import closed
        if closed(state.public):
            return "tail_only"
        reviews=getattr(state,'_wait_reviews',{})
        if any(dec(at)==state.now and state.trucks[tid]['active_operation'] is None for tid,at in reviews.items()):
            return 'waiting_review'
        recent=[e for e in reversed(state.trace) if dec(e['at'])==state.now]
        for tid,at in reviews.items():
            t=state.trucks[tid]
            if dec(at)>state.now and t['active_operation'] is None and any(
                (e['kind']=='pickup_unload_complete' and e.get('station')==t['location']) or
                (e['kind']=='drone_handling_done' and not e.get('loading',True) and state.drones[e['actor']]['home']==t['location'])
                for e in recent):
                return 'waiting_inventory_changed'
        # A finite known-order loading reservation executes without re-choosing.
        # Reconsider discretionary waiting only when an idle truck can depart.
        for action in state._actions.get(state.now, []):
            if action["kind"] in ("truck_continue_circuit", "truck_depart"):
                truck = state.trucks[action["actor"]]
                if truck["active_operation"] is None and truck["location"] in state.network["stations"]:
                    return "station_departure_or_wait"
        # A valid loading reservation due now executes directly. Completion of
        # another operation may expose an earlier batch/merge/departure choice.
        if not getattr(self.policy,'enhance',False):
            return None
        for e in recent:
            if e['kind'] not in ('truck_unload_done','truck_load_done'):
                continue
            tid=e['actor'];truck=state.trucks[tid]
            if truck['active_operation'] is not None:continue
            own=[a for t,rows in state._actions.items() if t>=state.now for a in rows if a.get('actor')==tid]
            if own and not any(dec(a['at'])==state.now and a['kind']=='truck_load' for a in own):
                return 'station_operation_choice'
        return None

    def on_event(self, environment):
        if self.tail.activated:
            require(not any(o["accepted"] is None for o in environment.orders.values()), "new request after public tail closure")
            return
        reason = self.decision_reason(environment)
        self.event_checks.append(dict(at=plain(environment.now), reason=reason,
                                      policy_called=reason not in (None, "tail_only")))
        if reason is None:
            return
        known = known_clone(environment)
        if reason == "tail_only":
            chosen = remaining_plan(known)
            log = dict(at=known.now, selected=chosen["id"], reason=reason, policy_called=False)
        else:
            environment._driver_calls[environment.now] = environment._driver_calls.get(environment.now, 0) + 1
            require(environment._driver_calls[environment.now] == 1, "duplicate online policy call at timestamp")
            chosen, log = self.policy.decide(known)
            log.update(reason=reason, policy_called=True)
        log['station_wait_memory'] = deepcopy(known._station_wait_memory)
        # Validate and atomically commit against the same current known state;
        # reveal-stream ownership stays with the environment.
        require(evaluate_plan(known, chosen, self.policy.waiting)["allowed"], "selected candidate failed final validation")
        next_state = _planning_copy(environment)
        install_plan(next_state, chosen)
        now_known = known_clone(next_state)
        if self.extend_horizon:
            next_state.stop = max(next_state.stop, _bound(now_known))
        incumbent = remaining_plan(now_known, chosen["id"])
        tail_plan = self.tail.select(now_known, incumbent, waiting=self.policy.waiting)
        if self.tail.activated:
            install_plan(next_state, tail_plan, admit=False)
        environment.__dict__.update(next_state.__dict__)
        self.decisions.append(plain(log))


def prepare_online(initial, configuration, network, *, waiting="WK", batch_permission="BM", full_day=False):
    from .kernel import Replay
    data = deepcopy(initial)
    data["policy"] = dict(batch_permission=batch_permission)
    end = data.get("public_information", {}).get("revelation_end")
    if end is not None and not data["public_information"].get("revelation_closed"):
        require(all(dec(e["at"]) <= dec(end) for e in data["external_events"] if e["kind"] == "request_reveal"),
                "world reveals requests after its declared revelation end")
        if not any(e["kind"] == "revelation_end" for e in data["external_events"]):
            data["external_events"].append(dict(at=end, kind="revelation_end"))
    state = Replay(configuration, network, data, full_day=full_day)
    plan = remaining_plan(state)
    # Install stops without adopting the pending admissions.
    install_plan(state, plan, admit=False)
    return state, OnlineCoordinator(MyopicPolicy(waiting), extend_horizon=full_day)
