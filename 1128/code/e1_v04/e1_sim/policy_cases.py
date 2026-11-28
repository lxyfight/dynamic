"""E1 Myopic/tail adapters. Fixed-pool fixtures stay distinct from full days."""
from copy import deepcopy
from decimal import Decimal

from .audit import audit_ledger
from .kernel import Replay, require
from .online import (TailManager, _new_truck, commit_decisions, evaluate_plan, install_plan,
                     known_clone, prepare_online, remaining_plan, select_best)
from .quantities import InvalidAction, dec, plain

POLICY_IDS = frozenset(("E12-W0", "E12-WK", "E14", "E18", "E22", "E27-success", "E27-failure",
                        "E27-W0_restriction", "E27-unknown_count"))


def _state(cfg, network, raw, *, reveal_now=False):
    data = deepcopy(raw)
    data["policy"] = {"batch_permission": (data.get("policy") or {}).get("batch_permission", "BM")}
    data["candidates"] = []
    data["actions"] = []
    data["stop_time"] = data["start_time"]
    if reveal_now:
        pending = [o["id"] for o in data["orders"] if o["accepted"] is None]
        if pending and not any(e["kind"] == "request_reveal" for e in data["external_events"]):
            data["external_events"].append(dict(at=data["start_time"], kind="request_reveal", orders=pending))
    engine = Replay(cfg, network, data)
    engine.run()
    return engine


def _tail_plan(raw):
    return dict(id=raw["id"], actions=deepcopy(raw["actions"]), new_trucks=[], decisions={})


def policy_observations(package, case):
    data = deepcopy(case["input"])
    cfg = deepcopy(package["configuration"]); cfg.update(case.get("parameter_overrides", {}))
    network = package["networks"][case["network"]]
    audit, diagnostics, trace, executed = None, {}, [], False
    if case["group"] == "E12":
        source = _state(cfg, network, data, reveal_now=True)
        known = known_clone(source)
        rows = []
        for c in data["candidates"]:
            plan = dict(id=c["id"], actions=[], new_trucks=[], decisions={"P1": c["accept"]})
            if c["accept"]:
                # Physical trajectory, not supplied fixture score/feasible flag.
                uid = data["drones"][0]["id"]
                home = data["drones"][0]["home"]
                from .online import _sortie_times
                returned, ready, ids = _sortie_times(known, home, known.now, ["P1"])
                require(ready == dec(c["pickup_ready"]), "fixture pickup-ready mismatch")
                plan["actions"] += [dict(at=known.now, kind="drone_sortie", actor=uid, visit_order=ids),
                                    dict(at=returned, kind="drone_unload", actor=uid, orders=ids)]
                actor = "T2" if c["new_trip"] else "T1"
                if c["new_trip"]:
                    plan["new_trucks"].append(_new_truck(actor))
                    plan["actions"] += [dict(at=known.now, kind="truck_continue_circuit", actor="T1"),
                                        dict(at=c["new_depart"], kind="truck_dispatch", actor=actor, delivery_orders=[])]
                plan["actions"] += [dict(at=c["load_start"], kind="truck_load", actor=actor, station=home, orders=["P1"]),
                                    dict(at=c["load_end"], kind="truck_continue_circuit", actor=actor, station=home)]
            else:
                plan["actions"].append(dict(at=known.now, kind="truck_continue_circuit", actor="T1"))
            rows.append(evaluate_plan(known, plan, data["policy"]["waiting"]))
        winner = select_best(rows)
        obs = source.observations()
        obs["decision"] = dict(selected=winner["id"])
        obs["candidate"] = {r["id"]: dict(score=r["score"], allowed=r["allowed"], feasible=r["feasible"]) for r in rows}
        diagnostics["candidate_traces"] = {r["id"]: r.get("trace", []) for r in rows}
        diagnostics["real_state_unchanged"] = source.public_snapshot() == known.public_snapshot()
    elif case["group"] in ("E18", "E27"):
        source = _state(cfg, network, data)
        known = known_clone(source)
        if "adopted_plan" in data:
            incumbent = _tail_plan(data["adopted_plan"])
            outcome = data["tail_replan_outcome"]
            proposals = [_tail_plan(outcome["candidate"])]
            search_status = outcome["status"]
        else:
            proposals = [_tail_plan(p) for p in data["tail_candidates"]]
            require(bool(proposals), "tail fixture requires a finite incumbent")
            incumbent = deepcopy(proposals[0]); incumbent["id"] = "original"
            search_status = "candidate"
        manager = TailManager()
        chosen = manager.select(known, incumbent, proposals, waiting=data["policy"]["waiting"], search_status=search_status)
        # Even a declared certificate must pass the physical completion oracle.
        checked = evaluate_plan(known, chosen, data["policy"]["waiting"])
        require(checked["feasible"] and checked["allowed"], "tail selected invalid schedule")
        install_plan(source, chosen)
        source.start = source.now; source.stop = dec(data["stop_time"]); source._has_run = False
        source.run()
        obs = source.observations(); obs["tail"] = manager.observations()
        trace, executed = source.trace, True
        audit_input = deepcopy(data); audit_input["stop_time"] = source.now
        audit = audit_ledger(cfg, network, audit_input, trace)
        audit["matches_online"] = not audit["errors"] and all(abs(dec(v)-dec(obs["ledger"][k])) <= dec("1e-6") for k,v in audit["ledger"].items())
        diagnostics["selected_actions"] = chosen["actions"]
    elif case["group"] == "E22":
        source = _state(cfg, network, data, reveal_now=True)
        before = source.public_snapshot()
        pending = set(source.orders)
        rows = []
        for c in data["candidates"]:
            decisions = {oid: oid in c["accept"] for oid in pending}
            trial = known_clone(source)
            feasible = True
            try:
                require(set(c["accept"]).isdisjoint(c["reject"]) and set(c["accept"]) | set(c["reject"]) == pending,
                        "incomplete candidate admission partition")
                commit_decisions(trial, decisions)
                ids = c.get("pickup_only_sortie_order", c["accept"])
                if ids:
                    trial.apply(dict(at=trial.now, kind="drone_sortie", actor=data["drones"][0]["id"], visit_order=ids))
            except InvalidAction:
                feasible = False
            # A/B are explicit external recovery contracts in this transaction
            # fixture only. Their supplied score is NOT Myopic economic J.
            contract = c.get("feasible", False)
            rows.append(dict(id=c["id"], feasible=feasible and contract, allowed=feasible and contract,
                             score=dec(c["fixture_score"]), plan=dict(decisions=decisions)))
        chosen = select_best(rows)
        unchanged = before == source.public_snapshot()
        commit_decisions(source, chosen["plan"]["decisions"])
        obs = source.observations()
        obs["events"]["reveal_batches"] = sum(e["kind"] == "reveal_batch" for e in source.trace)
        obs["decision"] = dict(selected=chosen["id"])
        obs["candidate"] = {r["id"]: dict(allowed=r["allowed"]) for r in rows}
        obs["candidate"]["validation_mutated_state"] = not unchanged
        trace = source.trace
        diagnostics["scope"] = "admission transaction with explicit external fixture recovery contracts; no full-day feasibility claim"
    elif case["group"] == "E14":
        worlds = []
        cutoff = dec(data["comparison_interval"]["to_exclusive"])
        for world in data["paired_worlds"]:
            raw = deepcopy(data)
            raw["orders"] += deepcopy(world["hidden_orders"])
            for o in world["hidden_orders"]:
                raw["external_events"].append(dict(at=o["release"], kind="request_reveal", orders=[o["id"]]))
            # Generate a finite initial recovery for the known ready order.
            truck = raw["trucks"][0]
            ids = sorted(l["order"] for l in raw["inventory"] if l["state"] == "available" and l["holder"] == truck["location"])
            q = sum(o["weight"] for o in raw["orders"] if o["id"] in ids)
            end = dec(raw["start_time"]) + dec(cfg["truck_handling_setup"]) + dec(cfg["truck_handling_per_kg"])*q
            raw["actions"] = [dict(at=raw["start_time"], kind="truck_load", actor=truck["id"], orders=ids, station=truck["location"]),
                              dict(at=end, kind="truck_continue_circuit", actor=truck["id"], station=truck["location"])]
            engine, driver = prepare_online(raw, cfg, network, waiting="WK")
            interval = dec(data["policy"]["review_interval"])
            checkpoints = []
            t = dec(raw["start_time"])+interval
            while t < cutoff:
                checkpoints.append(t); t += interval
            checkpoints.append(cutoff-dec("0.001"))
            engine.run(checkpoints, decision_hook=driver.on_event)
            logs = [l for l in driver.decisions if dec(l["at"]) < cutoff]
            trace_prefix = [e for e in engine.trace if e["at"] < cutoff]
            worlds.append(dict(decisions=logs, actions=[e for e in trace_prefix if e["kind"].endswith("started") or e["kind"] in ("accept","reject","truck_left_station")],
                               ledger=engine._snapshots[(cutoff-dec("0.001"),"after_closure")]["ledger"]))
        a,b = worlds
        require(bool(a["actions"]) and bool(a["decisions"]), "paired check must be nonvacuous")
        checks = dict(actions_equal=a["actions"] == b["actions"], ledger_equal=a["ledger"] == b["ledger"],
                      public_state_equal=[l["public_state"] for l in a["decisions"]] == [l["public_state"] for l in b["decisions"]],
                      candidate_lists_equal=[l["candidates"] for l in a["decisions"]] == [l["candidates"] for l in b["decisions"]],
                      value_features_equal=[l["value_features"] for l in a["decisions"]] == [l["value_features"] for l in b["decisions"]])
        obs = dict(paired=dict(pre_divergence=checks), ledger={k:0 for k in ("revenue",)+COSTS+("net",)})
        diagnostics["paired_execution"] = worlds
    else:
        raise InvalidAction("unsupported online fixture")
    return dict(observations=plain(obs), trace=plain(trace), independent_audit=plain(audit), diagnostics=plain(diagnostics), executed=executed)


from .kernel import COSTS
