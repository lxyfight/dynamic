"""Generated I0 -> public Myopic -> physical execution -> one-shot tail."""
from copy import deepcopy
from collections import defaultdict
import json

from .audit import audit_ledger
from .initializer import construct_i0, digest, public_delivery_input
from .kernel import COSTS, require
from .online import prepare_online
from .quantities import dec, plain


def make_world(package, releases=(1,), *, unknown_count=False, revelation_end=100):
    cases = {c['id']: c for c in package['cases']}
    source = cases['E01']['input']
    problem = public_delivery_input(package['configuration'], package['networks']['fixed_two_city'],
                                    source['orders'], source['drones'])
    result = construct_i0(problem)
    require(result['status'] == 'VERIFIED', 'I0 failed before online integration')
    world = deepcopy(result['plan']['replay_input'])
    world['stop_time'] = max(900, revelation_end)
    reveals = defaultdict(list)
    for i, at in enumerate(releases):
        p = deepcopy(next(o for o in cases['E02']['input']['orders'] if o['kind'] == 'P'))
        p.update(id=f'P{i+1}', release=at, deadline=float(at)+240, accepted=None)
        world['orders'].append(p)
        reveals[at].append(p['id'])
    world['external_events'] = [dict(at=t, kind='request_reveal', orders=ids) for t, ids in sorted(reveals.items())]
    world['public_information'] = dict(future_count=None if unknown_count else len(releases),
                                      revelation_closed=not releases and not unknown_count)
    if unknown_count:
        world['public_information']['revelation_end'] = revelation_end
    return world, result['plan_hash']


def run_world(package, world, *, waiting='WK', batch_permission='BM'):
    state, controller = prepare_online(world, package['configuration'], package['networks']['fixed_two_city'],
                                      waiting=waiting, batch_permission=batch_permission, full_day=True)
    state.run(decision_hook=controller.on_event, stop_at_terminal=True)
    require(state.is_terminal(), 'online run did not complete')
    initial = deepcopy(state.initial)
    initial['stop_time'] = state.now
    audit = audit_ledger(state.cfg, state.network, initial, state.trace)
    ledger = plain(dict(state.ledger, net=state.ledger['revenue']-sum(state.ledger[k] for k in COSTS)))
    require(not audit['errors'] and all(abs(dec(ledger[k])-dec(audit['ledger'][k])) < dec('1e-8') for k in ledger),
            'independent online ledger mismatch')
    report = dict(terminal=True, absorption_time=plain(state.absorption_time), ledger=ledger, independent_audit=audit,
                  tail=controller.tail.observations(), decisions=controller.decisions,
                  orders=plain(state.orders), trace=plain(state.trace),
                  truck_loading_batches={tid:t['loading_batches'] for tid,t in state.trucks.items()},
                  input_hash=digest(world))
    return state, controller, report


def run_online_examples(package, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    configs = [('WK_BM_one', 'WK', 'BM', [1], False), ('W0_BM_one', 'W0', 'BM', [1], False),
               ('WK_BM_two', 'WK', 'BM', [1,24.6], False), ('WK_B1_two', 'WK', 'B1', [1,24.6], False),
               ('WK_unknown_count', 'WK', 'BM', [1], True)]
    for name, waiting, batch, releases, unknown in configs:
        world, i0_hash = make_world(package, releases, unknown_count=unknown)
        state, controller, report = run_world(package, world, waiting=waiting, batch_permission=batch)
        report.update(example=name, i0_plan_hash=i0_hash, waiting=waiting, batch_permission=batch)
        # Replay the persisted input to verify the integration entry point itself.
        path = output/(name+'_input.json')
        path.write_text(json.dumps(world,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        _, _, repeated = run_world(package,json.loads(path.read_text(encoding='utf8')),waiting=waiting,batch_permission=batch)
        require(digest(report['trace']) == digest(repeated['trace']) and report['decisions'] == repeated['decisions'],
                'serialized world changed deterministic policy run')
        report['serialized_replay_passed'] = True
        (output/(name+'_result.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        rows.append(dict(example=name, terminal=True, serialized_replay_passed=True,
                         accepted=[oid for oid,o in state.orders.items() if o['kind']=='P' and o['accepted']],
                         rejected=[oid for oid,o in state.orders.items() if o['kind']=='P' and o['accepted'] is False],
                         absorption_time=report['absorption_time'], net=report['ledger']['net'],
                         max_batches=max(report['truck_loading_batches'].values(), default=0),
                         tail_replans=controller.tail.replans_after_activation))
    (output/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return rows
