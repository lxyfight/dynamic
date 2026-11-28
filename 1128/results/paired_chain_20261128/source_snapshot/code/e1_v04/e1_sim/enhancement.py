"""Bounded public-plan neighborhoods; all outputs require complete replay.

Search preserves every base candidate, including short-term costly waiting.
No neighborhood modifies live state or in-progress operations.
"""
from copy import deepcopy
from collections import defaultdict
from .online import evaluate_plan
from .quantities import dec


def variants(known, row):
    plan = row['plan']
    actions = plan['actions']
    # A discretionary reservation can be abandoned at an earlier real event.
    for tid, review in sorted(plan.get('wait_reviews', {}).items()):
        truck = known.trucks.get(tid)
        if not truck or truck['active_operation'] is not None:
            continue
        own = sorted((a for a in actions if a.get('actor') == tid), key=lambda a: dec(a['at']))
        if own and own[0]['kind'] == 'truck_continue_circuit' and dec(own[0]['at']) > known.now:
            p = deepcopy(plan)
            shift = dec(own[0]['at']) - known.now
            for a in p['actions']:
                if a.get('actor') == tid:
                    a['at'] = dec(a['at']) - shift
            p.get('wait_reviews', {}).pop(tid, None)
            yield 'depart_now', p

    # Group by actor AND station, never merge different stops or active loads.
    batches = defaultdict(list)
    for a in actions:
        if a['kind'] == 'truck_load':
            batches[a['actor'], a.get('station')].append(a)
    for (tid, station), loads in sorted(batches.items()):
        for first, second in zip(sorted(loads, key=lambda a: dec(a['at'])), sorted(loads, key=lambda a: dec(a['at']))[1:]):
            # Do not cross a departure: the same actor may revisit on another trip.
            if any(a['kind'] == 'truck_continue_circuit' and a.get('actor') == tid
                   and dec(first['at']) <= dec(a['at']) < dec(second['at']) for a in actions):
                continue
            ids = list(dict.fromkeys(first['orders'] + second['orders']))
            ready = max(dec(row['state'].orders[o]['available_at']) if row['state'].orders[o]['available_at'] is not None else known.now for o in ids)
            start = max(known.now, dec(first['at']), ready)
            end = start + known._handling('truck', sum(known.orders[o]['weight'] for o in ids))
            oldend = dec(second['at']) + known._handling('truck', sum(known.orders[o]['weight'] for o in second['orders']))
            p = deepcopy(plan)
            p['actions'] = [a for a in p['actions'] if a != first and a != second]
            for a in p['actions']:
                if a.get('actor') == tid and dec(a['at']) >= oldend:
                    a['at'] = dec(a['at']) + end - oldend
            p['actions'].append(dict(kind='truck_load', actor=tid, station=station, orders=ids, at=start))
            yield 'merge_batches', p

        # Exploit an idle gap by loading a ready subset before a later batch.
        for batch in loads:
            ids = batch['orders']
            if len(ids) < 2:
                continue
            available = {o: max(known.now, dec(row['state'].orders[o]['available_at'])) if row['state'].orders[o]['available_at'] is not None else known.now for o in ids}
            earliest = min(available.values())
            first_ids = [o for o in ids if available[o] == earliest]
            if len(first_ids) == len(ids):
                continue
            truck = known.trucks.get(tid)
            free = known.now if truck and truck['location'] == station and truck['active_operation'] is None else None
            for event in row['trace']:
                if event.get('actor') != tid or dec(event['at']) > dec(batch['at']):
                    continue
                if event['kind'] == 'truck_arrived' and event['destination'] == station:
                    free = dec(event['at'])
                elif event['kind'] in ('truck_load_done', 'truck_unload_done'):
                    free = dec(event['at'])
            if free is None:
                continue
            start = max(free, earliest)
            first_end = start + known._handling('truck', sum(known.orders[o]['weight'] for o in first_ids))
            if first_end > dec(batch['at']):
                continue
            rest = [o for o in ids if o not in first_ids]
            second_start = max(first_end, max(available[o] for o in rest))
            end = second_start + known._handling('truck', sum(known.orders[o]['weight'] for o in rest))
            oldend = dec(batch['at']) + known._handling('truck', sum(known.orders[o]['weight'] for o in ids))
            p = deepcopy(plan)
            p['actions'] = [a for a in p['actions'] if a != batch]
            for a in p['actions']:
                if a.get('actor') == tid and dec(a['at']) >= oldend:
                    a['at'] = dec(a['at']) + end - oldend
            p['actions'] += [dict(kind='truck_load', actor=tid, station=station, orders=first_ids, at=start),
                             dict(kind='truck_load', actor=tid, station=station, orders=rest, at=second_start)]
            yield 'split_batch', p

    # Reordering only commands still in the replaceable plan. Airborne sorties
    # have no such command and cannot enter this neighborhood.
    for i, a in enumerate(actions):
        if a['kind'] != 'drone_sortie':
            continue
        visits = a['visit_order']
        for j in range(len(visits) - 1):
            p = deepcopy(plan)
            seq = p['actions'][i]['visit_order']
            seq[j], seq[j + 1] = seq[j + 1], seq[j]
            yield 'sortie_swap', p


def enhance_pool(known, rows, budget=16, waiting='WK'):
    """Round-robin rows AND move types. Budget counts attempted full replays."""
    if budget <= 0:
        return list(rows), dict(attempted=0, accepted=0, by_move={})
    queues = defaultdict(list)
    for row in rows:
        for kind, plan in variants(known, row):
            queues[kind].append((row['id'], plan))
    extra = []
    counts = {}
    attempts = 0
    kinds = ['depart_now', 'merge_batches', 'split_batch', 'sortie_swap']
    while attempts < budget and any(queues.values()):
        for kind in kinds:
            if attempts >= budget:
                break
            if not queues[kind]:
                continue
            parent, p = queues[kind].pop(0)
            attempts += 1
            counts[kind] = counts.get(kind, 0) + 1
            p['id'] = parent + '_enh_' + kind + '_' + str(attempts)
            p['enhancement'] = dict(move=kind, parent=parent)
            checked = evaluate_plan(known, p, waiting)
            if checked['feasible'] and checked['allowed']:
                extra.append(checked)
    return list(rows) + extra, dict(attempted=attempts, accepted=len(extra), by_move=counts)
