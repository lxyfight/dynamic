"""One-truck first-action interventions with a frozen known-order service plan."""
from copy import deepcopy

from e1_sim.online import evaluate_plan
from e1_sim.quantities import dec, plain
from unified_vfa_policy import canonical_plan_digest, replay_wait_combination


def qualification(known, selected_row, truck, *, require_just_unloaded=True):
    """Require a true post-unload departure, not postponing reserved loading."""
    vehicle = known.trucks.get(truck)
    reasons = []
    if vehicle is None:
        return dict(eligible=False, reasons=['unknown_truck'], truck=truck)
    station = vehicle['location']
    own = sorted(((i,a) for i,a in enumerate(selected_row['plan']['actions'])
                  if a.get('actor')==truck), key=lambda x:(dec(x[1]['at']),x[0]))
    first = own[0] if own else None
    just_unloaded = any(e['kind']=='truck_unload_done' and e.get('actor')==truck
        and e.get('station')==station and dec(e['at'])==known.now for e in known.trace)
    live_review = dec(getattr(known,'_wait_reviews',{}).get(truck,known.now))>known.now
    reservation = bool(first and first[1]['kind']=='truck_load')
    if station not in known.network['stations']: reasons.append('not_at_station')
    if vehicle['active_operation'] is not None: reasons.append('not_idle')
    if require_just_unloaded and not just_unloaded: reasons.append('not_just_unloaded')
    if live_review: reasons.append('existing_active_review')
    if not first or first[1]['kind']!='truck_continue_circuit' or dec(first[1]['at'])!=known.now:
        reasons.append('first_action_not_immediate_departure')
    if not selected_row.get('feasible') or not selected_row.get('allowed'):
        reasons.append('base_not_verified_feasible')
    return dict(eligible=not reasons,reasons=reasons,truck=truck,station=station,
        at=float(known.now),just_unloaded=just_unloaded,live_review=live_review,
        known_cargo_reservation=reservation,
        first_command_deferred=bool(first and dec(first[1]['at'])>known.now),
        first_action_index=first[0] if first else None,
        first_action=plain(first[1]) if first else None)


def summarize_intervention(known, base_plan, branch_plan, truck, interval):
    """Certify same commands/admissions, allowing only necessary time delays."""
    if base_plan.get('decisions')!=branch_plan.get('decisions'):
        raise ValueError('intervention changed admissions')
    if base_plan.get('new_trucks')!=branch_plan.get('new_trucks'):
        raise ValueError('intervention changed truck dispatch resources')
    before,after=base_plan['actions'],branch_plan['actions']
    if len(before)!=len(after): raise ValueError('intervention changed command count')
    changes=[]
    for index,(old,new) in enumerate(zip(before,after)):
        if {k:v for k,v in old.items() if k!='at'}!={k:v for k,v in new.items() if k!='at'}:
            raise ValueError('intervention changed assignment, route, order or command identity')
        if dec(old['at'])!=dec(new['at']):
            if dec(new['at'])<dec(old['at']): raise ValueError('intervention advanced another command')
            actor=old.get('actor')
            if actor!=truck and actor not in known.drones:
                raise ValueError('intervention changed another truck action')
            changes.append(dict(index=index,actor=actor,kind=old['kind'],
                old_at=float(dec(old['at'])),new_at=float(dec(new['at'])),
                delay=float(dec(new['at'])-dec(old['at']))))
    expected_reviews=deepcopy(base_plan.get('wait_reviews',{}))
    if interval: expected_reviews[truck]=known.now+dec(interval)
    if branch_plan.get('wait_reviews',{})!=expected_reviews:
        raise ValueError('intervention changed another review timer')
    for name in set(base_plan)|set(branch_plan):
        if name not in ('id','actions','wait_reviews') and base_plan.get(name)!=branch_plan.get(name):
            raise ValueError('intervention changed non-timing plan field: '+name)
    return dict(truck=truck,interval=interval,admissions_unchanged=True,
        action_identities_unchanged=True,other_truck_actions_unchanged=True,
        strict_non_target_actions_unchanged=not any(c['actor']!=truck for c in changes),
        timing_only_changes=changes,base_hash=canonical_plan_digest(base_plan),
        branch_hash=canonical_plan_digest(branch_plan))


def build_branches(known, selected_row, truck, *, require_just_unloaded=True):
    """Return four physically checked continuations from the same chosen base.

    The wait is a review timer, not a forced minimum dwell. Legitimate earlier
    events remain eligible for the common subsequent online policy.
    """
    check=qualification(known,selected_row,truck,require_just_unloaded=require_just_unloaded)
    if not check['eligible']:
        raise ValueError('not a pure post-unload departure: '+','.join(check['reasons']))
    base=evaluate_plan(known,selected_row['plan'],'WK')
    if not base['feasible'] or not base['allowed']: raise ValueError('base replay failed')
    rows={0:base}
    for interval in (2,5,10):
        candidate,info=replay_wait_combination(known,base,5,
            [(truck,check['first_action_index'],'wait',interval)],(0,))
        if not candidate['feasible'] or not candidate['allowed']:
            raise ValueError('wait branch infeasible at interval '+str(interval)+': '+str(info))
        rows[interval]=candidate
    for interval,row in rows.items():
        row['intervention']=dict(qualification=check,**summarize_intervention(
            known,base['plan'],row['plan'],truck,interval))
        departure=next((e for e in row['trace'] if e['kind']=='road_started'
            and e.get('actor')==truck and e.get('source')==check['station']),None)
        if departure is None or dec(departure['at'])!=known.now+dec(interval):
            raise ValueError('known-only replay did not execute prescribed first departure time')
    return rows
