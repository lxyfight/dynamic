"""Fixed-structure Appendix A feasibility slice. No free reoptimization.

All used assignments/routes/quantities/times are fixed; unused assignments are
zero. This certifies mapped witnesses, not an optimized PI bound. Run with the
local Python interpreter containing gurobipy, passing input/output JSON paths.
"""
import json,sys,math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def solve(data):
    cfg,net,initial,events=data['configuration'],data['network'],data['initial'],data['trace']
    orders={o['id']:o for o in initial['orders']};homes={u['id']:u['home'] for u in initial['drones']}
    if initial['start_time']!=0 or any(l['state']!='depot' for l in initial['inventory']):
        raise ValueError('fixed full-day PI requires depot-start, empty station inventory')
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    m=gp.Model(env=env);m.Params.FeasibilityTol=1e-8
    times={};ends={};checks=[]
    def fixed(name,value,integer=False):
        if integer and (isinstance(value,bool) or int(value)!=value):raise ValueError('noninteger quantity '+name)
        return m.addVar(lb=float(value),ub=float(value),vtype=GRB.INTEGER if integer else GRB.CONTINUOUS,name=name)
    def check(condition,name):
        # Fixed discrete decisions are checked before constructing their active
        # continuous branch; a false condition makes the fixed model infeasible.
        m.addConstr(gp.LinExpr(0)<=0 if condition else gp.LinExpr(0)<=-1,name=name)
        checks.append(name)
    for k,e in enumerate(events):
        times[k]=fixed('t_'+str(k),e['at'])
        if 'end' in e:ends[k]=fixed('end_'+str(k),e['end'])
    bykind=defaultdict(list)
    for k,e in enumerate(events):bykind[e['kind']].append((k,e))
    def rows(kind,actor=None):return [(k,e) for k,e in bykind[kind] if actor is None or e.get('actor')==actor]
    served={};handled={};alloc=defaultdict(list);recover=defaultdict(list);dispatch={};returned={};truck_weight={};batchcounts=defaultdict(int)
    accepted={o['id']:o['accepted'] for o in orders.values()}
    for k,e in events_by_kind(events,['accept','reject']):accepted[e['order']]=e['kind']=='accept'
    for oid,o in orders.items():
        check(type(o['weight']) is int and 0<o['weight']<=cfg['drone_capacity'],'integer_request_'+oid)
        check(accepted[oid] in (True,False),'decided_'+oid)
        check(o['kind']!='D' or accepted[oid] is True,'mandatory_'+oid)
    for k,e in rows('truck_unload_started'):
        check(sum(e['quantities'].values())>0,'nonempty_unload_'+str(k))
        for oid,q in e['quantities'].items():
            check(orders[oid]['kind']=='D' and q>=0,'delivery_allocation_'+oid+'_'+str(k))
            alloc[oid].append((e['actor'],e['station'],k,fixed('chi_'+str(k)+'_'+oid,q,True)))
    for k,e in rows('truck_load_started'):
        batchcounts[e['actor'],e['station']]+=1
        check(bool(e['orders']) and len(set(e['orders']))==len(e['orders']),'nonempty_batch_'+str(k))
        for oid in e['orders']:
            check(orders[oid]['kind']=='P' and accepted[oid] is True,'pickup_recovery_'+oid)
            recover[oid].append((e['actor'],e['station'],k))
    for key,count in batchcounts.items():check(data['batch_permission']=='BM' or count<=1,'B1_'+str(key))
    for k,e in rows('truck_dispatched'):
        actor=e['actor'];check(actor not in dispatch,'unique_trip_'+actor);dispatch[actor]=k
        m.addConstr(times[k]<=cfg['truck_dispatch_cutoff'],name='dispatch_cutoff_'+actor)
        truck_weight[actor]=gp.quicksum(q for aa in alloc.values() for a,s,j,q in aa if a==actor)
        m.addConstr(truck_weight[actor]<=cfg['truck_capacity'],name='initial_capacity_'+actor)
        legs=rows('road_started',actor)
        check(len(legs)==len(net['arcs']),'whole_circuit_'+actor)
        arcmap={a['from']:a for a in net['arcs']};node='depot'
        previous=times[k]
        for j,leg in legs:
            arc=arcmap[node];check(leg['source']==node and leg['destination']==arc['to'],'road_arc_'+str(j))
            m.addConstr(ends[j]-times[j]==arc['min'],name='road_duration_'+str(j))
            m.addConstr(times[j]>=previous,name='road_order_'+str(j));previous=ends[j];node=arc['to']
            arrivals=[a for a,z in rows('truck_arrived',actor) if z['destination']==node]
            check(len(arrivals)==1,'arrival_record_'+str(j))
            if arrivals:m.addConstr(times[arrivals[0]]==ends[j],name='arrival_time_'+str(j))
        if legs:m.addConstr(times[legs[0][0]]==times[k],name='departure_road_start_'+actor)
        rr=[j for j,z in rows('truck_arrived',actor) if z['destination']=='depot']
        check(len(rr)==1,'one_return_'+actor)
        if rr:returned[actor]=rr[0];m.addConstr(times[rr[0]]==previous,name='return_'+actor)
    # Active operations are sequential on each physical resource.
    operations=defaultdict(list)
    for kind in ['road_started','truck_unload_started','truck_load_started','drone_handling_started','flight_started','service_started']:
        for k,e in rows(kind):operations[e['actor']].append(k)
    for actor,ids in operations.items():
        ids.sort(key=lambda k:(events[k]['at'],k))
        for left,right in zip(ids,ids[1:]):m.addConstr(ends[left]<=times[right],name='resource_'+str(left)+'_'+str(right))
    last_unload={};last_batch={}
    for k,e in enumerate(events):
        kind=e['kind'];actor=e.get('actor')
        if kind in ('truck_unload_started','truck_load_started'):
            check(actor in dispatch,'declared_trip_'+str(k))
            station=e['station'];legs=rows('road_started',actor)
            incoming=[j for j,z in legs if z['destination']==station];outgoing=[j for j,z in legs if z['source']==station]
            check(len(incoming)==len(outgoing)==1,'stop_'+str(k))
            if incoming and outgoing:
                m.addConstr(times[k]>=ends[incoming[0]],name='inside_stop_begin_'+str(k))
                m.addConstr(ends[k]<=times[outgoing[0]],name='inside_stop_end_'+str(k))
            key=actor,station
            if kind=='truck_unload_started':
                check(key not in last_unload and key not in last_batch,'single_unload_first_'+str(k));last_unload[key]=k
                amount=gp.quicksum(q for aa in alloc.values() for a,s,j,q in aa if j==k)
                truck_weight[actor]-=amount
            else:
                amount=sum(orders[o]['weight'] for o in e['orders']);truck_weight[actor]+=amount
                if key in last_unload:m.addConstr(times[k]>=ends[last_unload[key]],name='unload_before_load_'+str(k))
                if key in last_batch:m.addConstr(times[k]>=ends[last_batch[key]],name='batch_order_'+str(k))
                last_batch[key]=k
            m.addConstr(ends[k]-times[k]==cfg['truck_handling_setup']+cfg['truck_handling_per_kg']*amount,name='truck_handling_'+str(k))
            m.addConstr(truck_weight[actor]>=0,name='truck_nonnegative_'+str(k));m.addConstr(truck_weight[actor]<=cfg['truck_capacity'],name='truck_capacity_'+str(k))
        elif kind=='drone_handling_started':
            oid=e['order'];check(oid not in handled,'one_handling_'+oid);handled[oid]=k
            m.addConstr(ends[k]-times[k]==cfg['drone_handling_setup_per_request']+cfg['drone_handling_per_kg']*orders[oid]['weight'],name='drone_handling_'+oid)
            check(e['loading']==(orders[oid]['kind']=='D'),'handling_direction_'+oid)
        elif kind=='service_started':
            oid=e['order'];check(oid not in served,'once_service_'+oid);served[oid]=k
            m.addConstr(ends[k]-times[k]==cfg['customer_service_per_request'],name='service_duration_'+oid)
            m.addConstr(times[k]>=orders[oid]['release'],name='physical_release_'+oid)
            if orders[oid]['kind']=='D':m.addConstr(ends[k]<=orders[oid]['deadline'],name='delivery_deadline_'+oid)
    for oid,o in orders.items():
        check((oid in served)==bool(accepted[oid]),'admission_service_'+oid)
        if not accepted[oid]:check(not alloc[oid] and not recover[oid],'rejected_no_flow_'+oid);continue
        check(oid in handled,'handling_exists_'+oid)
        if oid not in handled:continue
        h=handled[oid];home=homes[events[h]['actor']];check(home in o['eligible_stations'],'eligible_'+oid)
        if o['kind']=='D':
            m.addConstr(gp.quicksum(q for a,s,k,q in alloc[oid])==o['weight'],name='integer_supply_balance_'+oid)
            for a,s,k,q in alloc[oid]:
                check(s==home,'supply_station_'+oid+'_'+str(k));m.addConstr(ends[k]<=times[h],name='D_available_'+oid+'_'+str(k))
        else:
            check(len(recover[oid])==1,'whole_pickup_'+oid)
            for a,s,k in recover[oid]:check(s==home,'recovery_station_'+oid);m.addConstr(ends[h]<=times[k],name='P_ready_'+oid)
        completion=[k for k,e in rows('order_completed') if e['order']==oid]
        check(len(completion)==1,'completion_record_'+oid)
        if completion and oid in served:
            if o['kind']=='D':m.addConstr(times[completion[0]]==ends[served[oid]],name='D_completion_'+oid)
            elif len(recover[oid])==1 and recover[oid][0][0] in returned:
                m.addConstr(times[completion[0]]==times[returned[recover[oid][0][0]]],name='P_depot_completion_'+oid)
    for uid,home in homes.items():
        flights=rows('drone_takeoff',uid);returns=rows('drone_returned',uid);previous_finish=None;allvisits=[]
        check(len(flights)==len(returns),'sortie_returns_'+uid)
        for index,((k,e),(r,re)) in enumerate(zip(flights,returns)):
            visits=e['orders'];allvisits+=visits;check(bool(visits),'nonempty_sortie_'+str(k))
            weight=sum(orders[o]['weight'] for o in visits if orders[o]['kind']=='D');check(weight<=cfg['drone_capacity'],'takeoff_capacity_'+str(k))
            xy=net['stations'][home];previous=times[k];starts=[times[k]];finishes=[times[r]]
            for oid in visits:
                check(oid in served and events[served[oid]]['actor']==uid,'sortie_assignment_'+oid)
                if oid not in served or oid not in handled:continue
                j=served[oid];h=handled[oid];o=orders[oid]
                check(events[h]['actor']==uid,'same_physical_drone_'+oid)
                travel=math.dist(xy,o['xy'])/cfg['drone_speed_km_min']
                m.addConstr(times[j]>=previous+travel,name='flight_sequence_'+oid)
                if o['kind']=='D':
                    m.addConstr(ends[h]<=times[k],name='preflight_load_'+oid);starts.append(times[h]);weight-=o['weight']
                else:
                    m.addConstr(times[h]>=times[r],name='postflight_unload_'+oid);finishes.append(ends[h]);weight+=o['weight']
                check(0<=weight<=cfg['drone_capacity'],'arc_payload_'+oid)
                xy=o['xy'];previous=ends[j]
            m.addConstr(times[r]>=previous+math.dist(xy,net['stations'][home])/cfg['drone_speed_km_min'],name='home_return_'+str(k))
            m.addConstr(times[r]-times[k]<=cfg['drone_endurance'],name='endurance_'+str(k))
            post=sorted([handled[o] for o in visits if orders[o]['kind']=='P' and o in handled],key=lambda j:(events[j]['at'],j))
            # A19 uses consecutive unloading from home return. A trajectory
            # with extra postflight idle needs an explicit dominance transform,
            # rather than being silently called an exact fixed mapping.
            if post:
                m.addConstr(times[post[0]]==times[r],name='A19_unload_from_return_'+str(k))
                for left,right in zip(post,post[1:]):m.addConstr(times[right]==ends[left],name='A19_consecutive_unload_'+str(right))
            if previous_finish is not None:
                for st in starts:
                    for fin in previous_finish:m.addConstr(st>=fin,name='physical_drone_reuse_'+str(k))
            previous_finish=finishes
        check(len(allvisits)==len(set(allvisits)),'no_revisit_request_'+uid)
        check(set(allvisits)=={o for o,j in served.items() if events[j]['actor']==uid},'route_service_match_'+uid)
    m.setObjective(0);m.optimize()
    certificate=[]
    for c in m.getConstrs():
        row=m.getRow(c)
        certificate.append(dict(name=c.ConstrName,sense=c.Sense,rhs=c.RHS,
            terms={row.getVar(i).VarName:row.getCoeff(i) for i in range(row.size())}))
    return dict(status='FEASIBLE' if m.Status==GRB.OPTIMAL else 'INFEASIBLE',solver_status=m.Status,
                variables=m.NumVars,constraints=m.NumConstrs,all_variables_fixed=all(v.LB==v.UB for v in m.getVars()),
                solver_version=gp.gurobi.version(),fixed_values={v.VarName:dict(value=v.LB,type=v.VType) for v in m.getVars()},linear_constraints=certificate,
                scope='fixed selected structure feasibility slice A3-A7; no global PI optimum/bound',free_reoptimization=False)

def events_by_kind(events,kinds):return [(k,e) for k,e in enumerate(events) if e['kind'] in kinds]

if __name__=='__main__':
    try:result=solve(json.load(open(sys.argv[1],encoding='utf8')))
    except Exception as exc:result=dict(status='ERROR',error=type(exc).__name__+': '+str(exc))
    with open(sys.argv[2],'w',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
