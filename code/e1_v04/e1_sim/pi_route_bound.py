"""Certified PI upper bound from an exhaustive sortie-cover relaxation.

NOT the full Appendix A optimizer. Every physically feasible day maps to this
relaxation with an objective no smaller than its true profit. Never use a
restricted route pool's optimum as an upper bound. Enumeration must finish.
Run with the configured Gurobi Python (single solver thread).
"""
import json,math,sys,time
from pathlib import Path
try:
    from .cost_model import per_sortie
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e1_sim.cost_model import per_sortie
import gurobipy as gp
from gurobipy import GRB


def routes(instance):
    cfg=instance['configuration'];orders=instance['input']['orders'];net=instance['network']
    cap=cfg['drone_capacity'];speed=cfg['drone_speed_km_min'];service=cfg['customer_service_per_request'];endurance=cfg['drone_endurance'];result=[]
    for station,home in net['stations'].items():
        local=[o for o in orders if station in o['eligible_stations']]
        xy=[home]+[o['xy'] for o in local];dist=[[math.dist(a,b) for b in xy] for a in xy]
        # For a prefix, the initial delivery load is not final until all later
        # deliveries are known. Check the whole route at each extension.
        def visit(path,remaining,km,dq,pq):
            if path:
                flight=km+dist[path[-1]+1][0]
                load=dq;ok=True
                for i in path:
                    load+=local[i]['weight']*(1 if local[i]['kind']=='P' else -1)
                    if load>cap:ok=False;break
                if ok and flight/speed+len(path)*service<=endurance+1e-9:
                    result.append(dict(station=station,orders=[local[i]['id'] for i in path],km=flight))
            last=path[-1]+1 if path else 0
            for i in remaining:
                o=local[i];nd=dq+(o['weight'] if o['kind']=='D' else 0);np=pq+(o['weight'] if o['kind']=='P' else 0)
                if nd>cap or np>cap:continue
                nk=km+dist[last][i+1]
                # Triangle inequality: this is a safe lower bound on every
                # extension's travel+service. No heuristic route truncation.
                if (nk+dist[i+1][0])/speed+(len(path)+1)*service>endurance+1e-9:continue
                visit(path+[i],[j for j in remaining if j!=i],nk,nd,np)
        visit([],list(range(len(local))),0.,0,0)
    return result


def solve(instance,seconds=60):
    begin=time.perf_counter();pool=routes(instance);enum_seconds=time.perf_counter()-begin
    cfg=instance['configuration'];net=instance['network'];orders={o['id']:o for o in instance['input']['orders']}
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start();m=gp.Model('PI_exhaustive_route_relaxation',env=env)
        m.Params.Threads=1;m.Params.TimeLimit=seconds;m.Params.MIPGap=1e-6;m.Params.Seed=17
        x=m.addVars(len(pool),vtype=GRB.BINARY,name='sortie');a=m.addVars(orders,vtype=GRB.BINARY,name='serve')
        z=m.addVars(net['stations'],vtype=GRB.BINARY,name='station_drone_used')
        d=m.addVars(net['stations'],vtype=GRB.BINARY,name='station_delivery_unload')
        p=m.addVars(net['stations'],vtype=GRB.BINARY,name='station_pickup_load')
        trips=m.addVar(vtype=GRB.INTEGER,lb=0,name='aggregate_trip_lower_bound')
        indexes={oid:[] for oid in orders};at={s:[] for s in net['stations']}
        for k,r in enumerate(pool):
            at[r['station']].append(k)
            for oid in r['orders']:indexes[oid].append(k)
        D={};P={}
        for oid,o in orders.items():
            m.addConstr(gp.quicksum(x[k] for k in indexes[oid])==a[oid])
            if o['kind']=='D':a[oid].LB=1
        for s in net['stations']:
            D[s]=gp.quicksum(orders[i]['weight']*x[k] for k in at[s] for i in pool[k]['orders'] if orders[i]['kind']=='D')
            P[s]=gp.quicksum(orders[i]['weight']*x[k] for k in at[s] for i in pool[k]['orders'] if orders[i]['kind']=='P')
            for k in at[s]:m.addConstr(x[k]<=z[s])
            m.addConstr(D[s]<=sum(o['weight'] for o in orders.values())*d[s]);m.addConstr(P[s]<=sum(o['weight'] for o in orders.values())*p[s])
        load=gp.quicksum(D.values());m.addConstr(load<=cfg['truck_capacity']*trips)
        for arc in net['arcs']:
            s=arc['to']
            if s=='depot':break
            load=load-D[s]+P[s];m.addConstr(load<=cfg['truck_capacity']*trips)
        travel=sum(a['min'] for a in net['arcs']);tripcost=sum(a['km']*(cfg['truck_distance_cost']+(cfg['toll_per_marked_km'] if a['tolled'] else 0)) for a in net['arcs'])
        revenue=gp.quicksum(cfg['revenue_per_kg']*o['weight']*a[oid] for oid,o in orders.items())
        rejection=gp.quicksum(cfg['rejection_per_kg']*o['weight']*(1-a[oid]) for oid,o in orders.items() if o['kind']=='P')
        handling=cfg['truck_handling_setup']*gp.quicksum(d[s]+p[s] for s in net['stations'])+cfg['truck_handling_per_kg']*gp.quicksum(D[s]+P[s] for s in net['stations'])
        cost=(tripcost+cfg['truck_time_cost']*travel)*trips+cfg['truck_time_cost']*handling+(cfg['drone_sortie_cost']*gp.quicksum(x.values()) if per_sortie(cfg) else cfg['drone_first_use_cost']*gp.quicksum(z.values()))+cfg['drone_distance_cost']*gp.quicksum(pool[k]['km']*x[k] for k in range(len(pool)))
        m.setObjective(revenue-rejection-cost,GRB.MAXIMIZE);m.optimize()
        bound=float(m.ObjBound) if m.Status not in [GRB.INFEASIBLE,GRB.INF_OR_UNBD,GRB.UNBOUNDED] else None
        if bound is None or not math.isfinite(bound):raise RuntimeError('No finite certified relaxation bound')
        return dict(instance=instance['id'],kind='PI_RELAXATION_UPPER_BOUND',upper_bound=bound+1e-5,solver_bound_raw=bound,
            relaxation_incumbent=float(m.ObjVal) if m.SolCount else None,solver_status=int(m.Status),solver_version=list(gp.gurobi.version()),
            gap=float(m.MIPGap) if m.SolCount else None,routes=len(pool),enumeration_complete=True,enumeration_seconds=enum_seconds,
            solve_seconds=m.Runtime,variables=m.NumVars,constraints=m.NumConstrs,
            dropped_constraints=['release/deadline timing','inter-sortie physical-drone overlap','truck-batch timing and waiting','pickup lateness'],
            retained=['whole request service','feasible station','complete same-home route','segment payload','travel+service endurance','aggregate circuit load lower bound','nonnegative operation and activation cost lower bounds'],
            limitation='The relaxation incumbent is NOT an executable PI schedule or a lower bound. Use independently replayed online profit as L. U-L combines policy suboptimality and relaxation looseness, not exact optimality gap.')


if __name__=='__main__':
    instance=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
    Path(sys.argv[2]).write_text(json.dumps(solve(instance,float(sys.argv[3]) if len(sys.argv)>3 else 60),indent=2),encoding='utf8')
