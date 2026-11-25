"""Fixed-variable PI witness bridge, distinct from free PI optimization."""
from copy import deepcopy
from pathlib import Path
import json,subprocess,tempfile,os,sys
from .quantities import plain
from .kernel import require
from .audit import audit_ledger

PI_PYTHON=Path(os.environ.get('E1_PI_PYTHON',sys.executable))

def fixed_feasibility(configuration,network,initial,trace,batch='BM'):
    data=plain(dict(configuration=configuration,network=network,initial=initial,trace=trace,batch_permission=batch))
    with tempfile.TemporaryDirectory(prefix='e1_fixed_pi_') as folder:
        source=Path(folder)/'input.json';target=Path(folder)/'output.json'
        source.write_text(json.dumps(data),encoding='utf8')
        subprocess.run([str(PI_PYTHON),str(Path(__file__).with_name('pi_fixed_solver.py')),str(source),str(target)],check=True,capture_output=True,timeout=60)
        return json.loads(target.read_text(encoding='utf8'))

def decode_commands(trace):
    commands=[];quantities={}
    for e in trace:
        if e['kind']=='truck_unload_started':
            q=quantities.setdefault(e['actor'],{})
            for oid,value in e['quantities'].items():q[oid]=q.get(oid,0)+value
    for e in trace:
        a=dict(at=e['at']);kind=e['kind']
        if kind in ('accept','reject'):a.update(kind=kind,orders=[e['order']])
        elif kind=='truck_dispatched':
            q=quantities.get(e['actor'],{});a.update(kind='truck_dispatch',actor=e['actor'],delivery_orders=list(q),quantities=q)
        elif kind=='truck_unload_started':a.update(kind='truck_unload',actor=e['actor'],station=e['station'],quantities=e['quantities'])
        elif kind=='truck_load_started':a.update(kind='truck_load',actor=e['actor'],station=e['station'],orders=e['orders'])
        elif kind=='truck_left_station':a.update(kind='truck_continue_circuit',actor=e['actor'],station=e['station'])
        elif kind=='drone_handling_started':a.update(kind='drone_load' if e['loading'] else 'drone_unload',actor=e['actor'],orders=[e['order']])
        elif kind=='drone_takeoff':a.update(kind='drone_sortie',actor=e['actor'],visit_order=e['orders'])
        else:continue
        commands.append(a)
    return commands

def mapping_observations(package,case):
    from .runner import build_replay
    witness=deepcopy(next(c for c in package['cases'] if c['id']==case['input']['witness_case']))
    cfg=deepcopy(package['configuration']);cfg.update(witness.get('parameter_overrides',{}))
    network=package['networks'][witness['network']]
    original=build_replay(package,witness);original.run()
    trace=plain(original.trace)
    first=fixed_feasibility(cfg,network,witness['input'],trace)
    require(first['status']=='FEASIBLE','execution to fixed PI infeasible: '+str(first))
    mapped=deepcopy(trace)
    for k,e in enumerate(mapped):
        e['at']=first['fixed_values']['t_'+str(k)]['value']
        if 'end' in e:e['end']=first['fixed_values']['end_'+str(k)]['value']
        if e['kind']=='truck_unload_started':
            e['quantities']={oid:int(first['fixed_values']['chi_'+str(k)+'_'+oid]['value']) for oid in e['quantities']}
    decoded=deepcopy(witness);decoded['input']['actions']=decode_commands(mapped)
    restored=build_replay(package,decoded);restored.run()
    require(plain(restored.trace)==trace,'decoded PI schedule changed physical trace')
    audit=audit_ledger(cfg,network,decoded['input'],restored.trace)
    require(not audit['errors'],'independent PI ledger failed')
    second=fixed_feasibility(cfg,network,decoded['input'],plain(restored.trace))
    require(second['status']=='FEASIBLE','reverse fixed PI infeasible')
    require(restored.observations()['ledger']==audit['ledger'],'PI/replay objective differs')
    audit['matches_online']=True
    return dict(mapping=dict(same_trace=dict(feasible=True,profit=audit['ledger']['net']),
                             physical_batch_permissions_matched=True,free_reoptimization_counts_as_validation=False),
                fixed_pi_forward=first,fixed_pi_reverse=second,
                fixed_witness=dict(initial=witness['input'],trace=trace,decoded_commands=decoded['input']['actions'],batch_permission='BM'),
                independent_audit=audit,ledger=audit['ledger'])
