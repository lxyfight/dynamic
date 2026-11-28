"""Complete current Myopic/scalar-VFA execution with a corrected common I0.

Only a fresh, unstarted instance may enter this initialization interface.
Online non-interruption and all commitments remain enforced by the kernel.
Pass theta=0 for Myopic; use an explicitly selected training-seed coefficient
for VFA. This entry point does not select a coefficient from test outcomes.
"""
from e1_sim.kernel import require
from e1_sim.quantities import dec
from profit_dispatch_stagger import stagger
from recovery_opportunity import run

def solve(instance,*,theta,initial_strategy='fixed'):
    raw=instance['input']
    require(dec(raw['start_time'])==0,'initialization requires a fresh instance')
    require(all(t['location']=='depot' and t['active_operation'] is None for t in raw['trucks']),
            'cannot stagger a truck whose operation has started')
    require(all(u['location']==u['home'] and u['active_operation'] is None and not u['onboard_weight'] for u in raw['drones']),
            'cannot reinitialize a committed drone operation')
    require(initial_strategy in ('fixed','stagger'),'unknown initial strategy')
    prepared=stagger(instance) if initial_strategy=='stagger' else instance
    result=run(prepared,theta)
    result.update(initial_rule=initial_strategy,initial_stagger=prepared.get('initial_stagger'),
        input_hash=prepared['input_hash'],i0_hash=prepared['i0_hash'],original_input_hash=instance['input_hash'])
    return result
