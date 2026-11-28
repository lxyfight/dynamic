"""Public conditional release sampler; owns no operating-world references."""
from decimal import ROUND_FLOOR
from random import Random
from .kernel import require
from .quantities import dec, quantity


def sample_releases(now, public, seed):
    from .information import unknown,sample_release_rows,expected_arrivals
    if unknown(public):
        rows=sample_release_rows(public,now,Random(seed))
        return dict(orders=[dict(**r,forecast_only=True) for r in rows],
                    sampled_counts_by_city={c:sum(r['city']==c for r in rows) for c in public['demand_model']['rates_by_city']},
                    expected_remaining_by_city={c:expected_arrivals(public,c,now) for c in public['demand_model']['rates_by_city']},
                    realized_counts_available=False)
    require(public['base_release_distribution']=='independent_discrete_uniform', 'unsupported release distribution')
    require(public['all_reveals_at_start_processed'] is True, 'process simultaneous reveals first')
    lo,hi=public['base_support']
    lo,hi=quantity(lo),quantity(hi)
    require(lo<=hi,'invalid release support')
    lo=max(lo,int(dec(now).to_integral_value(rounding=ROUND_FLOOR))+1)
    counts={}
    for city,total in sorted(public['total_by_city'].items()):
        total=quantity(total); seen=quantity(public['revealed_by_city'][city])
        require(seen<=total,'revealed count exceeds total')
        counts[city]=total-seen
    require(counts==public['remaining_by_city'] and sum(counts.values())==public['future_count'], 'inconsistent public counts')
    require(lo<=hi or sum(counts.values())==0,'positive remaining demand outside release support')
    rng=Random(seed)
    rows=[dict(city=city,release=rng.randint(lo,hi),forecast_only=True)
          for city,n in counts.items() for _ in range(n)]
    rows.sort(key=lambda r:(r['release'],r['city']))
    return dict(counts_by_city=counts,release_support=[lo,hi],orders=rows,
                expected_arrivals_at_next=[n/(hi-lo+1) if lo<=hi else 0 for n in counts.values()])
