"""Price the four frozen first-action plans using previously trained VFAs.

This is a known-state-only diagnostic, not another policy optimization: it
never regenerates admissions, sortie assignments, routes or base plans and
never reads the realized/resampled suffix into a feature or score.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from e1_sim.information import closed, pickup_deadline_lags
from e1_sim.initializer import digest
from e1_sim.online import evaluate_plan, known_clone
from e1_sim.quantities import plain
from run_feature_screen import write
from run_paired_wait_diagnostic import load_pickle
from unified_vfa_features import FEATURE_NAMES, features
from unified_vfa_policy import canonical_plan_digest, operating_rules, validate_model


ALIASES = ('chain', 'chain_late', 'full')


def score_sites(output, prior, expected_sites=32):
    output, prior = Path(output), Path(prior)
    sites = json.loads((output/'sites.json').read_text(encoding='utf8'))
    if len(sites) != expected_sites:
        raise ValueError('Capture incomplete: expected %s sites, found %s' % (expected_sites,len(sites)))
    models, hashes = {}, {}
    for alias in ALIASES:
        path = prior/'training'/alias/'selected_model.json'
        model = json.loads(path.read_text(encoding='utf8'))
        validate_model(model, 5, (2,5,10), 5)
        models[alias] = model
        hashes[alias] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            feature_set=model['feature_set'], weights=model['weights'], scales=model['scales'],
            selected_iteration=model['selection']['iteration'])
    results = []
    for site in sites:
        bundle = load_pickle(output/'snapshots'/(site['id']+'.pkl.gz'))
        known = known_clone(bundle['state'])
        assert not known._hidden
        assert not any(e[3] in ('request_reveal','revelation_end') for e in known._queue)
        assert set(o['id'] for o in known.initial['orders']) == set(known.orders)
        assert set(bundle['plans']) == {0,2,5,10}
        rules = operating_rules(known.cfg,pickup_deadline_lags(known.public['demand_model']['attributes']))
        assert all(model['operating_rules']==rules for model in models.values())
        row = dict(site_id=site['id'],case=site['case'],at=site['at'],truck=site['truck'],
            station=site['station'],stratum=site['qualification']['stratum'],
            known_state_hash=digest(plain(known.initial)),branches={},models={})
        for h in (0,2,5,10):
            candidate = evaluate_plan(known,bundle['plans'][h],'WK')
            assert candidate['feasible'] and candidate['allowed']
            x = list(features(known,candidate,5))
            row['branches'][str(h)] = dict(J=float(candidate['score']),features=x,
                canonical_plan_hash=canonical_plan_digest(bundle['plans'][h]))
        for alias, model in models.items():
            scores = {}
            for h, branch in row['branches'].items():
                contribution = ([0.]*len(FEATURE_NAMES) if closed(known.public) else
                    [float(w)*float(x)/float(s) for w,x,s in zip(
                        model['weights'],branch['features'],model['scales'])])
                future = sum(contribution)
                scores[h] = dict(J=branch['J'],W=future,score=branch['J']+future,
                                 feature_contributions=contribution)
            for h, price in scores.items():
                price['delta_vs_depart'] = price['score']-scores['0']['score']
            chosen = min((0,2,5,10),key=lambda h:(-scores[str(h)]['score'],h))
            row['models'][alias] = dict(recommended_h=chosen,scores=scores,
                chosen_margin_vs_depart=scores[str(chosen)]['delta_vs_depart'])
        results.append(row)
    summary = {alias:dict(choice_counts=dict(Counter(r['models'][alias]['recommended_h'] for r in results)),
        by_stratum={stratum:dict(Counter(r['models'][alias]['recommended_h']
            for r in results if r['stratum']==stratum)) for stratum in sorted({r['stratum'] for r in results})})
        for alias in ALIASES}
    value = dict(scope='same-known-state frozen-base first-action scores only',
        future_suffix_used=False, new_candidate_search=False, feature_horizon=5,
        feature_names=list(FEATURE_NAMES), model_hashes=hashes,
        tie_break='highest exact score, then smallest interval',
        site_count=len(results),summary=summary,sites=results)
    write(output/'chain_scores.json',value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('../../results/paired_wait_20261128'))
    parser.add_argument('--prior',type=Path,default=Path('../../results/feature_screen_20261128'))
    parser.add_argument('--expected-sites',type=int,default=32)
    args = parser.parse_args()
    result = score_sites(args.output,args.prior,args.expected_sites)
    print(json.dumps(dict(site_count=result['site_count'],summary=result['summary'],
                         output=str(args.output/'chain_scores.json')),ensure_ascii=False))


if __name__ == '__main__':
    main()
