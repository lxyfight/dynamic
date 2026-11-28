"""Explicit fee schema; historical E1 fixtures retain first-use accounting."""
from copy import deepcopy
from .quantities import dec, InvalidAction


def fee_key(cfg):
    keys = [k for k in ('drone_sortie_cost', 'drone_first_use_cost') if k in cfg]
    if len(keys) != 1:
        raise InvalidAction('exactly one drone fixed-fee schema is required')
    value = dec(cfg[keys[0]])
    if not value.is_finite() or value < 0:
        raise InvalidAction('invalid drone fixed fee')
    return keys[0]


def per_sortie(cfg):
    return fee_key(cfg) == 'drone_sortie_cost'


def departure_cost(cfg, already_used=False):
    key = fee_key(cfg)
    return dec(cfg[key]) if key == 'drone_sortie_cost' or not already_used else dec(0)


def current_configuration(legacy, fee=25):
    """Build new study settings. Persisted historical scenarios keep their own times."""
    cfg = deepcopy(legacy)
    cfg.pop('drone_first_use_cost', None)
    cfg['drone_sortie_cost'] = fee
    cfg['truck_handling_setup'] = 0
    fee_key(cfg)
    return cfg
