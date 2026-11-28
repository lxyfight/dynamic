"""Exact quantity contract; times and money are deliberately not quantized."""
from decimal import Decimal


class InvalidAction(ValueError):
    """A proposed physical action cannot be committed."""


class UnsupportedCase(ValueError):
    """An input requires semantics not implemented by this development stage."""


def quantity(value, *, positive=False):
    # JSON 10.0 is mathematically integral; bool is not a quantity.
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise InvalidAction("quantity must be numeric and integral")
    n = Decimal(str(value))
    if not n.is_finite() or n != n.to_integral_value() or n < int(positive):
        raise InvalidAction("quantity outside the integer domain")
    return int(n)


def dec(value):
    if isinstance(value, bool):
        raise InvalidAction("boolean is not a time or cost")
    n = Decimal(str(value))
    if not n.is_finite():
        raise InvalidAction("nonfinite numeric input")
    return n


def plain(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value
