"""Shared helper functions of mapping methods for several systems."""

from ... import exceptions as _exceptions
from ..._grammar import data_structures as _data_structures


def get_derivation_tree(grammar, data):
    """Parse the user-provided data and return a derivation tree if possible."""
    if isinstance(data, _data_structures.DerivationTree):
        dt = data
    elif isinstance(data, str):
        try:
            dt = grammar.parse_string(data)
        except Exception:
            _exceptions.raise_invalid_mapping_data1(data)
    else:
        _exceptions.raise_invalid_mapping_data2(data)
    return dt
