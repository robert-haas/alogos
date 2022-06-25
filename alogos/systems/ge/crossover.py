from . import representation as _representation
from .. import _shared


def two_point_length_preserving(grammar, genotype1, genotype2, parameters=None):
    return _shared.crossover.two_point_length_preserving(
        grammar, genotype1, genotype2, parameters, _representation)
