from . import default_parameters as _dp
from . import representation as _representation
from .. import _shared


def int_replacement_by_probability(grammar, genotype, parameters=None):
    return _shared.mutation.flat_int_replacement_by_prob(
        grammar, genotype, parameters, _dp, _representation)


def int_replacement_by_count(grammar, genotype, parameters=None):
    return _shared.mutation.flat_int_replacement_by_count(
        grammar, genotype, parameters, _dp, _representation)
