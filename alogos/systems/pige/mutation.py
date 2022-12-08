"""Mutation functions for piGE."""

from .. import _shared
from . import default_parameters as _dp
from . import representation as _representation


def int_replacement_by_probability(grammar, genotype, parameters=None):
    """Generate a new genotype by replacing int values by chance.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``mutation_int_replacement_probability`` (`int`) : Probability
          of changing a codon.
        - ``codon_size`` (`int`) : Number of bits in a codon.

    Returns
    -------
    genotype : `~.representation.Genotype`
        Mutated genotype.

    """
    return _shared.mutation.flat_int_replacement_by_prob(
        grammar, genotype, parameters, _dp, _representation
    )


def int_replacement_by_count(grammar, genotype, parameters=None):
    """Generate a new genotype by replacing a certain number of values.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``mutation_int_replacement_count`` (`int`) : Total number
          of codons that shall be replaced.
        - ``codon_size`` (`int`) : Number of bits in a codon.

    Returns
    -------
    genotype : `~.representation.Genotype`
        Mutated genotype.

    """
    return _shared.mutation.flat_int_replacement_by_count(
        grammar, genotype, parameters, _dp, _representation
    )
