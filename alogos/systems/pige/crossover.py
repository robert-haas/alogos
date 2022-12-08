"""Crossover functions for piGE."""

from .. import _shared
from . import representation as _representation


def two_point_length_preserving(grammar, genotype1, genotype2, parameters=None):
    """Generate new piGE genotypes by exchanging sequence parts.

    Select two random, but equally long subsequences in the two
    piGE genotypes and exchange them.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype1 : `~.representation.Genotype`
        Genotype of the first parent.
    genotype2 : `~.representation.Genotype`
        Genotype of the second parent.
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        No keyword-value pairs are considered by this function.
        This argument is only available to have a consistent interface.

    Returns
    -------
    genotype1 : `~.representation.Genotype`
        Genotype of the first child.
    genotype2 : `~.representation.Genotype`
        Genotype of the second child.

    """
    return _shared.crossover.two_point_length_preserving(
        grammar, genotype1, genotype2, parameters, _representation
    )
