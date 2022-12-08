"""Neighborhood functions to generate nearby genotypes for WHGE."""

from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import default_parameters as _default_parameters
from . import representation as _representation


def bit_flip(grammar, genotype, parameters=None):
    """Generate nearby genotypes by flipping n bits.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``neighborhood_distance`` (`int`) : The distance from the
          original genotype to a new genotype in terms of replaced
          int codons.
        - ``neighborhood_max_size`` (`int`) : Maximum number of
          neighbor genotypes to generate.

    Returns
    -------
    neighbors : `list` of `~.representation.Genotype` objects

    """
    # Parameter extraction
    distance = _get_given_or_default(
        "neighborhood_distance", parameters, _default_parameters
    )
    max_size = _get_given_or_default(
        "neighborhood_max_size", parameters, _default_parameters
    )

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Get alternative choices per position
    num_choices_per_pos = [1 for _ in range(len(genotype))]

    # Generate combinations
    combinations = _shared.neighborhood.generate_combinations(
        num_choices_per_pos, distance, max_size
    )
    for comb in combinations:
        print(" ", comb)

    # Construct neighborhood genotypes from combinations
    nbrs = []
    for comb in combinations:
        gen = genotype.copy()
        for i, val in enumerate(comb):
            if val > 0:
                gen.data[i] = not gen.data[i]
        nbrs.append(gen)
    return nbrs
