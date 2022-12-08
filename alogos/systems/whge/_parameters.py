"""Default parameters for WHGE."""

from ..._utilities.parametrization import ParameterCollection as _ParameterCollection


default_parameters = _ParameterCollection(
    {
        # General
        "genotype_length": 1_024,
        "max_expansions": 1_000_000,
        "max_depth": 3,
        # Individual initialization
        "init_ind_operator": "random_genotype",
        "init_ind_given_genotype": None,
        # Population initialization
        "init_pop_operator": "random_genotypes",
        "init_pop_size": 100,
        "init_pop_given_genotypes": None,
        "init_pop_unique_genotypes": False,
        "init_pop_unique_phenotypes": True,
        "init_pop_unique_max_tries": 100_000,
        # Mutation
        "mutation_operator": "bit_flip_by_probability",
        "mutation_bit_flip_probability": 0.01,
        "mutation_bit_flip_count": 2,
        # Crossover
        "crossover_operator": "two_point_length_preserving",
        # Neighborhood
        "neighborhood_operator": "bit_flip",
        "neighborhood_distance": 1,
        "neighborhood_max_size": None,
    }
)
"""Default parameters for WHGE.

These values can be changed to affect the default behavior
of WHGE when used in a search algorithm.
Note that the default values can also be overwritten by passing
parameters to the constructor of a search algorithm.

Parameters
----------
genotype_length : `int`, default=1_024
max_expansions : `int`, default=1_000_000
max_depth : `int`, default=3

init_ind_operator : `str`, default="random_genotype"
    Possible values:

    - ``"given_genotype"``
    - ``"random_genotype"``
init_ind_given_genotype : `~.representation.Genotype`, default=`None`

init_pop_operator : `str`, default="random_genotypes"
    Possible values:

    - ``"given_genotypes"``
    - ``"random_genotypes"``
init_pop_size : `int`, default=100
    This parameter is used by population initialization operators.
    Caution: Search methods like `~alogos.EvolutionaryAlgorithm` come
    with the parameter ``population_size``, which determines the size
    of the population both during initialization and during the search,
    therefore overwriting the value of ``init_pop_size``.
init_pop_given_genotypes : `list` of `~.representation.Genotype`, default=`None`
init_pop_unique_genotypes : `bool`, default=False
init_pop_unique_phenotypes : `bool`, default=True
init_pop_unique_max_tries : `int`, default=100000

mutation_operator : `str`, default="bit_flip_by_probability"
    Possible values:

    - ``"bit_flip_by_probability"``
    - ``"bit_flip_by_count"``
mutation_bit_flip_probability : `float`, default=0.01
mutation_bit_flip_count : `int`, default=2

crossover_operator : `str`, default="two_point_length_preserving"
    Possible values:

    - ``"two_point_length_preserving"``

neighborhood_operator : `str`, default="bit_flip"
    Possible values:

    - ``"bit_flip"``
neighborhood_distance : `int`, default=1
neighborhood_max_size : `int`, default=100

"""
