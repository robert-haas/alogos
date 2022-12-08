"""Default parameters for GE."""

from ..._utilities.parametrization import ParameterCollection as _ParameterCollection


default_parameters = _ParameterCollection(
    {
        # General
        "genotype_length": 10,
        "codon_size": 8,
        "max_wraps": 3,
        "max_expansions": 1_000_000,
        # Reverse mapping
        "codon_randomization": True,
        # Individual initialization
        "init_ind_operator": "random_genotype",
        "init_ind_given_genotype": None,
        "init_ind_given_derivation_tree": None,
        "init_ind_given_phenotype": None,
        "init_ind_random_valid_genotype_max_tries": 1_000,
        "init_ind_gp_grow_max_depth": 8,
        "init_ind_gp_full_max_depth": 8,
        "init_ind_pi_grow_max_depth": 8,
        "init_ind_ptc2_max_expansions": 50,
        # Population initialization
        "init_pop_operator": "random_genotypes",
        "init_pop_size": 100,
        "init_pop_given_genotypes": None,
        "init_pop_given_derivation_trees": None,
        "init_pop_given_phenotypes": None,
        "init_pop_gp_rhh_start_depth": 2,
        "init_pop_gp_rhh_end_depth": 17,
        "init_pop_pi_rhh_start_depth": 2,
        "init_pop_pi_rhh_end_depth": 17,
        "init_pop_ptc2_start_expansions": 20,
        "init_pop_ptc2_end_expansions": 240,
        "init_pop_unique_genotypes": False,
        "init_pop_unique_phenotypes": True,
        "init_pop_unique_max_tries": 100_000,
        # Mutation
        "mutation_operator": "int_replacement_by_probability",
        "mutation_int_replacement_probability": 0.01,
        "mutation_int_replacement_count": 1,
        # Crossover
        "crossover_operator": "two_point_length_preserving",
        # Neighborhood
        "neighborhood_operator": "int_replacement",
        "neighborhood_distance": 1,
        "neighborhood_max_size": 1_000,
        "neighborhood_only_terminals": False,
    }
)
"""Default parameters for GE.

These values can be changed to affect the default behavior
of GE when used in a search algorithm.
Note that the default values can also be overwritten by passing
parameters to the constructor of a search algorithm.

Parameters
----------
genotype_length : `int`, default=10
codon_size : `int`, default=8
max_wraps : `int`, default=3
max_expansions : `int`, default=1_000_000

init_ind_operator : `str`, default="random_genotype"
    Possible values:

    - ``"given_genotype"``
    - ``"given_derivation_tree"``
    - ``"given_phenotype"``
    - ``"random_genotype"``
    - ``"random_valid_genotype"``
    - ``"gp_grow_tree"``
    - ``"pi_grow_tree"``
    - ``"gp_full_tree"``
    - ``"ptc2_tree"``
init_ind_given_genotype : `~.representation.Genotype`, default=`None`
init_ind_given_derivation_tree : `~alogos._grammar.data_structures.DerivationTree`, default=`None`
init_ind_given_phenotype : `str`, default=`None`
init_ind_random_valid_genotype_max_tries : `int`, default=1_000
init_ind_gp_grow_max_depth : `int`, default=8
init_ind_gp_full_max_depth : `int`, default=8
init_ind_pi_grow_max_depth : `int`, default=8
init_ind_ptc2_max_expansions : `int`, default=50

init_pop_operator : `str`, default="random_genotype"
    Possible values:

    - ``"given_genotypes"``
    - ``"given_derivation_trees"``
    - ``"given_phenotypes"``
    - ``"random_genotypes"``
    - ``"gp_rhh"``
    - ``"pi_rhh"``
    - ``"ptc2"``
init_pop_size : `int`, default=100
    This parameter is used by population initialization operators.
    Caution: Search methods like `~alogos.EvolutionaryAlgorithm` come
    with the parameter ``population_size``, which determines the size
    of the population both during initialization and during the search,
    therefore overwriting the value of ``init_pop_size``.
init_pop_given_genotypes : `list` of `~.representation.Genotype`, default=`None`
init_pop_given_derivation_trees : `list` of `~alogos._grammar.data_structures.DerivationTree`, default=`None`
init_pop_given_phenotypes : `list` of `str`, default=`None`
init_pop_gp_rhh_start_depth : `int`, default=2
init_pop_gp_rhh_end_depth : `int`, default=17
init_pop_pi_rhh_start_depth : `int`, default=2
init_pop_pi_rhh_end_depth : `int`, default=17
init_pop_ptc2_start_expansions : `int`, default=20
init_pop_ptc2_end_expansions : `int`, default=240
init_pop_unique_genotypes : `bool`, default=False
init_pop_unique_phenotypes : `bool`, default=True
init_pop_unique_max_tries : `int`, default=100000

mutation_operator : `str`, default="int_replacement_by_probability"
    Possible values:

    - ``"int_replacement_by_probability"``
    - ``"int_replacement_by_count"``
mutation_int_replacement_probability : `float`, default=0.01
mutation_int_replacement_count : `int`, default=1

crossover_operator : `str`, default="two_point_length_preserving"
    Possible values:

    - ``"two_point_length_preserving"``

neighborhood_operator : `str`, default="int_replacement"
    Possible values:

    - ``"int_replacement"``
neighborhood_distance : `int`, default=1
neighborhood_max_size : `int`, default=100
neighborhood_only_terminals : `bool`, default=False

"""
