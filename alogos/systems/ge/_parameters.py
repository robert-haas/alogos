from ..._utilities.parametrization import ParameterCollection as _ParameterCollection


default_parameters = _ParameterCollection({
    # General
    'codon_size': 8,
    'genotype_length': 10,
    'population_size': 100,
    'max_expansions': 1000,
    'max_wraps': 3,

    # Representation

    # Reverse mapping
    'codon_randomization': True,

    # Individual initialization
    'init_ind_operator': 'random_genotype',
    'init_ind_given_genotype': None,
    'init_ind_given_derivation_tree': None,
    'init_ind_given_phenotype': None,
    'init_ind_random_valid_genotype_max_tries': 50000,
    'init_ind_grow_max_depth': 8,
    'init_ind_full_max_depth': 8,
    'init_ind_ptc2_max_expansions': 50,

    # Population initialization
    'init_pop_operator': 'random_genotypes',
    'init_pop_given_genotypes': None,
    'init_pop_given_derivation_trees': None,
    'init_pop_given_phenotypes': None,
    'init_pop_random_unique_genotypes': False,
    'init_pop_random_unique_phenotypes': True,
    'init_pop_random_unique_max_tries': 50000,
    'init_pop_rhh_start_depth': 2,
    'init_pop_rhh_end_depth': 17,
    'init_pop_rhh_with_pi_grow': True,
    'init_pop_ptc2_start_expansions': 20,
    'init_pop_ptc2_end_expansions': 240,

    # Mutation
    'mutation_operator': 'int_replacement_by_probability',
    'mutation_int_replacement_probability': 0.01,
    'mutation_int_replacement_count': 1,

    # Crossover
    'crossover_operator': 'two_point_length_preserving',

    # Neighborhood
    'neighborhood_operator': 'int_replacement',
    'neighborhood_distance': 1,
    'neighborhood_max_size': 1000,
    'neighborhood_only_terminals': False,
})
