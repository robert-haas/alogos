from ..._utilities.parametrization import ParameterCollection as _ParameterCollection


default_parameters = _ParameterCollection({
    # General
    'population_size': 100,
    'max_nodes': 10000,

    # Individual initialization
    'init_ind_operator': 'random_genotype',
    'init_ind_given_genotype': None,
    'init_ind_given_derivation_tree': None,
    'init_ind_given_phenotype': None,
    'init_ind_grow_max_depth': 8,
    'init_ind_full_max_depth': 8,
    'init_ind_ptc2_max_expansions': 50,

    # Population initialization
    'init_pop_operator': 'ptc2',
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
    'mutation_operator': 'subtree_replacement',
    'mutation_subtree_replacement_count': 1,

    # Crossover
    'crossover_operator': 'subtree_exchange',
    'crossover_subtree_replacement_count': 1,

    # Neighborhood
    'neighborhood_operator': 'subtree_replacement',
    'neighborhood_distance': 1,
    'neighborhood_max_size': 100,
    'neighborhood_only_terminals': False,
})


default_parameters_OLD = _ParameterCollection({
    'init_individual_operator': 'random_genotype',
    'init_population_operator': 'random_genotypes',
    'mutation_operator': 'subtree_replacement',
    'crossover_operator': 'subtree_exchange',

    'init_genotype': None,
    'init_genotypes': None,
    'init_derivation_tree': None,
    'init_pop_given_derivation_trees': None,
    'init_phenotype': None,
    'init_pop_given_phenotypes': None,
    'init_pop_random_unique_genotypes': False,
    'init_pop_random_unique_phenotypes': True,
    'init_pop_random_unique_max_tries': 50000,

    'init_grow_max_depth': 8,
    'init_full_max_depth': 8,

    'init_ptc2_max_expansions': 20,
    'init_pop_ptc2_start_expansions': 20,
    'init_pop_ptc2_end_expansions': 240,

    'init_pop_rhh_start_depth': 2,
    'init_pop_rhh_end_depth': 17,
    'init_pop_rhh_with_pi_grow': True,

    'codon_size': 8,
    'genotype_length': 10,

    'population_size': 100,

    'mutation_subtree_replacement_count': 1,

    'max_nodes': 20000,

    'neighborhood_operator': 'node_flip',
    'neighborhood_distance': 1,
    'neighborhood_max_size': 1000,
})
