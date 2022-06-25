from ..._utilities.parametrization import ParameterCollection as _ParameterCollection


default_parameters = _ParameterCollection(dict(
    # General
    population_size=100,
    offspring_size=100,
    verbose=False,

    # Storage
    database_on = False,
    database_location=':memory:',

    # Operators
    parent_selection_operator='truncation',
    parent_selection_tournament_size=3,
    parent_selection_rank_slope=1.5,
    survivor_selection_operator='tournament',
    survivor_selection_pooling='overlapping',
    survivor_selection_tournament_size=3,
    survivor_selection_rank_slope=1.5,
    elitism_on=False,

    # Evaluation
    gen_to_phe_cache_lookup_on=True,
    gen_to_phe_cache_size=100,

    phe_to_fit_database_lookup_on=True,
    phe_to_fit_cache_lookup_on=True,
    phe_to_fit_cache_size=100,

    # Termination
    max_generations=None,
    max_fitness_evaluations=None,
    max_runtime_in_seconds=None,
    max_or_min_fitness=None,
))
