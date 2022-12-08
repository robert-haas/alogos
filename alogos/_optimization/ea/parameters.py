"""Default parameters for an evolutionary algorithm."""

from ..._utilities.parametrization import ParameterCollection as _ParameterCollection


default_parameters = _ParameterCollection(
    dict(
        # General
        population_size=100,
        offspring_size=100,
        verbose=False,
        # Storage
        database_on=False,
        database_location=":memory:",
        # Operators
        generation_model="overlapping",
        parent_selection_operator="truncation",
        parent_selection_tournament_size=3,
        parent_selection_rank_slope=1.5,
        survivor_selection_operator="tournament",
        survivor_selection_tournament_size=3,
        survivor_selection_rank_slope=1.5,
        # Evaluation
        gen_to_phe_cache_lookup_on=True,
        gen_to_phe_cache_size=200,
        phe_to_fit_cache_lookup_on=True,
        phe_to_fit_cache_size=200,
        phe_to_fit_database_lookup_on=True,
        # Termination
        max_generations=None,
        max_fitness_evaluations=None,
        max_runtime_in_seconds=None,
        max_or_min_fitness=None,
    )
)
"""Default parameters for an evolutionary algorithm.

Parameters
----------
population_size : `int`, default=100
    This is the population size of the main population during
    initialization and during the run in surivor selection.
offspring_size : `int`, default=100
    This is the size of the offspring population used in
    parent selection, crossover and mutation.
verbose : `bool` or `int`, default=False
    Possible values:

    - If ``False`` or ``0``, no output is printed.
    - If ``True`` or ``1``, concise status messages about the run are
      printed.
    - If ``2`` or larger, detailed status messages about the run are
      printed.

database_on : `bool`, default=False
    If ``True``, data created during the run is stored in a SQLite
    database, which slows down the algorithm but allows later analysis
    and visualization via the ``database`` attribute of the algorithm
    object.
database_location : `str`, default=":memory:"
    Filepath for the SQLite database.
    Special case: ``":memory:"`` leads to the use of a faster but
    impermanent in-memory database instead of creating a file on disk.

generation_model : `str`, default="overlapping"
    Choice of the pool of individuals that the survivor selection
    operator acts on.

    Possible values:

    - ``"overlapping"``: Survivors are selected from the combined
      parent and offspring population.
    - ``"non_overlapping"``: Survivors are selected only from the
      offspring population. The individuals of the parent population
      are lost if they are not also part of the offspring population.
parent_selection_operator : `str`, default="truncation"
    Parent selection operator. The available options put a different
    selection pressure on the individuals of the search and therefore
    can lead to faster or slower convergence, which in turn influences
    how likely it is to get stuck in a local optimium.

    Possible values:

    - ``"uniform"``: Random choice of individuals independent of their
      fitness.
    - ``"truncation"``: Deterministically uses the best individuals.
    - ``"tournament"``: Randomly chooses some number of individuals
      determined by ``parent_selection_tournament_size`` and lets them
      compete for being selected by simply chosing the one with the best
      fitness.
    - ``"rank_proportional"``: Randomly choose individuals from a
      probability distribution calculated by the rank of each
      individual.
    - ``"fitness_proportional"``: Randomly choose individuals from a
      probability distribution calculated by the fitness of each
      individual. High fitness individuals can have a very high chance
      of being selected, possibly leading to premature convergence.

parent_selection_tournament_size : `int`, default=3
    Number of individuals sampled in each tournament. A higher value
    means higher selection pressure, because individuals with low
    fitness have a smaller chance to win in any tournament.
parent_selection_rank_slope : `float`, default=1.5
    Slope used for linear scaling in rank-proportional selection.
    A higher value means that individuals with a good rank get a
    higher probability of being selected and therefore higher
    selection pressure.
survivor_selection_operator : `str`, default="tournament"
    Survivor selection operator. See description and options for
    ``parent_selection_operator``, because the same methods are
    available for parent and survivor selection, though they may
    act on a different number of individuals.
survivor_selection_tournament_size : `int`, default=3
    See description for ``parent_selection_tournament_size``.
survivor_selection_rank_slope : `float`, default=1.5
    See description for ``parent_selection_rank_slope``.

gen_to_phe_cache_lookup_on : `bool`, default=True
    If ``True``, results of genotype-to-phenotype mappings are stored in
    a cache and can be reused instead of being recalculated when a the
    same genotype appears again.
gen_to_phe_cache_size : `int`, default=200
    Number of entries that can be stored in the LRU cache for
    genotype-to-phenotype mappings.
phe_to_fit_cache_lookup_on : `bool`, default=True
    If ``True``, results of phenotype-to-fitness mappings are stored in
    a cache and can be reused instead of being recalculated when a the
    same phenotype appears again. If the objective function is
    computationally demanding this can reduce the search time
    considerably!
phe_to_fit_cache_size : `int`, default=200
    Number of entries that can be stored in the LRU cache for
    phenotype-to-fitness mappings.
phe_to_fit_database_lookup_on : `bool`, default=True
    If ``True``, results of phenotype-to-fitness mappings are looked up
    in the database if 1) a database is active and 2) the phenotype is
    not present in the cache or there is no cache.

max_generations : `int`, default=None
    Stop criterion that halts the search after a given number of
    generations.
max_fitness_evaluations : `int`, default=None
    Stop criterion that halts the search after a given number of
    fitness evaluations with the objective function. Note that the
    actual number of fitness evaluations can be slightly higher,
    because all individuals of a generation are evaluated before
    halting.
max_runtime_in_seconds : `int`, default=None
    Stop criterion that halts the search after a given number of
    seconds.
max_or_min_fitness : `float`, default=None
    Stop criterion that halts the search if the fitness of the best
    individual exceeds a given threshold. If the objective is
    minimization, the search halts when the fitness gets lower than
    the given value. If the objective is maximization, the search halts
    when the fitness gets higher than the given value.

"""
