"""Shared initialization functions for several systems."""

from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default


def given_genotypes(grammar, parameters, dp, _representation, _init_individual):
    """Create a population from given genotypes."""
    # Parameter extraction
    init_pop_given_genotypes = _get_given_or_default(
        "init_pop_given_genotypes", parameters, dp
    )

    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    try:
        if not init_pop_given_genotypes:
            _exceptions.raise_no_genotypes_error()
        individuals = []
        for gt in init_pop_given_genotypes:
            parameters["init_ind_given_genotype"] = gt
            ind = _init_individual.given_genotype(grammar, parameters)
            individuals.append(ind)
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_from_gt_error()
    return population


def given_derivation_trees(grammar, parameters, dp, _representation, _init_individual):
    """Create a population from given derivation trees."""
    # Parameter extraction
    init_pop_given_derivation_trees = _get_given_or_default(
        "init_pop_given_derivation_trees", parameters, dp
    )

    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    try:
        if not init_pop_given_derivation_trees:
            _exceptions.raise_no_derivation_trees_error()
        individuals = []
        for dt in init_pop_given_derivation_trees:
            parameters["init_ind_given_derivation_tree"] = dt
            ind = _init_individual.given_derivation_tree(grammar, parameters)
            individuals.append(ind)
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_from_dt_error()
    return population


def given_phenotypes(grammar, parameters, dp, _representation, _init_individual):
    """Create a population from given phenotypes."""
    # Parameter extraction
    init_pop_given_phenotypes = _get_given_or_default(
        "init_pop_given_phenotypes", parameters, dp
    )

    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    try:
        if not init_pop_given_phenotypes:
            _exceptions.raise_no_phenotypes_error()
        individuals = []
        for phe in init_pop_given_phenotypes:
            parameters["init_ind_given_phenotype"] = phe
            ind = _init_individual.given_phenotype(grammar, parameters)
            individuals.append(ind)
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_from_phe_error()
    return population


def random_genotypes(grammar, parameters, dp, _representation, _init_individual):
    """Create a population from random genotypes.

    References
    ----------
    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__

         - 3.1 Random initialisation (RND): "The original implementations of GE
         used a random (RND) initialisation procedure. This simple procedure
         consists of generating random genotype arrays (either binary or integer)
         of a specified length, in a manner similar to random initialisation
         in (fixed-length) genetic algorithms."

    - 2018, Ryan, O'Neill, Collins: `Handbook of Grammatical Evolution
      <https://doi.org/10.1007/978-3-319-78717-6>`__

        - p. 11: "Originally, we used random initialisation for the GE population.
          However, as noted in [18, 26, 86], random initialisation can lead to
          very heavily biased initial populations. [...] What is crucial though,
          is to put some effort into ensuring good variation in that initial
          population, and to avoid simple random initialisation."

    """
    # Parameter extraction
    population_size = _get_given_or_default("init_pop_size", parameters, dp)
    max_tries = _get_given_or_default("init_pop_unique_max_tries", parameters, dp)
    ensure_uniqueness_func = _get_ensure_uniqueness_func(parameters, dp)

    try:
        # Create individuals (and optionally ensure their uniqueness)
        def create_ind_func():
            return _init_individual.random_genotype(grammar, parameters)

        individuals = []
        num_tries = 0
        known = set()
        for _ in range(population_size):
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)

        # Create population
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_rand_gt_error()
    return population


def gp_rhh(grammar, parameters, dp, _representation, _init_individual):
    """Create a population with GP RHH.

    Notes
    -----
    This approach was originally developed in genetic programming (GP),
    where it was termed "ramped half and half" (RHH).
    In grammatical evolution (GE), the adapted but conceptually
    identical approach is known as "sensible initialization".

    Two main variants of it can be found in GE literature:

    - Variant 1: It does not ensure that there are no duplicate trees.
    - Variant 2: It removes duplicate trees by comparing their linear
      genotypes. First the linear encoding uses the smallest possible
      integers to represent the rules for each step. To increase
      variability within the generated genotypes, an "unmod" operator
      is applied to each codon, which uses a random integer from the
      whole possible range, which still represents the same rule as the
      original smallest possible integer.

    References
    ----------
    - 2003, Ryan et al.: `Sensible Initialisation in Chorus
      <https://doi.org/10.1007/3-540-36599-0_37>`__

      - "This paper employs an approach similar to ramped half and half
        for Chorus. The abstract syntax trees used by Koza are replaced
        here by derivation trees, showing the sequence of derivations
        leading from the start symbol to a legal sentence of the
        grammar (a collection of only terminal symbols)."

    - 2016, Fagan et al.: `Exploring position independent
      initialisation in grammatical evolution
      <https://doi.org/10.1109/CEC.2016.7748331>`__

      - "It can be seen from Table II that both traditional Grow and
        traditional Ramped Half-and-Half generate excessive numbers of
        duplicate individuals. This is due to the propensity of the
        original Grow method towards generating small trees.
        With Full and PI-based methods, the creation of duplicate
        individuals reduces markedly as the ramping depth increases."

    """
    # Parameter extraction
    population_size = _get_given_or_default("init_pop_size", parameters, dp)
    max_tries = _get_given_or_default("init_pop_unique_max_tries", parameters, dp)
    start_depth = _get_given_or_default("init_pop_gp_rhh_start_depth", parameters, dp)
    end_depth = _get_given_or_default("init_pop_gp_rhh_end_depth", parameters, dp)
    ensure_uniqueness_func = _get_ensure_uniqueness_func(parameters, dp)
    if parameters is None:
        parameters = dict()

    try:
        # Create the ramp of depths: slight preference for larger values by starting from max
        ramped_depths = _create_ramp(start_depth, end_depth)
        n = len(ramped_depths)

        # Create individuals (and optionally ensure their uniqueness)
        individuals = []
        num_tries = 0
        known = set()
        i = 0
        for i in range(population_size // 2):
            # grow
            def create_ind_func():
                return _init_individual.gp_grow_tree(grammar, parameters)

            parameters["init_ind_gp_grow_max_depth"] = ramped_depths[i % n]
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)

            # full
            def create_ind_func():
                return _init_individual.gp_full_tree(grammar, parameters)

            parameters["init_ind_gp_full_max_depth"] = ramped_depths[i % n]
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)
        if population_size % 2 == 1:
            # grow: the last individual if population size is odd
            def create_ind_func():
                return _init_individual.gp_grow_tree(grammar, parameters)

            parameters["init_ind_gp_grow_max_depth"] = ramped_depths[i % n]
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)

        # Create population
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_gp_rhh_error()
    return population


def pi_rhh(grammar, parameters, dp, _representation, _init_individual):
    """Create a population with PI RHH."""
    # Parameter extraction
    population_size = _get_given_or_default("init_pop_size", parameters, dp)
    max_tries = _get_given_or_default("init_pop_unique_max_tries", parameters, dp)
    start_depth = _get_given_or_default("init_pop_pi_rhh_start_depth", parameters, dp)
    end_depth = _get_given_or_default("init_pop_pi_rhh_end_depth", parameters, dp)
    ensure_uniqueness_func = _get_ensure_uniqueness_func(parameters, dp)
    if parameters is None:
        parameters = dict()

    try:
        # Create the ramp of depths: slight preference for larger values by starting from max
        ramped_depths = _create_ramp(start_depth, end_depth)
        n = len(ramped_depths)

        # Create individuals (and optionally ensure their uniqueness)
        individuals = []
        num_tries = 0
        known = set()
        i = 0
        for i in range(population_size // 2):
            # grow
            def create_ind_func():
                return _init_individual.gp_grow_tree(grammar, parameters)

            parameters["init_ind_pi_grow_max_depth"] = ramped_depths[i % n]
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)

            # full
            def create_ind_func():
                return _init_individual.gp_full_tree(grammar, parameters)

            parameters["init_ind_gp_full_max_depth"] = ramped_depths[i % n]
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)
        if population_size % 2 == 1:
            # grow: the last individual if population size is odd
            def create_ind_func():
                return _init_individual.gp_grow_tree(grammar, parameters)

            parameters["init_ind_pi_grow_max_depth"] = ramped_depths[i % n]
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)

        # Create population
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_pi_rhh_error()
    return population


def ptc2(grammar, parameters, dp, _representation, _init_individual):
    """Create a population with Nicolau's PTC2 method.

    In 2000, Luke created a new population initialization approach
    called "Probabilistic tree-creation 2" (PTC2).
    In 2010, Harper adapted it to grammatical evolution.
    In 2017, Nicolau created a variant of Harper's PTC2 method, which
    is implemented here.

    References
    ----------
    - 2000, Luke: `Two Fast Tree-Creation Algorithms for Genetic Programming
      <https://doi.org/10.1109/4235.873237>`__

        - "PTC1 is a modification of GROW that allows the user to
          provide probabilities of appearance of functions in the tree,
          plus a desired expected tree size, and guarantees that, on
          average, trees will be of that size."

        - "With PTC2, the user provides a probability distribution of
          requested tree sizes. PTC2 guarantees that, once it has picked
          a random tree size from this distribution, it will generate
          and return a tree of that size or slightly larger."

    - 2010, Harper: `GE, explosive grammars and the lasting legacy of bad initialisation
      <https://doi.org/10.1109/CEC.2010.5586336>`__

        - "The PTC2 methodology is extended to GE and found to produce
          a more uniform distribution of parse trees."

        - "If the algorithm is called in a ramped way (i.e. starting
          with a low number of expansions, say 20, and increasing until
          say 240) then a large number of trees of different size and
          shapes will be generated."

    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__:

        - 3.3 Probabilistic tree-creation 2 (PTC2)
        - 3.6 Probabilistic tree-creation 2 with depth limit (PTC2D)

    - 2018, Ryan, O'Neill, Collins: `Handbook of Grammatical Evolution
      <https://doi.org/10.1007/978-3-319-78717-6>`__

        - p. 13: "More recent work on initialisation includes that of
          Nicolau, who demonstrated that across the problems examined
          in their study, a variant of Harper’s PTC2 consistently
          outperforms other initialisations"

    """
    population_size = _get_given_or_default("init_pop_size", parameters, dp)
    max_tries = _get_given_or_default("init_pop_unique_max_tries", parameters, dp)
    start_expansions = _get_given_or_default(
        "init_pop_ptc2_start_expansions", parameters, dp
    )
    end_expansions = _get_given_or_default(
        "init_pop_ptc2_end_expansions", parameters, dp
    )
    ensure_uniqueness_func = _get_ensure_uniqueness_func(parameters, dp)
    if parameters is None:
        parameters = dict()

    try:
        # Create the ramp of expansions: slight preference for larger values by starting from max
        ramped_exp = _create_ramp(start_expansions, end_expansions)
        n = len(ramped_exp)

        # Create individuals (and optionally ensure their uniqueness)
        individuals = []
        num_tries = 0
        known = set()
        for i in range(population_size):

            def create_ind_func():
                return _init_individual.ptc2_tree(grammar, parameters)

            parameters["init_ind_ptc2_max_expansions"] = ramped_exp[i % n]
            ind, num_tries = ensure_uniqueness_func(
                create_ind_func, known, population_size, num_tries, max_tries
            )
            individuals.append(ind)

        # Create population
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_ptc2_error()
    return population


# Helper functions


def _create_ramp(start, end):
    """Create a ramp of values.

    Downstream it can lead to a slight preference for larger values
    because it starts with the maximum and goes towards the minimum.
    The number of individuals does not need to be divisable by the
    length of the list of values.

    """
    if start > end:
        raise ValueError(
            "For creating a ramp, the chosen start value of {} is bigger than the chosen end value of {}.".format(
                start, end
            )
        )
    return list(range(end, start - 1, -1))


def _get_ensure_uniqueness_func(parameters, dp):
    """Return a function that ensures some aspect of a new individual is unique."""
    # Parameter extraction
    unique_phe = _get_given_or_default("init_pop_unique_phenotypes", parameters, dp)
    unique_gen = _get_given_or_default("init_pop_unique_genotypes", parameters, dp)

    # Transformation
    if unique_phe:
        ensure_uniqueness_func = _ind_with_unique_phenotype
    elif unique_gen:
        ensure_uniqueness_func = _ind_with_unique_genotype
    else:
        ensure_uniqueness_func = _ind_without_uniqueness
    return ensure_uniqueness_func


def _ind_without_uniqueness(
    create_ind_func, known, population_size, num_tries, max_tries
):
    """Generate a new individual without ensuring anything is unique."""
    ind = create_ind_func()
    new_num_tries = num_tries + 1
    return ind, new_num_tries


def _ind_with_unique_genotype(
    create_ind_func, known_genotypes, population_size, num_tries, max_tries
):
    """Generate a new individual and ensure its genotype is unique."""
    for new_num_tries in range(num_tries, max_tries + 1):  # noqa: B007
        ind = create_ind_func()
        gen = ind.genotype
        if gen not in known_genotypes:
            known_genotypes.add(gen)
            break
    else:
        _exceptions.raise_init_pop_unique_gen_error(
            len(known_genotypes), population_size, new_num_tries
        )
    return ind, new_num_tries


def _ind_with_unique_phenotype(
    create_ind_func, known_phenotypes, population_size, num_tries, max_tries
):
    """Generate a new individual and ensure its phenotype is unique."""
    for new_num_tries in range(num_tries, max_tries):  # noqa: B007
        ind = create_ind_func()
        phe = ind.phenotype
        if phe not in known_phenotypes:
            known_phenotypes.add(phe)
            break
    else:
        _exceptions.raise_init_pop_unique_phe_error(
            len(known_phenotypes), population_size, new_num_tries
        )
    return ind, new_num_tries
