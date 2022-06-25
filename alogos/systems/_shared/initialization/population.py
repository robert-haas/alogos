from ...._utilities.parametrization import get_given_or_default as _get_given_or_default
from .... import exceptions as _exceptions


def given_genotypes(grammar, parameters, dp, _representation, _individual):
    """Create a population from given genotypes.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    genotypes : list of genotypes, where each genotype is a list of int
    parameters : TODO

    """
    # Parameter extraction
    init_pop_given_genotypes = _get_given_or_default('init_pop_given_genotypes', parameters, dp)

    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    try:
        if not init_pop_given_genotypes:
            _exceptions.raise_no_genotypes_error()
        individuals = []
        for gt in init_pop_given_genotypes:
            parameters['init_ind_given_genotype'] = gt
            ind = _individual.given_genotype(grammar, parameters)
            individuals.append(ind)
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_from_gt_error()
    return population


def given_derivation_trees(grammar, parameters, dp, _representation, _individual):
    """Create a population from given derivation trees.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    derivatoin_trees : list of derivation trees TODO
    parameters : TODO

    """
    # Parameter extraction
    init_pop_given_derivation_trees = _get_given_or_default(
        'init_pop_given_derivation_trees', parameters, dp)

    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    try:
        if not init_pop_given_derivation_trees:
            _exceptions.raise_no_derivation_trees_error()
        individuals = []
        for dt in init_pop_given_derivation_trees:
            parameters['init_ind_given_derivation_tree'] = dt
            ind = _individual.given_derivation_tree(grammar, parameters)
            individuals.append(ind)
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_from_dt_error()
    return population


def given_phenotypes(grammar, parameters, dp, _representation, _individual):
    """Create a population from given phenotypes.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    phenotypes : list of phenotypes, where each phenotype is a string
    parameters : TODO

    """
    # Parameter extraction
    init_pop_given_phenotypes = _get_given_or_default('init_pop_given_phenotypes', parameters, dp)

    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    try:
        if not init_pop_given_phenotypes:
            _exceptions.raise_no_phenotypes_error()
        individuals = []
        for phe in init_pop_given_phenotypes:
            parameters['init_ind_given_phenotype'] = phe
            ind = _individual.given_phenotype(grammar, parameters)
            individuals.append(ind)
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_from_phe_error()
    return population


def random_genotypes(grammar, parameters, dp, _representation, _individual):
    """Create a population from random genotypes.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

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
    population_size = _get_given_or_default('population_size', parameters, dp)
    unique_phenotypes = _get_given_or_default('init_pop_random_unique_phenotypes', parameters, dp)
    unique_genotypes = _get_given_or_default('init_pop_random_unique_genotypes', parameters, dp)
    max_tries = _get_given_or_default('init_pop_random_unique_max_tries', parameters, dp)

    # Transformation
    try:
        individuals = []
        # - Unique phenotypes
        if unique_phenotypes:
            unique_phenotypes = set()
            for _ in range(max_tries):
                ind = _individual.random_genotype(grammar, parameters)
                phe = ind.phenotype
                if phe not in unique_phenotypes:
                    unique_phenotypes.add(phe)
                    individuals.append(ind)
                if len(individuals) == population_size:
                    break
            else:
                message = (
                    'Failed to find enough unique phenotypes after {} tries.'.format(max_tries))
                raise ValueError(message)
        # - Unique genotypes
        elif unique_genotypes:
            unique_genotypes = set()
            for _ in range(max_tries):
                ind = _individual.random_genotype(grammar, parameters)
                gen = str(ind.genotype)
                if gen not in unique_genotypes:
                    unique_genotypes.add(gen)
                    individuals.append(ind)
                if len(individuals) == population_size:
                    break
            else:
                message = (
                    'Failed to find enough unique genotypes after {} tries.'.format(max_tries))
                raise ValueError(message)
        # - No filter
        else:
            individuals = [_individual.random_genotype(grammar, parameters)
                           for _ in range(population_size)]
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_rand_gt_error()
    return population


def rhh(grammar, parameters, dp, _representation, _individual):
    """Use "ramped-half and half" from GP, also known as "sensible initialization" in GE.

    There seem to be two variants in GE literature:

    - Variant 1 does not ensure that there are no duplicate trees.
    - Variant 2 removes duplicates by comparing their linear genotypes before
      applying an "unmod" operator to generate more variability in the codons.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

    Returns
    -------
    TODO

    References
    ----------
    TODO - ref to docs

    """
    # TODO: parameters to control whether only unique genotypes/phenotypes/not, as in random

    # Parameter extraction
    population_size = _get_given_or_default('population_size', parameters, dp)
    start_depth = _get_given_or_default('init_pop_rhh_start_depth', parameters, dp)
    end_depth = _get_given_or_default('init_pop_rhh_end_depth', parameters, dp)
    with_pi_grow = _get_given_or_default('init_pop_rhh_with_pi_grow', parameters, dp)

    # Argument processing
    if parameters is None:
        parameters = dict()
    chosen_grow_method = _individual.pi_grow_tree if with_pi_grow else _individual.grow_tree

    try:
        # Create the ramp of depths: slight preference for larger values by starting from max
        ramped_depths = _create_ramp(start_depth, end_depth)

        # Create individuals
        n = len(ramped_depths)
        individuals = []
        i = 0
        for i in range(population_size // 2):
            # grow
            parameters['init_ind_grow_max_depth'] = ramped_depths[i % n]
            ind = chosen_grow_method(grammar, parameters)
            individuals.append(ind)
            # full
            parameters['init_ind_full_max_depth'] = ramped_depths[i % n]
            ind = _individual.full_tree(grammar, parameters)
            individuals.append(ind)
        if population_size % 2 == 1:
            # grow: the last individual if population size is odd
            parameters['init_ind_grow_max_depth'] = ramped_depths[(i + 1) % n]
            ind = chosen_grow_method(grammar, parameters)
            individuals.append(ind)

        # Create population
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_rhh_error()
    return population


def ptc2(grammar, parameters, dp, _representation, _individual):
    """Use "Probabilistic tree-creation 2" (PTC2).

    TODO: Description

    References
    ----------
    - 2000, Luke: `Two Fast Tree-Creation Algorithms for Genetic Programming
      <https://doi.org/10.1109/4235.873237>`__

        - "PTC1 is a modification of GROW that allows the user to provide probabilities
          of appearance of functions in the tree, plus a desired expected tree size,
          and guarantees that, on average, trees will be of that size."

        - "With PTC2, the user provides a probability distribution of requested tree sizes.
          PTC2 guarantees that, once it has picked a random tree size from this distribution,
          it will generate and return a tree of that size or slightly larger."

    - 2010, Harper: `GE, explosive grammars and the lasting legacy of bad initialisation
      <https://doi.org/10.1109/CEC.2010.5586336>`__

        - "The PTC2 methodology is extended to GE and found to produce a more
          uniform distribution of parse trees."

        - "If the algorithm is called in a ramped way (i.e. starting with a low number
          of expansions, say 20, and increasing until say 240) then a large number of
          trees of different size and shapes will be generated."

    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__:

        - 3.3 Probabilistic tree-creation 2 (PTC2)
        - 3.6 Probabilistic tree-creation 2 with depth limit (PTC2D)

    - 2018, Ryan, O'Neill, Collins: `Handbook of Grammatical Evolution
      <https://doi.org/10.1007/978-3-319-78717-6>`__

        - p. 13: "More recent work on initialisation includes that of Nicolau,
          who demonstrated that across the problems examined in their study,
          a variant of Harper’s PTC2 consistently outperforms other initialisations"

    """
    # Parameter extraction
    population_size = _get_given_or_default('population_size', parameters, dp)
    start_expansions = _get_given_or_default('init_pop_ptc2_start_expansions', parameters, dp)
    end_expansions = _get_given_or_default('init_pop_ptc2_end_expansions', parameters, dp)

    # Argument processing
    if parameters is None:
        parameters = dict()

    try:
        # Create the ramp of expansions
        ramped_exp = _create_ramp(start_expansions, end_expansions)

        # Create individuals
        n = len(ramped_exp)
        individuals = []
        for i in range(population_size):
            parameters['init_ind_ptc2_max_expansions'] = ramped_exp[i % n]
            ind = _individual.ptc2_tree(grammar, parameters)
            individuals.append(ind)

        # Create population
        population = _representation.Population(individuals)
    except Exception:
        _exceptions.raise_init_pop_ptc2_error()
    return population


def _create_ramp(start, end):
    """Create a ramp of values with slight preference for larger ones by starting from max."""
    return list(range(end, start-1, -1))
