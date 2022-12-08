"""Initialization functions to generate a population for GE."""

from .._shared import init_population as _init_population
from . import default_parameters as _dp
from . import init_individual as _init_individual
from . import representation as _representation


def given_genotypes(grammar, parameters=None):
    """Create a population from given genotypes.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``init_pop_given_genotypes`` (`list` of `~.representation.Genotype` objects or data that can be converted to a genotype) : A
          list of genotypes, which are used to initialize the
          individuals of the population. Note that the length
          of this list determines the size of the generated population.

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    """
    return _init_population.given_genotypes(
        grammar, parameters, _dp, _representation, _init_individual
    )


def given_derivation_trees(grammar, parameters=None):
    """Create a population from given derivation trees.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``init_pop_given_derivation_trees`` (`list` of `~alogos._grammar.data_structures.DerivationTree`) : A
          list of derivation trees, which are used to initialize the
          individuals of the population. Note that the length
          of this list determines the size of the generated population.

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    """
    return _init_population.given_derivation_trees(
        grammar, parameters, _dp, _representation, _init_individual
    )


def given_phenotypes(grammar, parameters=None):
    """Create a population from given phenotypes.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``init_pop_given_phenotypes`` (`list` of `str`) : A
          list of phenotypes, which are used to initialize the
          individuals of the population. Note that the length
          of this list determines the size of the generated population.

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    """
    return _init_population.given_phenotypes(
        grammar, parameters, _dp, _representation, _init_individual
    )


def random_genotypes(grammar, parameters=None):
    """Create a population from random genotypes.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``population_size`` (`int`)
        - ``init_pop_unique_genotypes`` (`bool`)
        - ``init_pop_unique_phenotypes`` (`bool`)
        - ``init_pop_unique_max_tries`` (`int`)

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    """
    return _init_population.random_genotypes(
        grammar, parameters, _dp, _representation, _init_individual
    )


def gp_rhh(grammar, parameters=None):
    """Create a population with GP RHH.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``population_size`` (`int`)
        - ``init_pop_unique_genotypes`` (`bool`)
        - ``init_pop_unique_phenotypes`` (`bool`)
        - ``init_pop_unique_max_tries`` (`int`)
        - ``init_pop_gp_rhh_start_depth`` (`int`)
        - ``init_pop_gp_rhh_end_depth`` (`int`)

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    Notes
    -----
    See `~.init_population.gp_rhh`.

    """
    return _init_population.gp_rhh(
        grammar, parameters, _dp, _representation, _init_individual
    )


def pi_rhh(grammar, parameters=None):
    """Create a population with PI RHH.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``population_size`` (`int`)
        - ``init_pop_unique_genotypes`` (`bool`)
        - ``init_pop_unique_phenotypes`` (`bool`)
        - ``init_pop_unique_max_tries`` (`int`)
        - ``init_pop_pi_rhh_start_depth`` (`int`)
        - ``init_pop_pi_rhh_end_depth`` (`int`)

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    Notes
    -----
    See `~.init_population.pi_rhh`.

    """
    return _init_population.pi_rhh(
        grammar, parameters, _dp, _representation, _init_individual
    )


def ptc2(grammar, parameters=None):
    """Create a population with PTC2.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``population_size`` (`int`)
        - ``init_pop_unique_genotypes`` (`bool`)
        - ``init_pop_unique_phenotypes`` (`bool`)
        - ``init_pop_unique_max_tries`` (`int`)

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    Notes
    -----
    See `~.init_population.ptc2`.

    """
    return _init_population.ptc2(
        grammar, parameters, _dp, _representation, _init_individual
    )
