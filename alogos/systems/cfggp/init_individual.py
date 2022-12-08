"""Initialization functions to create a CFG-GP individual."""

from .._shared import init_individual as _init_individual
from . import default_parameters as _dp
from . import mapping as _mapping
from . import representation as _representation


def given_genotype(grammar, parameters=None):
    """Create an individual from a given genotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``init_ind_given_genotype`` (`~.representation.Genotype`) : A
          genotype or data that can be converted to a genotype.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    Notes
    -----
    The genotype is converted to a derivation tree and phenotype with
    the `~.mapping.forward` mapping function of this system.

    """
    return _init_individual.given_genotype(
        grammar, parameters, _dp, _representation, _mapping
    )


def given_derivation_tree(grammar, parameters=None):
    """Create an individual from a given derivation tree.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``init_ind_given_derivation_tree`` (`~alogos._grammar.data_structures.DerivationTree`): A
          derivation tree, which can be created with a `~alogos.Grammar`
          by using `~alogos.Grammar.parse_string` or
          `~alogos.Grammar.generate_derivation_tree`.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    Notes
    -----
    The leaf nodes of the derivation tree are read to create the
    phenotype.
    The phenotype is converted to a genotype with
    the `~.mapping.reverse` mapping function of this system.

    """
    return _init_individual.given_derivation_tree(
        grammar, parameters, _dp, _representation, _mapping
    )


def given_phenotype(grammar, parameters=None):
    """Create an individual from a given phenotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``init_ind_given_phenotype`` (`str`) : A phenotype, i.e. a
          string which is part of the grammar's language.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    Notes
    -----
    The phenotype is converted to a genotype and derivation tree with
    the `~.mapping.reverse` mapping function of this system.

    """
    return _init_individual.given_phenotype(
        grammar, parameters, _dp, _representation, _mapping
    )


def random_genotype(grammar, parameters=None):
    """Create an individual from a random genotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        No keyword-value pairs are considered by this function.
        This argument is only available to have a consistent interface.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    """
    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    random_genotype = _representation.Genotype(grammar.generate_derivation_tree())
    parameters["init_ind_given_genotype"] = random_genotype
    return given_genotype(grammar, parameters)


def gp_grow_tree(grammar, parameters=None):
    """Create an individual from a random tree grown."""
    return _init_individual.gp_grow_tree(
        grammar, parameters, _dp, _representation, _mapping
    )


def pi_grow_tree(grammar, parameters=None):
    """Create an individual from a random tree grown in a position-independently fashion."""
    return _init_individual.pi_grow_tree(
        grammar, parameters, _dp, _representation, _mapping
    )


def gp_full_tree(grammar, parameters=None):
    """Create an individual from a random tree that is grown fully to a maximum depth."""
    return _init_individual.gp_full_tree(
        grammar, parameters, _dp, _representation, _mapping
    )


def ptc2_tree(grammar, parameters=None):
    """Create an individual from a tree grown with Nicolau's "PTC2".

    The original PTC2 method for growing random trees was invented by
    Sean Luke in 2000. Some slightly modified variants were
    introduced later by other authors.
    This function implements a PTC2 variant described by Miguel Nicolau
    in 2017 in section "3.3 Probabilistic tree-creation 2 (PTC2)" of
    the publication. It restricts tree size not with a maximum tree
    depth but rather with a maximum number of expansions and if
    possible remains strictly below this limit.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``init_ind_ptc2_max_expansions`` (`int`): The maximum number
          of nonterminal expansions that may be used to grow the tree.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    Notes
    -----
    The leaf nodes of the derivation tree are read to create the
    phenotype.
    The phenotype is converted to a genotype with
    the `~.mapping.reverse` mapping function of this system.

    References
    ----------
    - 2000, Luke: `Two Fast Tree-Creation Algorithms for Genetic
      Programming <https://doi.org/10.1109/4235.873237>`__

        - "PTC1 is a modification of GROW that allows the user to
          provide probabilities of appearance of functions in the tree,
          plus a desired expected tree size, and guarantees that, on
          average, trees will be of that size."

        - "With PTC2, the user provides a probability distribution of
          requested tree sizes. PTC2 guarantees that, once it has picked
          a random tree size from this distribution, it will generate
          and return a tree of that size or slightly larger."

    - 2010, Harper: `GE, explosive grammars and the lasting legacy of
      bad initialisation
      <https://doi.org/10.1109/CEC.2010.5586336>`__

        - "The PTC2 methodology is extended to GE and found to produce
          a more uniform distribution of parse trees."

        - "If the algorithm is called in a ramped way (i.e. starting
          with a low number of expansions, say 20, and increasing until
          say 240) then a large number of trees of different size and
          shapes will be generated."

    - 2017, Nicolau: `Understanding grammatical evolution:
      initialisation <https://doi.org/10.1007/s10710-017-9309-9>`__:

        - 3.3 Probabilistic tree-creation 2 (PTC2)
        - 3.6 Probabilistic tree-creation 2 with depth limit (PTC2D)

    - 2018, Ryan, O'Neill, Collins: `Handbook of Grammatical Evolution
      <https://doi.org/10.1007/978-3-319-78717-6>`__

        - p. 13: "More recent work on initialisation includes that of
          Nicolau, who demonstrated that across the problems examined
          in their study, a variant of Harper’s PTC2 consistently
          outperforms other initialisations"

    """
    return _init_individual.ptc2_tree(
        grammar, parameters, _dp, _representation, _mapping
    )
