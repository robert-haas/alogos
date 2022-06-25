from .. import default_parameters as _dp
from .. import representation as _representation
from .. import mapping as _mapping
from ..._shared.initialization import individual as _individual


def given_genotype(grammar, parameters=None):
    """Create an individual from a given genotype.

    The derivation tree and phenotype are calculated
    with :func:`forward mapping <.mapping.forward>`.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO - init_genotype

    """
    return _individual.given_genotype(grammar, parameters, _dp, _representation, _mapping)


def given_derivation_tree(grammar, parameters=None):
    """Create an individual from a given derivation tree.

    The phenotype is read from the leaf nodes of the derivation tree.
    A genotype (with randomized codon values by "unmod" operation)
    is calculated with :func:`reverse mapping <.mapping.reverse>`.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO - init_derivation_tree

    """
    return _individual.given_derivation_tree(grammar, parameters, _dp, _representation, _mapping)


def given_phenotype(grammar, parameters=None):
    """Create an individual from a given phenotype.

    The genotype and derivation tree are calculated
    with :func:`reverse mapping <.mapping.reverse>`.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO - init_phenotype

    """
    return _individual.given_phenotype(grammar, parameters, _dp, _representation, _mapping)


def random_genotype(grammar, parameters=None):
    """Create an individual from a random genotype.

    The random genotype is a fixed-length list of int which are drawn
    independently from a uniform distribution of numbers in the
    interval `[0, 2**codon_size)`.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

    References
    ----------
    - 2016, Nicolau, Fenton: `Managing Repetition in Grammar-Based Genetic Programming
      <https://doi.org/10.1145/2908812.2908904>`__

    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__

        - 3.1 Random initialisation (RND)

    """
    return _individual.random_genotype(grammar, parameters, _dp, _representation, _mapping)


def random_valid_genotype(grammar, parameters=None):
    """Create an individual from a random genotype that can be mapped to a phenotype.

    It repeatedly calls :meth:`ind_by_random_genotype` until it returns
    a genotype that can successfully be mapped to a phenotype or the
    maximum number of tries is reached.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

    """
    return _individual.random_valid_genotype(grammar, parameters, _dp, _representation, _mapping)


def grow_tree(grammar, parameters=None):
    """Create an individual from a random tree grown."""
    return _individual.grow_tree(grammar, parameters, _dp, _representation, _mapping)


def pi_grow_tree(grammar, parameters=None):
    """Create an individual from a random tree grown in a position-independently fashion."""
    return _individual.pi_grow_tree(grammar, parameters, _dp, _representation, _mapping)


def full_tree(grammar, parameters=None):
    """Create an individual from a random tree that is grown fully to a maximum depth."""
    return _individual.full_tree(grammar, parameters, _dp, _representation, _mapping)


def ptc2_tree(grammar, parameters=None):
    """Create an individual from a random tree that is grown by PTC2."""
    return _individual.ptc2_tree(grammar, parameters, _dp, _representation, _mapping)
