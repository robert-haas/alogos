from .. import default_parameters as _dp
from .. import representation as _representation
from .. import mapping as _mapping
from ..._shared.initialization import individual as _individual


def given_genotype(grammar, parameters=None):
    """Create an individual from a given genotype.

    The derivation tree and phenotype are calculated
    with :func:`forward <.mapping.forward>` mapping.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

    """
    return _individual.given_genotype(grammar, parameters, _dp, _representation, _mapping)


def given_derivation_tree(grammar, parameters=None):
    """Create an individual from a given derivation tree.

    The phenotype is read from the leaf nodes of the derivation tree.
    A genotype (with randomized codon values by "unmod" operation)
    is calculated with :func:`reverse <.mapping.reverse>` mapping.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

    """
    return _individual.given_derivation_tree(grammar, parameters, _dp, _representation, _mapping)


def given_phenotype(grammar, parameters=None):
    """Create an individual from a given phenotype.

    The genotype and derivation tree are calculated
    with :func:`reverse <.mapping.reverse>` mapping.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

    """
    return _individual.given_phenotype(grammar, parameters, _dp, _representation, _mapping)


def random_genotype(grammar, parameters=None):
    """Create an individual from a random genotype."""
    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    random_genotype = _representation.Genotype(grammar.generate_derivation_tree())
    parameters['init_ind_given_genotype'] = random_genotype
    return given_genotype(grammar, parameters)


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
