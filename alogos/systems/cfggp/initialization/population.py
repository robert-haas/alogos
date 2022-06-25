from . import individual as _individual
from .. import default_parameters as _dp
from .. import representation as _representation
from .. import mapping as _mapping
from ..._shared.initialization import population as _population


def given_genotypes(grammar, parameters=None):
    """Create a population from given genotypes.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO - list of genotypes, where each genotype is a list of int

    """
    return _population.given_genotypes(grammar, parameters, _dp, _representation, _individual)


def given_derivation_trees(grammar, parameters=None):
    """Create a population from given derivation trees.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO - list of derivation trees TODO

    """
    return _population.given_derivation_trees(
        grammar, parameters, _dp, _representation, _individual)


def given_phenotypes(grammar, parameters=None):
    """Create a population from given phenotypes.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO - list of phenotypes, where each phenotype is a string

    """
    return _population.given_phenotypes(grammar, parameters, _dp, _representation, _individual)


def random_genotypes(grammar, parameters=None):
    """Create a population from random genotypes.

    Parameters
    ----------
    grammar : :ref:`Grammar <grammar>`
    parameters : TODO

    """
    return _population.random_genotypes(grammar, parameters, _dp, _representation, _individual)


def rhh(grammar, parameters=None):
    """Ramped half and half or half ramping"""
    return _population.rhh(grammar, parameters, _dp, _representation, _individual)


def ptc2(grammar, parameters=None):
    """Use "probabilistic tree-creation 2" (PTC2)."""
    return _population.ptc2(grammar, parameters, _dp, _representation, _individual)
