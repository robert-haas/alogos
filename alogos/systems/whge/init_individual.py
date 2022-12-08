"""Initialization functions to create a WHGE individual."""

import random as _random

from bitarray.util import int2ba as _int2ba

from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
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

        - ``init_ind_given_genotype`` : Data for a WHGE
          `~.representation.Genotype`.

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


def random_genotype(grammar, parameters=None):
    """Create an individual from a random genotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``genotype_length`` : Number of bits in the WHGE
          `~.representation.Genotype`.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    """
    # Parameter extraction
    gl = _get_given_or_default("genotype_length", parameters, _dp)

    # Argument processing
    if parameters is None:
        parameters = dict()

    # Transformation
    try:
        rand_int = _random.getrandbits(gl)
        rand_bitstring = _int2ba(rand_int, gl)
        rand_genotype = _representation.Genotype(rand_bitstring)
    except Exception:
        _exceptions.raise_init_ind_rand_gt_error()
    parameters["init_ind_given_genotype"] = rand_genotype
    return given_genotype(grammar, parameters)
