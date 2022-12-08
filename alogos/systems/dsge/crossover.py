"""Crossover functions for DSGE."""

import random as _random

from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from . import default_parameters as _default_parameters
from . import repair as _repair
from . import representation as _representation


# Shortcuts for minor speedup
_GT = _representation.Genotype
_rc = _random.choice


def gene_swap(grammar, genotype1, genotype2, parameters=None):
    """Generate new DSGE genotypes by exchanging a random gene.

    Each DSGE genotype contains the same number of genes. Randomly
    select one gene and exchange it between the two genotypes.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype1 : `~.representation.Genotype`
        Genotype of the first parent.
    genotype2 : `~.representation.Genotype`
        Genotype of the second parent.
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``repair_after_crossover`` (`bool`): {`True`, `False`}

    Returns
    -------
    genotype1 : `~.representation.Genotype`
        Genotype of the first child.
    genotype2 : `~.representation.Genotype`
        Genotype of the second child.

    References
    ----------
    - Software implementations by the authors of the approach

        - Python: `dsge
          <https://github.com/nunolourenco/dsge>`__

            - `core/sge.py
              <https://github.com/nunolourenco/dsge/blob/master/src/core/sge.py>`__:
              ``def crossover(p1, p2)`` is the implementation of the
              crossover operator

    - Papers

        - Lourenço et al. in 2018:
          `Structured Grammatical Evolution: A Dynamic Approach
          <https://doi.org/10.1007/978-3-319-78717-6_6>`__

            - p. 145: "It starts by creating a random binary mask and
              the offspring are created by selecting the parents genes
              based on the mask values. Recombination does not modify
              the lists inside the genes. This is similar to what
              happens with uniform crossover for binary
              representations."

    """
    # Parameter extraction
    repair = _get_given_or_default(
        "repair_after_crossover", parameters, _default_parameters
    )

    # Argument processing
    if not isinstance(genotype1, _GT):
        genotype1 = _GT(genotype1)
    if not isinstance(genotype2, _GT):
        genotype2 = _GT(genotype2)

    # Crossover: For each position use the gene of one parent according to a random binary mask
    d1 = genotype1.data
    d2 = genotype2.data
    vals = (True, False)
    mask = [_rc(vals) for _ in range(len(d1))]
    n1 = tuple(d1[i] if x else d2[i] for i, x in enumerate(mask))
    n2 = tuple(d2[i] if x else d1[i] for i, x in enumerate(mask))

    # Optional repair of the new genotypes (either here and/or later)
    if repair:
        gt1 = _repair.fix_genotype(grammar, n1, parameters, raise_errors=False)
        gt2 = _repair.fix_genotype(grammar, n2, parameters, raise_errors=False)
    else:
        gt1 = _GT(n1)
        gt2 = _GT(n2)
    return gt1, gt2
