"""Shared crossover functions for several systems."""

from random import randint as _ri

from ... import exceptions as _exceptions


def two_point_length_preserving(grammar, gt1, gt2, parameters, representation):
    """Generate new genotypes by exchanging sequence parts.

    Select two random, but equally long subsequences in the two
    genotypes and exchange them.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype1 : Genotype
        Genotype of the first parent.
    genotype2 : Genotype
        Genotype of the second parent.
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        No keyword-value pairs are considered by this function.
        This argument is only available to have a consistent interface.
    representation : module
        Representation module of the system that calls this generic
        function. This module contains the specific
        `~.representation.Genotype` subclass of the system, which is
        used here to create the child genotypes.

    Returns
    -------
    genotype1 : Genotype
        Genotype of the first child.
    genotype2 : Genotype
        Genotype of the second child.

    """
    # Argument processing
    _GT = representation.Genotype
    if not isinstance(gt1, _GT):
        gt1 = _GT(gt1)
    if not isinstance(gt2, _GT):
        gt2 = _GT(gt2)
    d1 = gt1.data
    d2 = gt2.data
    l1 = len(d1)
    l2 = len(d2)
    if l1 != l2:
        _exceptions.raise_crossover_lp_error1(l1, l2)
    if l1 < 2:
        _exceptions.raise_crossover_lp_error2()

    # Get a random segment in genotype 1: choose two valid random points
    s1, e1 = _get_two_different_points(l1)

    # Get a random segment in genotype 2: choose a valid start position for a same-sized segment
    lseg = e1 - s1
    s2 = _ri(0, l2 - lseg)
    e2 = s2 + lseg

    # Crossover: Swap two randomly positioned, but equally long segments
    n1 = d1[:s1] + d2[s2:e2] + d1[e1:]
    n2 = d2[:s2] + d1[s1:e1] + d2[e2:]
    return _GT(n1), _GT(n2)


def _get_two_different_points(n):
    """Get two different numbers between 0 and n-1 to use as indices."""
    while True:
        p1 = _ri(0, n)
        p2 = _ri(0, n)
        if p1 == p2:
            continue
        if (p1 == 0 and p2 == n) or (p1 == n and p2 == 0):
            continue
        break
    if p1 > p2:
        p1, p2 = p2, p1
    return p1, p2
