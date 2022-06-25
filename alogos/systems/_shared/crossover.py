from random import randint as _ri

from ... import exceptions as _exceptions


def two_point_length_preserving(grammar, gt1, gt2, parameters, _representation):
    # Argument processing
    _GT = _representation.Genotype
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


def _get_two_different_points(l):
    while True:
        p1 = _ri(0, l)
        p2 = _ri(0, l)
        if p1 == p2:
            continue
        if (p1 == 0 and p2 == l) or (p1 == l and p2 == 0):
            continue
        break
    if p1 > p2:
        p1, p2 = p2, p1
    return p1, p2
