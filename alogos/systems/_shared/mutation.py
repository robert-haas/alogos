"""Shared mutation functions for several systems."""

from random import randint as _ri
from random import random as _rf
from random import sample as _rs

from ..._utilities.parametrization import get_given_or_default as _get


def flat_int_replacement_by_prob(
    grammar, gt, parameters, default_parameters, _representation
):
    """Generate a new genotype by replacing int codons by chance."""
    # Parameter extraction
    p = _get("mutation_int_replacement_probability", parameters, default_parameters)
    cs = _get("codon_size", parameters, default_parameters)

    # Argument processing
    if not isinstance(gt, _representation.Genotype):
        gt = _representation.Genotype(gt)

    # Mutation: Randomly decide for each position in the genotype whether it shall be modified
    top = 2**cs - 1
    data = tuple(_ri(0, top) if _rf() < p else x for x in gt.data)
    return _representation.Genotype(data)


def flat_int_replacement_by_count(
    grammar, gt, parameters, default_parameters, _representation
):
    """Generate a new genotype by replacing a number of int codons."""
    # Parameter extraction
    flip_count = _get("mutation_int_replacement_count", parameters, default_parameters)
    codon_size = _get("codon_size", parameters, default_parameters)

    # Argument processing
    if not isinstance(gt, _representation.Genotype):
        gt = _representation.Genotype(gt)

    # Mutation: Choose n different positions to flip
    top = 2**codon_size - 1
    le = len(gt)
    pos = _rs(range(le), flip_count) if flip_count < le else set(range(le))
    data = tuple(_ri(0, top) if i in pos else x for i, x in enumerate(gt.data))
    return _representation.Genotype(data)
