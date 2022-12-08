"""Mutation functions for WHGE."""

import random as _random

from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from . import default_parameters as _dp
from . import representation as _representation


def bit_flip_by_probability(grammar, genotype, parameters=None):
    """Mutate a genotype by random bit flips with a certain probability.

    The probability for a bit flip is considered independently for
    each position in the genotype, regardless of the genotype length.

    Caution: Mutation is performed in-place for performance reasons,
    which means that the provided genotype is modified. If this is
    not desired, a copy needs to be made beforehand.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``mutation_bit_flip_probability`` (`int`) : Probability
          of changing a bit.

    References
    ----------
    - Medvet in 2017:
      `Hierarchical Grammatical Evolution
      <https://doi.org/10.1145/3067695.3075972>`__

        - p. 250: "we performed 30 independent executions of the evolutionary search
          [...] with the following evolution parameters: [...]
          bit flip mutation with p_mut = 0.01 and 0.2 rate"

    - Bartoli, Castelli, Medvet in 2018:
      `Weighted Hierarchical Grammatical Evolution
      <https://doi.org/10.1109/TCYB.2018.2876563>`__

        - p. 7, Table 1: "Mutation op. bit flip w. p_mut = 0.01"

    - Reference implementation in Java: `evolved-ge
      <https://github.com/ericmedvet/evolved-ge>`__

        - `ProbabilisticMutation.java
          <https://github.com/ericmedvet/evolved-ge/blob/master/src/main/java/it/units/malelab/ege/ge/operator/ProbabilisticMutation.java>`__

            - The algorithm loops over each bit of the genotype, in each
              run generating a random number between 0.0 and 1.0 and if
              it is smaller than p_mut the current bit is flipped.

        - `Folder with example scripts
          <https://github.com/ericmedvet/evolved-ge/blob/master/src/main/java/it/units/malelab/ege>`__

            - Using ``ProbabilisticMutation`` with p_mut 0.01 and rate 0.2:
              ``DeepExperimenter.java``,
              ``DeepDistributedExperimenter.java``,
              ``GOM.java``,
              ``MapperGenerationExperimenter.java``,
              ``MapperGenerationDistributedExperimenter.java``

    """
    # Note about the implementation:
    # An attempted speedup with
    # numpy.random.choice([True, False], size=len(genotype), p=(probability, 1.0-probability))`
    # to generate a mask of positions to flip turned out to be slower
    # than simply using Python's `random.random() < probability` in a
    # for loop.

    # Parameter extraction
    probability = _get_given_or_default(
        "mutation_bit_flip_probability", parameters, _dp
    )

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Mutation: For each bit decide randomly whether it is flipped
    data = genotype.data.copy()
    for i in range(len(data)):
        if _random.random() < probability:
            data[i] = not data[i]
    return _representation.Genotype(data)


def bit_flip_by_count(grammar, genotype, parameters=None):
    """Mutate a genotype by n random bit flips."""
    # Parameter extraction
    flip_count = _get_given_or_default("mutation_bit_flip_count", parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Mutation: Choose n different positions to flip
    data = genotype.data.copy()
    num_pos = len(data)
    if flip_count > num_pos:
        positions = range(num_pos)
    else:
        positions = _random.sample(range(num_pos), flip_count)
    for i in positions:
        data[i] = not data[i]
    return _representation.Genotype(data)
