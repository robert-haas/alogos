# Generation models implemented as functions with identical interface


def overlapping(parent_population, offspring_population, parameters):
    """Use an overlapping generation model.

    Notes
    -----
    - Overlapping pooling together with rank-based survivor selection
      is equivalent to "(lambda + mu) selection" [1]_.
    - Overlapping pooling together with tournament survivor selection
      is equivalent to "round-robin tournament".

    - "However, a much more significant effect on selection pressure
      occurs when using an EA with an overlapping-generation model such
      as a "steady-state GA", a "µ + λ" ES, or any EP algorithm.
      In this case, parents and offspring compete with each other for
      survival. The combination of a larger selection pool (m + n) and
      the fact that, as evolution proceeds, the m parents provide
      stronger and stronger competition, results in a significant
      increase in selection pressure over a non-overlapping version of
      the same EA." [2]_

    References
    ----------
    .. [1] Eiben p. 89
    .. [2] DeJong p. 59

    """
    return parent_population + offspring_population


def non_overlapping(parent_population, offspring_population, parameters):
    """Use a non-overlapping generation model.

    Notes
    -----
    This is also known as "generational model" of population
    management [2]_ or (lambda, mu) selection [3]_.

    - "With non-overlapping models, the entire parent population dies
      off each generation and the offspring only compete with each other
      for survival. Historical examples of non-overlapping EAs include
      "generational GAs" and the "µ, λ" variation of ESs.
      In non-overlapping models, if the offspring population size n is
      significantly larger than the parent population size m
      (e.g., traditional ESs), then competition for survival
      increases." [4]_

    References
    ----------
    .. [2] Eiben p. 79
    .. [3] Eiben p. 89
    .. [4] DeJong p. 59

    """
    return offspring_population
