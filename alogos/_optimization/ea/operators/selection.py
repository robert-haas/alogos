import random as _random
from functools import lru_cache as _lru_cache
from math import isfinite as _isfinite
from math import isnan as _isnan

from ...._utilities.operating_system import NEWLINE as _NEWLINE


# Selection operators implemented as functions with identical interface


def uniform(individuals, sample_size, objective, parameters, state):
    """Perform uniform selection via uniform sampling with replacement.

    References
    ----------
    - Eiben, Introduction to Evolutionary Computing (2e 2015): p. 86

    """
    sel_inds = _uniform_sampling_with_replacement(individuals, sample_size)
    return sel_inds


def truncation(individuals, sample_size, objective, parameters, state):
    """Perform truncation selection via deterministic cut-off.

    Given a population, return the best <proportion> of them.

    """
    # Argument processing
    num_ind = len(individuals)

    # Case 1: Pass-through
    if sample_size == num_ind:
        sel_inds = individuals

    # Case 2: All individuals multiple times + best individuals as rest
    elif sample_size > num_ind:
        srt_inds = _sort_individuals(individuals, reverse=(objective == "max"))
        sel_inds = [srt_inds[i % num_ind] for i in range(sample_size)]

    # Case 3: Best individuals
    else:
        srt_inds = _sort_individuals(individuals, reverse=(objective == "max"))
        sel_inds = srt_inds[:sample_size]
    return sel_inds


def tournament(individuals, sample_size, objective, parameters, state):
    """Perform tournament selection via sampling with replacement.

    Given a population, draw <tournament_size> competitors randomly and select the single best of
    them.

    """
    # Parameter processing
    ts = parameters.parent_selection_tournament_size

    # Perform tournaments and keep the winner of each
    sel_inds = []
    if objective == "min":
        for _ in range(sample_size):
            competitors = _uniform_sampling_with_replacement(individuals, ts)
            winner = competitors[0]
            for competitor in competitors[1:]:
                if competitor.less_than(winner, objective):
                    winner = competitor
            sel_inds.append(winner)
    else:
        for _ in range(sample_size):
            competitors = _uniform_sampling_with_replacement(individuals, ts)
            winner = competitors[0]
            for competitor in competitors[1:]:
                if competitor.greater_than(winner, objective):
                    winner = competitor
            sel_inds.append(winner)
    return sel_inds


def rank_proportional(individuals, sample_size, objective, parameters, state):
    """Perform rank-proportional selection with linear scaling."""
    # Sorting
    srt_inds = _sort_individuals(individuals, reverse=(objective == "max"))

    # Probability calculation based on ranks
    num_inds = len(srt_inds)
    mu = len(srt_inds)
    eta_plus = 1.5
    probabilities = _calculate_rank_probabilities(num_inds, mu, eta_plus)

    # Sampling
    sel_inds = _stochastic_universal_sampling(srt_inds, probabilities, sample_size)
    return sel_inds


def fitness_proportional(individuals, sample_size, objective, parameters, state):
    """Perform fitness-proportional selection with linear scaling.

    Considerations for special float values:

    - NaN values are ignored, i.e. the individual has 0.0% chance of being selected.
    - +Inf values are 1) ignored in minimization or 2) replaced by a large positive number in
      maximization.
    - -Inf values are 1) ignored in maximization or 2) replaced by a large negative number in
      minimization.

    """
    # Preprocessing: Ignore NaN, +Inf and -Inf (to not have effect on scaling and probabilities)
    fitnesses, usage_tracking = _preprocess_nonfinite_fitnesses(individuals, objective)

    # Linear scaling
    scaled_fitnesses = _linear_scaling(fitnesses, objective)

    # Probability calculation based on scaled fitness values
    probabilities = _calculate_fitness_probabilities(scaled_fitnesses, usage_tracking)

    # Sampling
    sel_inds = _stochastic_universal_sampling(individuals, probabilities, sample_size)
    return sel_inds


# Functions called by some operators


@_lru_cache(maxsize=8)
def _calculate_rank_probabilities(num_inds, mu, eta_plus):
    """Calculate probabilities for rank-proportional selection.

    Note that the probabilities stay the same if mu and eta_plus do not
    change, therefore they can be cached to prevent recalculation.

    """
    # Use ranks from 0 to n-1 (in range) instead of 1 to n => formula without -1
    mu_inv = 1.0 / mu
    mu_minus_1 = mu - 1.0
    eta_minus = 2.0 - eta_plus
    eta_diff = eta_plus - eta_minus
    probabilities = [
        mu_inv * (eta_minus + eta_diff * r_i / mu_minus_1) for r_i in range(num_inds)
    ]
    return probabilities


def _preprocess_nonfinite_fitnesses(individuals, objective):
    float_max = 1.7976931348623157e308
    float_large_pos = float_max / 10e7
    float_min = -1.7976931348623157e308
    float_large_neg = float_min / 10e7
    fitnesses = []
    usage_tracking = []
    for ind in individuals:
        fitness = ind.fitness
        if _isfinite(fitness):
            # Finite number: Used in all cases
            fitnesses.append(fitness)
            usage_tracking.append(True)
        elif _isnan(fitness):
            # NaN: Ignored in all cases
            usage_tracking.append(False)
        elif fitness > 0.0:
            # +Inf
            if objective == "min":
                # Ignored in minimization, considered as no fitness
                usage_tracking.append(False)
            else:
                # Used in maximization, considered as extremely good fitness
                fitnesses.append(float_large_pos)
                usage_tracking.append(True)
        else:
            # -Inf
            if objective == "max":
                # Ignored in maximization, considered as no fitness
                usage_tracking.append(False)
            else:
                # Used in minimization, considered as extremely good fitness
                fitnesses.append(float_large_neg)
                usage_tracking.append(True)
    if not fitnesses:
        fitnesses = [0.0 for _ in range(len(individuals))]
        usage_tracking = [True for _ in range(len(individuals))]
    return fitnesses, usage_tracking


def _linear_scaling(fitnesses, objective):
    """Shift the input range to a positive one from 0 (worst) to some positive value (best)."""
    if objective == "max":
        worst_fitness = min(fitnesses)
        scaled_fitnesses = [fitness - worst_fitness for fitness in fitnesses]
    else:
        worst_fitness = max(fitnesses)
        scaled_fitnesses = [-fitness + worst_fitness for fitness in fitnesses]
    return scaled_fitnesses


def _calculate_fitness_probabilities(scaled_fitnesses, usage_tracking):
    sum_fitnesses = sum(scaled_fitnesses)
    if sum_fitnesses <= 10e-20:
        # Use uniform sampling if all fitnesses are practically equal, i.e. nearly 0 after scaling.
        num_ind = len(usage_tracking)
        probabilities = [1.0 / num_ind for _ in range(num_ind)]
    else:
        # Otherwise calculate a probability distribution for non-uniform sampling based on fitness
        probabilities = []
        i = 0
        for used in usage_tracking:
            if used:
                fitness = scaled_fitnesses[i]
                i += 1
                probability = fitness / sum_fitnesses
                probabilities.append(probability)
            else:
                probabilities.append(0.0)
    return probabilities


# Custom sorting algorithm: NaN values need to be treated properly depending on objective


def _sort_individuals(individuals, reverse=False):
    # Separate: individuals with a number as fitness and those with NaN
    inds_val = [ind for ind in individuals if ind.fitness == ind.fitness]
    inds_nan = [ind for ind in individuals if ind.fitness != ind.fitness]

    # Sort
    inds_val_sort = sorted(inds_val, key=lambda x: x.fitness, reverse=reverse)

    # Recombine
    return inds_val_sort + inds_nan


# Sampling algorithms: draw individuals from a population according to a probability distribution


def _uniform_sampling_with_replacement(population, sample_size):
    """Uniform sampling with replacement."""
    selected_individuals = [_random.choice(population) for _ in range(sample_size)]
    return selected_individuals


def _stochastic_universal_sampling(population, probabilities, sample_size):
    """Stochastic universal sampling (SUS) for drawing individuals from a probability distribution.

    A sampling algorithm by James Baker, designed as improvement to roulette wheel sampling.

    References
    ----------
    - Bäck, Handbook of Evolutionary Computation (1997): p. C2.2:4
    - Eiben, Introduction to Evolutionary Computing (2e 2015): p. 84

    """
    lambda_inv = 1.0 / sample_size

    # Draw from uniform distribution
    rand_uniform = _random.uniform(0.0, lambda_inv)

    # Decide which individual gets how many children (=copies of itself)
    num_children = []
    cumulative_sum = 0.0
    for idx in range(len(population)):
        num_children.append(0)
        cumulative_sum += probabilities[idx]
        while rand_uniform < cumulative_sum:
            num_children[idx] += 1
            rand_uniform += lambda_inv

    # Construct the population of selected individuals
    selected_individuals = []
    for idx, num in enumerate(num_children):
        for _ in range(num):
            selected_individuals.append(population[idx])

    if len(selected_individuals) != sample_size:
        message = (
            "Stochastic uniform sampling failed.{nl}"
            "Sum of provided probabilities: {prob}{nl}"
            "Number of sampled individuals: {ind}{nl}"
            "Wanted sample size: {spl}{nl}".format(
                nl=_NEWLINE,
                prob=sum(probabilities),
                ind=len(selected_individuals),
                spl=sample_size,
            )
        )
        raise ValueError(message)
    return selected_individuals
