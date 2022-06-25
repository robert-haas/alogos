# Pooling operators implemented as functions with identical interface

def overlapping(parent_population, offspring_population, parameters):
    """Overlapping generations, also known as (lambda + mu) selection"""
    return parent_population + offspring_population


def non_overlapping(parent_population, offspring_population, parameters):
    """Non-overlapping generations, also known as (lambda, mu) selection"""
    return offspring_population


def steady_state(parent_population, offspring_population, parameters):
    """Steady-state population management, also known as   TODO """
    worst_parent_idx = _find_worst_individual_index(parent_population, parameters.objective)
    best_offspring_idx = _find_best_individual_index(offspring_population, parameters.objective)
    parent_population[worst_parent_idx] = offspring_population[best_offspring_idx]
    return parent_population


# Helper functions

def _find_worst_individual_index(population, objective):
    worst_ind_idx = 0
    if objective == 'min':
        for idx, ind in enumerate(population):
            if population[idx].greater_than(population[worst_ind_idx], objective):
                worst_ind_idx = idx
    else:
        for idx, ind in enumerate(population):
            if population[idx].less_than(population[worst_ind_idx], objective):
                worst_ind_idx = idx
    return worst_ind_idx


def _find_best_individual_index(population, objective):
    best_ind_idx = 0
    if objective == 'min':
        for idx, ind in enumerate(population):
            if population[idx].less_than(population[best_ind_idx], objective):
                best_ind_idx = idx
    else:
        for idx, ind in enumerate(population):
            if population[idx].greater_than(population[best_ind_idx], objective):
                best_ind_idx = idx
    return best_ind_idx
