def elitism(old_population, offspring_population, survivor_population, parameters, state):
    # Determine best individual
    if True:
        best_ind = min(min(old_population), min(offspring_population))
    else:
        best_ind = max(max(old_population), max(offspring_population))
    # If it is not in the survivor population, replace the worst individual with it
    if best_ind not in survivor_population:
        worst_ind = survivor_population[0]
        worst_ind_index = 0
        if True:  # TODO: settings.general._minimization_on:
            for idx, ind in enumerate(survivor_population):
                if ind > worst_ind:
                    worst_ind = ind
                    worst_ind_index = idx
        else:
            for idx, ind in enumerate(survivor_population):
                if ind < worst_ind:
                    worst_ind = ind
                    worst_ind_index = idx
    return survivor_population
