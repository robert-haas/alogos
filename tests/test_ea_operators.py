import itertools
import random

import pytest
import shared

import alogos as al


@pytest.mark.parametrize(
    "system, crossover_operators, mutation_operators, op_params",
    [
        (
            "cfggp",
            ["subtree_exchange"],
            ["subtree_replacement"],
            dict(
                # TODO: mutation and crossover modifying parameters
            ),
        ),
        (
            "cfggpst",
            ["subtree_exchange"],
            ["subtree_replacement"],
            dict(
                # TODO: mutation and crossover modifying parameters
            ),
        ),
        (
            "ge",
            ["two_point_length_preserving"],
            ["int_replacement_by_probability", "int_replacement_by_count"],
            dict(
                mutation_int_replacement_probability=0.13,
                mutation_int_replacement_count=2,
                # TODO: crossover modifying parameters
            ),
        ),
    ],
)
def test_operators_all_combinations(
    system, crossover_operators, mutation_operators, op_params
):
    parent_selection_operators = survivor_selection_operators = [
        "uniform",
        "truncation",
        "tournament",
        "rank_proportional",
        "fitness_proportional",
    ]
    generation_model_operators = ["overlapping", "non_overlapping"]

    # Test all combinations of available operators
    num_generations = 5
    population_size = 5
    offspring_size = random.choice(
        [population_size, population_size - 1, population_size + 1]
    )
    operators = [
        parent_selection_operators,
        crossover_operators,
        mutation_operators,
        generation_model_operators,
        survivor_selection_operators,
    ]
    init_params = dict(
        init_pop_given_phenotypes=["+9.999"] * population_size,
        init_pop_operator="given_phenotypes",
    )

    print(system)
    print(
        "parent_selection",
        "crossover",
        "mutation",
        "generation model",
        "survivor_selection",
        sep="\t",
    )
    for par, cross, mut, gm, surv in itertools.product(*operators):
        print(par, cross, mut, gm, surv, sep="\t")

        ea = al.EvolutionaryAlgorithm(
            grammar=shared.GRAMMAR_FLOAT,
            objective_function=shared.OBJ_FUN_FLOAT,
            objective="min",
            system=system,
            population_size=population_size,
            offspring_size=offspring_size,
            max_generations=num_generations,
            parent_selection_operator=par,
            crossover_operator=cross,
            mutation_operator=mut,
            generation_model=gm,
            survivor_selection_operator=surv,
            **op_params,
            **init_params,
        )
        best_individual = ea.run()
        assert best_individual.fitness < 9.7


def test_operator_unknown():
    grammar = al.Grammar("<S> ::= 1 | 2 | 3 | 4 | 5")
    kwargs = dict(
        grammar=grammar,
        objective_function=lambda s: 42,
        objective="max",
        max_generations=2,
        population_size=3,
        offspring_size=3,
    )
    with pytest.raises(al.exceptions.ParameterError):
        ea = al.EvolutionaryAlgorithm(**kwargs, parent_selection_operator="nonsense")
        ea.run()
    with pytest.raises(al.exceptions.ParameterError):
        ea = al.EvolutionaryAlgorithm(**kwargs, mutation_operator="nonsense")
        ea.run()
    with pytest.raises(al.exceptions.ParameterError):
        ea = al.EvolutionaryAlgorithm(**kwargs, crossover_operator="nonsense")
        ea.run()
    with pytest.raises(al.exceptions.ParameterError):
        ea = al.EvolutionaryAlgorithm(**kwargs, survivor_selection_operator="nonsense")
        ea.run()


def test_operators_improvement():
    # Test search with some meaningful settings
    combinations = [
        [
            "truncation",
            "two_point_length_preserving",
            "int_replacement_by_count",
            "overlapping",
            "tournament",
            8,
            8,
        ],
        # ['truncation', 'two_point', 'uniform', 'overlapping', 'tournament', 8, 8],
        # ['truncation', 'one_point', 'one_point', 'non_overlapping', 'tournament', 8, 8],
        # ['uniform', 'one_point', 'one_point', 'overlapping', 'truncation', 8, 4],
        # ['uniform', 'two_point', 'best_neighborhood', 'overlapping', 'truncation', 1, 1]
    ]
    for par, cross, mut, gm, surv, popsize, offsize in combinations:
        # Setup
        print(par, cross, mut, gm, surv, popsize, offsize)
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT,
            shared.OBJ_FUN_FLOAT,
            "min",
            "ge",
            init_pop_given_phenotypes=["(+9.998,+9.999)"] * popsize,
            parent_selection_operator=par,
            crossover_operator=cross,
            mutation_operator=mut,
            generation_model=gm,
            survivor_selection_operator=surv,
            population_size=popsize,
            offspring_size=offsize,
            max_fitness_evaluations=200,
        )

        # Run
        first_best_ind = ea.step()
        final_best_ind = ea.run()

        # Check
        if final_best_ind.greater_than(first_best_ind, "min") and mut is not None:
            raise ValueError(
                "Best fitness in last generation ({}) is not better than in first "
                "({}).".format(final_best_ind.fitness, first_best_ind.fitness)
            )


@pytest.mark.xfail  # TODO: parameter crossover_subtree_replacement_count is not used yet
def test_operators_no_change_if_only_crossover_and_probability_0():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=4,
        population_size=10,
    )
    ea.parameters.mutation_operator = None
    ea.parameters.crossover_subtree_replacement_count = 0
    old_best_fitness = None
    for _ in range(10):
        ea.step()
        best_fitness = ea.state.best_individual.fitness
        if old_best_fitness:
            assert best_fitness == old_best_fitness
        old_best_fitness = best_fitness


@pytest.mark.xfail  # TODO: parameter crossover_subtree_replacement_count is not used yet
def test_operators_no_change_if_only_mutation_and_probability_0():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=4,
        population_size=10,
    )
    ea.parameters.crossover_operator = None
    ea.parameters.mutation_subtree_replacement_count = 0
    old_best_fitness = None
    for _ in range(10):
        ea.step()
        best_fitness = ea.state.best_individual.fitness
        if old_best_fitness:
            assert best_fitness == old_best_fitness
        old_best_fitness = best_fitness


def test_operators_no_variation_1():
    with pytest.raises(al.exceptions.ParameterError):
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT,
            shared.OBJ_FUN_FLOAT,
            "min",
            max_generations=4,
            crossover_operator=None,
            mutation_operator=None,
        )
        ea.run()


def test_operators_no_variation_2():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=4
    )
    ea.parameters.crossover_operator = None
    ea.parameters.mutation_operator = None
    with pytest.raises(al.exceptions.ParameterError):
        ea.run()


def test_operators_no_variation_3():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=4
    )
    ea.parameters.crossover_operator = None
    ea.parameters.mutation_operator = None
    with pytest.raises(al.exceptions.ParameterError):
        ea.step()
        ea.step()


def test_operators_general():
    num_gen = 3
    pop_size = 7
    ea = al.EvolutionaryAlgorithm(shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min")
    ea.parameters.population_size = pop_size
    ea.parameters.max_generations = num_gen

    # Full run
    ea.run()
    assert ea.state.generation == num_gen
    assert len(ea.state.population) == pop_size

    # Reset
    ea.reset()

    # Run step by step
    for i in range(num_gen):
        assert ea.state.generation == i
        ea.step()
        assert ea.state.generation == i + 1
        assert len(ea.state.population) == pop_size
    assert ea.state.generation == num_gen
