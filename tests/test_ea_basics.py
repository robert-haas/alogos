import pytest
import shared

import alogos as al


def test_api():
    # Initialization with minimal required arguments
    # - grammar, objective function, objective
    ea = al.EvolutionaryAlgorithm(shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min")
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, objective="max"
    )
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, objective_function=shared.OBJ_FUN_FLOAT, objective="min"
    )
    ea = al.EvolutionaryAlgorithm(
        grammar=shared.GRAMMAR_FLOAT,
        objective_function=shared.OBJ_FUN_FLOAT,
        objective="min",
    )
    shared.check_ea_algorithm(ea)

    # Invalid require argument
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(101010, shared.OBJ_FUN_FLOAT, "min")
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(shared.GRAMMAR_FLOAT, 101010, "max")
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, 101010)
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "nonsense")

    # Missing stop criterion
    with pytest.raises(al.exceptions.ParameterError):
        ea = al.EvolutionaryAlgorithm(shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "max")
        ea.run()

    # Initialization with all ea-specific parameters
    # - system
    for system in ("cfggp", "cfggpst", "dsge", "ge", "pige", "whge"):
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system
        )
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "max", system=system
        )
        shared.check_ea_algorithm(ea)

    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system=101010
        )
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "max", system="nonsense"
        )

    # - evaluator
    def evaluator(func, args):
        return [func(arg) for arg in args]

    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system, evaluator
    )
    shared.check_ea_algorithm(ea)

    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "max", evaluator=101010
        )
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", evaluator="nonsense"
        )

    # - database_on
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", database_on=True
    )
    shared.check_ea_algorithm(ea)

    # - verbose
    for vb in (False, True, 0, 1, 2, 3):
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT,
            shared.OBJ_FUN_FLOAT,
            "min",
            verbose=vb,
        )
        shared.check_ea_algorithm(ea)

    # - others
    for key, val in al._optimization.ea.parameters.default_parameters.items():
        param = {key: val}
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system, **param
        )
        shared.check_ea_algorithm(ea)

        param = {"nonsense": 101010}
        with pytest.raises(al.exceptions.ParameterError):
            al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "max", system, **param
            )

    # Initialization with system-specific parameters
    for system in ("cfggp", "cfggpst", "dsge", "ge", "pige", "whge"):
        module = getattr(al.systems, system)
        for key, val in module.default_parameters.items():
            param = {key: val}
            ea = al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system, **param
            )
            shared.check_ea_algorithm(ea)

            param = {"nonsense": 101010}
            with pytest.raises(al.exceptions.ParameterError):
                al.EvolutionaryAlgorithm(
                    shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system, **param
                )


def test_reset():
    # init
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=2
    )
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=False)

    # run
    ea.run()
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=True)

    # reset
    ea.reset()
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=False)

    # run
    ea.step()
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=True)

    # reset
    ea.reset()
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=False)


def test_is_stop_criterion_met():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=2
    )
    assert not ea.is_stop_criterion_met()
    assert ea.state.num_generations == 0

    ea.step()
    assert not ea.is_stop_criterion_met()
    assert ea.state.num_generations == 1

    ea.step()
    assert ea.is_stop_criterion_met()
    assert ea.state.num_generations == 2

    ea.step()
    assert ea.is_stop_criterion_met()
    assert ea.state.num_generations == 3

    ea.reset()
    assert not ea.is_stop_criterion_met()
    assert ea.state.num_generations == 0

    ea.run()
    assert ea.is_stop_criterion_met()
    assert ea.state.num_generations == 2


def test_state():
    # init
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=2
    )
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=False)

    # run
    ea.step()
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=True)

    # run continued
    ea.step()
    ea.step()
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=True)

    # reset
    ea.reset()
    shared.check_ea_algorithm(ea)
    shared.check_ea_state(ea, has_run=False)


def test_usual_run_1():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT_DETAILS,
        "min",
        database_on=True,
        max_generations=3,
    )
    ea.run()
    assert ea.state.best_individual.fitness >= 0.0
    shared.check_ea_algorithm(ea)


def test_usual_run_2():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE,
        shared.OBJ_FUN_TUPLE,
        "min",
        database_on=True,
        max_fitness_evaluations=321,
    )
    ea.run()
    shared.check_ea_algorithm(ea)


def test_usual_run_3():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE,
        shared.OBJ_FUN_TUPLE,
        "min",
        database_on=True,
        max_runtime_in_seconds=0.14,
    )
    ea.run()
    shared.check_ea_algorithm(ea)


def test_usual_run_4():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE,
        shared.OBJ_FUN_TUPLE,
        "min",
        database_on=True,
        max_or_min_fitness=0.01,
    )
    ea.run()
    shared.check_ea_algorithm(ea)
