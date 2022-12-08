import shared

import alogos as al


def test_evaluation_caching():
    for cache1 in (True, False):
        for cache2 in (True, False):
            ea = al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=20
            )
            ea.parameters.gen_to_phe_cache_lookup_on = cache1
            ea.parameters.phe_to_fit_cache_lookup_on = cache2
            ea.parameters.evaluator = lambda f, al: [f(a) for a in al]
            ea.run()


def test_phenotype_fitness_evaluation_1():
    def func_returning_float(string):
        x, y = eval(string)
        z = x**2 + y**2
        return z

    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE, func_returning_float, "max", max_generations=10
    )
    ea.run()
    fitness = ea.state.best_individual.fitness
    details = ea.state.best_individual.details
    assert fitness > 0.0
    assert isinstance(details, dict)
    assert len(details) == 3
    assert details["evaluation"] is None


def test_phenotype_fitness_evaluation_2():
    some_list = [42, "cde", None, [], ("x", 2)]
    some_tuple = tuple(some_list)
    some_dict = dict(a=42, b="cde", c=None, d=[], e=("x", 2))

    for extra_val in [42, 3.14, "abc", [], (), None, some_list, some_tuple, some_dict]:

        def func_returning_dict(string):
            x, y = eval(string)
            z = x**2 + y**2
            return z, extra_val  # noqa: B023

        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_TUPLE, func_returning_dict, "max", max_generations=10
        )
        ea.run()
        fitness = ea.state.best_individual.fitness
        details = ea.state.best_individual.details
        assert fitness > 0.0
        assert isinstance(details, dict)
        assert len(details) == 3
        assert details["evaluation"] == extra_val


def test_phenotype_fitness_evaluation_3():
    def func_returning_nothing(string):
        pass

    for objective, inf_val in [("max", "-inf"), ("min", "+inf")]:
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_TUPLE, func_returning_nothing, objective, max_generations=10
        )
        ea.run()
        fitness = ea.state.best_individual.fitness
        details = ea.state.best_individual.details
        assert fitness == float(inf_val)
        assert isinstance(details, dict)
        assert len(details) == 3
        assert (
            details["evaluation"]
            == "ValueError: Returned fitness value is not a number: None"
        )


def test_phenotype_fitness_evaluation_4():
    def func_raising_exception(string):
        raise ValueError("Something bad happened")
        return 42

    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE, func_raising_exception, "max", max_generations=10
    )
    ea.run()
    fitness = ea.state.best_individual.fitness
    details = ea.state.best_individual.details
    assert fitness == float("-inf")
    assert isinstance(details, dict)
    assert len(details) == 3
    assert details["evaluation"] == "ValueError: Something bad happened"


def test_phenotype_fitness_evaluation_5():
    def func_returning_not_a_number(string):
        return "Nonsense", dict(b=4, x=8, z=22)

    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE, func_returning_not_a_number, "min", max_generations=10
    )
    ea.run()
    fitness = ea.state.best_individual.fitness
    details = ea.state.best_individual.details
    assert fitness == float("inf")
    assert isinstance(details, dict)
    assert len(details) == 3
    assert (
        details["evaluation"]
        == "ValueError: Returned fitness value is not a number: Nonsense"
    )
