import time

import pytest
import shared
import unified_map as um

import alogos as al


# Parameters of the evolutionary algorithm itself


def test_parameter_grammar():
    # Minimal
    grammar = al.Grammar(bnf_text="<S> ::= 1")
    ea = al.EvolutionaryAlgorithm(
        grammar,
        lambda s: int(s),
        "min",
        max_generations=1,
        init_pop_unique_phenotypes=False,
    )
    best_ind = ea.run()
    assert best_ind.fitness == 1.0

    # Error: invalid type
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(
            "nonsense", shared.OBJ_FUN_FLOAT, "min", max_generations=1
        )


def test_parameter_objective_function():
    # Lambda function
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        lambda string: float(string) ** 2,
        "min",
        max_or_min_fitness=0.0,
    )
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Function
    def square(string):
        return float(string) ** 2

    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, square, "min", max_or_min_fitness=0.0
    )
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Method
    class Carrier:
        def square(self, string):
            return float(string) ** 2

    carrier = Carrier()
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, carrier.square, "min", max_or_min_fitness=0.0
    )
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Callable class
    class Squarer:
        def __call__(self, string):
            return float(string) ** 2

    squarer = Squarer()
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, squarer, "min", max_or_min_fitness=0.0
    )
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Error: class that is not callable
    class NonSquarer:
        pass

    non_squarer = NonSquarer()
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, non_squarer, "min", max_or_min_fitness=0.0
        )


def test_parameter_population_size():
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text)

    for ps in range(1, 10):
        for os in range(1, 10):
            ea = al.EvolutionaryAlgorithm(
                grammar,
                lambda s: s.count("1"),
                "max",
                max_generations=int(1000 / ps / os),
                population_size=ps,
                offspring_size=os,
            )
            ind = ea.run()
            assert ind.fitness > 8


def test_parameter_objective():
    bnf = """
    <S> ::= <A>     <B>
    <A> ::= <digit>
    <B> ::=        <digit>
    <digit> ::= 0 | 1 |      2| 3 |4|5 | 6 |7
              | 8 |   9
    """
    grammar = al.Grammar(bnf_text=bnf)

    def obj_fun(string):
        return int(string)

    # min
    ea = al.EvolutionaryAlgorithm(grammar, obj_fun, "min", max_generations=50)
    ea.run()
    assert ea.state.best_individual.fitness == 0.0

    # max
    ea = al.EvolutionaryAlgorithm(grammar, obj_fun, "max", max_generations=50)
    ea.run()
    assert ea.state.best_individual.fitness == 99.0

    # Error: invalid type
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, 42, max_generations=1
        )

    # Error: invalid value
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "nonsense", max_generations=1
        )


def test_parameter_system():
    # Chosen system influences available parameters
    systems = ("cfggp", "cfggpst", "dsge", "ge", "pige", "whge")
    params_repr = set()
    for system in systems:
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system
        )
        params_repr.add(
            str(ea.parameters)
        )  # parameters are specific for the chosen system
    assert len(params_repr) == len(
        systems
    )  # therefore the set has as many different entries

    # Error: invalid type
    for system in (0, 0.0, [], {}):
        with pytest.raises(TypeError):
            al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", system
            )

    # Error: invalid value
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", "nonsense"
        )


def test_parameter_evaluator():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE,
        shared.OBJ_FUN_TUPLE_SLOW,
        "min",
        max_generations=3,
        population_size=8,
    )
    ea.parameters.evaluator = um.univariate.serial.for_loop

    # 1) Run with default serial evaluator
    start = time.time()
    ea.run()
    stop = time.time()
    t1 = stop - start

    # 2) Run with parallel evaluator
    ea.reset()
    ea.parameters.evaluator = um.univariate.parallel.multiprocessing
    start = time.time()
    ea.run()
    stop = time.time()
    t2 = stop - start
    assert t2 < t1

    # 3) Run with another parallel evaluator and setting evaluator in EA parameters
    ea.reset()
    ea.parameters.evaluator = um.univariate.parallel.futures
    start = time.time()
    ea.run()
    stop = time.time()
    t3 = stop - start
    assert t3 < t1

    # 4) Run with another parallel evaluator and passing evaluator on EA initialization
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE,
        shared.OBJ_FUN_TUPLE_SLOW,
        "min",
        max_generations=3,
        population_size=8,
        evaluator=um.univariate.parallel.joblib,
    )
    start = time.time()
    ea.run()
    stop = time.time()
    t4 = stop - start
    assert t4 < t1


def test_parameter_verbose(capsys):
    # default
    def func():
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=2
        )
        ea.step()

    shared.prints_nothing_to_stdout(func, capsys)

    # off
    for verb in (False, 0):

        def func():
            ea = al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT,
                shared.OBJ_FUN_FLOAT,
                "min",
                max_generations=2,
                verbose=verb,  # noqa: B023
            )
            ea.step()

        shared.prints_nothing_to_stdout(func, capsys)

    # on
    for verb in (True, 1):

        def func():
            ea = al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT,
                shared.OBJ_FUN_FLOAT,
                "min",
                max_generations=2,
                verbose=verb,  # noqa: B023
            )
            ea.step()

        shared.prints_to_stdout(
            func, capsys, partial_message="Progress         Generations"
        )

    # high level
    for verb in (2, 3, 4):

        def func():
            ea = al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT,
                shared.OBJ_FUN_FLOAT,
                "min",
                max_generations=2,
                verbose=verb,  # noqa: B023
            )
            ea.step()

        shared.prints_to_stdout(func, capsys, partial_message="╭─ Run started ── ")

    # Error: invalid value
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT,
            shared.OBJ_FUN_FLOAT,
            "min",
            max_generations=2,
            verbose=-1,
        )

    # Error: invalid type
    for verbose in (2.0, None, [], {}, "nonsense"):
        with pytest.raises(TypeError):
            al.EvolutionaryAlgorithm(
                shared.GRAMMAR_FLOAT,
                shared.OBJ_FUN_FLOAT,
                "min",
                max_generations=2,
                verbose=verbose,
            )


# Parameters of the G3P systems


def test_parameter_init_pop_given_genotypes():
    population_size = 10
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        system="ge",
        database_on=True,
        max_generations=5,
        population_size=population_size,
        verbose=10,
        init_pop_operator="given_genotypes",
    )

    # Case 1: Fewer genotypes than population_size
    ea.parameters.init_pop_given_genotypes = [(0, 1, 2)]
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_genotypes) == len(pop0) < population_size
    for ind in pop0:
        assert ind.genotype.data in ea.parameters.init_pop_given_genotypes

    # Case 2: More genotypes than population_size
    ea.reset()
    ea.parameters.init_pop_given_genotypes = [(0, 1, 2), (3, 4, 5), (7, 8, 9)] * 20
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_genotypes) == len(pop0) > population_size
    for ind in pop0:
        assert ind.genotype.data in ea.parameters.init_pop_given_genotypes

    # Case 3: Number of genotypes is equal to population_size
    ea.reset()
    ea.parameters.init_pop_given_genotypes = [(0, 1, 2), (3, 4, 5)] * 5
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_genotypes) == len(pop0) == population_size
    for ind in pop0:
        assert ind.genotype.data in ea.parameters.init_pop_given_genotypes


def test_parameter_init_pop_given_genotypes_error_because_not_iterable():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=5,
        init_pop_given_genotypes=42,
        init_pop_operator="given_genotypes",
    )
    with pytest.raises(al.exceptions.InitializationError):
        ea.step()


def test_parameter_init_pop_given_genotypes_error_because_one_element_not_valid():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        system="ge",
        max_generations=5,
    )
    ea.parameters.population_size = 10
    ea.parameters.init_pop_operator = "given_genotypes"
    ea.parameters.init_pop_given_genotypes = [(0, 1, 2)] * 10
    ea.parameters.init_pop_given_genotypes[3] = []
    with pytest.raises(al.exceptions.InitializationError):
        ea.run()


def test_parameter_init_pop_given_phenotypes():
    population_size = 10
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE,
        shared.OBJ_FUN_TUPLE,
        "min",
        max_generations=5,
        population_size=population_size,
        verbose=10,
        database_on=True,
        init_pop_operator="given_phenotypes",
    )

    # Case 1: Fewer genotypes than population_size
    ea.parameters.init_pop_given_phenotypes = ["(+1.112,-2.914)"]
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0) < population_size
    for ind in pop0:
        assert ind.phenotype in ea.parameters.init_pop_given_phenotypes

    # Case 2: More genotypes than population_size
    ea.reset()
    ea.parameters.init_pop_given_phenotypes = [
        "(+1.112,-2.914)",
        "(-1.122,-3.143)",
    ] * 40
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0) > population_size
    for ind in pop0:
        assert ind.phenotype in ea.parameters.init_pop_given_phenotypes

    # Case 3: Number of genotypes is equal to population_size
    ea.reset()
    ea.parameters.init_pop_given_phenotypes = ["(-9.120,+2.622)", "(+5.671,+3.674)"] * 5
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0) == population_size
    for ind in pop0:
        assert ind.phenotype in ea.parameters.init_pop_given_phenotypes


def test_parameter_init_pop_given_phenotypes_error_because_one_element_not_valid():
    population_size = 10
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_TUPLE,
        shared.OBJ_FUN_TUPLE,
        "min",
        max_generations=5,
        population_size=population_size,
        verbose=10,
        init_pop_operator="given_phenotypes",
    )
    ea.parameters.init_pop_given_phenotypes = [
        "(+1.112,-2.914)",
        "invalid, parser fails",
    ]
    with pytest.raises(al.exceptions.InitializationError):
        ea.run()


def test_parameter_init_pop_given_phenotypes_error_because_system_does_not_have_reverse_mapping():
    for sys in ("cfggp", "cfggpst", "dsge", "ge", "pige", "whge"):
        population_size = 10
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_TUPLE,
            shared.OBJ_FUN_TUPLE,
            "min",
            system=sys,
            max_generations=5,
            population_size=population_size,
            verbose=10,
            init_pop_operator="given_phenotypes",
        )
        if sys == "whge":
            with pytest.raises(al.exceptions.ParameterError):
                ea.parameters.init_pop_given_phenotypes = ["(-9.120,+2.622)"]
        else:
            ea.parameters.init_pop_given_phenotypes = ["(-9.120,+2.622)"]


# - init_pop_given_genotypes > init_pop_given_phenotypes


def test_init_pop_given_genotypes_or_given_phenotypes_are_used_depending_on_operator_choice():
    for op in ("given_genotypes", "given_phenotypes"):
        population_size = 10
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT,
            shared.OBJ_FUN_FLOAT,
            "min",
            system="ge",
            max_generations=3,
            population_size=population_size,
            verbose=10,
            database_on=True,
        )
        ea.parameters.init_pop_operator = op
        ea.parameters.init_pop_given_genotypes = [[1, 2, 3], [4, 3, 2], [8, 9, 10]]
        ea.parameters.init_pop_given_phenotypes = ["+1.112", "-2.913"]
        ea.run()

        pop0 = ea.database.individuals(generation_range=0)
        if op == "given_genotypes":
            assert len(ea.parameters.init_pop_given_genotypes) == len(pop0)
            assert len(ea.parameters.init_pop_given_phenotypes) != len(pop0)
        else:
            assert len(ea.parameters.init_pop_given_genotypes) != len(pop0)
            assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0)
