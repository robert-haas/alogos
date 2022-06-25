import itertools
import os
import random
import sqlite3
import time
from collections.abc import Callable
from numbers import Number

import pytest
import unified_map as um

import alogos as al

import shared


# Shared

BNF_FLOAT = """
<number> ::= <sign><digit>.<digits>
<sign> ::= +|-
<digits> ::= <digit><digit><digit>
<digit> ::= 0|1|2|3|4|5|6|7|8|9
"""

GRAMMAR_FLOAT = al.Grammar(bnf_text=BNF_FLOAT)


def OBJ_FUN_FLOAT(string):
    x = float(string)
    return abs(x-0.4321)


def OBJ_FUN_FLOAT_DETAILS(string):
    x = float(string)
    return abs(x-0.4321), x  # returns string interpretation as details


BNF_TUPLE = """
<tuple> ::= (<number>, <number>)
<number> ::= <sign><digit>.<digits>
<sign> ::= +|-
<digits> ::= <digit><digit><digit>
<digit> ::= 0|1|2|3|4|5|6|7|8|9
"""

GRAMMAR_TUPLE = al.Grammar(bnf_text=BNF_TUPLE)


def OBJ_FUN_TUPLE(string):
    x, y = eval(string)
    z = (x+0.1234)**2 + (y+0.6123)**2
    return z


def OBJ_FUN_TUPLE_SLOW(string):
    # Artificial delay for performance comparisons of serial versus parallel evaluation
    time.sleep(0.01)
    x, y = eval(string)
    z = (x+0.1234)**2 + (y+0.6123)**2
    return z


def check_algorithm(ea):
    # Type
    assert isinstance(ea, al.EvolutionaryAlgorithm)

    # Attributes
    assert isinstance(ea.default_parameters, al._utilities.parametrization.ParameterCollection)
    assert isinstance(ea.parameters, al._utilities.parametrization.ParameterCollection)
    assert isinstance(ea.parameters.grammar, al.Grammar)
    assert isinstance(ea.parameters.objective_function, Callable)
    assert isinstance(ea.parameters.objective, str)
    assert ea.parameters.objective in ('min', 'max')
    assert not isinstance(ea.parameters.system, str)
    assert isinstance(ea.parameters.evaluator, Callable)

    assert isinstance(ea.state, al._optimization.ea.state.State)
    if ea.state.generation == 0:
        assert ea.state.best_individual is None
    else:
        assert isinstance(
            ea.state.best_individual, ea.parameters.system.representation.Individual)

    if ea.parameters.database_on:
        assert isinstance(ea.database, al._optimization.ea.database.Database)
    else:
        assert ea.database is None

    # Methods
    assert isinstance(ea.is_stop_criterion_met, Callable)
    assert isinstance(ea.reset, Callable)
    assert isinstance(ea.run, Callable)
    assert isinstance(ea.step, Callable)

    # Representations
    s1 = repr(ea)
    s2 = str(ea)
    s3 = shared.call_repr_pretty(ea, cycle=True)
    s4 = shared.call_repr_pretty(ea, cycle=False)
    s5 = repr(ea.parameters)
    s6 = str(ea.parameters)
    s7 = shared.call_repr_pretty(ea.parameters, cycle=True)
    s8 = shared.call_repr_pretty(ea.parameters, cycle=False)
    s9 = repr(ea.state)
    s10 = str(ea.state)
    s11 = shared.call_repr_pretty(ea.state, cycle=True)
    s12 = shared.call_repr_pretty(ea.state, cycle=False)
    if ea.parameters.database_on:
        s13 = repr(ea.database)
        s14 = str(ea.database)
        s15 = shared.call_repr_pretty(ea.database, cycle=True)
        s16 = shared.call_repr_pretty(ea.database, cycle=False)
    else:
        s13, s14, s15, s16 = s9, s10, s11, s12
    for string in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16):
        assert isinstance(string, str)
        assert string
    assert s1.startswith('<EvolutionaryAlgorithm object at') and s1.endswith('>')
    assert s5.startswith('<ParameterCollection object at') and s5.endswith('>')
    assert s9.startswith('<EvolutionaryAlgorithmState object at') and s9.endswith('>')
    if ea.parameters.database_on:
        assert s13.startswith('<EvolutionaryAlgorithmDatabase object at') and s13.endswith('>')

    # Database
    if ea.database is not None:
        check_database(ea)

    # State
    has_run = ea.state.generation > 0
    check_state(ea, has_run)


def check_state(ea, has_run):
    if has_run:
        assert ea.state.generation > 0
        assert ea.state.num_gen_to_phe_evaluations > 0
        assert ea.state.num_phe_to_fit_evaluations > 0
        assert ea.state.best_individual is not None
        assert ea.state.max_individual is not None
        assert ea.state.min_individual is not None
        assert ea.state.population is not None
    else:
        assert ea.state.generation == 0
        assert ea.state.num_gen_to_phe_evaluations == 0
        assert ea.state.num_phe_to_fit_evaluations == 0
        assert ea.state.best_individual is None
        assert ea.state.max_individual is None
        assert ea.state.min_individual is None
        assert ea.state.population is None


def check_database(ea):
    db = ea.database
    system = db._deserializer._system
    for gen_range in [None, 0, (0, 1), [0, 1], (0, None), (None, 0), [None, None]]:
        for only_main in (False, True):
            # Counts
            num_gen = db.num_generations()
            num_ind = db.num_individuals(gen_range, only_main)
            num_gt = db.num_genotypes(gen_range, only_main)
            num_phe = db.num_phenotypes(gen_range, only_main)
            num_fit = db.num_fitnesses(gen_range, only_main)
            num_det = db.num_details(gen_range, only_main)
            num_gen_phe = db.num_gen_to_phe_evaluations()
            num_phe_fit = db.num_phe_to_fit_evaluations()
            counts = (
                num_gen, num_ind, num_gt, num_phe, num_fit, num_det, num_gen_phe, num_phe_fit)
            for val in counts:
                assert isinstance(val, int)
                if num_gen == 0:
                    assert val == 0
                else:
                    assert val > 0

            # Generation
            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db.generation_first()
                with pytest.raises(al.exceptions.DatabaseError):
                    db.generation_last()
            else:
                gen_first = db.generation_first()
                gen_last = db.generation_last()
                if gen_first is not None or gen_last is not None:
                    assert gen_last >= gen_first
                    for val in (gen_first, gen_last):
                        assert isinstance(val, int)
                        assert val >= 0

            # Individual
            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db._individual_max_id(gen_range, only_main)
            else:
                ind_max_id = db._individual_max_id(gen_range, only_main)
                for val in (ind_max_id,):
                    if val is not None:
                        assert isinstance(val, int)
                        assert val >= 0

            ind = db.individuals(gen_range, only_main)
            some_fitness = 0.0 if len(ind) == 0 else ind[0].fitness
            ind_exact = db.individuals_with_given_fitness(some_fitness, gen_range, only_main)
            ind_min = db.individuals_with_min_fitness(gen_range, only_main)
            ind_max = db.individuals_with_max_fitness(gen_range, only_main)
            ind_low = db.individuals_with_low_fitness(
                generation_range=gen_range, only_main=only_main)
            ind_low2 = db.individuals_with_low_fitness(2, gen_range, only_main)
            ind_high = db.individuals_with_high_fitness(
                generation_range=gen_range, only_main=only_main)
            ind_high2 = db.individuals_with_high_fitness(2, gen_range, only_main)
            vals = (ind, ind_exact, ind_min, ind_max, ind_low, ind_low2, ind_high, ind_high2)
            for val in vals:
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                    assert all(isinstance(ind, system.representation.Individual) for ind in val)

            # Population
            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db.population_size_min()
                with pytest.raises(al.exceptions.DatabaseError):
                    db.population_size_max()
            else:
                pop_size_min = db.population_size_min()
                pop_size_max = db.population_size_max()
                for val in (pop_size_min, pop_size_max):
                    if val is not None:
                        assert isinstance(val, int)
                        assert val >= 1

            # Genotype
            gt = db.genotypes(gen_range, only_main)
            gt_exact = db.genotypes_with_given_fitness(some_fitness, gen_range, only_main)
            gt_min = db.genotypes_with_min_fitness(gen_range, only_main)
            gt_max = db.genotypes_with_max_fitness(gen_range, only_main)
            for val in (gt, gt_exact, gt_min, gt_max):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                    assert all(isinstance(gt, system.representation.Genotype) for gt in val)

            # Phenotype
            phe = db.phenotypes(gen_range, only_main)
            phe_exact = db.phenotypes_with_given_fitness(some_fitness, gen_range, only_main)
            phe_min = db.phenotypes_with_min_fitness(gen_range, only_main)
            phe_max = db.phenotypes_with_max_fitness(gen_range, only_main)
            for val in (phe, phe_exact, phe_min, phe_max):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                    assert all(isinstance(phe, str) for phe in val)

            # Details
            det = db.details(gen_range, only_main)
            det_exact = db.details_with_given_fitness(some_fitness, gen_range, only_main)
            det_min = db.details_with_min_fitness(gen_range, only_main)
            det_max = db.details_with_max_fitness(gen_range, only_main)
            for val in (det, det_exact, det_min, det_max):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0

            # Fitness
            fitnesses = db.fitnesses(gen_range, only_main)
            assert isinstance(fitnesses, list)
            if num_gen == 0:
                assert len(fitnesses) == 0
            else:
                assert len(fitnesses) > 0
                assert all(isinstance(val, float) for val in fitnesses)

            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_min(gen_range, only_main)
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_max(gen_range, only_main)
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_min_after_num_evals(2)
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_max_after_num_evals(2)
            else:
                fit_min = db.fitness_min(gen_range, only_main)
                fit_max = db.fitness_max(gen_range, only_main)
                fit_min2 = db.fitness_min_after_num_evals(2)
                fit_max2 = db.fitness_max_after_num_evals(2)
                for val in (fit_min, fit_max, fit_min2, fit_max2):
                    assert isinstance(val, float)
                    assert val == val  # not NaN

            n = 17
            # Genotype-phenotype evaluations
            gt_phe_map1 = db.gen_to_phe_evaluations()
            gt_phe_map2 = db.gen_to_phe_evaluations(n)
            assert gt_phe_map1[:n] == gt_phe_map2
            for val in (gt_phe_map1, gt_phe_map2):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                # Type of genotypes
                assert all(isinstance(row[0], system.representation.Genotype) for row in val)
                # Type of phenotypes
                assert all(isinstance(row[1], str) for row in val)

            # Phenotye-fitness evaluations
            for wd in (True, False):
                phe_fit_map1 = db.phe_to_fit_evaluations(with_details=wd)
                phe_fit_map2 = db.phe_to_fit_evaluations(n, wd)
                assert phe_fit_map1[:n] == phe_fit_map2
                for val in (phe_fit_map1, phe_fit_map2):
                    assert isinstance(val, list)
                    if num_gen == 0:
                        assert len(val) == 0
                    else:
                        assert len(val) > 0
                    # Length of a row depends on with_details being True or False
                    if wd:
                        assert all(len(row) == 3 for row in val)
                    else:
                        assert all(len(row) == 2 for row in val)
                    # Type of phenotypes
                    assert all(isinstance(row[0], str) for row in val)
                    # Type of fitnesses
                    assert all(isinstance(row[1], float) for row in val)

            # Connections between some of these queries
            assert len(set(gt)) == num_gt
            assert len(set(phe)) == num_phe
            assert len([str(val) for val in det]) == num_det
            assert len(set(fitnesses)) == num_fit

            for ind, gt, phe in zip(ind_min, gt_min, phe_min):
                assert ind.genotype == gt
                assert ind.phenotype == phe
                assert ind.fitness == fit_min
            for ind, gt, phe in zip(ind_max, gt_max, phe_max):
                assert ind.genotype == gt
                assert ind.phenotype == phe
                assert ind.fitness == fit_max

            # Checks against some values determined from to_list and to_dataframe
            if random.choice([True, False]):  # random order of calling to_list and to_dataframe
                data = ea.database.to_list(gen_range, only_main)
                df = ea.database.to_dataframe(gen_range, only_main)
            else:
                df = ea.database.to_dataframe(gen_range, only_main)
                data = ea.database.to_list(gen_range, only_main)

            if gen_range is None and len(df) > 0:
                assert len(set([row[2] for row in data])) == df['generation'].nunique() == num_gen
                assert data[0][2] == min(df['generation']) == db.generation_first()
                assert data[-1][2] == max(df['generation']) == db.generation_last()

            assert len(data) == len(df) == num_ind
            assert len(set([row[4] for row in data])) == df['genotype'].nunique() == num_gt
            assert len(set([row[5] for row in data])) == df['phenotype'].nunique() == num_phe
            assert df['fitness'].nunique(dropna=False) == num_fit
            assert df['details'].nunique(dropna=False) == num_det

            assert all(isinstance(pi, list) for pi in df['parent_ids'])
            assert all(isinstance(gn, int) for gn in df['generation'])
            known_labels = ('main', 'parent_selection', 'crossover', 'mutation')
            assert all(label in known_labels for label in df['label'])






# Tests

def test_api():
    # Initialization with minimal required arguments
    # - grammar, objective function, objective
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min')
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, objective='max')
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, objective_function=OBJ_FUN_FLOAT, objective='min')
    ea = al.EvolutionaryAlgorithm(
        grammar=GRAMMAR_FLOAT, objective_function=OBJ_FUN_FLOAT, objective='min')
    check_algorithm(ea)

    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(101010, OBJ_FUN_FLOAT, 'min')
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, 101010, 'max')
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 101010)
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'nonsense')

    # Initialization with all ea-specific parameters
    # - system
    for system in ('cfggp', 'cfggpst', 'ge'):
        ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system)
        ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'max', system=system)
        check_algorithm(ea)

    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system=101010)
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'max', system='nonsense')

    # - evaluator
    def evaluator(func, args):
        return [func(arg) for arg in args]

    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system, evaluator)
    check_algorithm(ea)

    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'max', evaluator=101010)
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', evaluator='nonsense')

    # - database_on
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', database_on=True)
    check_algorithm(ea)

    # - others
    for key, val in al.EvolutionaryAlgorithm.default_parameters.items():
        param = {key: val}
        ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system, **param)
        check_algorithm(ea)

        param = {'nonsense': 101010}
        with pytest.raises(al.exceptions.ParameterError):
            al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'max', system, **param)

    # Initialization with system-specific parameters
    for system in ('cfggp', 'cfggpst', 'ge'):
        module = getattr(al.systems, system)
        for key, val in module.default_parameters.items():
            param = {key: val}
            ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system, **param)
            check_algorithm(ea)

            param = {'nonsense': 101010}
            with pytest.raises(al.exceptions.ParameterError):
                al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system, **param)


def test_reset():
    # init
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2)
    check_algorithm(ea)
    check_state(ea, has_run=False)

    # run
    ea.run()
    check_algorithm(ea)
    check_state(ea, has_run=True)

    # reset
    ea.reset()
    check_algorithm(ea)
    check_state(ea, has_run=False)

    # run
    ea.step()
    check_algorithm(ea)
    check_state(ea, has_run=True)

    # reset
    ea.reset()
    check_algorithm(ea)
    check_state(ea, has_run=False)


def test_is_stop_criterion_met():
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2)
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


def test_usual_run_1():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT_DETAILS, 'min', database_on=True, max_generations=3)
    ea.run()
    assert ea.state.best_individual.fitness >= 0.0
    check_algorithm(ea)


def test_usual_run_2():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_TUPLE, OBJ_FUN_TUPLE, 'min', database_on=True, max_fitness_evaluations=321)
    ea.run()
    check_algorithm(ea)


def test_usual_run_3():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_TUPLE, OBJ_FUN_TUPLE, 'min', database_on=True, max_runtime_in_seconds=0.14)
    ea.run()
    check_algorithm(ea)


def test_usual_run_4():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_TUPLE, OBJ_FUN_TUPLE, 'min', database_on=True, max_or_min_fitness=0.01)
    ea.run()
    check_algorithm(ea)



# Parameters of the algorithm

def test_parameter_grammar():
    # Minimal
    grammar = al.Grammar(bnf_text='<S> ::= 1')
    ea = al.EvolutionaryAlgorithm(grammar, lambda s: int(s), 'min', max_generations=1)
    best_ind = ea.run()
    assert best_ind.fitness == 1.0

    # Error: invalid type
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm('nonsense', OBJ_FUN_FLOAT, 'min', max_generations=1)


def test_parameter_objective_function():
    # Lambda function
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, lambda string: float(string)**2, 'min', max_or_min_fitness=0.0)
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Function
    def square(string):
        return float(string)**2

    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, square, 'min', max_or_min_fitness=0.0)
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Method
    class Carrier:
        def square(self, string):
            return float(string)**2

    carrier = Carrier()
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, carrier.square, 'min', max_or_min_fitness=0.0)
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Callable class
    class Squarer:
        def __call__(self, string):
            return float(string)**2

    squarer = Squarer()
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, squarer, 'min', max_or_min_fitness=0.0)
    best_ind = ea.run()
    assert best_ind.fitness == 0.0

    # Error: class that is not callable
    class NonSquarer:
        pass

    non_squarer = NonSquarer()
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, non_squarer, 'min', max_or_min_fitness=0.0)


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
    ea = al.EvolutionaryAlgorithm(grammar, obj_fun, 'min', max_generations=50)
    ea.run()
    assert ea.state.best_individual.fitness == 0.0

    # max
    ea = al.EvolutionaryAlgorithm(grammar, obj_fun, 'max', max_generations=50)
    ea.run()
    assert ea.state.best_individual.fitness == 99.0

    # Error: invalid type
    with pytest.raises(TypeError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 42, max_generations=1)

    # Error: invalid value
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'nonsense', max_generations=1)


def test_parameter_system():
    # Chosen system influences available parameters
    systems = ('cfggp', 'cfggpst', 'ge')
    params_repr = set()
    for system in systems:
        ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system)
        params_repr.add(str(ea.parameters))  # parameters are specific for the chosen system
    assert len(params_repr) == len(systems)  # therefore the set has as many different entries

    # Error: invalid type
    for system in (0, 0.0, [], {}):
        with pytest.raises(TypeError):
            al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system)

    # Error: invalid value
    with pytest.raises(ValueError):
        al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', 'nonsense')


def test_parameter_evaluator():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_TUPLE,
        OBJ_FUN_TUPLE_SLOW,
        'min',
        max_generations=3,
        population_size=8,
    )

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
        GRAMMAR_TUPLE,
        OBJ_FUN_TUPLE_SLOW,
        'min',
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
        ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2)
        ea.step()
    shared.prints_nothing_to_stdout(func, capsys)

    # off
    for verbose in (False, 0):
        def func():
            ea = al.EvolutionaryAlgorithm(
                GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2, verbose=verbose)
            ea.step()
        shared.prints_nothing_to_stdout(func, capsys)

    # on
    for verbose in (True, 1, 2, 3, 4):
        def func():
            ea = al.EvolutionaryAlgorithm(
                GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2, verbose=verbose)
            ea.step()
        shared.prints_to_stdout(func, capsys, partial_message='Progress         Generations')

    # high level
    def func():
        ea = al.EvolutionaryAlgorithm(
            GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2, verbose=2)
        ea.run()
    shared.prints_to_stdout(func, capsys, partial_message='Progress         Generations')
    # TODO: Different verbosity levels still available?

    # Error: invalid value
    with pytest.raises(ValueError):
        ea = al.EvolutionaryAlgorithm(
            GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2, verbose=-1)

    # Error: invalid type
    for verbose in (2.0, None, [], {}, 'nonsense'):
        with pytest.raises(TypeError):
            ea = al.EvolutionaryAlgorithm(
                GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2, verbose=verbose)



# Parameters of the G3P systems

def test_parameter_init_pop_given_genotypes():
    population_size = 10
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT,
        OBJ_FUN_FLOAT,
        'min',
        system='ge',
        database_on=True,
        max_generations=5,
        population_size=population_size,
        verbose=10,
        init_pop_operator='given_genotypes',
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
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=5,
        init_pop_given_genotypes=42, init_pop_operator='given_genotypes')
    with pytest.raises(al.exceptions.InitializationError):
        ea.step()


def test_parameter_init_pop_given_genotypes_error_because_one_element_not_valid():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', system='ge', max_generations=5)
    ea.parameters.population_size = 10
    ea.parameters.init_pop_operator = 'given_genotypes'
    ea.parameters.init_pop_given_genotypes = [(0, 1, 2)] * 10
    ea.parameters.init_pop_given_genotypes[3] = []
    with pytest.raises(al.exceptions.InitializationError):
        ea.run()


def test_parameter_init_pop_given_phenotypes():
    population_size = 10
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_TUPLE,
        OBJ_FUN_TUPLE,
        'min',
        max_generations=5,
        population_size=population_size,
        verbose=10,
        database_on=True,
        init_pop_operator='given_phenotypes',
    )

    # Case 1: Fewer genotypes than population_size
    ea.parameters.init_pop_given_phenotypes = ['(+1.112,-2.914)']
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0) < population_size
    for ind in pop0:
        assert ind.phenotype in ea.parameters.init_pop_given_phenotypes

    # Case 2: More genotypes than population_size
    ea.reset()
    ea.parameters.init_pop_given_phenotypes = ['(+1.112,-2.914)', '(-1.122,-3.143)'] * 40
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0) > population_size
    for ind in pop0:
        assert ind.phenotype in ea.parameters.init_pop_given_phenotypes

    # Case 3: Number of genotypes is equal to population_size
    ea.reset()
    ea.parameters.init_pop_given_phenotypes = ['(-9.120,+2.622)', '(+5.671,+3.674)'] * 5
    ea.run()
    pop0 = ea.database.individuals(generation_range=0)
    assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0) == population_size
    for ind in pop0:
        assert ind.phenotype in ea.parameters.init_pop_given_phenotypes


def test_parameter_init_pop_given_phenotypes_error_because_one_element_not_valid():
    population_size = 10
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_TUPLE,
        OBJ_FUN_TUPLE,
        'min',
        max_generations=5,
        population_size=population_size,
        verbose=10,
        init_pop_operator='given_phenotypes',
    )
    ea.parameters.init_pop_given_phenotypes = ['(+1.112,-2.914)', 'invalid, parser fails']
    with pytest.raises(al.exceptions.InitializationError):
        ea.run()


def test_parameter_init_pop_given_phenotypes_error_because_system_does_not_have_reverse_mapping():
    for sys in ('cfggp', 'cfggpst', 'ge'):
        population_size = 10
        ea = al.EvolutionaryAlgorithm(
            GRAMMAR_TUPLE,
            OBJ_FUN_TUPLE,
            'min',
            system=sys,
            max_generations=5,
            population_size=population_size,
            verbose=10,
            init_pop_operator='given_phenotypes',
        )
        ea.parameters.init_pop_given_phenotypes = ['(-9.120,+2.622)']


# - init_pop_given_genotypes > init_pop_given_phenotypes

def test_init_pop_given_genotypes_or_given_phenotypes_are_used_depending_on_operator_choice():
    for op in ('given_genotypes', 'given_phenotypes'):
        population_size = 10
        ea = al.EvolutionaryAlgorithm(
            GRAMMAR_FLOAT,
            OBJ_FUN_FLOAT,
            'min',
            system='ge',
            max_generations=3,
            population_size=population_size,
            verbose=10,
            database_on=True,
        )
        ea.parameters.init_pop_operator = op
        ea.parameters.init_pop_given_genotypes = [[1, 2, 3], [4, 3, 2], [8, 9, 10]]
        ea.parameters.init_pop_given_phenotypes = ['+1.112', '-2.913']
        ea.run()

        pop0 = ea.database.individuals(generation_range=0)
        if op == 'given_genotypes':
            assert len(ea.parameters.init_pop_given_genotypes) == len(pop0)
            assert len(ea.parameters.init_pop_given_phenotypes) != len(pop0)
        else:
            assert len(ea.parameters.init_pop_given_genotypes) != len(pop0)
            assert len(ea.parameters.init_pop_given_phenotypes) == len(pop0)



# Operators

@pytest.mark.parametrize(
    'system, crossover_operators, mutation_operators, op_params',
    [
        (
            'cfggp',
            ['subtree_exchange'],
            ['subtree_replacement'],
            dict(
                mutation_subtree_replacement_count=2,
                # TODO: crossover probability missing
            ),
        ),
        (
            'cfggpst',
            ['subtree_exchange'],
            ['subtree_replacement'],
            dict(
                mutation_subtree_replacement_count=2,
            ),
        ),
        (
            'ge',
            ['two_point_length_preserving'],
            ['int_replacement_by_probability', 'int_replacement_by_count'],
            dict(
                mutation_int_replacement_probability=0.13,
                mutation_int_replacement_count=2,
                # TODO: crossover probability missing
            ),
        ),
    ]
)
def test_operators_all_combinations(system, crossover_operators, mutation_operators, op_params):
    parent_selection_operators = survivor_selection_operators = [
        'uniform', 'truncation', 'tournament', 'rank_proportional', 'fitness_proportional']
    pooling_operators = [
        'overlapping', 'non_overlapping', 'steady_state']

    # Test all combinations of available operators
    num_generations = 5
    population_size = 5
    offspring_size = random.choice([population_size, population_size-1, population_size+1])
    operators = [
        parent_selection_operators, crossover_operators, mutation_operators,
        pooling_operators, survivor_selection_operators]
    init_params = dict(
        init_pop_given_phenotypes=['+9.999']*population_size,
        init_pop_operator='given_phenotypes',
    )

    print(system)
    print('parent_selection', 'crossover', 'mutation', 'pool', 'survivor_selection', sep='\t')
    for par, cross, mut, pool, surv in itertools.product(*operators):
        print(par, cross, mut, pool, surv, sep='\t')

        ea = al.EvolutionaryAlgorithm(
            grammar=GRAMMAR_FLOAT,
            objective_function=OBJ_FUN_FLOAT,
            objective='min',
            system=system,

            population_size=population_size,
            offspring_size=offspring_size,
            max_generations=num_generations,

            parent_selection_operator=par,
            crossover_operator=cross,
            mutation_operator=mut,
            survivor_selection_pooling=pool,
            survivor_selection_operator=surv,

            **op_params,
            **init_params,
        )
        best_individual = ea.run()
        assert best_individual.fitness < 9.7


def test_operators_improvement():
    # Test search with some meaningful settings
    combinations = [
        ['truncation', 'two_point_length_preserving', 'int_replacement_by_count', 'overlapping', 'tournament', 8, 8],
        #['truncation', 'two_point', 'uniform', 'overlapping', 'tournament', 8, 8],
        #['truncation', 'one_point', 'one_point', 'non_overlapping', 'tournament', 8, 8],
        #['uniform', 'one_point', 'one_point', 'overlapping', 'truncation', 8, 4],
        #['uniform', 'two_point', 'best_neighborhood', 'overlapping', 'truncation', 1, 1]
    ]
    for par, cross, mut, pool, surv, popsize, offsize in combinations:
        # Setup
        print(par, cross, mut, pool, surv, popsize, offsize)
        ea = al.EvolutionaryAlgorithm(
            GRAMMAR_FLOAT,
            OBJ_FUN_FLOAT,
            'min',
            'ge',
            init_pop_given_phenotypes=['(+9.998,+9.999)']*popsize,
            parent_selection_operator=par,
            crossover_operator=cross,
            mutation_operator=mut,
            survivor_selection_pooling=pool,
            survivor_selection_operator=surv,
            population_size=popsize,
            offspring_size=offsize,
            max_fitness_evaluations=200,
        )

        # Run
        first_best_ind = ea.step()
        final_best_ind = ea.run()

        # Check
        if final_best_ind.greater_than(first_best_ind, 'min') and mut is not None:
            raise ValueError('Best fitness in last generation ({}) is not better than in first '
                             '({}).'.format(final_best_ind.fitness, first_best_ind.fitness))


@pytest.mark.xfail  # TODO: parameter crossover_subtree_replacement_count is not used yet
def test_operators_no_change_if_only_crossover_and_probability_0():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=4, population_size=10)
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
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=4, population_size=10)
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
            GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=4,
            crossover_operator=None, mutation_operator=None)
        ea.run()


def test_operators_no_variation_2():
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=4)
    ea.parameters.crossover_operator = None
    ea.parameters.mutation_operator = None
    with pytest.raises(al.exceptions.ParameterError):
        ea.run()


def test_operators_no_variation_3():
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=4)
    ea.parameters.crossover_operator = None
    ea.parameters.mutation_operator = None
    with pytest.raises(al.exceptions.ParameterError):
        ea.step()
        ea.step()


def test_operators_general():
    num_gen = 3
    pop_size = 7
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min')
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


def test_evaluation_caching():
    for cache1 in (True, False):
        for cache2 in (True, False):
            ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=20)
            ea.parameters.gen_to_phe_cache_lookup_on = cache1
            ea.parameters.phe_to_fit_cache_lookup_on = cache2
            ea.parameters.evaluator = lambda f, al: [f(a) for a in al]
            ea.run()


def test_phenotype_fitness_evaluation_1():
    def func_returning_float(string):
        x, y = eval(string)
        z = x**2 + y**2
        return z

    ea = al.EvolutionaryAlgorithm(GRAMMAR_TUPLE, func_returning_float, 'max', max_generations=10)
    ea.run()
    fitness = ea.state.best_individual.fitness
    details = ea.state.best_individual.details
    assert fitness > 0.0
    assert isinstance(details, dict)
    assert len(details) == 3
    assert details['evaluation'] is None


def test_phenotype_fitness_evaluation_2():
    some_list = [42, 'cde', None, [], ('x', 2)]
    some_tuple = tuple(some_list)
    some_dict = dict(a=42, b='cde', c=None, d=[], e=('x', 2))

    for extra_val in [42, 3.14, 'abc', [], (), None, some_list, some_tuple, some_dict]:
        def func_returning_dict(string):
            x, y = eval(string)
            z = x**2 + y**2
            return z, extra_val

        ea = al.EvolutionaryAlgorithm(GRAMMAR_TUPLE, func_returning_dict, 'max', max_generations=10)
        ea.run()
        fitness = ea.state.best_individual.fitness
        details = ea.state.best_individual.details
        assert fitness > 0.0
        assert isinstance(details, dict)
        assert len(details) == 3
        assert details['evaluation'] == extra_val


def test_phenotype_fitness_evaluation_3():
    def func_returning_nothing(string):
        pass

    for objective, inf_val in [('max', '-inf'), ('min', '+inf')]:
        ea = al.EvolutionaryAlgorithm(
            GRAMMAR_TUPLE, func_returning_nothing, objective, max_generations=10)
        ea.run()
        fitness = ea.state.best_individual.fitness
        details = ea.state.best_individual.details
        assert fitness == float(inf_val)
        assert isinstance(details, dict)
        assert len(details) == 3
        assert details['evaluation'] == 'ValueError: Returned fitness value is not a number: None'


def test_phenotype_fitness_evaluation_4():
    def func_raising_exception(string):
        raise ValueError('Something bad happened')
        return 42

    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_TUPLE, func_raising_exception, 'max', max_generations=10)
    ea.run()
    fitness = ea.state.best_individual.fitness
    details = ea.state.best_individual.details
    assert fitness == float('-inf')
    assert isinstance(details, dict)
    assert len(details) == 3
    assert details['evaluation'] == 'ValueError: Something bad happened'


def test_phenotype_fitness_evaluation_5():
    def func_returning_not_a_number(string):
        return 'Nonsense', dict(b=4, x=8, z=22)

    ea = al.EvolutionaryAlgorithm(GRAMMAR_TUPLE, func_returning_not_a_number, 'min', max_generations=10)
    ea.run()
    fitness = ea.state.best_individual.fitness
    details = ea.state.best_individual.details
    assert fitness == float('inf')
    assert isinstance(details, dict)
    assert len(details) == 3
    assert details['evaluation'] == 'ValueError: Returned fitness value is not a number: Nonsense'





# State

def test_state():
    # init
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=2)
    check_algorithm(ea)
    check_state(ea, has_run=False)

    # run
    ea.step()
    check_algorithm(ea)
    check_state(ea, has_run=True)

    # run continued
    ea.step()
    ea.step()
    check_algorithm(ea)
    check_state(ea, has_run=True)

    # reset
    ea.reset()
    check_algorithm(ea)
    check_state(ea, has_run=False)





# Database

def test_database_on_and_location(tmpdir):
    # No database
    # - implicit (default)
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=1)
    assert ea.database is None
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    # - explicit
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=1, database_on=False)
    assert ea.database is None
    ea.run()
    assert ea.state.best_individual.fitness >= 0

    # SQLite database in RAM
    # - implicit (default if database_on=True)
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=1, database_on=True)
    assert ea.database is not None
    check_database(ea)
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    check_database(ea)
    # - explicit
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=1,
        database_on=True, database_location=':memory:')
    assert ea.database is not None
    check_database(ea)
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    check_database(ea)

    # SQLite database on disk
    filepath = os.path.join(tmpdir.strpath, 'test.db')
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=1,
        database_on=True, database_location=filepath)
    assert ea.database is not None
    check_database(ea)
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0
    check_database(ea)


def test_database_export_sql_creates_valid_sql_file(tmpdir):
    # See if the SQLite3 database file is created and can be read
    def check_database_file(filepath):
        assert os.path.isfile(filepath)
        connection = sqlite3.connect(filepath)
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM full_search')
        data = cursor.fetchall()
        assert len(data) > 20

    filepath = os.path.join(tmpdir.strpath, 'dummy1.db')
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT,
        OBJ_FUN_FLOAT,
        'min',
        database_on=True,
        database_location=filepath,
        max_generations=1,
    )
    ea.run()
    check_database_file(filepath)


@pytest.mark.xfail  # TODO: failure modes not clear at the moment
def test_database_export_sql_errors(tmpdir):
    def func():
        filepath = os.path.join(tmpdir.strpath, 'dummy1.db')
        ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min',
                                      database_on=True, database_location=filepath)
        ea.database.export_sql(filepath)

    shared.emits_error(
            function=func,
            error_type=al.exceptions.DatabaseError,
            expected_message=message)


def test_database_generation_range_errors():
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, lambda s: float(s), 'min', database_on=True, max_generations=2)
    ea.run()

    method_names = [
        'num_individuals',
        'num_genotypes',
        'num_phenotypes',
        'num_fitnesses',
        'individuals',
        'individuals_with_min_fitness',
        'individuals_with_max_fitness',
        'individuals_with_low_fitness',
        'individuals_with_high_fitness',
        'genotypes',
        'genotypes_with_min_fitness',
        'genotypes_with_max_fitness',
        'phenotypes',
        'phenotypes_with_min_fitness',
        'phenotypes_with_max_fitness',
        'details',
        'details_with_min_fitness',
        'details_with_max_fitness',
        'fitnesses',
        'fitness_min',
        'fitness_max',
    ]
    for method_name in method_names:
        method = getattr(ea.database, method_name)

        for generation_range in (3.14, ):
            with pytest.raises(TypeError):
                method(generation_range=generation_range)
        for generation_range in ('x', ('x', 5), (5, 'y'), (3.14, 5), (5, 3.14), (1,), (1, 2, 3)):
            with pytest.raises(ValueError):
                method(generation_range=[1, 2, 3])


def test_database_fitness_min_and_fitness_max_errors():
    ea = al.EvolutionaryAlgorithm(GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', database_on=True)
    ea.step()

    # Expected errors
    methods = [
        ea.database.fitness_min_after_num_evals,
        ea.database.fitness_max_after_num_evals,
    ]
    for fun in methods:
        value = fun(1)
        assert isinstance(value, float)
        assert value == value  # not NaN
        with pytest.raises(TypeError):
            fun('nonsense')
        with pytest.raises(ValueError):
            fun(0)
        with pytest.raises(ValueError):
            fun(-1)


@pytest.mark.skip  # TODO: reasons for db failures not clear at the moment
@pytest.mark.parametrize(
    'params',
    [
        (dict()),
        (dict(phe_to_fit_cache_size=3)),
        (dict(gen_to_phe_cache_size=3)),
        (dict(parent_selection_operator='rank_proportional')),
        (dict(parent_selection_operator='tournament')),
        (dict(parent_selection_operator='truncation')),
        (dict(parent_selection_operator='uniform')),
        (dict(crossover_operator=None)),
        (dict(mutation_operator=None)),
        (dict(survivor_selection_operator='rank_proportional')),
        (dict(survivor_selection_operator='tournament')),
        (dict(survivor_selection_operator='truncation')),
        (dict(survivor_selection_operator='uniform')),
    ]
)
def test_database_genotype_to_phenotype_and_phenotype_to_fitness_evaluation_order(params):
    bnf = """
    <tuple> ::= <number>
    <number> ::= <sign><digit>.<digit> | nan | inf | -inf
    <sign> ::= +|-
    <digit> ::= 0|1|2|3|4|5|6|7|8|9
    """
    grammar = al.Grammar(bnf_text=bnf)

    max_generations = 21

    for cache_on in (True, False):
        for db_on in (True, False):
            for system in ('ge', 'cfggp', 'cfggpst'):
                # 1) Algorithm with caching and db

                # Objective function
                tracked = []
                def obj_fun(string):
                    phe = string
                    fit = float(string)
                    tracked.append((phe, fit))
                    return fit

                # Algorithm
                ea = al.EvolutionaryAlgorithm(
                    grammar, obj_fun, 'min',
                    system=system,
                    max_generations=max_generations,
                    population_size=random.choice(range(13, 23)),
                    offspring_size=random.choice(range(13, 23)),
                    database_on=True,
                    phe_to_fit_database_lookup_on=db_on,
                    phe_to_fit_cache_lookup_on=cache_on,
                    **params)
                ea.run()

                # 1) Genotype-to-phenotype evaluations
                # - Evaluations queried from database by dedicated method
                db_eval_unique = ea.database.gen_to_phe_evaluations()
                num_db_eval_unique = ea.database.num_gen_to_phe_evaluations()
                assert len(db_eval_unique) == num_db_eval_unique

                # - Evaluations extracted from dataframe
                df = ea.database.to_dataframe()
                if 'mutation_operator' not in params or params['mutation_operator'] is not None:
                    # Crossover individuals are evaluated only if they are not changed by mutation
                    df = df[df['label'] != 'crossover']
                df_gen_phe_values = df[['genotype', 'phenotype']].values.tolist()
                df_eval = [(gen, phe) for gen, phe in df_gen_phe_values]
                df_eval_unique = shared.filter_list_unique(df_eval)
                num_df_eval = len(df_eval)
                num_df_eval_unique = len(df_eval_unique)
                assert num_df_eval > num_df_eval_unique
                assert df_eval != df_eval_unique

                # - Evaluation number stored in state (tracks actual objective function calls)
                num_state_eval = ea.state.num_gen_to_phe_evaluations

                # Consistency between results retrieved with different methods
                assert num_db_eval_unique == num_df_eval_unique
                assert db_eval_unique == df_eval_unique
                if 'gen_to_phe_cache_size' in params:
                    assert num_state_eval > num_db_eval_unique  # due to partial lookup
                else:
                    assert num_state_eval == num_db_eval_unique  # due to full lookup

                # 2) Phenotype-to-fitness evaluations
                # - Evaluations tracked directly in objective function calls
                tracked_eval = [(phe, fit if fit==fit else float('inf'))  # NaN to +Inf as in EA
                                for phe, fit in tracked]
                tracked_eval_unique = shared.filter_list_unique(tracked_eval)
                num_tracked_eval = len(tracked_eval)
                num_tracked_eval_unique = len(tracked_eval_unique)
                if 'phe_to_fit_cache_size' in params:
                    if db_on:
                        assert num_tracked_eval == num_tracked_eval_unique  # db lookup
                        assert tracked_eval == tracked_eval_unique
                    else:
                        assert num_tracked_eval > num_tracked_eval_unique  # limited cache lookup
                        assert tracked_eval != tracked_eval_unique
                elif cache_on or db_on:
                    assert num_tracked_eval == num_tracked_eval_unique  # cache or db lookup
                    assert tracked_eval == tracked_eval_unique
                else:
                    assert num_tracked_eval > num_tracked_eval_unique  # no lookup, repeated calc
                    assert tracked_eval != tracked_eval_unique

                # - Evaluations queried from database by dedicated method
                db_eval_unique = ea.database.phe_to_fit_evaluations()
                num_db_eval_unique = ea.database.num_phe_to_fit_evaluations()
                assert len(db_eval_unique) == num_db_eval_unique

                # - Evaluations extracted from dataframe
                df = ea.database.to_dataframe()
                if 'mutation_operator' not in params or params['mutation_operator'] is not None:
                    # Crossover individuals are evaluated only if they are not changed by mutation
                    df = df[df['label'] != 'crossover']
                df_phe_fit_values = df[['phenotype', 'fitness']].values.tolist()
                df_eval = [(phe, fit) for phe, fit in df_phe_fit_values]
                df_eval_unique = shared.filter_list_unique(df_eval)
                num_df_eval = len(df_eval)
                num_df_eval_unique = len(df_eval_unique)
                assert num_df_eval > num_df_eval_unique
                assert df_eval != df_eval_unique

                # - Evaluation number stored in state (tracks actual objective function calls)
                num_state_eval = ea.state.num_phe_to_fit_evaluations

                # Consistency between results retrieved with different methods
                assert num_tracked_eval == num_state_eval
                assert num_tracked_eval_unique == num_db_eval_unique == num_df_eval_unique
                if 'phe_to_fit_cache_size' in params:
                    if db_on:
                        assert num_tracked_eval == num_tracked_eval_unique  # due to full lookup
                        assert tracked_eval == tracked_eval_unique == db_eval_unique \
                            == df_eval_unique
                    else:
                        assert num_tracked_eval > num_tracked_eval_unique  # due to partial lookup
                        assert tracked_eval != tracked_eval_unique == db_eval_unique \
                            == df_eval_unique
                elif cache_on or db_on:
                    assert num_tracked_eval == num_tracked_eval_unique  # due to full lookup
                    assert tracked_eval == tracked_eval_unique == db_eval_unique == df_eval_unique
                else:
                    assert num_tracked_eval > num_tracked_eval_unique  # due to no lookup
                    assert tracked_eval != tracked_eval_unique == db_eval_unique == df_eval_unique

                # Consistency with fitness_min_after_num_evals and fitness_max_after_num_evals
                num_evaluations = ea.database.num_phe_to_fit_evaluations()
                tracked_fitnesses = [fit for phe, fit in tracked_eval_unique]
                db_fitnesses = [fit for phe, fit in db_eval_unique]
                for n in range(1, num_evaluations, 7):
                    fit_min = ea.database.fitness_min_after_num_evals(n)
                    fit_max = ea.database.fitness_max_after_num_evals(n)
                    phe_fit_n = ea.database.phe_to_fit_evaluations(num_evaluations=n)
                    phe_fit_det_n = ea.database.phe_to_fit_evaluations(
                        num_evaluations=n, with_details=True)
                    fit_min2 = min(fit for phe, fit in phe_fit_n)
                    fit_min3 = min(fit for phe, fit, det in phe_fit_det_n)
                    fit_max2 = max(fit for phe, fit in phe_fit_n)
                    fit_max3 = max(fit for phe, fit, det in phe_fit_det_n)
                    assert fit_min == fit_min2 == fit_min3 == \
                        min(tracked_fitnesses[:n]) == min(db_fitnesses[:n])
                    assert fit_max == fit_max2 == fit_max3 == \
                        max(tracked_fitnesses[:n]) == max(db_fitnesses[:n])


def test_plot_genealogy(tmpdir):
    ea = al.EvolutionaryAlgorithm(
        GRAMMAR_FLOAT, OBJ_FUN_FLOAT, 'min', max_generations=3, database_on=True)
    ea.run()
    for backend in (None, 'd3', 'vis', 'three'):
        if backend:
            fig = ea.database.plot_genealogy(backend=backend)
        else:
            fig = ea.database.plot_genealogy()
        
        filepath = os.path.join(tmpdir.strpath, str(backend) + '.html')
        assert fig.to_html()
