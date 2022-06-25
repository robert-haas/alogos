import ast
import copy
import itertools
import json
import math
import os
import random

import pytest

import alogos as al

import shared


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, 'in')


# Shared

def check_genotype(gt):
    assert isinstance(gt, al.systems.ge.representation.Genotype)
    assert isinstance(gt.data, tuple)
    assert all(isinstance(codon, int) for codon in gt.data)
    assert len(gt) == len(gt.data) > 0


def check_phenotype(phenotype):
    # TODO: depends on representation of invalid phenotype
    if phenotype is not None:
        assert isinstance(phenotype, str)
        assert len(phenotype) > 0


def check_fitness(fitness):
    assert isinstance(fitness, float)


def check_individual(ind):
    assert isinstance(ind, al.systems.ge.representation.Individual)
    check_genotype(ind.genotype)
    check_phenotype(ind.phenotype)
    check_fitness(ind.fitness)


def check_population(pop):
    assert isinstance(pop, al.systems.ge.representation.Population)
    assert len(pop) > 0
    for ind in pop:
        check_individual(ind)


# Representation

def test_representation_genotype():
    # Genotypic data of four types:
    # 1) tuple of int 2) string thereof 3) list of int 4) string thereof
    data_variants = (
        (0,),
        (42,),
        (0, 8, 15),
        '(0,)',
        '(42,)',
        '(0, 8, 15)',
        [0],
        [42],
        [0, 8, 15],
        '[0]',
        '[42]',
        '[0, 8, 15]',
    )
    for data in data_variants:
        gt = al.systems.ge.representation.Genotype(data)
        check_genotype(gt)
        # Printing
        assert isinstance(str(gt), str)
        assert isinstance(repr(gt), str)
        assert repr(gt).startswith('<GE genotype at ')
        p1 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p1, False)
        assert p1.string == str(gt)
        p2 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p2, True)
        assert p2.string == '...'
        # Length
        assert len(gt) > 0
        if isinstance(data, (tuple, list)):
            assert len(gt) == len(data)
        # Copying and equality
        gt2 = gt.copy()
        gt3 = copy.copy(gt)
        gt4 = copy.deepcopy(gt)
        assert id(gt) != id(gt2) != id(gt3) != id(gt4)  # new Genotype object
        assert id(gt.data) == id(gt2.data) == id(gt3.data) == id(gt4.data)  # same tuple
        assert gt != 'nonsense'
        assert not gt == 'nonsense'
        assert gt == gt2 == gt3 == gt4
        assert len(gt) == len(gt2) == len(gt3) == len(gt4)
        gt = al.systems.ge.representation.Genotype((1, 2, 3, 4, 5, 6))
        assert gt != gt2 == gt3 == gt4
        assert len(gt) != len(gt2) == len(gt3) == len(gt4)
        # Usage as key
        some_dict = dict()
        some_set = set()
        for i, gt in enumerate([gt, gt2, gt3, gt4]):
            some_dict[gt] = i
            some_set.add(gt)
        assert len(some_dict) == len(some_set) == 2
        # Immutability
        with pytest.raises(al.exceptions.GenotypeError):
            gt.data = 'anything'

    invalid_data_variants = (
        '',
        (),
        [],
        False,
        True,
        None,
        1,
        3.14,
    )
    for data in invalid_data_variants:
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.ge.representation.Genotype(data)


def test_representation_individual():
    data_variants = (
        [],
        ['gt'],
        ['gt', 'phe'],
        ['gt', 'phe', 'fit'],
        ['gt', 'phe', 'fit', 'det'],
    )
    for data in data_variants:
        ind = al.systems.ge.representation.Individual(*data)
        # Member types
        assert ind.genotype is None if len(data) < 1 else data[0]
        assert ind.phenotype is None if len(data) < 2 else data[1]
        assert math.isnan(ind.fitness) if len(data) < 3 else data[2]
        assert isinstance(ind.details, dict) if len(data) < 4 else data[3]
        # Printing
        assert isinstance(str(ind), str)
        assert isinstance(repr(ind), str)
        assert str(ind).startswith('GE individual:')
        assert repr(ind).startswith('<GE individual object at ')
        p1 = shared.MockPrettyPrinter()
        ind._repr_pretty_(p1, False)
        assert p1.string == str(ind)
        p2 = shared.MockPrettyPrinter()
        ind._repr_pretty_(p2, True)
        assert p2.string == '...'
        # Copying
        ind2 = ind.copy()
        ind3 = copy.copy(ind)
        ind4 = copy.deepcopy(ind)
        ind.genotype = 42
        ind.phenotype = 42
        ind.fitness = 42
        ind.details = 42
        assert ind.genotype != ind2.genotype == ind3.genotype == ind4.genotype
        assert ind.phenotype != ind2.phenotype == ind3.phenotype == ind4.phenotype
        assert ind.fitness != ind2.fitness
        assert ind.fitness != ind3.fitness
        assert ind.fitness != ind4.fitness
        assert ind.details != ind2.details == ind3.details == ind4.details
        # Usage as key
        some_dict = dict()
        some_set = set()
        for i, individual in enumerate([ind, ind2, ind3, ind4]):
            some_dict[individual] = i
            some_set.add(individual)

    # Fitness comparison
    # - Case 1: two numbers
    ind1 = al.systems.ge.representation.Individual(fitness=1)
    ind2 = al.systems.ge.representation.Individual(fitness=2)
    assert ind1.less_than(ind2, 'min')
    assert ind1.less_than(ind2, 'max')
    assert ind2.greater_than(ind1, 'min')
    assert ind2.greater_than(ind1, 'max')
    # - Case 2: number and NaN
    ind1 = al.systems.ge.representation.Individual(fitness=1)
    ind2 = al.systems.ge.representation.Individual(fitness=float('nan'))
    assert ind1.less_than(ind2, 'min')
    assert not ind1.less_than(ind2, 'max')
    assert ind2.greater_than(ind1, 'min')
    assert not ind2.greater_than(ind1, 'max')
    # - Case 3: NaN and number
    ind1 = al.systems.ge.representation.Individual(fitness=float('nan'))
    ind2 = al.systems.ge.representation.Individual(fitness=2)
    assert not ind1.less_than(ind2, 'min')
    assert ind1.less_than(ind2, 'max')
    assert not ind2.greater_than(ind1, 'min')
    assert ind2.greater_than(ind1, 'max')
    # - Case 4: NaN and NaN
    ind1 = al.systems.ge.representation.Individual(fitness=float('nan'))
    ind2 = al.systems.ge.representation.Individual(fitness=float('nan'))
    assert not ind1.less_than(ind2, 'min')
    assert not ind1.less_than(ind2, 'max')
    assert not ind2.greater_than(ind1, 'min')
    assert not ind2.greater_than(ind1, 'max')
    # Invalid objective - check removed in methods for performance improvement
    # with pytest.raises(ValueError):
    #     assert ind1.less_than(ind2, 'nonsense')
    # with pytest.raises(ValueError):
    #     assert ind2.greater_than(ind1, 'nonsense')


def test_representation_population():
    data_variants = (
        [],
        [al.systems.ge.representation.Individual('gt1')],
        [al.systems.ge.representation.Individual('gt1'),
         al.systems.ge.representation.Individual('gt2')],
    )
    for data in data_variants:
        pop = al.systems.ge.representation.Population(data)
        # Member types
        assert isinstance(pop.individuals, list)
        # Printing
        assert isinstance(str(pop), str)
        assert isinstance(repr(pop), str)
        assert str(pop).startswith('GE population:')
        assert repr(pop).startswith('<GE population at')
        p1 = shared.MockPrettyPrinter()
        pop._repr_pretty_(p1, False)
        assert p1.string == str(pop)
        p2 = shared.MockPrettyPrinter()
        pop._repr_pretty_(p2, True)
        assert p2.string == '...'
        # Length
        assert len(pop) == len(data)
        # Copying
        pop2 = pop.copy()
        pop3 = copy.copy(pop)
        pop4 = copy.deepcopy(pop)
        assert id(pop.individuals) != id(pop2.individuals) != id(pop3.individuals) \
            != id(pop4.individuals)
        pop.individuals = [
            al.systems.ge.representation.Individual('gt3'),
            al.systems.ge.representation.Individual('gt4'),
            al.systems.ge.representation.Individual('gt5'),
        ]
        assert len(pop) != len(pop2) == len(pop3) == len(pop4)
        # Get, set and delete an item
        if len(pop) > 1:
            # Get
            ind = pop[0]
            ind.genotype = 42
            with pytest.raises(TypeError):
                pop['a']
            with pytest.raises(IndexError):
                pop[300]
            # Set
            pop[0] = ind
            with pytest.raises(TypeError):
                pop[0] = 'abc'
            # Delete
            l1 = len(pop)
            del pop[0]
            l2 = len(pop)
            assert l2 == l1 - 1
            with pytest.raises(TypeError):
                del pop['a']
            with pytest.raises(IndexError):
                del pop[300]
        # Iteration
        for ind in pop:
            pass
        # Concatenation
        pop2 = pop + pop
        assert isinstance(pop2, al.systems.ge.representation.Population)
        assert len(pop) * 2 == len(pop2)
        # Counts
        assert isinstance(pop.num_unique_genotypes, int)
        assert isinstance(pop.num_unique_phenotypes, int)
        assert isinstance(pop.num_unique_fitnesses, int)

    invalid_data_variants = (
        None,
        False,
        True,
        '',
        3,
        3.14,
        '123',
    )
    for data in invalid_data_variants:
        with pytest.raises(TypeError):
            al.systems.ge.representation.Population(data)


# Initialization

def test_initialize_individual():
    # Number of repetitions for methods with randomness
    num_repetitions = 20

    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte> | <bytes> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Method: given_genotype
    valid_genotypes = [
        (0, 1, 2),
        '(0, 1, 2)',
        [0, 1, 2],
        '[0, 1, 2]',
        '[0,1,2]',
    ]
    for gt in valid_genotypes:
        parameters = dict(init_ind_given_genotype=gt)
        ind = al.systems.ge.initialization.individual.given_genotype(grammar, parameters)
        check_individual(ind)
        assert ind.genotype.data == tuple(eval(str(gt)))
    # Parameter: init_ind_given_genotype not valid
    invalid_genotypes = [None, False, True, (), [], '', 'abc', 3, 3.14]
    for gt in invalid_genotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_genotype=gt)
            al.systems.ge.initialization.individual.given_genotype(grammar, parameters)
    # Parameter: init_ind_given_genotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.ge.initialization.individual.given_genotype(grammar)

    # Method: given_derivation_tree
    valid_derivation_trees = [
        grammar.generate_derivation_tree('ge', '[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2]'),
        grammar.generate_derivation_tree(
            'ge', [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4]),
    ]
    for dt in valid_derivation_trees:
        parameters = dict(init_ind_given_derivation_tree=dt)
        ind = al.systems.ge.initialization.individual.given_derivation_tree(grammar, parameters)
        check_individual(ind)
        ind_dt = ind.details['derivation_tree']
        assert isinstance(ind_dt, al._grammar.data_structures.DerivationTree)
        assert ind_dt == dt
    # Parameter: init_ind_given_derivation_tree not valid
    invalid_derivation_trees = [None, False, True, '', 'abc', 3, 3.14, (0, 1, 2)]
    for dt in invalid_derivation_trees:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_derivation_tree=dt)
            al.systems.ge.initialization.individual.given_derivation_tree(grammar, parameters)
    # Parameter: init_ind_given_derivation_tree not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.ge.initialization.individual.given_derivation_tree(grammar)

    # Method: given_phenotype
    valid_phenotypes = ['11110000', '1111000011110000']
    for phe in valid_phenotypes:
        parameters = dict(init_ind_given_phenotype=phe)
        ind = al.systems.ge.initialization.individual.given_phenotype(grammar, parameters)
        check_individual(ind)
        assert ind.phenotype == phe
    # Parameter: init_ind_given_phenotype not valid
    invalid_phenotypes = [None, False, True, '', 'abc', 3, 3.14, (0, 1, 2)]
    for phe in invalid_phenotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_phenotype=phe)
            al.systems.ge.initialization.individual.given_phenotype(grammar, parameters)
    # Parameter: init_ind_given_phenotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.ge.initialization.individual.given_phenotype(grammar)

    # Method: random_genotype
    for _ in range(num_repetitions):
        ind = al.systems.ge.initialization.individual.random_genotype(grammar)
        check_individual(ind)
    # Parameter: genotype_length
    parameters = dict(genotype_length=21)
    ind = al.systems.ge.initialization.individual.random_genotype(grammar, parameters)
    check_individual(ind)
    assert len(ind.genotype) == len(ind.genotype.data) == 21
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.ge.initialization.individual.random_genotype(grammar, parameters)
    # Parameter: codon_size
    parameters = dict(codon_size=1)
    ind = al.systems.ge.initialization.individual.random_genotype(grammar, parameters)
    check_individual(ind)
    assert all(codon in (0, 1) for codon in ind.genotype.data)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(codon_size=0)
        al.systems.ge.initialization.individual.random_genotype(grammar, parameters)

    # Method: random_valid_genotype
    for _ in range(num_repetitions):
        ind = al.systems.ge.initialization.individual.random_valid_genotype(grammar)
        check_individual(ind)
    # Parameter: genotype_length
    parameters = dict(genotype_length=21)
    ind = al.systems.ge.initialization.individual.random_valid_genotype(grammar, parameters)
    check_individual(ind)
    assert len(ind.genotype) == len(ind.genotype.data) == 21
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.ge.initialization.individual.random_valid_genotype(grammar, parameters)
    # Parameter: codon_size
    for cs, possible_vals in [(1, [0, 1]), (2, [0, 1, 2, 3]), (3, [0, 1, 2, 3, 4, 5, 6, 7])]:
        for _ in range(num_repetitions):
            parameters = dict(codon_size=cs, genotype_length=3000)
            ind = al.systems.ge.initialization.individual.random_genotype(grammar, parameters)
            check_individual(ind)
            assert all(codon in possible_vals for codon in ind.genotype.data)
            assert max(ind.genotype.data) == max(possible_vals)  # can fail, very low probability
            assert min(ind.genotype.data) == min(possible_vals)  # can fail, very low probability
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(codon_size=0)
        al.systems.ge.initialization.individual.random_valid_genotype(grammar, parameters)
    # Parameter: init_ind_random_valid_genotype_max_tries
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_ind_random_valid_genotype_max_tries=0)
        al.systems.ge.initialization.individual.random_valid_genotype(grammar, parameters)

    # Method: grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.ge.initialization.individual.grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_grow_max_depth
    ind1 = al.systems.ge.initialization.individual.grow_tree(
        grammar, dict(init_ind_grow_max_depth=0))
    for _ in range(num_repetitions):
        ind2 = al.systems.ge.initialization.individual.grow_tree(
            grammar, dict(init_ind_grow_max_depth=5))
        assert ind1.details['derivation_tree'].depth() <= \
            ind2.details['derivation_tree'].depth()

    # Method: pi_grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.ge.initialization.individual.pi_grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_grow_max_depth
    ind1 = al.systems.ge.initialization.individual.pi_grow_tree(
        grammar, dict(init_ind_grow_max_depth=0))
    for _ in range(num_repetitions):
        ind2 = al.systems.ge.initialization.individual.pi_grow_tree(
            grammar, dict(init_ind_grow_max_depth=5))
        assert ind1.details['derivation_tree'].depth() <= \
            ind2.details['derivation_tree'].depth()

    # Method: full_tree
    for _ in range(num_repetitions):
        ind = al.systems.ge.initialization.individual.full_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_full_max_depth
    ind1 = al.systems.ge.initialization.individual.full_tree(
        grammar, dict(init_ind_full_max_depth=0))
    for _ in range(num_repetitions):
        ind2 = al.systems.ge.initialization.individual.full_tree(
            grammar, dict(init_ind_full_max_depth=5))
        assert ind1.details['derivation_tree'].depth() <= \
            ind2.details['derivation_tree'].depth()

    # Method: ptc2_tree
    for _ in range(num_repetitions):
        ind = al.systems.ge.initialization.individual.ptc2_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_ptc2_max_expansions
    ind1 = al.systems.ge.initialization.individual.ptc2_tree(
        grammar, dict(init_ind_ptc2_max_expansions=0))
    for _ in range(num_repetitions):
        ind2 = al.systems.ge.initialization.individual.ptc2_tree(
            grammar, dict(init_ind_ptc2_max_expansions=100))
        assert ind1.details['derivation_tree'].num_expansions() <= \
            ind2.details['derivation_tree'].num_expansions()


def test_initialize_population():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Method: given_genotypes
    valid_genotype_collections = [
        [[0]],
        [[1]],
        [[2, 5, 7, 11, 13], [2, 4, 6, 8], [1, 2, 3, 4, 5, 6, 7]],
        ['[0]'],
        ['[1]'],
        ['[2, 5, 7, 11, 13]', '[2, 4, 6, 8]', '[1, 2, 3, 4, 5, 6, 7]'],
        [[0], '[1]'],
        [[2, 5, 7, 11, 13], '[2, 4, 6, 8]', [1, 2, 3, 4, 5, 6, 7]],
    ]
    for gts in valid_genotype_collections:
        parameters = dict(init_pop_given_genotypes=gts)
        pop = al.systems.ge.initialization.population.given_genotypes(grammar, parameters)
        check_population(pop)
        assert len(pop) == len(gts)
    # Parameter: init_pop_given_genotypes not valid
    invalid_genotype_collections = [
        None,
        [],
        [None],
        [0],
        [3.14],
        [[0, 1], 1],
        [1, '[0, 1]'],
    ]
    for gts in invalid_genotype_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_genotypes=gts)
            al.systems.ge.initialization.population.given_genotypes(grammar, parameters)
    # Parameter: init_pop_given_genotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.ge.initialization.population.given_genotypes(grammar)

    # Method: given_derivation_trees
    valid_derivation_tree_collections = [
        [grammar.generate_derivation_tree('ge', [0, 7, 11]),
         grammar.generate_derivation_tree('ge', '[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2]')],
    ]
    for dts in valid_derivation_tree_collections:
        parameters = dict(init_pop_given_derivation_trees=dts)
        pop = al.systems.ge.initialization.population.given_derivation_trees(grammar, parameters)
        check_population(pop)
        assert len(pop) == len(dts)
        for ind in pop:
            ind_dt = ind.details['derivation_tree']
            assert isinstance(ind_dt, al._grammar.data_structures.DerivationTree)
            assert ind_dt in dts
    # Parameter: init_pop_given_derivation_trees not valid
    invalid_derivation_tree_collections = [
        None,
        [],
        [None],
        [3.14],
        [[0, 1], 1],
        [1, '[0, 1]'],
    ]
    for dts in invalid_derivation_tree_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_derivation_trees=dts)
            al.systems.ge.initialization.population.given_derivation_trees(grammar, parameters)
    # Parameter: init_pop_given_derivation_trees not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.ge.initialization.population.given_derivation_trees(grammar)

    # Method: given_phenotypes
    valid_phenotype_collections = [
        ['00000000', '11111111'],
        ['00000000', '11111111', '00000000', '11111111', '0000111100001111', '1111000011110000'],
    ]
    for pts in valid_phenotype_collections:
        parameters = dict(init_pop_given_phenotypes=pts)
        pop = al.systems.ge.initialization.population.given_phenotypes(grammar, parameters)
        check_population(pop)
        assert len(pop) == len(pts)
    # Parameter: init_pop_given_phenotypes not valid
    invalid_phenotype_collections = [
        None,
        [],
        [None],
        ['000000001', '11111111'],
        ['00000000', '111111110'],
    ]
    for pts in invalid_phenotype_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_phenotypes=pts)
            al.systems.ge.initialization.population.given_derivation_trees(grammar, parameters)
    # Parameter: init_pop_given_phenotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.ge.initialization.population.given_phenotypes(grammar)

    # Method: random_genotypes
    n = 10
    for _ in range(n):
        pop = al.systems.ge.initialization.population.random_genotypes(grammar)
        check_population(pop)
        assert len(pop) == al.systems.ge.default_parameters.population_size
        # Parameters: population_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(population_size=chosen_pop_size)
            pop = al.systems.ge.initialization.population.random_genotypes(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameter: init_pop_random_unique_genotypes, init_pop_random_unique_phenotypes
    for unique_gen in (True, False):
        for unique_phe in (True, False):
            params = dict(
                population_size=10,
                init_pop_random_unique_max_tries=500,
                init_pop_random_unique_genotypes=unique_gen,
                init_pop_random_unique_phenotypes=unique_phe,
            )
            pop = al.systems.ge.initialization.population.random_genotypes(grammar, params)
            check_population(pop)
            assert len(pop) == 10
            if unique_gen or unique_phe:
                params['population_size'] = 30
                params['genotype_length'] = 2
                params['codon_size'] = 2
                with pytest.raises(al.exceptions.InitializationError):
                    al.systems.ge.initialization.population.random_genotypes(grammar, params)
    # Parameter: init_pop_random_unique_max_tries
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_random_unique_max_tries=0)
        al.systems.ge.initialization.population.random_genotypes(grammar, parameters)
    # Parameter: genotype_length
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.ge.initialization.population.random_genotypes(grammar, parameters)

    # Method: rhh (=ramped half and half)
    for _ in range(n):
        pop = al.systems.ge.initialization.population.rhh(grammar)
        check_population(pop)
        assert len(pop) == al.systems.ge.default_parameters.population_size
        # Parameters: population_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(population_size=chosen_pop_size)
            pop = al.systems.ge.initialization.population.rhh(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_rhh_with_pi_grow
    for use_pi_grow in (True, False):
        parameters = dict(init_pop_rhh_with_pi_grow=use_pi_grow)
        pop = al.systems.ge.initialization.population.rhh(grammar, parameters)
        check_population(pop)
    # Parameters: init_pop_rhh_start_depth, init_pop_rhh_end_depth
    parameters = dict(init_pop_rhh_start_depth=3, init_pop_rhh_end_depth=4)
    pop = al.systems.ge.initialization.population.rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_rhh_start_depth=5, init_pop_rhh_end_depth=3)
        pop = al.systems.ge.initialization.population.rhh(grammar, parameters)

    # Method: ptc2 (=probabilistic tree creation 2)
    for _ in range(n):
        pop = al.systems.ge.initialization.population.ptc2(grammar)
        check_population(pop)
        assert len(pop) == al.systems.ge.default_parameters.population_size
        # Parameters: population_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(population_size=chosen_pop_size)
            pop = al.systems.ge.initialization.population.ptc2(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_ptc2_start_expansions, init_pop_ptc2_end_expansions
    parameters = dict(init_pop_ptc2_start_expansions=10, init_pop_ptc2_end_expansions=50)
    pop = al.systems.ge.initialization.population.rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_ptc2_start_expansions=50, init_pop_ptc2_end_expansions=10)
        pop = al.systems.ge.initialization.population.ptc2(grammar, parameters)


# Mutation

def test_mutation():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types: 1) list of int, 2) string thereof, 3) Genotype class
    genotypes = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        '(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)',
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]',
        al.systems.ge.representation.Genotype((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        al.systems.ge.representation.Genotype('(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)'),
        al.systems.ge.representation.Genotype([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        al.systems.ge.representation.Genotype('[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'),
    )

    # Mutation (guaranteed to change the genotype due to parameter choice)
    methods = (
        al.systems.ge.mutation.int_replacement_by_probability,
        al.systems.ge.mutation.int_replacement_by_count,
    )
    params = dict(mutation_int_replacement_probability=1.0, mutation_int_replacement_count=2)
    for method in methods:
        for gt in genotypes:
            # Without parameters (=using defaults)
            method(grammar, copy.copy(gt))
            # With parameters
            gt2 = method(grammar, copy.copy(gt), params)
            gt3 = method(grammar, copy.copy(gt), parameters=params)
            gt4 = method(grammar, genotype=copy.copy(gt), parameters=params)
            gt5 = method(grammar=grammar, genotype=copy.copy(gt), parameters=params)
            assert gt != gt2
            assert gt != gt3
            assert gt != gt4
            assert gt != gt5

    # Mutation (guaranteed NOT to change the genotype due to different parameter choice)
    params = dict(mutation_int_replacement_probability=0.0, mutation_int_replacement_count=0)
    for method in methods:
        for gt in genotypes:
            # With parameters
            gt2 = method(grammar, copy.copy(gt), params)
            gt3 = method(grammar, copy.copy(gt), parameters=params)
            gt4 = method(grammar, genotype=copy.copy(gt), parameters=params)
            gt5 = method(grammar=grammar, genotype=copy.copy(gt), parameters=params)
            assert gt2 == gt3 == gt4 == gt5


def test_mutation_count():
    # Grammar
    bnf_text = '<bit> ::= 1 | 0'
    grammar = al.Grammar(bnf_text=bnf_text)

    # Mutation
    for _ in range(50):
        for genotype in ([-1], [-1, -1], [-1, -1, -1], [-1, -1, -1, -1], [-1]*50, [-1]*100):
            # Parameter: mutation_int_replacement_count
            for mutation_int_replacement_count in range(10):
                parameters = dict(
                    mutation_int_replacement_count=mutation_int_replacement_count,
                    codon_size=8,
                )
                gt_copy = copy.copy(genotype)
                gt_mut = al.systems.ge.mutation.int_replacement_by_count(
                    grammar, gt_copy, parameters)
                # Check expected number of int flips for different cases
                num_changed_codons = sum(codon != -1 for codon in gt_mut.data)
                if mutation_int_replacement_count == 0:
                    assert gt_mut.data == tuple([-1] * len(genotype))
                elif mutation_int_replacement_count >= len(genotype):
                    assert num_changed_codons == len(genotype)
                else:
                    assert num_changed_codons == mutation_int_replacement_count


# Crossover

def test_crossover():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types: 1) list of int, 2) string thereof, 3) Genotype class
    genotypes = (
        [0, 1, 2] * 10,
        str([3, 4, 5] * 10),
        al.systems.ge.representation.Genotype(str([6, 7, 8] * 10)),
    )

    # Crossover
    methods = (
        al.systems.ge.crossover.two_point_length_preserving,
    )
    def perform_checks(gt1, gt2, gt3, gt4):
        if not isinstance(gt1, al.systems.ge.representation.Genotype):
            gt1 = al.systems.ge.representation.Genotype(copy.copy(gt1))
        if not isinstance(gt2, al.systems.ge.representation.Genotype):
            gt2 = al.systems.ge.representation.Genotype(copy.copy(gt2))
        assert gt1 == gt1
        assert gt1 != gt2
        assert gt3 != gt1
        assert gt3 != gt2
        assert gt4 != gt1
        assert gt4 != gt2
        for codon_val in range(1, 6):
            cnt_in_parents = gt1.data.count(codon_val) + gt2.data.count(codon_val)
            cnt_in_children = gt3.data.count(codon_val) + gt4.data.count(codon_val)
            assert cnt_in_parents == cnt_in_children

    params = dict()
    for _ in range(50):
        for two_genotypes in itertools.combinations(genotypes, 2):
            for method in methods:
                gt1, gt2 = two_genotypes
                gt3, gt4 = method(
                    grammar, copy.copy(gt1), copy.copy(gt2), params)
                perform_checks(gt1, gt2, gt3, gt4)
                gt3, gt4 = method(
                    grammar, copy.copy(gt1), copy.copy(gt2),
                    parameters=params)
                perform_checks(gt1, gt2, gt3, gt4)
                gt3, gt4 = method(
                    grammar, copy.copy(gt1),
                    genotype2=copy.copy(gt2), parameters=params)
                perform_checks(gt1, gt2, gt3, gt4)
                gt3, gt4 = method(
                    grammar,
                    genotype1=copy.copy(gt1), genotype2=copy.copy(gt2), parameters=params)
                perform_checks(gt1, gt2, gt3, gt4)
                gt3, gt4 = method(
                    grammar=grammar, genotype1=copy.copy(gt1), genotype2=copy.copy(gt2),
                    parameters=params)
                perform_checks(gt1, gt2, gt3, gt4)


def test_crossover_fails():
    # Grammar
    bnf_text = '<bit> ::= 1 | 0'
    grammar = al.Grammar(bnf_text=bnf_text)

    # Crossover
    # - invalid genotype types
    gt_valid = [1, 2, 3, 4, 5]
    for gt_invalid in [None, False, True, '', 0, 1, 3.14, '101']:
        al.systems.ge.crossover.two_point_length_preserving(grammar, gt_valid, gt_valid)
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.ge.crossover.two_point_length_preserving(grammar, gt_valid, gt_invalid)
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.ge.crossover.two_point_length_preserving(grammar, gt_invalid, gt_valid)
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.ge.crossover.two_point_length_preserving(grammar, gt_invalid, gt_invalid)
    # - too short genotype
    gt1 = [0]
    gt2 = [1]
    with pytest.raises(al.exceptions.CrossoverError):
        al.systems.ge.crossover.two_point_length_preserving(grammar, gt1, gt2)

    # - genotypes with different length
    gt1 = [0, 1, 2, 3]
    gt2 = [0, 1, 2, 3, 4]
    with pytest.raises(al.exceptions.CrossoverError):
        al.systems.ge.crossover.two_point_length_preserving(grammar, gt1, gt2)


# Neighborhood

def test_neighborhood_api():
    # Grammar
    bnf_text = """
    <S> ::= <A> | <B> | <C>
    <A> ::= <AA>
    <AA> ::= 1 | 2 | 3 | 4
    <B> ::= a | b | c
    <C> ::= <CC> | <CCC>
    <CC> ::= X | Y
    <CCC> ::= Z
    """
    gr = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types: 1) list of int, 2) string thereof, 3) Genotype class
    genotypes = (
        [0, 1, 2],
        str([3, 4, 5] * 3),
        al.systems.ge.representation.Genotype(str([6, 7, 8, 9] * 2)),
    )

    # Neighborhood
    for gt in genotypes:
        phe = al.systems.ge.mapping.forward(gr, gt)
        # Default
        nh1 = al.systems.ge.neighborhood.int_replacement(gr, gt)
        nh2 = al.systems.ge.neighborhood.int_replacement(gr, genotype=gt)
        nh3 = al.systems.ge.neighborhood.int_replacement(grammar=gr, genotype=gt)
        nh4 = al.systems.ge.neighborhood.int_replacement(gr, gt, dict())
        nh5 = al.systems.ge.neighborhood.int_replacement(gr, gt, parameters=dict())
        nh6 = al.systems.ge.neighborhood.int_replacement(gr, genotype=gt, parameters=dict())
        nh7 = al.systems.ge.neighborhood.int_replacement(
            grammar=gr, genotype=gt, parameters=dict())
        assert nh1 == nh2 == nh3 == nh4 == nh5 == nh6 == nh7
        for new_gt in nh1:
            check_genotype(new_gt)
            new_phe = al.systems.ge.mapping.forward(gr, new_gt)
            assert new_phe != phe


@pytest.mark.parametrize(
    'bnf, genotype, phenotype',
    [
        (shared.BNF5, [0], '1'),
        (shared.BNF5, [1], '2'),
        (shared.BNF5, [2], '3'),
        (shared.BNF5, [3], '4'),
        (shared.BNF5, [4], '5'),
        (shared.BNF6, [0], '1'),
        (shared.BNF6, [0, 0], '1'),
        (shared.BNF6, [0, 1], '2'),
        (shared.BNF6, [1, 0], '3'),
        (shared.BNF6, [1, 1], '4'),
        (shared.BNF6, [2], '5'),
        (shared.BNF7, [0], 'ac1'),
        (shared.BNF7, [1], 'bf8'),
        (shared.BNF7, [0, 1, 1], 'ad4'),
        (shared.BNF9, [0], 'a'),
        (shared.BNF9, [0, 1, 1, 2], 'bc'),
        (shared.BNF9, [1], '22'),
        (shared.BNF9, [1, 0, 2], '3'),
        (shared.BNF9, [1, 1, 0, 2], '13'),
    ]
)
def test_neighborhood_reachability_in_finite_languages(bnf, genotype, phenotype):
    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.ge.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.ge.mapping.forward(grammar, gt)
    assert phe == phenotype

    # Language
    language_gr = grammar.generate_language()

    # Neighborhood
    params = [
        dict(),
        dict(neighborhood_max_size=1),
        dict(neighborhood_max_size=2),
        dict(neighborhood_max_size=5),
    ]
    for param in params:
        language_nbr = set()
        genotypes_seen = set()
        genotypes_new = [gt]
        i = 0
        i_max = 1_000_000
        while len(language_nbr) < len(language_gr) and i < i_max:
            genotypes_nbrs = set()
            for gen in genotypes_new:
                nbrs = al.systems.ge.neighborhood.int_replacement(grammar, gen, param)
                if 'neighborhood_max_size' in param:
                    assert len(nbrs) <= param['neighborhood_max_size']
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                try:
                    phe = al.systems.ge.mapping.forward(grammar, gen, param)
                    language_nbr.add(phe)
                except al.exceptions.MappingError:
                    continue
            genotypes_new = [x for x in genotypes_nbrs if x not in genotypes_seen]
            if not genotypes_new:
                genotypes_new = genotypes_nbrs
            i += 1
        assert i < i_max
        assert set(language_gr) == language_nbr


@pytest.mark.parametrize(
    'bnf, genotype, phenotype, strings_given',
    [
        (shared.BNF10, (85,15), '1x', ('2x', '3x', '4y', '5y', '6y', '7')),
        (shared.BNF11, (195,105), '1', ('2', '3', '4', '22', '33', '44')),
        (shared.BNF12, (10,119), '131', ('242', '2332', '22422', '21312', '223322')),
    ]
)
def test_neighborhood_reachability_in_infinite_languages(bnf, genotype, phenotype, strings_given):
    # TODO: del
    # Use some infinite grammars (with recursive rules) and see if the stop criteria act
    # properly to prevent overly complex construcitons
    # Perhaps set a time limit instead of sending the test suit in a stochastic sleep

    # TODO: write proper warning in neighborhood func docs - recursive grammars need a low stop criterion (max_wraps fails, max_expansions works), otherwise very long genotypes are produced most of the time

    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.ge.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.ge.mapping.forward(grammar, gt)
    assert phe == phenotype

    # Neighborhood
    strings_given = set(strings_given)
    params = [
        dict(),  # required time depends on the default parameters (stop criteria values)
        dict(max_expansions=10),
        dict(max_expansions=10, neighborhood_max_size=3),
        dict(max_wraps=1),
        dict(max_wraps=1, neighborhood_max_size=3),
    ]
    for param in params:
        strings_seen = set()
        genotypes_seen = set()
        genotypes_new = [gt]
        i = 0
        i_max = 1_000_000
        while not strings_given.issubset(strings_seen) and i < i_max:
            genotypes_nbrs = set()
            for gen in genotypes_new:
                # Neighborhood generation
                nbrs = al.systems.ge.neighborhood.int_replacement(grammar, gen, param)
                if 'neighborhood_max_size' in param:
                    assert len(nbrs) <= param['neighborhood_max_size']
                # Genotype management
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                # Phenotype generation
                try:
                    phe = al.systems.ge.mapping.forward(grammar, gen, param)
                    strings_seen.add(phe)
                except al.exceptions.MappingError:
                    continue
            genotypes_new = [x for x in genotypes_nbrs if x not in genotypes_seen]
            if not genotypes_new:
                genotypes_new = genotypes_nbrs
            i += 1
        assert strings_given.issubset(strings_seen)
        assert i < i_max


def test_neighborhood_parameter_distance():
    # Grammar
    ebnf_text = """
S = S S | A | B
A = X | Y
X = "1" | "2"
Y = "2" | "1"
B = "a" | "b"
"""
    grammar = al.Grammar(ebnf_text=ebnf_text)

    # Genotype from string parsing
    dt = grammar.parse_string('1a1a')
    gt = al.systems.ge.mapping.reverse(grammar, dt)

    # Neighborhood in different distances when changing only terminals
    # - distance 1
    parameters=dict(neighborhood_only_terminals=True)
    nbrs_gt = al.systems.ge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.ge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {'1a1b', '1a2a', '1b1a', '2a1a'}

    # - distance 2
    parameters=dict(neighborhood_distance=2, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.ge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.ge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {'1a2b', '1b1b', '1b2a', '2a1b', '2a2a', '2b1a'}

    # - distance 3
    parameters=dict(neighborhood_distance=3, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.ge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.ge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {'1b2b', '2a2b', '2b1b', '2b2a'}

    # - distance 4
    parameters=dict(neighborhood_distance=4, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.ge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.ge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {'2b2b'}

    # - distance 5 and greater
    for dist in range(5, 20):
        parameters=dict(neighborhood_distance=dist, neighborhood_only_terminals=True)
        nbrs_gt = al.systems.ge.neighborhood.int_replacement(grammar, gt, parameters)
        nbrs = [al.systems.ge.mapping.forward(grammar, gt) for gt in nbrs_gt]
        assert nbrs == []  # TODO: why empty and not max changes?


@pytest.mark.parametrize(
    'bnf, gt, phe, phe_neighbors',
    [
        (shared.BNF1, [0], '0', ('1', '2')),
        (shared.BNF1, [0, 0], '0', ('1', '2')),
        (shared.BNF1, [1], '1', ('0', '2')),
        (shared.BNF1, [1, 1], '1', ('0', '2')),
        (shared.BNF1, [2], '2', ('0', '1')),
        (shared.BNF1, [3], '0', ('1', '2')),
        (shared.BNF1, [4], '1', ('0', '2')),
        (shared.BNF1, [5], '2', ('0', '1')),
        (shared.BNF2, [0], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF2, [0, 0], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF2, [0, 0, 0], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF2, [1], '1b', ('0b', '2b', '1a', '1c')),
        (shared.BNF2, [1, 1], '1b', ('0b', '2b', '1a', '1c')),
        (shared.BNF2, [1, 1, 1], '1b', ('0b', '2b', '1a', '1c')),
        (shared.BNF2, [2], '2c', ('0c', '1c', '2a', '2b')),
        (shared.BNF2, [2, 2], '2c', ('0c', '1c', '2a', '2b')),
        (shared.BNF2, [2, 2, 2], '2c', ('0c', '1c', '2a', '2b')),
        (shared.BNF2, [3], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF2, [3, 6], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF2, [6, 3, 123123], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF2, [0, 1], '0b', ('1b', '2b', '0a', '0c')),
        (shared.BNF2, [1, 2], '1c', ('0c', '2c', '1a', '1b')),
        (shared.BNF3, [0], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF3, [0, 0], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF3, [0, 0, 0], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF3, [1], '1b', ('0b', '2b', '1a', '1c')),
        (shared.BNF3, [1, 1], '1b', ('0b', '2b', '1a', '1c')),
        (shared.BNF3, [1, 1, 1], '1b', ('0b', '2b', '1a', '1c')),
        (shared.BNF3, [2], '2c', ('0c', '1c', '2a', '2b')),
        (shared.BNF3, [2, 2], '2c', ('0c', '1c', '2a', '2b')),
        (shared.BNF3, [2, 2, 2], '2c', ('0c', '1c', '2a', '2b')),
        (shared.BNF3, [3], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF3, [3, 6], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF3, [6, 3, 123123], '0a', ('1a', '2a', '0b', '0c')),
        (shared.BNF3, [0, 1], '0b', ('1b', '2b', '0a', '0c')),
        (shared.BNF3, [1, 2], '1c', ('0c', '2c', '1a', '1b')),
        (shared.BNF4, [0], '00000000', ('10000000', '01000000', '00100000', '00010000',
                                        '00001000', '00000100', '00000010', '00000001')),
        (shared.BNF4, [1], '11111111', ('01111111', '10111111', '11011111', '11101111',
                                        '11110111', '11111011', '11111101', '11111110')),
        (shared.BNF4, [0, 1], '01010101', ('11010101', '00010101', '01110101', '01000101',
                                           '01011101', '01010001', '01010111', '01010100')),
        (shared.BNF4, [2, 5, 3], '01101101', ('11101101', '00101101', '01001101', '01111101',
                                              '01100101', '01101001', '01101111', '01101100')),
    ]
)
def test_neighborhood_parameter_max_size(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    parameters = dict(max_wraps=50, max_expansions=1000)
    assert phe == al.systems.ge.mapping.forward(gr, gt, parameters)

    # Neighborhood
    nbrs = al.systems.ge.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [al.systems.ge.mapping.forward(gr, nbr_gt, parameters) for nbr_gt in nbrs]
    assert set(nbrs_phe) == set(phe_neighbors)

    # Parameter: neighborhood_max_size
    parameters['neighborhood_max_size'] = None
    nbrs = al.systems.ge.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [al.systems.ge.mapping.forward(gr, nbr_gt, parameters) for nbr_gt in nbrs]
    assert set(nbrs_phe) == set(phe_neighbors)

    for max_size in range(1, 5):
        parameters['neighborhood_max_size'] = max_size
        nbrs_phe = set()
        for _ in range(100):
            nbrs = al.systems.ge.neighborhood.int_replacement(gr, gt, parameters)
            assert len(nbrs) <= max_size
            for nbr_gt in nbrs:
                nbr_phe = al.systems.ge.mapping.forward(gr, nbr_gt, parameters)
                assert nbr_phe in phe_neighbors
                nbrs_phe.add(nbr_phe)
        assert nbrs_phe == set(phe_neighbors)


@pytest.mark.parametrize(
    'bnf, gt, phe, phe_neighbors',
    [
        (shared.BNF5, [0], '1', ('2', '3', '4', '5')),
        (shared.BNF5, [1], '2', ('1', '3', '4', '5')),
        (shared.BNF5, [2], '3', ('1', '2', '4', '5')),
        (shared.BNF5, [3], '4', ('1', '2', '3', '5')),
        (shared.BNF5, [4], '5', ('1', '2', '3', '4')),
        (shared.BNF6, [0, 0], '1', ('2', '5')),
        (shared.BNF6, [0, 1], '2', ('1', '5')),
        (shared.BNF6, [1, 0], '3', ('4', '5')),
        (shared.BNF6, [1, 1], '4', ('3', '5')),
        (shared.BNF6, [2], '5', ()),
        (shared.BNF7, [0], 'ac1', ('be5', 'ad3', 'ac2')),
        (shared.BNF7, [1], 'bf8', ('ad4', 'be6', 'bf7')),
        (shared.BNF7, [0, 1, 1], 'ad4', ('bf8', 'ac2', 'ad3')),
        (shared.BNF8, [0], '<S><S><S><S>',
         ('a0g', '1g', 'a0ga0g', '1g1g1g<S>', 'a0ga0g<A>0<B>', '1g1g1<B><S><S>')),
        (shared.BNF8, [1], 't', ('t0g', '1c', 'a')),
        (shared.BNF8, [2, 0, 1], 'a0c', ('1g', 't0c', 'a0g')),
    ]
)
def test_neighborhood_parameter_only_terminals(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    parameters = dict(neighborhood_only_terminals=True, max_wraps=2)
    assert phe == al.systems.ge.mapping.forward(gr, gt, parameters, raise_errors=False)

    # Neighborhood
    nbrs = al.systems.ge.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [al.systems.ge.mapping.forward(gr, nbr_gt, parameters, raise_errors=False)
                for nbr_gt in nbrs]
    assert set(nbrs_phe) == set(phe_neighbors)


# Mapping

def test_mapping_forward_api():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypic data of five types:
    # 1) tuple of int 2) string thereof 3) list of int 4) string thereof 5) Genotype object
    tup = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    data_variants = (
        tup,
        str(tup),
        list(tup),
        str(list(tup)),
        al.systems.ge.representation.Genotype(tup),
    )

    # Forward mapping
    parameters = dict(
        codon_size=4,
        max_wraps=0,
        max_expansions=3,
    )
    kwargs = dict(
        verbose=False,
        raise_errors=False,
        return_derivation_tree=False,
    )
    for data in data_variants:
        for vb in (True, False):
            kwargs['verbose'] = vb

            # Method of Grammar class
            string1 = grammar.generate_string(
                'ge', data, parameters, **kwargs)
            string2 = grammar.generate_string(
                'ge', data, parameters=parameters, **kwargs)
            string3 = grammar.generate_string(
                method='ge', genotype=data, parameters=parameters, **kwargs)
            assert string1
            assert string1 == string2 == string3

            # Method of DerivationTree class
            dt1 = grammar.generate_derivation_tree(
                'ge', data, parameters, **kwargs)
            dt2 = grammar.generate_derivation_tree(
                'ge', data, parameters=parameters, **kwargs)
            dt3 = grammar.generate_derivation_tree(
                method='ge', genotype=data, parameters=parameters, **kwargs)
            assert string1 == dt1.string() == dt2.string() == dt3.string()

            # Functions in mapping module
            string4 = al.systems.ge.mapping.forward(
                grammar, data, parameters, **kwargs)
            string5 = al.systems.ge.mapping.forward(
                grammar, data, parameters=parameters, **kwargs)
            string6 = al.systems.ge.mapping.forward(
                grammar=grammar, genotype=data, parameters=parameters, **kwargs)
            assert string1 == string4 == string5 == string6

            kwargs['return_derivation_tree'] = True
            phe, dt4 = al.systems.ge.mapping.forward(
                grammar, data, parameters, **kwargs)
            phe, dt5 = al.systems.ge.mapping.forward(
                grammar, data, parameters=parameters, **kwargs)
            phe, dt6 = al.systems.ge.mapping.forward(
                grammar=grammar, genotype=data, parameters=parameters, **kwargs)
            kwargs['return_derivation_tree'] = False
            assert string1 == dt4.string() == dt5.string() == dt6.string()

            # Same with errors when reaching wrap or expansion limit
            kwargs['raise_errors'] = True
            with pytest.raises(al.exceptions.MappingError):
                grammar.generate_string(
                    method='ge', genotype=data, parameters=parameters, **kwargs)
            with pytest.raises(al.exceptions.MappingError):
                grammar.generate_derivation_tree(
                    method='ge', genotype=data, parameters=parameters, **kwargs)
            with pytest.raises(al.exceptions.MappingError):
                al.systems.ge.mapping.forward(
                    grammar, genotype=data, parameters=parameters, **kwargs)
            with pytest.raises(al.exceptions.MappingError):
                al.systems.ge.mapping.forward(
                    grammar, genotype=data, parameters=parameters, **kwargs)
            kwargs['raise_errors'] = False


def test_mapping_reverse_api():
    # Number of repetitions for methods with randomness
    num_repetitions = 10

    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Random derivation trees and corresponding strings
    random_dts, random_strings = [], []
    for _ in range(num_repetitions):
        dt = grammar.generate_derivation_tree()
        string = dt.string()
        assert isinstance(string, str)
        assert isinstance(dt, al._grammar.data_structures.DerivationTree)
        random_dts.append(dt)
        random_strings.append(string)
    for _ in range(num_repetitions):
        string = grammar.generate_string()
        dt = grammar.parse_string(string)
        assert isinstance(string, str)
        assert isinstance(dt, al._grammar.data_structures.DerivationTree)
        random_dts.append(dt)
        random_strings.append(string)

    # Reverse mapping
    parameters = dict(
        codon_size=8,
        max_wraps=None,
        max_expansions=None,
        codon_randomization=False,
    )
    # Functions in mapping module
    p1 = dict()
    p2 = dict(codon_size=4)
    p3 = dict(codon_randomization=False)
    p4 = dict(codon_size=8, codon_randomization=False)
    p5 = dict(codon_size=5, codon_randomization=True)
    for parameters in (p1, p2, p3, p4, p5):
        for dt, string in zip(random_dts, random_strings):
            gt1 = al.systems.ge.mapping.reverse(grammar, string)
            gt2 = al.systems.ge.mapping.reverse(grammar, dt)
            gt3 = al.systems.ge.mapping.reverse(grammar, string, parameters)
            gt4 = al.systems.ge.mapping.reverse(grammar, string, parameters, False)
            gt5, dt5 = al.systems.ge.mapping.reverse(grammar, string, parameters, True)
            gt6 = al.systems.ge.mapping.reverse(grammar, phenotype_or_derivation_tree=string)
            gt7 = al.systems.ge.mapping.reverse(grammar, phenotype_or_derivation_tree=dt)
            gt8 = al.systems.ge.mapping.reverse(grammar, phenotype_or_derivation_tree=string,
                                                parameters=parameters)
            gt9 = al.systems.ge.mapping.reverse(grammar, phenotype_or_derivation_tree=dt,
                                                parameters=parameters)
            gt10 = al.systems.ge.mapping.reverse(grammar, phenotype_or_derivation_tree=string,
                                                 parameters=parameters,
                                                 return_derivation_tree=False)
            gt11, dt11 = al.systems.ge.mapping.reverse(grammar, phenotype_or_derivation_tree=dt,
                                                       parameters=parameters,
                                                       return_derivation_tree=True)
            for gt in (gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10, gt11):
                # Check if reverse mapping resulted in a valid genotype
                check_genotype(gt)
                # Check if genotype allows to reproduce the original string via forward mapping
                string_from_fwd_map = grammar.generate_string('ge', gt)
                assert string_from_fwd_map == string


def test_mapping_errors():
    bnf_text = '<S> ::= <S><S> | 1 | 2 | 3'
    grammar = al.Grammar(bnf_text=bnf_text)
    # Invalid input: a string that is not part of the grammar's language
    string = '4'
    with pytest.raises(al.exceptions.MappingError):
        al.systems.ge.mapping.reverse(grammar, string)
    # Invalid input: a derivation tree with an unknown nonterminal
    dt = grammar.generate_derivation_tree()
    dt.root_node.symbol = al._grammar.data_structures.NonterminalSymbol('nonsense')
    with pytest.raises(al.exceptions.MappingError):
        al.systems.ge.mapping.reverse(grammar, dt)
    # Invalid input: a derivation tree with an unknown derivation (no corresponding rule)
    dt = grammar.generate_derivation_tree()
    dt.leaf_nodes()[0].symbol = al._grammar.data_structures.TerminalSymbol('nonsense')
    with pytest.raises(al.exceptions.MappingError):
        al.systems.ge.mapping.reverse(grammar, dt)
    # Parameter: codon_size
    string = '111222333'
    parameters = dict(codon_size=1)
    with pytest.raises(al.exceptions.MappingError):
        al.systems.ge.mapping.reverse(grammar, string, parameters)
    parameters = dict(codon_size=2)
    gt = al.systems.ge.mapping.reverse(grammar, string, parameters)
    assert string == al.systems.ge.mapping.forward(grammar, gt)


def test_mapping_stop_criteria():
    bnf_text = """
    <S> ::= <A><B><C><D>
    <A> ::= 0 | 1
    <B> ::= x | y
    <C> ::= + | -
    <D> ::= ( | )
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for vb in (False, True):
        # Default
        string = grammar.generate_string('ge', [0], verbose=vb)
        assert string == '0x+('
        # Parameter: max_wraps
        for mw in range(10):
            params = dict(max_wraps=mw, max_expansions=None)
            if mw < 3:
                with pytest.raises(al.exceptions.MappingError):
                    grammar.generate_string('ge', [0], params, verbose=vb)
                sentential_form = grammar.generate_string(
                    'ge', [0], params, verbose=vb, raise_errors=False)
                assert '<' in sentential_form
                assert '>' in sentential_form
            else:
                string = grammar.generate_string('ge', [0], params, verbose=vb)
                assert '<' not in string
                assert '>' not in string
        # Parameter: max_expansions
        for me in range(20):
            params = dict(max_wraps=None, max_expansions=me)
            if me < 5:
                with pytest.raises(al.exceptions.MappingError):
                    grammar.generate_string('ge', [0], params, verbose=vb)
                sentential_form = grammar.generate_string(
                    'ge', [0], params, verbose=vb, raise_errors=False)
                assert '<' in sentential_form
                assert '>' in sentential_form
            else:
                string = grammar.generate_string('ge', [0], params, verbose=vb)
                assert '<' not in string
                assert '>' not in string
        # Parameter: max_wraps and max_expansions
        params = dict(max_wraps=None, max_expansions=None)
        with pytest.raises(al.exceptions.MappingError):
            grammar.generate_string('ge', [0], params, verbose=vb)


def test_mapping_reverse_randomized():
    bnf_text = """
    <S> ::= <S><S> | <A> | <B> | a | b
    <A> ::= 1 | 2 | 3
    <B> ::= X | Y
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for string1 in ('a1X1a', '123XYba', 'Yb', '3a2b1X2Y3a', 'ababa1', 'X1Y3Xb'):
        randomized = set()
        nonrandomized = set()
        for rc in (True, False):
            for _ in range(20):
                parameters = dict(codon_randomization=rc)
                genotype = al.systems.ge.mapping.reverse(grammar, string1, parameters)
                string2 = al.systems.ge.mapping.forward(grammar, genotype, parameters)
                assert string1 == string2
                if rc:
                    randomized.add(str(genotype))
                else:
                    nonrandomized.add(str(genotype))
        assert len(randomized) > 1
        assert len(nonrandomized) == 1


def test_mapping_forward_by_hand():
    bnf_text = """
    <A> ::= <B><C><D>
    <B> ::= 7 | 8 | 9
    <C> ::= x | y
    <D> ::= R | S
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_genotype_phenotype_map = [
        ([0], '7xR'),
        ([1], '8yS'),
        ([2], '9xR'),
        ([0, 0], '7xR'),
        ([1, 0], '8xS'),
        ([0, 1], '7yR'),
        ([1, 1], '8yS'),
        ([2, 0], '9xR'),
        ([2, 1], '9yR'),
        ([2, 2], '9xR'),
        ([0, 0, 0], '7xR'),
        ([1, 0, 0], '8xR'),
        ([0, 1, 0], '7yR'),
        ([1, 1, 0], '8yR'),
        ([0, 0, 1], '7xS'),
        ([1, 0, 1], '8xS'),
        ([0, 1, 1], '7yS'),
        ([1, 1, 1], '8yS'),
        ([2, 0, 0], '9xR'),
        ([2, 1, 0], '9yR'),
        ([2, 0, 1], '9xS'),
        ([2, 1, 1], '9yS'),
        ([2, 1, 1, 33], '9yS'),
        ([2, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], '9yS'),
    ]
    for genotype, expected_phenotype in expected_genotype_phenotype_map:
        phenotype = grammar.generate_string(method='ge', genotype=genotype)
        dt = grammar.generate_derivation_tree(method='ge', genotype=genotype)
        assert phenotype == dt.string() == expected_phenotype


def test_mapping_forward_and_reverse_automated():
    bnf_text = """
    <data> ::= <byte> | <data><byte>
    <byte> ::= <bit><bit><bit><bit><bit><bit><bit><bit>
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for _ in range(100):
        string1 = grammar.generate_string()
        for cs in (3, 5, 7):
            for rc in (True, False):
                parameters = dict(
                    codon_size=cs,
                    codon_randomization=rc,
                )
                # Reverse map: string1 -> genotype
                genotype = al.systems.ge.mapping.reverse(grammar, string1, parameters)
                # Forward map: genotype -> string2
                string2 = al.systems.ge.mapping.forward(grammar, genotype, parameters)
                assert string1 == string2


def test_mapping_forward_and_reverse_by_hand1():
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | - | * | /
    <v> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for string1 in ('1+1', '9-4', '7*5-3', '9*8/7+6-5', '3+4/9-1*8', '1+2+3+4+5-6-7*8/9'):
        for cs in (4, 6, 9):
            for rc in (True, False):
                parameters = dict(
                    codon_size=cs,
                    codon_randomization=rc,
                )
                # Reverse map: string1 -> genotype
                genotype = al.systems.ge.mapping.reverse(grammar, string1, parameters)
                # Forward map: genotype -> string2
                string2 = al.systems.ge.mapping.forward(grammar, genotype, parameters)
                assert string2 == string1


def test_mapping_forward_against_book_2003_example():
    # References
    # - Paper 2001: https://doi.org/10.1109/4235.942529 - contains an incorrect codon
    # - Book 2003: https://doi.org/10.1007/978-1-4615-0447-4 - used here (pp. 37-42)
    # - Paper ?: http://ncra.ucd.ie/papers/ICAI06_GDE.pdf
    bnf_text = """
    <expr> ::= <expr><op><expr>
             | ( <expr><op><expr> )
             | <pre-op> ( <expr> )
             | <var>
    <op> ::= +
           | -
           | /
           | *
    <pre-op> ::= Sin
    <var> ::= X
            | 1.0
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    genotype = [220, 40, 16, 203, 101, 53, 202, 203, 102, 55, 220, 202,
                19, 130, 37, 202, 203, 32, 39, 202, 203, 102]
    phenotype = grammar.generate_string(method='ge', genotype=genotype)
    assert phenotype == '1.0-Sin(X)*Sin(X)-Sin(X)*Sin(X)'

    tree = grammar.generate_derivation_tree(method='ge', genotype=genotype)
    assert tree.string() == '1.0-Sin(X)*Sin(X)-Sin(X)*Sin(X)'
    expected_derivation = """<expr>
    => <expr><op><expr>
    => <expr><op><expr><op><expr>
    => <expr><op><expr><op><expr><op><expr>
    => <var><op><expr><op><expr><op><expr>
    => 1.0<op><expr><op><expr><op><expr>
    => 1.0-<expr><op><expr><op><expr>
    => 1.0-<pre-op>(<expr>)<op><expr><op><expr>
    => 1.0-Sin(<expr>)<op><expr><op><expr>
    => 1.0-Sin(<var>)<op><expr><op><expr>
    => 1.0-Sin(X)<op><expr><op><expr>
    => 1.0-Sin(X)*<expr><op><expr>
    => 1.0-Sin(X)*<expr><op><expr><op><expr>
    => 1.0-Sin(X)*<pre-op>(<expr>)<op><expr><op><expr>
    => 1.0-Sin(X)*Sin(<expr>)<op><expr><op><expr>
    => 1.0-Sin(X)*Sin(<var>)<op><expr><op><expr>
    => 1.0-Sin(X)*Sin(X)<op><expr><op><expr>
    => 1.0-Sin(X)*Sin(X)-<expr><op><expr>
    => 1.0-Sin(X)*Sin(X)-<pre-op>(<expr>)<op><expr>
    => 1.0-Sin(X)*Sin(X)-Sin(<expr>)<op><expr>
    => 1.0-Sin(X)*Sin(X)-Sin(<var>)<op><expr>
    => 1.0-Sin(X)*Sin(X)-Sin(X)<op><expr>
    => 1.0-Sin(X)*Sin(X)-Sin(X)*<expr>
    => 1.0-Sin(X)*Sin(X)-Sin(X)*<pre-op>(<expr>)
    => 1.0-Sin(X)*Sin(X)-Sin(X)*Sin(<expr>)
    => 1.0-Sin(X)*Sin(X)-Sin(X)*Sin(<var>)
    => 1.0-Sin(X)*Sin(X)-Sin(X)*Sin(X)""".replace('    ', '')
    assert tree.derivation() == expected_derivation
    assert tree.derivation(separate_lines=False) == expected_derivation.replace('\n', ' ')


def test_mapping_forward_against_paper_2010_example():
    # References
    # - Paper 2010: https://doi.org/10.1109/CEC.2010.5586204
    # - Paper 2011: https://doi.org/10.1007/978-3-642-20407-4_25

    # Create grammar
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | *
    <v> ::= 0.5 | 5
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Check if genotype is mapped by Python to the same phenotype as in paper
    genotype = [12, 8, 3, 11, 7, 6, 11, 8, 4, 3, 3, 11, 15, 7, 9, 8, 10, 3, 7, 4]
    phenotype = grammar.generate_string(method='ge', genotype=genotype)
    assert phenotype == '5*0.5+5*5'


def test_mapping_forward_against_paper_2017_example1():
    # References
    # - https://doi.org/10.1007/s10710-017-9309-9 - Table 1 and Fig. 1

    # Create grammar
    bnf_text = """
    <code> ::= <code><line> | <line>
    <line> ::= <ifte> | <loop> | <action>
    <ifte> ::= if <cond> { <code> } else { <code> }
    <loop> ::= while ( <cond> ) { <code> }
    <cond> ::= c1 | c2
    <action> ::= a1; | a2;
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Check if genotype is mapped by Python to the same phenotype as in paper
    genotype = [8, 5, 4, 0, 7, 5, 6, 2, 7, 6, 2, 9, 7, 4, 8]
    phenotype = grammar.generate_string(method='ge', genotype=genotype)
    assert phenotype == 'while(c1){a1;}a2;'


def test_mapping_forward_against_paper_2017_example2():
    # References
    # - https://doi.org/10.1007/s10710-017-9309-9 - Fig. 4, Fig. 5 and Fig. 6

    # Create grammar
    bnf_text = """
    <e> ::= + <e> <e>
          | - <e> <e>
          | * <e> <e>
          | / <e> <e>
          | x
          | x
          | 1.0
          | 1.0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Check if genotype is mapped by Python to the same phenotype as in paper
    genotype = [8, 2, 0, 7, 5, 9, 6, 5, 3, 3, 7, 4, 8, 7, 5]
    phenotype = grammar.generate_string(method='ge', genotype=genotype)
    assert phenotype == '+*+1.0x-1.0x//1.0x+1.0x'

    # Check if genotype is mapped by Python to the same phenotype as in paper
    genotype = [0, 2, 4, 9, 7, 5, 5, 9, 3, 6, 2, 9, 7, 4, 8]
    phenotype = grammar.generate_string(method='ge', genotype=genotype)
    assert phenotype == '+*x-1.0xx'


def test_mapping_forward_against_java_reference_implementation():
    # References
    # - https://github.com/robert-haas/GE-mapping-reference
    def nt_to_str(sym):
        return '<{}>'.format(sym)

    directory = os.path.join(IN_DIR, 'mappings', 'geva_reduced')
    filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    assert len(filepaths) == 20
    for filepath in sorted(filepaths):
        # Read data from JSON file
        with open(filepath) as file_handle:
            data = json.load(file_handle)
        bnf_text = data['grammar']['bnf']
        start_symbol = data['grammar']['start_symbol']
        nonterminals = data['grammar']['nonterminals']
        terminals = data['grammar']['terminals']
        parameters = dict(
            codon_size=int(data['parameters']['codon_size']),
            max_wraps=int(data['parameters']['max_wraps']),
            max_expansions=None,
        )
        gen_phe_map = data['genotype_to_phenotype_mappings']
        # Create grammar
        grammar = al.Grammar(bnf_text=bnf_text)
        assert nt_to_str(grammar.start_symbol) == start_symbol
        assert list(nt_to_str(nt) for nt in grammar.nonterminal_symbols) == nonterminals
        assert set(str(ts) for ts in grammar.terminal_symbols) == set(terminals)
        # Check if each genotype is mapped to the same phenotype in Python and Java
        for i, (gen, phe) in enumerate(list(data['genotype_to_phenotype_mappings'].items())):
            genotype = list(eval(gen))
            # Fast implementation (default)
            try:
                phe_calc_fast = al.systems.ge.mapping.forward(
                    genotype=genotype, grammar=grammar, parameters=parameters)
            except al.exceptions.MappingError:
                phe_calc_fast = 'MappingException'
            assert phe == phe_calc_fast
            # Slow implementation
            try:
                dt = al.systems.ge.mapping._forward_slow(
                    grammar, genotype, parameters['max_expansions'], parameters['max_wraps'],
                    raise_errors=True, verbose=False)
                phe_calc_slow = dt.string()
            except al.exceptions.MappingError:
                phe_calc_slow = 'MappingException'
            assert phe == phe_calc_slow
            # Reverse and forward mapping: phenotype -> (randomized) genotype -> same phenotype
            if phe != 'MappingException':
                genotype_rev = al.systems.ge.mapping.reverse(grammar, phe)
                phe_calc_rev = al.systems.ge.mapping.forward(grammar, genotype_rev, parameters)
                assert phe == phe_calc_rev
