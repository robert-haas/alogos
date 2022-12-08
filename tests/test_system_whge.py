import copy
import itertools
import json
import math
import os
import random

import bitarray
import pytest
import shared

import alogos as al


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, "in")


# Shared


def check_genotype(gt):
    assert isinstance(gt, al.systems.whge.representation.Genotype)
    assert isinstance(gt.data, bitarray.bitarray)
    assert len(gt) == len(gt.data) > 0


def check_phenotype(phe):
    if phe is not None:
        assert isinstance(phe, str)
        assert len(phe) > 0


def check_fitness(fitness):
    assert isinstance(fitness, float)


def check_individual(ind):
    assert isinstance(ind, al.systems.whge.representation.Individual)
    check_genotype(ind.genotype)
    check_phenotype(ind.phenotype)
    check_fitness(ind.fitness)


def check_population(pop):
    assert isinstance(pop, al.systems.whge.representation.Population)
    assert len(pop) > 0
    for ind in pop:
        check_individual(ind)


# Tests

# Representation


def test_representation_genotype():
    # Genotypic data of two types:
    # 1) bitarray 2) string of 0 and 1 characters
    data_variants = (
        bitarray.bitarray("0"),
        bitarray.bitarray("1"),
        bitarray.bitarray("001010010101"),
        "0",
        "1",
        "10101010101",
    )
    for data in data_variants:
        gt = al.systems.whge.representation.Genotype(data)
        check_genotype(gt)
        # Printing
        assert isinstance(str(gt), str)
        assert isinstance(repr(gt), str)
        assert repr(gt).startswith("<WHGE genotype at ")
        p1 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p1, False)
        assert p1.string == str(gt)
        p2 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p2, True)
        assert p2.string == "..."
        # Length
        assert len(gt) == len(data)
        if isinstance(data, bitarray.bitarray):
            assert len(gt) == len(data)
        # Copying and equality
        gt2 = gt.copy()
        gt3 = copy.copy(gt)
        gt4 = copy.deepcopy(gt)
        assert id(gt) != id(gt2) != id(gt3) != id(gt4)  # new Genotype object
        assert (
            id(gt.data) != id(gt2.data) != id(gt3.data) != id(gt4.data)
        )  # new bitarray
        assert gt != "nonsense"
        assert not gt == "nonsense"
        assert gt == gt2 == gt3 == gt4
        assert len(gt) == len(gt2) == len(gt3) == len(gt4)
        gt = al.systems.whge.representation.Genotype("10")
        assert gt != gt2 == gt3 == gt4
        assert len(gt) != len(gt2) == len(gt3) == len(gt4)
        # Usage as key
        some_dict = dict()
        some_set = set()
        for i, g in enumerate([gt, gt2, gt3, gt4]):
            some_dict[g] = i
            some_set.add(g)
        assert len(some_dict) == len(some_set) == 2
        # Immutability
        with pytest.raises(al.exceptions.GenotypeError):
            gt.data = "anything"

    invalid_data_variants = (
        "",
        (),
        [],
        False,
        True,
        None,
        1,
        3.14,
        "012",
        [0, 1],
        (0, 1),
    )
    for data in invalid_data_variants:
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.whge.representation.Genotype(data)


def test_representation_individual():
    data_variants = (
        [],
        ["gt"],
        ["gt", "phe"],
        ["gt", "phe", "fit"],
        ["gt", "phe", "fit", "det"],
    )
    for data in data_variants:
        ind = al.systems.whge.representation.Individual(*data)
        # Member types
        assert ind.genotype is None if len(data) < 1 else data[0]
        assert ind.phenotype is None if len(data) < 2 else data[1]
        assert math.isnan(ind.fitness) if len(data) < 3 else data[2]
        assert isinstance(ind.details, dict) if len(data) < 4 else data[3]
        # Printing
        assert isinstance(str(ind), str)
        assert isinstance(repr(ind), str)
        assert str(ind).startswith("WHGE individual:")
        assert repr(ind).startswith("<WHGE individual object at ")
        p1 = shared.MockPrettyPrinter()
        ind._repr_pretty_(p1, False)
        assert p1.string == str(ind)
        p2 = shared.MockPrettyPrinter()
        ind._repr_pretty_(p2, True)
        assert p2.string == "..."
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
    ind1 = al.systems.whge.representation.Individual(fitness=1)
    ind2 = al.systems.whge.representation.Individual(fitness=2)
    assert ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 2: number and NaN
    ind1 = al.systems.whge.representation.Individual(fitness=1)
    ind2 = al.systems.whge.representation.Individual(fitness=float("nan"))
    assert ind1.less_than(ind2, "min")
    assert not ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert not ind2.greater_than(ind1, "max")
    # - Case 3: NaN and number
    ind1 = al.systems.whge.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.whge.representation.Individual(fitness=2)
    assert not ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert not ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 4: NaN and NaN
    ind1 = al.systems.whge.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.whge.representation.Individual(fitness=float("nan"))
    assert not ind1.less_than(ind2, "min")
    assert not ind1.less_than(ind2, "max")
    assert not ind2.greater_than(ind1, "min")
    assert not ind2.greater_than(ind1, "max")
    # Invalid objective - this check was removed for performance improvement
    # with pytest.raises(ValueError):
    #     assert ind1.less_than(ind2, 'nonsense')
    # with pytest.raises(ValueError):
    #     assert ind2.greater_than(ind1, 'nonsense')


def test_representation_population():
    data_variants = (
        [],
        [al.systems.whge.representation.Individual("gt1")],
        [
            al.systems.whge.representation.Individual("gt1"),
            al.systems.whge.representation.Individual("gt2"),
        ],
    )
    for data in data_variants:
        pop = al.systems.whge.representation.Population(data)
        # Member types
        assert isinstance(pop.individuals, list)
        # Printing
        assert isinstance(str(pop), str)
        assert isinstance(repr(pop), str)
        assert str(pop).startswith("WHGE population:")
        assert repr(pop).startswith("<WHGE population at")
        p1 = shared.MockPrettyPrinter()
        pop._repr_pretty_(p1, False)
        assert p1.string == str(pop)
        p2 = shared.MockPrettyPrinter()
        pop._repr_pretty_(p2, True)
        assert p2.string == "..."
        # Length
        assert len(pop) == len(data)
        # Copying
        pop2 = pop.copy()
        pop3 = copy.copy(pop)
        pop4 = copy.deepcopy(pop)
        assert (
            id(pop.individuals)
            != id(pop2.individuals)
            != id(pop3.individuals)
            != id(pop4.individuals)
        )
        pop.individuals = [
            al.systems.whge.representation.Individual("gt3"),
            al.systems.whge.representation.Individual("gt4"),
            al.systems.whge.representation.Individual("gt5"),
        ]
        assert len(pop) != len(pop2) == len(pop3) == len(pop4)
        # Get, set and delete an item
        if len(pop) > 1:
            # Get
            ind = pop[0]
            ind.genotype = 42
            with pytest.raises(TypeError):
                pop["a"]
            with pytest.raises(IndexError):
                pop[300]
            # Set
            pop[0] = ind
            with pytest.raises(TypeError):
                pop[0] = "abc"
            # Delete
            l1 = len(pop)
            del pop[0]
            l2 = len(pop)
            assert l2 == l1 - 1
            with pytest.raises(TypeError):
                del pop["a"]
            with pytest.raises(IndexError):
                del pop[300]
        # Iteration
        for _ in pop:
            pass
        # Concatenation
        pop2 = pop + pop
        assert isinstance(pop2, al.systems.whge.representation.Population)
        assert len(pop) * 2 == len(pop2)
        # Counts
        assert isinstance(pop.num_unique_genotypes, int)
        assert isinstance(pop.num_unique_phenotypes, int)
        assert isinstance(pop.num_unique_fitnesses, int)

    invalid_data_variants = (
        None,
        False,
        True,
        "",
        3,
        3.14,
        "123",
    )
    for data in invalid_data_variants:
        with pytest.raises(TypeError):
            al.systems.whge.representation.Population(data)


# Initialization


def test_initialize_individual():
    # Number of repetitions for methods with randomness
    num_repetitions = 20

    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Method: given_genotype
    valid_genotypes = [
        "0",
        "01010101",
        bitarray.bitarray("1"),
        bitarray.bitarray("01010101"),
    ]
    for gt in valid_genotypes:
        parameters = dict(init_ind_given_genotype=gt)
        ind = al.systems.whge.init_individual.given_genotype(grammar, parameters)
        check_individual(ind)
        if isinstance(gt, str):
            assert ind.genotype.data == bitarray.bitarray(gt)
            assert str(ind.genotype) == gt
        else:
            assert ind.genotype.data == gt
            assert str(ind.genotype) == gt.to01()
    # Parameter: init_ind_given_genotype not valid
    invalid_genotypes = [
        None,
        False,
        True,
        "",
        "abc",
        bitarray.bitarray(""),
        3,
        3.14,
        (0, 1, 2),
    ]
    for gt in invalid_genotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_genotype=gt)
            al.systems.whge.init_individual.given_genotype(grammar, parameters)
    # Parameter: init_ind_given_genotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.whge.init_individual.given_genotype(grammar)

    # Method: random_genotype
    for _ in range(num_repetitions):
        ind = al.systems.whge.init_individual.random_genotype(grammar)
        check_individual(ind)
    # Parameter: genotype_length
    parameters = dict(genotype_length=21)
    ind = al.systems.whge.init_individual.random_genotype(grammar, parameters)
    check_individual(ind)
    assert len(ind.genotype) == len(ind.genotype.data) == 21
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.whge.init_individual.random_genotype(grammar, parameters)

    # Note: Several methods are not available for WHGE because there
    # is no effective reverse mapping procedure, hence phenotypes and
    # derivation trees can not be used as input, because they would
    # need to be mapped back to a genotype


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
        ["0"],
        ["0", "1", "00", "01", "10", "11"],
        [bitarray.bitarray("0")],
        [bitarray.bitarray("0"), bitarray.bitarray("1"), bitarray.bitarray("00")],
    ]
    for gts in valid_genotype_collections:
        parameters = dict(init_pop_given_genotypes=gts)
        pop = al.systems.whge.init_population.given_genotypes(grammar, parameters)
        check_population(pop)
        assert len(pop) == len(gts)
    # Parameter: init_pop_given_genotypes not valid
    invalid_genotype_collections = [
        None,
        [],
        [None],
        [0],
        [3.14],
        ["0", "1", 2],
        [("0", "1"), ("3",)],
    ]
    for gts in invalid_genotype_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_genotypes=gts)
            al.systems.whge.init_population.given_genotypes(grammar, parameters)
    # Parameter: init_pop_given_genotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.whge.init_population.given_genotypes(grammar)

    # Method: random_genotypes
    for _ in range(20):
        pop = al.systems.whge.init_population.random_genotypes(grammar)
        check_population(pop)
        assert len(pop) == al.systems.whge.default_parameters.init_pop_size
        # Parameters: init_pop_size
        parameters = dict(init_pop_size=5)
        pop = al.systems.whge.init_population.random_genotypes(grammar, parameters)
        check_population(pop)
        assert len(pop) == 5
    # Parameter: init_pop_unique_genotypes, init_pop_unique_phenotypes
    for unique_gen in (True, False):
        for unique_phe in (True, False):
            params = dict(
                init_pop_size=10,
                init_pop_unique_max_tries=500,
                init_pop_unique_genotypes=unique_gen,
                init_pop_unique_phenotypes=unique_phe,
            )
            pop = al.systems.whge.init_population.random_genotypes(grammar, params)
            check_population(pop)
            assert len(pop) == 10
            if unique_gen or unique_phe:
                params["genotype_length"] = 3
                with pytest.raises(al.exceptions.InitializationError):
                    al.systems.whge.init_population.random_genotypes(grammar, params)
    # Parameter: init_pop_unique_max_tries
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_unique_max_tries=0)
        al.systems.whge.init_population.random_genotypes(grammar, parameters)
    # Parameter: genotype_length
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.whge.init_population.random_genotypes(grammar, parameters)


# Mutation


def test_mutation():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types: 1) string of bits, 2) bitarray, 3) Genotype class
    genotypes = (
        "1" * 512,
        bitarray.bitarray("0" * 512),
        al.systems.whge.representation.Genotype("10" * 256),
    )

    # Mutation (guaranteed to change the genotype due to parameter choice)
    methods = (
        al.systems.whge.mutation.bit_flip_by_probability,
        al.systems.whge.mutation.bit_flip_by_count,
    )
    params = dict(mutation_bit_flip_probability=1.0, mutation_bit_flip_count=2)
    for method in methods:
        for gt in genotypes:
            # Without parameters (=using defaults)
            gt1 = method(grammar, copy.copy(gt))
            # With parameters
            gt2 = method(grammar, copy.copy(gt), params)
            gt3 = method(grammar, copy.copy(gt), parameters=params)
            gt4 = method(grammar, genotype=copy.copy(gt), parameters=params)
            gt5 = method(grammar=grammar, genotype=copy.copy(gt), parameters=params)
            # Check that resulting genotypes are valid
            check_genotype(gt1)
            check_genotype(gt2)
            check_genotype(gt3)
            check_genotype(gt4)
            check_genotype(gt5)
            # Check that resulting genotypes are different from input genotype
            if isinstance(gt, str):
                gt_bitarr = bitarray.bitarray(gt)
            elif isinstance(gt, al.systems.whge.representation.Genotype):
                gt_bitarr = gt.data
            else:
                gt_bitarr = gt
            assert gt1.data != gt_bitarr
            assert gt2.data != gt_bitarr
            assert gt3.data != gt_bitarr
            assert gt4.data != gt_bitarr
            assert gt5.data != gt_bitarr

    # Mutation (guaranteed to -not- change the genotype due to different parameter choice)
    params = dict(mutation_bit_flip_probability=0.0, mutation_bit_flip_count=0)
    for method in methods:
        for gt in genotypes:
            # With parameters (required here)
            gt2 = method(grammar, copy.copy(gt), params)
            gt3 = method(grammar, copy.copy(gt), parameters=params)
            gt4 = method(grammar, genotype=copy.copy(gt), parameters=params)
            gt5 = method(grammar=grammar, genotype=copy.copy(gt), parameters=params)
            # Check that resulting genotypes are valid
            check_genotype(gt2)
            check_genotype(gt3)
            check_genotype(gt4)
            check_genotype(gt5)
            # Check that resulting genotypes are identical to input genotype
            if isinstance(gt, str):
                gt_bitarr = bitarray.bitarray(gt)
            elif isinstance(gt, al.systems.whge.representation.Genotype):
                gt_bitarr = gt.data
            else:
                gt_bitarr = gt
            assert gt2.data == gt_bitarr
            assert gt3.data == gt_bitarr
            assert gt4.data == gt_bitarr
            assert gt5.data == gt_bitarr


def test_mutation_count():
    # Grammar
    bnf_text = "<bit> ::= 1 | 0"
    grammar = al.Grammar(bnf_text=bnf_text)

    # Mutation
    for _ in range(50):
        for genotype in ("0", "00", "000", "0000", "00000", "0" * 50, "0" * 100):
            # Parameter: bit_flip_count
            for bit_flip_count in range(10):
                parameters = dict(mutation_bit_flip_count=bit_flip_count)
                gt_copy = copy.copy(genotype)
                gt_mut = al.systems.whge.mutation.bit_flip_by_count(
                    grammar, gt_copy, parameters
                )
                check_genotype(gt_mut)
                # Check expected number of bit flips for different cases
                num_changed_codons = gt_mut.data.count()
                if bit_flip_count == 0:
                    assert gt_mut.data == bitarray.bitarray("0" * len(genotype))
                elif bit_flip_count >= len(genotype):
                    assert num_changed_codons == len(genotype)
                else:
                    assert num_changed_codons == bit_flip_count


# Crossover


def test_crossover_api():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types: 1) string of bits, 2) bitarray, 3) Genotype class
    genotypes = (
        "1" * 512,
        bitarray.bitarray("0" * 512),
        al.systems.whge.representation.Genotype("0" * 512),
    )

    # Crossover
    methods = (al.systems.whge.crossover.two_point_length_preserving,)

    def perform_checks(gt1, gt2, gt3, gt4):
        # Ensure input genotypes are of correct type
        if not isinstance(gt1, al.systems.whge.representation.Genotype):
            gt1 = al.systems.whge.representation.Genotype(copy.copy(gt1))
        if not isinstance(gt2, al.systems.whge.representation.Genotype):
            gt2 = al.systems.whge.representation.Genotype(copy.copy(gt2))
        # General checks
        check_genotype(gt1)
        check_genotype(gt2)
        check_genotype(gt3)
        check_genotype(gt4)
        # Specific checks for crossover
        assert gt1 == gt1
        assert gt1 != gt2
        assert gt3 != gt1
        assert gt3 != gt2
        assert gt4 != gt1
        assert gt4 != gt2
        num_1_parents = str(gt1).count("1") + str(gt2).count("1")
        num_1_children = str(gt3).count("1") + str(gt4).count("1")
        assert num_1_parents == num_1_children
        num_0_parents = str(gt1).count("0") + str(gt2).count("0")
        num_0_children = str(gt3).count("0") + str(gt4).count("0")
        assert num_0_parents == num_0_children

    params = dict()
    for _ in range(50):
        for two_genotypes in [
            (genotypes[0], genotypes[1]),
            (genotypes[0], genotypes[2]),
        ]:
            for method in methods:
                gt1, gt2 = two_genotypes

                gt3, gt4 = method(grammar, copy.copy(gt1), copy.copy(gt2), params)
                perform_checks(gt1, gt2, gt3, gt4)

                gt3, gt4 = method(
                    grammar, copy.copy(gt1), copy.copy(gt2), parameters=params
                )
                perform_checks(gt1, gt2, gt3, gt4)

                gt3, gt4 = method(
                    grammar, copy.copy(gt1), genotype2=copy.copy(gt2), parameters=params
                )
                perform_checks(gt1, gt2, gt3, gt4)

                gt3, gt4 = method(
                    grammar,
                    genotype1=copy.copy(gt1),
                    genotype2=copy.copy(gt2),
                    parameters=params,
                )
                perform_checks(gt1, gt2, gt3, gt4)

                gt3, gt4 = method(
                    grammar=grammar,
                    genotype1=copy.copy(gt1),
                    genotype2=copy.copy(gt2),
                    parameters=params,
                )
                perform_checks(gt1, gt2, gt3, gt4)


def test_crossover_fails():
    # Grammar
    bnf_text = "<bit> ::= 1 | 0"
    grammar = al.Grammar(bnf_text=bnf_text)

    # Crossover
    # - invalid genotype types
    gt_valid = "101010101"
    for gt_invalid in [None, False, True, "", 0, 1, 3.14, (1, 0, 1), [1, 0, 1]]:
        al.systems.whge.crossover.two_point_length_preserving(
            grammar, gt_valid, gt_valid
        )
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.whge.crossover.two_point_length_preserving(
                grammar, gt_valid, gt_invalid
            )
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.whge.crossover.two_point_length_preserving(
                grammar, gt_invalid, gt_valid
            )
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.whge.crossover.two_point_length_preserving(
                grammar, gt_invalid, gt_invalid
            )
    # - too short genotype
    gt1 = "0"
    gt2 = "1"
    with pytest.raises(al.exceptions.OperatorError):
        al.systems.whge.crossover.two_point_length_preserving(grammar, gt1, gt2)

    # - genotypes with different length
    gt1 = "000000"
    gt2 = "111111111"
    with pytest.raises(al.exceptions.OperatorError):
        al.systems.whge.crossover.two_point_length_preserving(grammar, gt1, gt2)


# Neighborhood


def test_neighborhood_api():
    # Grammar
    bnf_text = "<S> ::= 1 | 2"
    gr = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types: 1) string of bits, 2) bitarray, 3) Genotype class
    n_bits = 10
    genotypes = (
        "1" * n_bits,
        bitarray.bitarray("0" * n_bits),
        al.systems.whge.representation.Genotype("1" * n_bits),
    )

    # Neighborhood
    for gt in genotypes:
        # Method: bit_flip
        # Default
        nh1 = al.systems.whge.neighborhood.bit_flip(gr, gt)
        nh2 = al.systems.whge.neighborhood.bit_flip(gr, genotype=gt)
        nh3 = al.systems.whge.neighborhood.bit_flip(grammar=gr, genotype=gt)
        nh4 = al.systems.whge.neighborhood.bit_flip(gr, gt, dict())
        nh5 = al.systems.whge.neighborhood.bit_flip(gr, gt, parameters=dict())
        nh6 = al.systems.whge.neighborhood.bit_flip(gr, genotype=gt, parameters=dict())
        nh7 = al.systems.whge.neighborhood.bit_flip(
            grammar=gr, genotype=gt, parameters=dict()
        )
        nh8 = al.systems.whge.neighborhood.bit_flip(
            grammar=gr, genotype=gt, parameters=dict(neighborhood_distance=1)
        )
        assert nh1 == nh2 == nh3 == nh4 == nh5 == nh6 == nh7 == nh8
        assert len(nh1) == len(gt)
        for new_gt in nh1:
            check_genotype(new_gt)
        # Parameter: neighborhood_distance
        for dist in range(1, 5):
            parameters = dict(neighborhood_distance=dist)
            nh = al.systems.whge.neighborhood.bit_flip(gr, gt, parameters)
            assert len(nh) == len(list(itertools.combinations(range(len(gt)), dist)))
            for new_gt in nh:
                check_genotype(new_gt)
        # Parameter: neighborhood_max_size
        for max_size in (1, 2, 5, 10, 15, 20, 50):
            parameters = dict(neighborhood_max_size=max_size)
            nh = al.systems.whge.neighborhood.bit_flip(gr, gt, parameters)
            assert len(nh) <= max_size
            for new_gt in nh:
                check_genotype(new_gt)


@pytest.mark.parametrize(
    "bnf, genotype, phenotype",
    [
        (shared.BNF5, "00000000", "1"),
        (shared.BNF5, "00110000", "2"),
        (shared.BNF5, "00001100", "3"),
        (shared.BNF5, "00000011", "4"),
        (shared.BNF5, "01111111", "5"),
        (shared.BNF6, "00000000", "1"),
        (shared.BNF6, "01001000", "2"),
        (shared.BNF6, "01011000", "3"),
        (shared.BNF6, "00001000", "4"),
        (shared.BNF6, "00000011", "5"),
    ],
)
def test_neighborhood_reachability_in_finite_languages(bnf, genotype, phenotype):
    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.whge.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.whge.mapping.forward(grammar, gt)
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
                nbrs = al.systems.whge.neighborhood.bit_flip(grammar, gen, param)
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                try:
                    phe = al.systems.whge.mapping.forward(grammar, gen, param)
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
    "bnf, genotype, phenotype, strings_given",
    [
        (shared.BNF10, "00100000", "1x", ("2x", "3x", "4y", "5y", "6y", "7")),
        (shared.BNF11, "00100100", "1", ("2", "3", "4", "22", "33", "44")),
        (
            shared.BNF12,
            "0110001000101100",
            "131",
            ("242", "2332", "22422", "21312", "223322"),
        ),
    ],
)
def test_neighborhood_reachability_in_infinite_languages(
    bnf, genotype, phenotype, strings_given
):
    strings_given = set(strings_given)

    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.whge.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.whge.mapping.forward(grammar, gt)
    assert phe == phenotype

    # Neighborhood
    strings_given = set(strings_given)
    params = [
        dict(),
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
                nbrs = al.systems.whge.neighborhood.bit_flip(grammar, gen, param)
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                phe = al.systems.whge.mapping.forward(grammar, gen, param)
                strings_seen.add(phe)
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
    gt = al.systems.whge.representation.Genotype("0000")

    # Neighborhood in different distances when changing only terminals
    # - distance 1
    parameters = dict(neighborhood_only_terminals=True)
    nbrs_gt = al.systems.whge.neighborhood.bit_flip(grammar, gt, parameters)
    assert set(str(gt) for gt in nbrs_gt) == {"0001", "0010", "0100", "1000"}

    # - distance 2
    parameters = dict(neighborhood_distance=2, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.whge.neighborhood.bit_flip(grammar, gt, parameters)
    assert set(str(gt) for gt in nbrs_gt) == {
        "0011",
        "0101",
        "1001",
        "0110",
        "1010",
        "1100",
    }

    # - distance 3
    parameters = dict(neighborhood_distance=3, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.whge.neighborhood.bit_flip(grammar, gt, parameters)
    assert set(str(gt) for gt in nbrs_gt) == {"0111", "1011", "1101", "1110"}

    # - distance 4
    parameters = dict(neighborhood_distance=4, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.whge.neighborhood.bit_flip(grammar, gt, parameters)
    assert set(str(gt) for gt in nbrs_gt) == {"1111"}

    # - distance 5 and greater
    for dist in range(5, 20):
        parameters = dict(neighborhood_distance=dist, neighborhood_only_terminals=True)
        nbrs_gt = al.systems.whge.neighborhood.bit_flip(grammar, gt, parameters)
        assert nbrs_gt == []


def test_neighborhood_parameter_max_size():
    # Grammar
    bnf = "<S> ::= 0 | 1"
    gr = al.Grammar(bnf_text=bnf)

    for num_bits in (1, 2, 5, 10, 20):
        # Genotype
        bits = ("0", "1")
        gt = "".join(random.choice(bits) for _ in range(num_bits))

        # Neighborhood
        nbrs = al.systems.whge.neighborhood.bit_flip(gr, gt)
        assert len(nbrs) == num_bits

        # Parameter: neighborhood_max_size
        parameters = dict(neighborhood_max_size=None)
        nbrs = al.systems.whge.neighborhood.bit_flip(gr, gt, parameters)
        assert len(nbrs) == num_bits

        for max_size in range(1, 5):
            parameters = dict(neighborhood_max_size=max_size)
            nbrs_gts = set()
            for _ in range(1000):
                nbrs = al.systems.whge.neighborhood.bit_flip(gr, gt, parameters)
                assert len(nbrs) <= max_size
                for nbr_gt in nbrs:
                    nbrs_gts.add(str(nbr_gt))
            assert len(nbrs_gts) == num_bits


# Mapping


def test_mapping_forward_api():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypic data of three types:
    # 1) bitarray 2) string of 0 and 1 characters 3) Genotype object
    bit_string = (
        "0000 1000 0100 1100 0010 1010 0110 1110 "
        "0001 1001 0101 1101 0011 1011 0111 1111"
    ).replace(" ", "")
    data_variants = (
        bitarray.bitarray(bit_string),
        bit_string,
        al.systems.whge.representation.Genotype(bit_string),
    )

    # Forward WHGE mapping
    parameters = dict(
        max_expansions=3,
        max_depth=0,
    )
    kwargs = dict(
        verbose=False,
        raise_errors=False,
        return_derivation_tree=False,
    )
    for data in data_variants:
        for vb in (True, False):
            kwargs["verbose"] = vb

            # Method of Grammar class
            string1 = grammar.generate_string("whge", data, parameters, **kwargs)
            string2 = grammar.generate_string(
                "whge", data, parameters=parameters, **kwargs
            )
            string3 = grammar.generate_string(
                method="whge", genotype=data, parameters=parameters, **kwargs
            )
            assert string1
            assert string1 == string2 == string3

            # Method of DerivationTree class
            dt1 = grammar.generate_derivation_tree("whge", data, parameters, **kwargs)
            dt2 = grammar.generate_derivation_tree(
                "whge", data, parameters=parameters, **kwargs
            )
            dt3 = grammar.generate_derivation_tree(
                method="whge", genotype=data, parameters=parameters, **kwargs
            )
            assert string1 == dt1.string() == dt2.string() == dt3.string()

            # Functions in mapping module
            string4 = al.systems.whge.mapping.forward(
                grammar, data, parameters, **kwargs
            )
            string5 = al.systems.whge.mapping.forward(
                grammar, data, parameters=parameters, **kwargs
            )
            string6 = al.systems.whge.mapping.forward(
                grammar=grammar, genotype=data, parameters=parameters, **kwargs
            )
            assert string1 == string4 == string5 == string6

            kwargs["return_derivation_tree"] = True
            phe, dt4 = al.systems.whge.mapping.forward(
                grammar, data, parameters, **kwargs
            )
            phe, dt5 = al.systems.whge.mapping.forward(
                grammar, data, parameters=parameters, **kwargs
            )
            phe, dt6 = al.systems.whge.mapping.forward(
                grammar=grammar, genotype=data, parameters=parameters, **kwargs
            )
            kwargs["return_derivation_tree"] = False
            assert string1 == dt4.string() == dt5.string() == dt6.string()

            # - Same with errors when reaching wrap or expansion limit
            kwargs["raise_errors"] = True
            with pytest.raises(al.exceptions.MappingError):
                grammar.generate_string(
                    method="whge", genotype=data, parameters=parameters, **kwargs
                )
            with pytest.raises(al.exceptions.MappingError):
                grammar.generate_derivation_tree(
                    method="whge", genotype=data, parameters=parameters, **kwargs
                )
            with pytest.raises(al.exceptions.MappingError):
                al.systems.whge.mapping.forward(
                    grammar, genotype=data, parameters=parameters, **kwargs
                )
            with pytest.raises(al.exceptions.MappingError):
                al.systems.whge.mapping.forward(
                    grammar, genotype=data, parameters=parameters, **kwargs
                )
            kwargs["raise_errors"] = False


def test_mapping_forward_special_case():
    # Reaches parts of the code (stopping the recursion) that other tests do not
    grammar = al.Grammar(bnf_text="<S> ::= <S><S> | 1")
    for verbose in (True, False):
        al.systems.whge.mapping.forward(grammar, "0000", verbose=verbose)


def test_mapping_forward_against_paper_2018_example():
    # References
    # - https://doi.org/10.1109/TCYB.2018.2876563
    bnf_text = """
    <expr> ::= ( <expr> <op> <expr> ) | <num> | <var>
    <op> ::= + | - | * | /
    <var> ::= x | y
    <num> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    genotype = "11100111-11110000-10100101-01110001-01100101-00000111".replace("-", "")
    for vb in (True, False):
        grammar.generate_string("whge", genotype)
        grammar.generate_string("whge", genotype, dict(max_depth=1))
        phenotype_md1 = grammar.generate_string(
            "whge", genotype, dict(max_depth=1), verbose=vb
        )
        phenotype_md2 = grammar.generate_string(
            "whge", genotype, dict(max_depth=2), verbose=vb
        )
        phenotype_md3 = grammar.generate_string(
            "whge", genotype, dict(max_depth=3), verbose=vb
        )
        assert phenotype_md1 == "((y/y)/x)"  # phenotype from Java implementation
        assert phenotype_md2 == "((y*2)*(3-y))"  # phenotype from paper
        assert phenotype_md3 == "((y-2)/(3-y))"  # phenotype from Java implementation

        # Check if inner workings of WHGE behave as expected (cached calculations within grammar)
        assert shared.conv_keys_to_str(grammar._cache["whge"]["sd"]) == {
            "expr": [1, 2],
            "op": [0, 1, 2, 3],
            "var": [0, 1],
            "num": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        assert shared.conv_keys_to_str(grammar._cache["whge"]["ep_d1"]) == {
            "expr": 5,
            "op": 2,
            "num": 4,
            "var": 1,
        }
        assert shared.conv_keys_to_str(grammar._cache["whge"]["ep_d2"]) == {
            "expr": 7,
            "op": 2,
            "num": 4,
            "var": 1,
        }
        assert shared.conv_keys_to_str(grammar._cache["whge"]["ep_d3"]) == {
            "expr": 8,
            "op": 2,
            "num": 4,
            "var": 1,
        }

    # Check if further genotypes are mapped to expected phenotypes
    for vb in (True, False):
        genotype = "1".replace("-", "")
        phenotype_md1 = grammar.generate_string(
            "whge", genotype, dict(max_depth=1), verbose=vb
        )
        phenotype_md2 = grammar.generate_string(
            "whge", genotype, dict(max_depth=2), verbose=vb
        )
        phenotype_md3 = grammar.generate_string(
            "whge", genotype, dict(max_depth=3), verbose=vb
        )
        assert phenotype_md1 == "y"  # from Java implementation
        assert phenotype_md2 == "y"  # from Java implementation
        assert phenotype_md3 == "y"  # from Java implementation

        genotype = "0".replace("-", "")
        phenotype_md1 = grammar.generate_string(
            "whge", genotype, dict(max_depth=1), verbose=vb
        )
        phenotype_md2 = grammar.generate_string(
            "whge", genotype, dict(max_depth=2), verbose=vb
        )
        phenotype_md3 = grammar.generate_string(
            "whge", genotype, dict(max_depth=3), verbose=vb
        )
        assert phenotype_md1 == "0"  # from Java implementation
        assert phenotype_md2 == "0"  # from Java implementation
        assert phenotype_md3 == "0"  # from Java implementation

        genotype = "111101011".replace("-", "")
        phenotype_md1 = grammar.generate_string(
            "whge", genotype, dict(max_depth=1), verbose=vb
        )
        phenotype_md2 = grammar.generate_string(
            "whge", genotype, dict(max_depth=2), verbose=vb
        )
        phenotype_md3 = grammar.generate_string(
            "whge", genotype, dict(max_depth=3), verbose=vb
        )
        assert phenotype_md1 == "(3-1)"  # from Java implementation
        assert phenotype_md2 == "(3-1)"  # from Java implementation
        assert phenotype_md3 == "(3+y)"  # from Java implementation


def test_mapping_forward_against_java_implementation_with_single_example():
    bnf_text = """
    <expression> ::= <term> | <term> + <expression>
    <term>       ::= <factor> | <factor> * <term>
    <factor>     ::= <constant> | <variable> | ( <expression> )
    <variable>   ::= x | y | z
    <constant>   ::= <digit> | <digit> <constant>
    <digit>      ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Check if genotypes produce expected phenotypes
    genotype = "11100111-11110000-10100101-01110001-01100101-00000111".replace("-", "")
    phenotype_md1 = grammar.generate_string("whge", genotype, dict(max_depth=1))
    phenotype_md2 = grammar.generate_string("whge", genotype, dict(max_depth=2))
    phenotype_md3 = grammar.generate_string("whge", genotype, dict(max_depth=3))
    assert phenotype_md1 == "1"  # from Java implementation
    assert phenotype_md2 == "1"  # from Java implementation
    assert phenotype_md3 == "1"  # from Java implementation

    # Check if inner workings of WHGE behave as expected (cached computations stored in grammar)
    assert shared.conv_keys_to_str(grammar._cache["whge"]["sd"]) == {
        "expression": [0],
        "term": [0],
        "factor": [1],
        "variable": [0, 1, 2],
        "constant": [0],
        "digit": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    assert shared.conv_keys_to_str(grammar._cache["whge"]["ep_d1"]) == {
        "factor": 4,
        "term": 4,
        "variable": 2,
        "expression": 3,
        "constant": 5,
        "digit": 4,
    }
    assert shared.conv_keys_to_str(grammar._cache["whge"]["ep_d2"]) == {
        "factor": 6,
        "term": 5,
        "variable": 2,
        "expression": 5,
        "constant": 6,
        "digit": 4,
    }
    assert shared.conv_keys_to_str(grammar._cache["whge"]["ep_d3"]) == {
        "factor": 7,
        "term": 7,
        "variable": 2,
        "expression": 7,
        "constant": 6,
        "digit": 4,
    }

    # Check if further genotypes produce expected phenotypes
    genotype = "1111111-11111111-11111111-11111111-11111111".replace("-", "")
    phenotype_md1 = grammar.generate_string("whge", genotype, dict(max_depth=1))
    phenotype_md2 = grammar.generate_string("whge", genotype, dict(max_depth=2))
    phenotype_md3 = grammar.generate_string("whge", genotype, dict(max_depth=3))
    assert phenotype_md1 == "55*31*1*x+(1+x+x)*x*y*x"  # from Java implementation
    assert phenotype_md2 == "x+7+1*x+y*x"  # from Java implementation
    assert phenotype_md3 == "x+7+1*x+y*x"  # from Java implementation

    genotype = "1111111-11100111-11110111-11111".replace("-", "")
    phenotype_md1 = grammar.generate_string("whge", genotype, dict(max_depth=1))
    phenotype_md2 = grammar.generate_string("whge", genotype, dict(max_depth=2))
    phenotype_md3 = grammar.generate_string("whge", genotype, dict(max_depth=3))
    assert phenotype_md1 == "x+x*y*y+y*y+y"  # from Java implementation
    assert phenotype_md2 == "x+z+1+y+y"  # from Java implementation
    assert phenotype_md3 == "x+z+1+y+y"  # from Java implementation


def test_mapping_forward_against_java_implementation_with_a_lot_of_grammars_and_genotypes():
    def nt_to_str(sym):
        return "<{}>".format(sym)

    directory = os.path.join(IN_DIR, "mappings", "evolved-ge_reduced")
    filepaths = [
        os.path.join(directory, filename) for filename in os.listdir(directory)
    ]
    assert len(filepaths) == 20 * 3
    for filepath in sorted(filepaths):
        # Read data from JSON file
        with open(filepath) as file_handle:
            data = json.load(file_handle)
        bnf_text = data["grammar"]["bnf"]
        start_symbol = data["grammar"]["start_symbol"]
        nonterminals = data["grammar"]["nonterminals"]
        max_depth = int(data["parameters"]["max_depth"])
        # Create grammar
        grammar = al.Grammar(bnf_text=bnf_text)
        assert nt_to_str(grammar.start_symbol) == start_symbol
        assert list(nt_to_str(nt) for nt in grammar.nonterminal_symbols) == nonterminals
        # Check if each genotype is mapped to the same phenotype in Python and Java
        for gen, phe in data["genotype_to_phenotype_mappings"].items():
            # Fast implementation (default)
            phe_calc_fast = al.systems.whge.mapping.forward(
                grammar, gen, dict(max_depth=max_depth)
            )
            assert phe == phe_calc_fast
            # Slow implementation (call depends on internals if not via verbose=True)
            sd = grammar._lookup_or_calc(
                "whge",
                "sd",
                al.systems.whge._cached_calculations.shortest_distances,
                grammar,
            )
            ep = grammar._lookup_or_calc(
                "whge",
                "ep_d{}".format(max_depth),
                al.systems.whge._cached_calculations.expressive_powers,
                grammar,
                max_depth,
            )
            dt = al.systems.whge.mapping._forward_slow(
                grammar,
                al.systems.whge.representation.Genotype(gen).data,
                max_expansions=None,
                shortest_distances=sd,
                expressive_powers=ep,
                verbose=False,
                raise_errors=False,
            )
            phe_calc_slow = dt.string()
            assert phe == phe_calc_slow
