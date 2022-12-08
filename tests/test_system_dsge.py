import copy
import itertools
import json
import math
import os
import random

import pytest
import shared

import alogos as al


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, "in")


# Shared


def check_genotype(gt):
    assert isinstance(gt, al.systems.dsge.representation.Genotype)
    assert isinstance(gt.data, tuple)
    for gene in gt.data:
        assert isinstance(gene, tuple)
        for codon in gene:
            assert isinstance(codon, int)
    assert len(gt) == len(gt.data) > 0


def check_phenotype(phe):
    if phe is not None:
        assert isinstance(phe, str)
        assert len(phe) > 0


def check_fitness(fitness):
    assert isinstance(fitness, float)


def check_individual(ind):
    assert isinstance(ind, al.systems.dsge.representation.Individual)
    check_genotype(ind.genotype)
    check_phenotype(ind.phenotype)
    check_fitness(ind.fitness)


def check_population(pop):
    assert isinstance(pop, al.systems.dsge.representation.Population)
    assert len(pop) > 0
    for ind in pop:
        check_individual(ind)


# Representation


def test_representation_genotype():
    # Genotypic data of four types:
    # 1) tuple of tuples of int 2) string thereof 3) list of lists of int 4) string thereof
    data_variants = (
        ((0, 1), (0,)),
        ((42,),),
        ((0,), (8, 15)),
        "((0, 1), (0,))",
        "((42,),)",
        "((0,), (8, 15))",
        [[0, 1], [0]],
        [[42]],
        [[0], [8, 15]],
        "[[0, 1], [0]]",
        "[[42]]",
        "[[0], [8, 15]]",
    )
    for data in data_variants:
        gt = al.systems.dsge.representation.Genotype(data)
        check_genotype(gt)
        # Printing
        assert isinstance(str(gt), str)
        assert isinstance(repr(gt), str)
        assert repr(gt).startswith("<DSGE genotype at ")
        p1 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p1, False)
        assert p1.string == str(gt)
        p2 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p2, True)
        assert p2.string == "..."
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
        assert gt != "nonsense"
        assert not gt == "nonsense"
        assert gt == gt2 == gt3 == gt4
        assert len(gt) == len(gt2) == len(gt3) == len(gt4)
        gt = al.systems.dsge.representation.Genotype(((1, 2, 3, 4, 5, 6), (7, 8), (9)))
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
    )
    for data in invalid_data_variants:
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.dsge.representation.Genotype(data)


def test_representation_individual():
    data_variants = (
        [],
        ["gt"],
        ["gt", "phe"],
        ["gt", "phe", "fit"],
        ["gt", "phe", "fit", "det"],
    )
    for data in data_variants:
        ind = al.systems.dsge.representation.Individual(*data)
        # Member types
        assert ind.genotype is None if len(data) < 1 else data[0]
        assert ind.phenotype is None if len(data) < 2 else data[1]
        assert math.isnan(ind.fitness) if len(data) < 3 else data[2]
        assert isinstance(ind.details, dict) if len(data) < 4 else data[3]
        # Printing
        assert isinstance(str(ind), str)
        assert isinstance(repr(ind), str)
        assert str(ind).startswith("DSGE individual:")
        assert repr(ind).startswith("<DSGE individual object at ")
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
    ind1 = al.systems.dsge.representation.Individual(fitness=1)
    ind2 = al.systems.dsge.representation.Individual(fitness=2)
    assert ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 2: number and NaN
    ind1 = al.systems.dsge.representation.Individual(fitness=1)
    ind2 = al.systems.dsge.representation.Individual(fitness=float("nan"))
    assert ind1.less_than(ind2, "min")
    assert not ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert not ind2.greater_than(ind1, "max")
    # - Case 3: NaN and number
    ind1 = al.systems.dsge.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.dsge.representation.Individual(fitness=2)
    assert not ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert not ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 4: NaN and NaN
    ind1 = al.systems.dsge.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.dsge.representation.Individual(fitness=float("nan"))
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
        [al.systems.dsge.representation.Individual("gt1")],
        [
            al.systems.dsge.representation.Individual("gt1"),
            al.systems.dsge.representation.Individual("gt2"),
        ],
    )
    for data in data_variants:
        pop = al.systems.dsge.representation.Population(data)
        # Member types
        assert isinstance(pop.individuals, list)
        # Printing
        assert isinstance(str(pop), str)
        assert isinstance(repr(pop), str)
        assert str(pop).startswith("DSGE population:")
        assert repr(pop).startswith("<DSGE population at")
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
            al.systems.dsge.representation.Individual("gt3"),
            al.systems.dsge.representation.Individual("gt4"),
            al.systems.dsge.representation.Individual("gt5"),
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
        assert isinstance(pop2, al.systems.dsge.representation.Population)
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
            al.systems.dsge.representation.Population(data)


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
        ((0,), (0,), (0, 0, 0, 0, 1, 1, 1, 1)),
        ((0,), (0,), (0, 0, 0, 0, 1, 1, 1)),
        "((1,), (0, 0), (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1))",
        [[0], [0], [0, 0, 0, 0, 1, 1, 1, 1]],
        [[0], [0], [0, 0, 0, 0, 1, 1, 1]],
        "[[1], [0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]",
    ]
    for gt in valid_genotypes:
        parameters = dict(init_ind_given_genotype=gt)
        ind = al.systems.dsge.init_individual.given_genotype(grammar, parameters)
        check_individual(ind)
        assert ind.genotype.data == tuple(tuple(gene) for gene in eval(str(gt)))
    # Parameter: init_ind_given_genotype not valid
    invalid_genotypes = [
        None,
        False,
        True,
        [],
        (),
        "",
        "abc",
        3,
        3.14,
    ]
    for gt in invalid_genotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_genotype=gt)
            al.systems.dsge.init_individual.given_genotype(grammar, parameters)
    # Parameter: init_ind_given_genotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_individual.given_genotype(grammar)

    # Method: given_derivation_tree
    valid_derivation_trees = [
        grammar.generate_derivation_tree(
            "ge", "[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]"
        ),
        grammar.generate_derivation_tree(
            "ge", (5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4)
        ),
    ]
    for dt in valid_derivation_trees:
        parameters = dict(init_ind_given_derivation_tree=dt)
        ind = al.systems.dsge.init_individual.given_derivation_tree(grammar, parameters)
        check_individual(ind)
        ind_dt = ind.details["derivation_tree"]
        assert isinstance(ind_dt, al._grammar.data_structures.DerivationTree)
        assert ind_dt == dt
    # Parameter: init_ind_given_derivation_tree not valid
    invalid_derivation_trees = [None, False, True, "", "abc", 3, 3.14, (0, 1, 2)]
    for dt in invalid_derivation_trees:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_derivation_tree=dt)
            al.systems.dsge.init_individual.given_derivation_tree(grammar, parameters)
    # Parameter: init_ind_given_derivation_tree not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_individual.given_derivation_tree(grammar)

    # Method: given_phenotype
    valid_phenotypes = ["11110000", "1111000011110000"]
    for phe in valid_phenotypes:
        parameters = dict(init_ind_given_phenotype=phe)
        ind = al.systems.dsge.init_individual.given_phenotype(grammar, parameters)
        check_individual(ind)
        assert ind.phenotype == phe
    # Parameter: init_ind_given_phenotype not valid
    invalid_phenotypes = [None, False, True, "", "abc", 3, 3.14, (0, 1, 2)]
    for phe in invalid_phenotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_phenotype=phe)
            al.systems.dsge.init_individual.given_phenotype(grammar, parameters)
    # Parameter: init_ind_given_phenotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_individual.given_phenotype(grammar)

    # Method: random_genotype
    for _ in range(num_repetitions):
        ind = al.systems.dsge.init_individual.random_genotype(grammar)
        check_individual(ind)
    # Parameter: init_depth
    for depth in (0, 5, 10, 20, 40):
        parameters = dict(init_depth=depth)
        ind = al.systems.dsge.init_individual.random_genotype(grammar, parameters)
        check_individual(ind)
        assert len(ind.genotype) == len(ind.genotype.data)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_depth="a")
        al.systems.dsge.init_individual.random_genotype(grammar, parameters)

    # Method: gp_grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.dsge.init_individual.gp_grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_grow_max_depth
    al.systems.dsge.init_individual.gp_grow_tree(
        grammar, dict(init_ind_gp_grow_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.dsge.init_individual.gp_grow_tree(
            grammar, dict(init_ind_gp_grow_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_individual.gp_grow_tree(
            grammar, dict(init_ind_gp_grow_max_depth="nonsense")
        )

    # Method: gp_full_tree
    for _ in range(num_repetitions):
        ind = al.systems.dsge.init_individual.gp_full_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_full_max_depth
    al.systems.dsge.init_individual.gp_full_tree(
        grammar, dict(init_ind_gp_full_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.dsge.init_individual.gp_full_tree(
            grammar, dict(init_ind_gp_full_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_individual.gp_full_tree(
            grammar, dict(init_ind_gp_full_max_depth="nonsense")
        )

    # Method: pi_grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.dsge.init_individual.pi_grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_grow_max_depth
    al.systems.dsge.init_individual.pi_grow_tree(
        grammar, dict(init_ind_pi_grow_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.dsge.init_individual.pi_grow_tree(
            grammar, dict(init_ind_pi_grow_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_individual.pi_grow_tree(
            grammar, dict(init_ind_pi_grow_max_depth="nonsense")
        )

    # Method: ptc2
    for _ in range(num_repetitions):
        ind = al.systems.dsge.init_individual.ptc2_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_ptc2_max_expansions
    al.systems.dsge.init_individual.ptc2_tree(
        grammar, dict(init_ind_ptc2_max_expansions=0)
    )
    for _ in range(num_repetitions):
        al.systems.dsge.init_individual.ptc2_tree(
            grammar, dict(init_ind_ptc2_max_expansions=100)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_individual.ptc2_tree(
            grammar, dict(init_ind_ptc2_max_expansions="nonsense")
        )


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
        [
            [[0], [0], [0, 0, 0, 0, 0, 0, 0, 0]],
            [[1], [0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        ],
        [
            "((0,), (0,), (1, 1, 1, 1, 1, 1, 0, 0))",
            "[[1], [0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]",
        ],
        [
            ((0,), (0,), (1, 0, 0, 0, 0, 0, 1, 0)),
            "[[0], [0], [1, 1, 1, 1, 0, 0, 1, 1]]",
            "[[1], [0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]",
            [[1], [0, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]],
        ],
    ]
    for gts in valid_genotype_collections:
        parameters = dict(init_pop_given_genotypes=gts)
        pop = al.systems.dsge.init_population.given_genotypes(grammar, parameters)
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
        [1, "[0, 1]"],
    ]
    for gts in invalid_genotype_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_genotypes=gts)
            al.systems.dsge.init_population.given_genotypes(grammar, parameters)
    # Parameter: init_pop_given_genotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_population.given_genotypes(grammar)

    # Method: given_derivation_trees
    valid_derivation_tree_collections = [
        [
            grammar.generate_derivation_tree("whge", "11001"),
            grammar.generate_derivation_tree(
                "ge", "[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2]"
            ),
        ],
        [
            grammar.generate_derivation_tree("whge", "10"),
            grammar.generate_derivation_tree("ge", [0, 7, 11]),
        ],
    ]
    for dts in valid_derivation_tree_collections:
        parameters = dict(init_pop_given_derivation_trees=dts)
        pop = al.systems.dsge.init_population.given_derivation_trees(
            grammar, parameters
        )
        check_population(pop)
        assert len(pop) == len(dts)
        for ind in pop:
            ind_dt = ind.details["derivation_tree"]
            assert isinstance(ind_dt, al._grammar.data_structures.DerivationTree)
            assert ind_dt in dts
    # Parameter: init_pop_given_derivation_trees not valid
    invalid_derivation_tree_collections = [
        None,
        [],
        [None],
        [grammar.generate_derivation_tree("whge", "11001"), 0],
        [3.14],
        [[0, 1], 1],
        [1, "[0, 1]"],
    ]
    for dts in invalid_derivation_tree_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_derivation_trees=dts)
            al.systems.dsge.init_population.given_derivation_trees(grammar, parameters)
    # Parameter: init_pop_given_derivation_trees not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_population.given_derivation_trees(grammar)

    # Method: given_phenotypes
    valid_phenotype_collections = [
        ["00000000", "11111111"],
        [
            "00000000",
            "11111111",
            "00000000",
            "11111111",
            "0000111100001111",
            "1111000011110000",
        ],
    ]
    for pts in valid_phenotype_collections:
        parameters = dict(init_pop_given_phenotypes=pts)
        pop = al.systems.dsge.init_population.given_phenotypes(grammar, parameters)
        check_population(pop)
        assert len(pop) == len(pts)
    # Parameter: init_pop_given_phenotypes not valid
    invalid_phenotype_collections = [
        None,
        [],
        [None],
        ["000000001", "11111111"],
        ["00000000", "111111110"],
    ]
    for pts in invalid_phenotype_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_phenotypes=pts)
            al.systems.dsge.init_population.given_derivation_trees(grammar, parameters)
    # Parameter: init_pop_given_phenotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.dsge.init_population.given_phenotypes(grammar)

    # Method: random_genotypes
    n = 10
    for _ in range(n):
        pop = al.systems.dsge.init_population.random_genotypes(grammar)
        check_population(pop)
        assert len(pop) == al.systems.dsge.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.dsge.init_population.random_genotypes(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameter: init_pop_unique_genotypes, init_pop_unique_phenotypes
    for unique_gen in (True, False):
        for unique_phe in (True, False):
            params = dict(
                init_pop_size=10,
                init_pop_unique_max_tries=500,
                init_pop_unique_genotypes=unique_gen,
                init_pop_unique_phenotypes=unique_phe,
            )
            pop = al.systems.dsge.init_population.random_genotypes(grammar, params)
            check_population(pop)
            assert len(pop) == 10
            if unique_gen or unique_phe:
                params["init_pop_size"] = 10000
                with pytest.raises(al.exceptions.InitializationError):
                    al.systems.dsge.init_population.random_genotypes(grammar, params)
    # Parameter: init_pop_unique_max_tries
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_unique_max_tries=0)
        al.systems.dsge.init_population.random_genotypes(grammar, parameters)

    # Method: gp_rhh (=GP's ramped half and half)
    for _ in range(n):
        pop = al.systems.dsge.init_population.gp_rhh(grammar)
        check_population(pop)
        assert len(pop) == al.systems.dsge.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.dsge.init_population.gp_rhh(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_gp_rhh_start_depth, init_pop_gp_rhh_end_depth
    parameters = dict(init_pop_gp_rhh_start_depth=3, init_pop_gp_rhh_end_depth=4)
    pop = al.systems.dsge.init_population.gp_rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_gp_rhh_start_depth=5, init_pop_gp_rhh_end_depth=3)
        pop = al.systems.dsge.init_population.gp_rhh(grammar, parameters)

    # Method: pi_rhh (=position-independent ramped half and half)
    for _ in range(n):
        pop = al.systems.dsge.init_population.pi_rhh(grammar)
        check_population(pop)
        assert len(pop) == al.systems.dsge.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.dsge.init_population.pi_rhh(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_pi_rhh_start_depth, init_pop_pi_rhh_end_depth
    parameters = dict(init_pop_pi_rhh_start_depth=3, init_pop_pi_rhh_end_depth=4)
    pop = al.systems.dsge.init_population.pi_rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_pi_rhh_start_depth=5, init_pop_pi_rhh_end_depth=3)
        pop = al.systems.dsge.init_population.pi_rhh(grammar, parameters)

    # Method: ptc2 (=probabilistic tree creation 2)
    for _ in range(n):
        pop = al.systems.dsge.init_population.ptc2(grammar)
        check_population(pop)
        assert len(pop) == al.systems.dsge.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.dsge.init_population.ptc2(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_ptc2_start_expansions, init_pop_ptc2_end_expansions
    parameters = dict(
        init_pop_ptc2_start_expansions=10, init_pop_ptc2_end_expansions=50
    )
    pop = al.systems.dsge.init_population.ptc2(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(
            init_pop_ptc2_start_expansions=50, init_pop_ptc2_end_expansions=10
        )
        pop = al.systems.dsge.init_population.ptc2(grammar, parameters)


# Mutation


def test_mutation1():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte> | <bytes> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes of four types:
    # 1) tuple of tuples of int 2) string thereof 3) list of lists of int, 4) string thereof
    genotypes = (
        ((0,), (0,), (1, 0, 0, 0, 0, 0, 1, 0)),
        ((0,), (0,), (0, 0, 0, 0, 0, 0, 0, 0)),
        ((1,), (0, 0), (0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0)),
        (
            (2,),
            (0, 0, 0),
            (1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0),
        ),
        "((0,), (0,), (1, 0, 0, 0, 0, 0, 1, 0))",
        [[0], [0], [1, 0, 0, 0, 0, 0, 1, 0]],
        "[[0], [0], [1, 0, 0, 0, 0, 0, 1, 0]]",
        al.systems.dsge.representation.Genotype([[0], [0], [1, 0, 0, 0, 0, 0, 1, 0]]),
        al.systems.dsge.representation.Genotype(
            "((0,), (0,), (1, 0, 1, 1, 0, 0, 0, 1))"
        ),
    )

    # Mutation (guaranteed to change the genotype due to parameter choice)
    methods = (
        al.systems.dsge.mutation.int_replacement_by_probability,
        al.systems.dsge.mutation.int_replacement_by_count,
    )
    p1 = dict(
        mutation_int_replacement_probability=1.0, mutation_int_replacement_count=2
    )
    p2 = dict(
        mutation_int_replacement_probability=1.0,
        mutation_int_replacement_count=2,
        repair_after_mutation=True,
    )
    p3 = dict(
        mutation_int_replacement_probability=1.0,
        mutation_int_replacement_count=2,
        max_depth=1,
    )
    for params in (p1, p2, p3):
        for method in methods:
            for gt in genotypes:
                # Without parameters (=using defaults, mutation happens with some probability)
                method(grammar, copy.deepcopy(gt))
                # With parameters (mutation happens with a probability of 100%)
                gt2 = method(grammar, copy.deepcopy(gt), params)
                gt3 = method(grammar, copy.deepcopy(gt), parameters=params)
                gt4 = method(grammar, genotype=copy.deepcopy(gt), parameters=params)
                gt5 = method(
                    grammar=grammar, genotype=copy.deepcopy(gt), parameters=params
                )
                assert gt != gt2
                assert gt != gt3
                assert gt != gt4
                assert gt != gt5

    # Mutation (guaranteed to -not- change the genotype due to different parameter choice)
    p1 = dict(
        mutation_int_replacement_probability=0.0, mutation_int_replacement_count=0
    )
    p2 = dict(
        mutation_int_replacement_probability=0.0,
        mutation_int_replacement_count=0,
        repair_after_mutation=True,
    )
    p3 = dict(
        mutation_int_replacement_probability=0.0,
        mutation_int_replacement_count=0,
        max_depth=1,
    )
    for params in (p1, p2, p3):
        for method in methods:
            for gt in genotypes:
                # With parameters (mutation happens with a probability of 0%)
                gt2 = method(grammar, copy.copy(gt), params)
                gt3 = method(grammar, copy.copy(gt), parameters=params)
                gt4 = method(grammar, genotype=copy.copy(gt), parameters=params)
                gt5 = method(grammar=grammar, genotype=copy.copy(gt), parameters=params)
                assert gt2 == gt3 == gt4 == gt5


def test_mutation2():
    # Grammar
    bnf_text = """
    <S> ::= <A><B>
    <A> ::= <A1> | <A2> | d
    <A1> ::= a | b
    <A2> ::= c
    <B> ::= <B1> | <B2> | /
    <B1> ::= + | <B1a> | <B1b>
    <B1a> ::= -
    <B1b> ::= <B1c> | /
    <B1c> ::= +
    <B2> ::= *
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Language
    language = grammar.generate_language()
    assert len(language) == 4 * 4

    # Genotypes of four types:
    # 1) tuple of tuples of int 2) string thereof 3) list of lists of int, 4) string thereof
    genotypes = (
        ((0,), (0,), (0,), (), (0,), (0,), (), (), (), ()),
        "((0,),(0,),(1,),(),(0,),(1,),(0,),(),(),())",
        [[0], [1], [], [0], [1], [], [], [], [], [0]],
        "[[0], [2], [], [], [2], [], [], [], [], []]",
    )

    # Mutation
    methods = (
        al.systems.dsge.mutation.int_replacement_by_probability,
        al.systems.dsge.mutation.int_replacement_by_count,
    )
    params = dict(
        repair_after_mutation=True,
        mutation_int_replacement_probability=0.15,
        mutation_int_replacement_count=1,
    )
    for method in methods:
        for gt in genotypes:
            strings = set()
            for _ in range(3000):
                gt_mut1 = method(grammar, gt, params)
                gt_mut = method(
                    grammar, gt_mut1, params
                )  # second mutation to reach everything
                assert isinstance(gt_mut, al.systems.dsge.representation.Genotype)
                string = al.systems.dsge.mapping.forward(grammar, gt_mut)
                assert isinstance(string, str)
                strings.add(string)
            # This test can fail with a very low probability
            assert len(strings) == len(language)


def test_mutation_count():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 1 | 0
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Mutation
    genotypes = (
        ((0,), (0,), (0, 0, 0, 0, 0, 0, 0, 0)),
        ((1,), (0, 0), (0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0)),
        (
            (2,),
            (0, 0, 0),
            (1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0),
        ),
        "((0,), (0,), (0, 0, 0, 0, 0, 0, 0, 0))",
        [[0], [0], [0, 0, 0, 0, 0, 0, 0, 0]],
        "[[0], [0], [0, 0, 0, 0, 0, 0, 0, 0]]",
        al.systems.dsge.representation.Genotype(
            [[0], [0], [0, 0, 0, 0, 0, 0, 0, 0]],
        ),
    )
    for _ in range(50):
        for genotype in genotypes:
            # Parameter: mutation_int_replacement_count
            for mutation_int_replacement_count in range(5):
                parameters = dict(
                    mutation_int_replacement_count=mutation_int_replacement_count
                )
                gt_copy = copy.deepcopy(genotype)
                gt_mut = al.systems.dsge.mutation.int_replacement_by_count(
                    grammar, gt_copy, parameters
                )
                # Check expected number of int flips for different cases
                num_changed_codons = 0
                if isinstance(genotype, al.systems.dsge.representation.Genotype):
                    gt_ori = copy.deepcopy(genotype)
                else:
                    gt_ori = al.systems.dsge.representation.Genotype(
                        copy.deepcopy(genotype)
                    )
                for g1, g2 in zip(gt_ori.data, gt_mut.data):
                    for codon1, codon2 in zip(g1, g2):
                        if codon1 != codon2:
                            num_changed_codons += 1
                assert num_changed_codons == mutation_int_replacement_count


# Crossover


def test_crossover_api():
    # Grammar
    bnf_text = """
    <S> ::= 1<A> | 2<A>
    <A> ::= 3<B><C> | 4<C><B>
    <B> ::= 5 | 6 | 7
    <C> ::= 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types: 1) list of int, 2) string thereof, 3) Genotype class
    genotypes = (
        [[0], [0], [0], [0]],
        "[[1], [1], [2], [1]]",
        al.systems.dsge.representation.Genotype([[0], [1], [0], [1]]),
    )

    # Crossover
    methods = (al.systems.dsge.crossover.gene_swap,)

    def perform_checks(gt1, gt2, gt3, gt4):
        if not isinstance(gt1, al.systems.dsge.representation.Genotype):
            gt1 = al.systems.dsge.representation.Genotype(copy.copy(gt1))
        if not isinstance(gt2, al.systems.dsge.representation.Genotype):
            gt2 = al.systems.dsge.representation.Genotype(copy.copy(gt2))
        assert gt1 == gt1
        assert gt2 == gt2
        assert gt1 != gt2
        assert (
            (gt1 != gt3 and gt1 != gt4 and gt2 != gt3 and gt2 != gt4)
            or (gt1 == gt3 and gt1 != gt4 and gt2 != gt3 and gt2 == gt4)
            or (gt1 != gt3 and gt1 == gt4 and gt2 == gt3 and gt2 != gt4)
        )
        assert len(gt1) == len(gt2) == len(gt3) == len(gt4)

    params = dict()
    for _ in range(200):
        for two_genotypes in itertools.combinations(genotypes, 2):
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


def test_crossover_parameter_repair_after_crossover():
    # Grammar
    bnf_text = """
    <S> ::= 0 <A> | 1 <A> <A>
    <A> ::= a | b
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes
    dt1 = grammar.parse_string("0a")
    dt2 = grammar.parse_string("1ab")
    gt1 = al.systems.dsge.mapping.reverse(grammar, dt1)
    gt2 = al.systems.dsge.mapping.reverse(grammar, dt2)

    # Crossover
    # - with repair
    parameters = dict(repair_after_crossover=True)
    for _ in range(1000):
        gt3, gt4 = al.systems.dsge.crossover.gene_swap(grammar, gt1, gt2, parameters)
        phe3 = al.systems.dsge.mapping.forward(grammar, gt3)
        phe4 = al.systems.dsge.mapping.forward(grammar, gt4)
        assert isinstance(phe3, str)
        assert isinstance(phe4, str)

    # - without repiar
    parameters = dict(repair_after_crossover=False)
    with pytest.raises(al.exceptions.MappingError):
        for _ in range(1000):
            gt3, gt4 = al.systems.dsge.crossover.gene_swap(
                grammar, gt1, gt2, parameters
            )
            phe3 = al.systems.dsge.mapping.forward(grammar, gt3)
            phe4 = al.systems.dsge.mapping.forward(grammar, gt4)
            assert isinstance(phe3, str)
            assert isinstance(phe4, str)


def test_crossover_fails():
    # Grammar
    bnf_text = "<bit> ::= 1 | 0"
    grammar = al.Grammar(bnf_text=bnf_text)

    # Crossover
    # - invalid genotype types
    gt_valid = [[0]]
    for gt_invalid in [None, False, True, "", (), [], 0, 1, 3.14, "101"]:
        al.systems.dsge.crossover.gene_swap(grammar, gt_valid, gt_valid)
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.dsge.crossover.gene_swap(grammar, gt_valid, gt_invalid)
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.dsge.crossover.gene_swap(grammar, gt_invalid, gt_valid)
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.dsge.crossover.gene_swap(grammar, gt_invalid, gt_invalid)


# Repair


def test_repair():
    # Grammars
    bnf_text = """
    <S> ::= <A><B> | <B><C> | <C><A> | <A><B><C> | <C><B><A>
    <A> ::= <AA>
    <AA> ::= 1 2 | 2 3 | 3 4 5 | 4 5 6 7 | <AA> 8 <AA>
    <B> ::= a | b | c
    <C> ::= <CC> | <CCC>
    <CC> ::= X | Y
    <CCC> ::= Z
    """
    gr1 = al.Grammar(bnf_text=bnf_text)

    bnf_text = """
    <expr> ::= (<expr><op><expr>) | <pre-op>(<expr>) | <var>
    <op> ::= + | - | * | / | **
    <pre-op> ::= sin | cos | exp | log | sqrt
    <var> ::= x | y | <number>
    <number> ::= <digit> . <digit>
    <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    gr2 = al.Grammar(bnf_text=bnf_text)

    for gr in [gr1, gr2]:
        num_nt = len(gr.nonterminal_symbols)
        for _ in range(5):
            # Random invalid genotype
            gt1 = [
                [random.randint(0, 100) for _ in range(random.randint(0, 100))]
                for _ in range(num_nt)
            ]

            # Mapping fails
            with pytest.raises(al.exceptions.MappingError):
                al.systems.dsge.mapping.forward(gr, gt1)

            # Repair
            gt2 = al.systems.dsge.repair.fix_genotype(gr, gt1)

            params = dict(repair_with_random_choices=True)
            gt3 = al.systems.dsge.repair.fix_genotype(gr, gt1, params)

            params = dict(repair_with_random_choices=False)
            gt4 = al.systems.dsge.repair.fix_genotype(gr, gt1, params)

            # Mapping works
            al.systems.dsge.mapping.forward(gr, gt2)
            al.systems.dsge.mapping.forward(gr, gt3)
            al.systems.dsge.mapping.forward(gr, gt4)


def test_repair_fail1():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <bytes><byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotype that can always be repaired when more than 9 expansions are allowed
    gt = [[0], [0, 0], [1, 1, 1, 1, 0, 0, 0]]
    al.systems.dsge.repair.fix_genotype(grammar, gt)
    for re in (True, False):
        for me in range(20):
            params = dict(max_expansions=me)
            if re is True and me < 10:
                with pytest.raises(al.exceptions.GenotypeError):
                    al.systems.dsge.repair.fix_genotype(
                        grammar, gt, params, raise_errors=re
                    )
            else:
                al.systems.dsge.repair.fix_genotype(
                    grammar, gt, params, raise_errors=re
                )

    # Genotype that can be repaired with different number of expansions, depending on random
    # choice of first production rule
    gt = [[], [0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]
    cnt_success, cnt_fail = 0, 0
    for _ in range(100):
        try:
            al.systems.dsge.repair.fix_genotype(grammar, gt, dict(max_expansions=10))
            cnt_success += 1
        except al.exceptions.GenotypeError:
            cnt_fail += 1
    assert cnt_success > 10
    assert cnt_fail > 10


def test_repair_fail2():
    bnf_text = """
    <expr> ::= (<expr><op><expr>) | <pre-op>(<expr>) | <var>
    <op> ::= + | - | * | / | **
    <pre-op> ::= sin | cos | exp | log | sqrt
    <var> ::= x | y | <number>
    <number> ::= <digit> . <digit>
    <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    for _ in range(20):
        # Random derivation tree
        dt = grammar.generate_derivation_tree()
        string1 = dt.string()
        # Genotype
        gt = al.systems.dsge.mapping.reverse(grammar, dt)
        # Mapping works
        string2 = al.systems.dsge.mapping.forward(grammar, gt)
        assert string1 == string2
        # Genotype broken
        while True:
            gene_idx = random.randint(0, len(gt) - 1)
            try:
                data = list(list(gene) for gene in gt.data)
                data[gene_idx].pop()  # remove last codon
                gt_defect = al.systems.dsge.representation.Genotype(data)
                break
            except IndexError:
                pass
        # Mapping fails
        with pytest.raises(al.exceptions.MappingError):
            al.systems.dsge.mapping.forward(grammar, gt_defect)
        # Genotype repaired
        gt_repaired = al.systems.dsge.repair.fix_genotype(grammar, gt_defect)
        # Mapping works again
        al.systems.dsge.mapping.forward(grammar, gt_repaired)


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

    # Genotypes of three types: 1) list of list of int, 2) string thereof, 3) Genotype class
    genotypes = (
        [[0], [0], [2], [], [], [], []],
        "[[0], [0], [2], [], [], [], []]",
        al.systems.dsge.representation.Genotype([[0], [0], [2], [], [], [], []]),
    )

    # Neighborhood
    for gt in genotypes:
        phe = al.systems.dsge.mapping.forward(gr, gt)
        # Default
        nh1 = al.systems.dsge.neighborhood.int_replacement(gr, gt)
        nh2 = al.systems.dsge.neighborhood.int_replacement(gr, genotype=gt)
        nh3 = al.systems.dsge.neighborhood.int_replacement(grammar=gr, genotype=gt)
        nh4 = al.systems.dsge.neighborhood.int_replacement(gr, gt, dict())
        nh5 = al.systems.dsge.neighborhood.int_replacement(gr, gt, parameters=dict())
        nh6 = al.systems.dsge.neighborhood.int_replacement(
            gr, genotype=gt, parameters=dict()
        )
        nh7 = al.systems.dsge.neighborhood.int_replacement(
            grammar=gr, genotype=gt, parameters=dict()
        )
        assert nh1 == nh2 == nh3 == nh4 == nh5 == nh6 == nh7
        for new_gt in nh1:
            check_genotype(new_gt)
            print(new_gt)
            new_phe = al.systems.dsge.mapping.forward(gr, new_gt)
            assert new_phe != phe


@pytest.mark.parametrize(
    "bnf, genotype, phenotype",
    [
        (shared.BNF5, ((0,),), "1"),
        (shared.BNF5, ((1,),), "2"),
        (shared.BNF5, ((2,),), "3"),
        (shared.BNF5, ((3,),), "4"),
        (shared.BNF5, ((4,),), "5"),
        (shared.BNF6, ((0,), (0,), ()), "1"),
        (shared.BNF6, ((0,), (1,), ()), "2"),
        (shared.BNF6, ((1,), (), (0,)), "3"),
        (shared.BNF6, ((1,), (), (1,)), "4"),
        (shared.BNF6, ((2,), (), ()), "5"),
        (shared.BNF7, ((0,), (0,), (), (0,), (), (), ()), "ac1"),
        (shared.BNF7, ((1,), (), (1,), (), (), (), (1,)), "bf8"),
        (shared.BNF7, ((0,), (1,), (), (), (1,), (), ()), "ad4"),
        (shared.BNF9, ((0,), (), (0,), (), (0,)), "a"),
        (shared.BNF9, ((0,), (), (1,), (), (1, 2)), "bc"),
        (shared.BNF9, ((1,), (1,), (), (1, 1), ()), "22"),
        (shared.BNF9, ((1,), (0,), (), (2,), ()), "3"),
        (shared.BNF9, ((1,), (1,), (), (0, 2), ()), "13"),
    ],
)
def test_neighborhood_reachability_in_finite_languages(bnf, genotype, phenotype):
    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.dsge.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.dsge.mapping.forward(grammar, gt)
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
                nbrs = al.systems.dsge.neighborhood.int_replacement(grammar, gen, param)
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                try:
                    phe = al.systems.dsge.mapping.forward(grammar, gen, param)
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
        (shared.BNF10, ((1,), (0,), (), ()), "1x", ("2x", "3x", "4y", "5y", "6y", "7")),
        (shared.BNF11, ((0,), (1,), (), ()), "1", ("2", "3", "4", "22", "33", "44")),
    ],
)
def test_neighborhood_reachability_in_infinite_languages(
    bnf, genotype, phenotype, strings_given
):
    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.dsge.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.dsge.mapping.forward(grammar, gt)
    assert phe == phenotype

    # Neighborhood
    strings_given = set(strings_given)
    params = [
        dict(max_depth=3),
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
                nbrs = al.systems.dsge.neighborhood.int_replacement(grammar, gen, param)
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                # Genotype management
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                # Phenotype generation
                phe = al.systems.dsge.mapping.forward(grammar, gen, param)
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
    dt = grammar.parse_string("1a1a")
    gt = al.systems.dsge.mapping.reverse(grammar, dt)

    # Neighborhood in different distances when changing only terminals
    # - distance 1
    parameters = dict(neighborhood_only_terminals=True)
    nbrs_gt = al.systems.dsge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.dsge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1a1b", "1a2a", "1b1a", "2a1a"}

    # - distance 2
    parameters = dict(neighborhood_distance=2, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.dsge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.dsge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1a2b", "1b1b", "1b2a", "2a1b", "2a2a", "2b1a"}

    # - distance 3
    parameters = dict(neighborhood_distance=3, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.dsge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.dsge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1b2b", "2a2b", "2b1b", "2b2a"}

    # - distance 4
    parameters = dict(neighborhood_distance=4, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.dsge.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.dsge.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"2b2b"}

    # - distance 5 and greater
    for dist in range(5, 20):
        parameters = dict(neighborhood_distance=dist, neighborhood_only_terminals=True)
        nbrs_gt = al.systems.dsge.neighborhood.int_replacement(grammar, gt, parameters)
        nbrs = [al.systems.dsge.mapping.forward(grammar, gt) for gt in nbrs_gt]
        assert nbrs == []


@pytest.mark.parametrize(
    "bnf, gt, phe, phe_neighbors",
    [
        (shared.BNF1, [[0]], "0", ("1", "2")),
        (shared.BNF1, [[1]], "1", ("0", "2")),
        (shared.BNF1, [[2]], "2", ("0", "1")),
        (shared.BNF2, [[0], [0], [0]], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF2, [[0], [1], [1]], "1b", ("0b", "2b", "1a", "1c")),
        (shared.BNF2, [[0], [2], [2]], "2c", ("0c", "1c", "2a", "2b")),
        (shared.BNF2, [[0], [0], [1]], "0b", ("1b", "2b", "0a", "0c")),
        (shared.BNF2, [[0], [1], [2]], "1c", ("0c", "2c", "1a", "1b")),
        (
            shared.BNF3,
            [[0], [0], [0], [0], [0], [0], [0]],
            "0a",
            ("1a", "2a", "0b", "0c"),
        ),
        (
            shared.BNF3,
            [[0], [0], [1], [0], [0], [0], [1]],
            "1b",
            ("0b", "2b", "1a", "1c"),
        ),
        (
            shared.BNF3,
            [[0], [0], [2], [0], [0], [0], [2]],
            "2c",
            ("0c", "1c", "2a", "2b"),
        ),
        (
            shared.BNF3,
            [[0], [0], [0], [0], [0], [0], [1]],
            "0b",
            ("1b", "2b", "0a", "0c"),
        ),
        (
            shared.BNF3,
            [[0], [0], [1], [0], [0], [0], [2]],
            "1c",
            ("0c", "2c", "1a", "1b"),
        ),
        (
            shared.BNF4,
            [[0], [0, 0, 0, 0, 0, 0, 0, 0]],
            "00000000",
            (
                "10000000",
                "01000000",
                "00100000",
                "00010000",
                "00001000",
                "00000100",
                "00000010",
                "00000001",
            ),
        ),
        (
            shared.BNF4,
            [[0], [1, 1, 1, 1, 1, 1, 1, 1]],
            "11111111",
            (
                "01111111",
                "10111111",
                "11011111",
                "11101111",
                "11110111",
                "11111011",
                "11111101",
                "11111110",
            ),
        ),
        (
            shared.BNF4,
            [[0], [0, 1, 0, 1, 0, 1, 0, 1]],
            "01010101",
            (
                "11010101",
                "00010101",
                "01110101",
                "01000101",
                "01011101",
                "01010001",
                "01010111",
                "01010100",
            ),
        ),
        (
            shared.BNF4,
            [[0], [0, 1, 1, 0, 1, 1, 0, 1]],
            "01101101",
            (
                "11101101",
                "00101101",
                "01001101",
                "01111101",
                "01100101",
                "01101001",
                "01101111",
                "01101100",
            ),
        ),
    ],
)
def test_neighborhood_parameter_max_size(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    assert phe == al.systems.dsge.mapping.forward(gr, gt)

    # Neighborhood
    nbrs = al.systems.dsge.neighborhood.int_replacement(gr, gt)
    nbrs_phe = [al.systems.dsge.mapping.forward(gr, nbr_gt) for nbr_gt in nbrs]
    assert set(nbrs_phe) == set(phe_neighbors)

    # Parameter: neighborhood_max_size
    parameters = dict(neighborhood_max_size=None)
    nbrs = al.systems.dsge.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [al.systems.dsge.mapping.forward(gr, nbr_gt) for nbr_gt in nbrs]
    assert set(nbrs_phe) == set(phe_neighbors)

    for max_size in range(1, 5):
        parameters = dict(neighborhood_max_size=max_size)
        nbrs_phe = set()
        for _ in range(100):
            nbrs = al.systems.dsge.neighborhood.int_replacement(gr, gt, parameters)
            assert len(nbrs) <= max_size
            for nbr_gt in nbrs:
                nbr_phe = al.systems.dsge.mapping.forward(gr, nbr_gt)
                assert nbr_phe in phe_neighbors
                nbrs_phe.add(nbr_phe)
        assert nbrs_phe == set(phe_neighbors)


@pytest.mark.parametrize(
    "bnf, gt, phe, phe_neighbors",
    [
        (shared.BNF5, ((0,),), "1", ("2", "3", "4", "5")),
        (shared.BNF5, ((1,),), "2", ("1", "3", "4", "5")),
        (shared.BNF5, ((2,),), "3", ("1", "2", "4", "5")),
        (shared.BNF5, ((3,),), "4", ("1", "2", "3", "5")),
        (shared.BNF5, ((4,),), "5", ("1", "2", "3", "4")),
        (shared.BNF6, ((0,), (0,), ()), "1", ("2", "5")),
        (shared.BNF6, ((0,), (1,), ()), "2", ("1", "5")),
        (shared.BNF6, ((1,), (), (0,)), "3", ("4", "5")),
        (shared.BNF6, ((1,), (), (1,)), "4", ("3", "5")),
        (shared.BNF6, ((2,), (), ()), "5", ()),
        (shared.BNF7, ((0,), (0,), (), (0,), (), (), ()), "ac1", ("be5", "ad3", "ac2")),
        (shared.BNF7, ((1,), (), (1,), (), (), (), (1,)), "bf8", ("ac1", "be5", "bf7")),
        (shared.BNF7, ((0,), (1,), (), (), (1,), (), ()), "ad4", ("be5", "ac1", "ad3")),
        (shared.BNF8, ((1,), (1,), ()), "t", ("t0g", "1g", "a")),
        (shared.BNF8, ((2,), (0,), (1,)), "a0c", ("1c", "t0c", "a0g")),
    ],
)
def test_neighborhood_parameter_only_terminals(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    parameters = dict(neighborhood_only_terminals=True)
    assert phe == al.systems.dsge.mapping.forward(
        gr, gt, parameters, raise_errors=False
    )

    # Neighborhood
    nbrs = al.systems.dsge.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [
        al.systems.dsge.mapping.forward(gr, nbr_gt, parameters, raise_errors=False)
        for nbr_gt in nbrs
    ]
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

    # Genotypic data of four types:
    # 1) tuple of tuples of int 2) string thereof 3) list of lists of int 4) string thereof
    # 5) Genotype object
    tup = ((1,), (0, 0), (1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1))
    data_variants = (
        tup,
        str(tup),
        list(tup),
        str(list(tup)),
        al.systems.dsge.representation.Genotype(tup),
    )

    # Forward mapping
    parameters = dict(
        max_expansions=3,
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
            string1 = grammar.generate_string("dsge", data, parameters, **kwargs)
            string2 = grammar.generate_string(
                "dsge", data, parameters=parameters, **kwargs
            )
            string3 = grammar.generate_string(
                method="dsge", genotype=data, parameters=parameters, **kwargs
            )
            assert string1
            assert string1 == string2 == string3

            # Method of DerivationTree class
            dt1 = grammar.generate_derivation_tree("dsge", data, parameters, **kwargs)
            dt2 = grammar.generate_derivation_tree(
                "dsge", data, parameters=parameters, **kwargs
            )
            dt3 = grammar.generate_derivation_tree(
                method="dsge", genotype=data, parameters=parameters, **kwargs
            )
            assert string1 == dt1.string() == dt2.string() == dt3.string()

            # Functions in mapping module
            string4 = al.systems.dsge.mapping.forward(
                grammar, data, parameters, **kwargs
            )
            string5 = al.systems.dsge.mapping.forward(
                grammar, data, parameters=parameters, **kwargs
            )
            string6 = al.systems.dsge.mapping.forward(
                grammar=grammar, genotype=data, parameters=parameters, **kwargs
            )
            assert string1 == string4 == string5 == string6

            kwargs["return_derivation_tree"] = True
            phe, dt4 = al.systems.dsge.mapping.forward(
                grammar, data, parameters, **kwargs
            )
            phe, dt5 = al.systems.dsge.mapping.forward(
                grammar, data, parameters=parameters, **kwargs
            )
            phe, dt6 = al.systems.dsge.mapping.forward(
                grammar=grammar, genotype=data, parameters=parameters, **kwargs
            )
            kwargs["return_derivation_tree"] = False
            assert string1 == dt4.string() == dt5.string() == dt6.string()

            # Same with errors when reaching expansion limit
            kwargs["raise_errors"] = True
            with pytest.raises(al.exceptions.MappingError):
                grammar.generate_string(
                    method="dsge", genotype=data, parameters=parameters, **kwargs
                )
            with pytest.raises(al.exceptions.MappingError):
                grammar.generate_derivation_tree(
                    method="dsge", genotype=data, parameters=parameters, **kwargs
                )
            with pytest.raises(al.exceptions.MappingError):
                al.systems.dsge.mapping.forward(
                    grammar, genotype=data, parameters=parameters, **kwargs
                )
            with pytest.raises(al.exceptions.MappingError):
                al.systems.dsge.mapping.forward(
                    grammar, genotype=data, parameters=parameters, **kwargs
                )
            kwargs["raise_errors"] = False


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
        max_expansions=None,
    )
    # Functions in mapping module
    p1 = dict()
    p2 = dict(max_depth=5)
    p3 = dict(repair_with_random_choices=False)
    p4 = dict(max_depth=1, repair_with_random_choices=False)
    for parameters in (p1, p2, p3, p4):
        for dt, string in zip(random_dts, random_strings):
            gt1 = al.systems.dsge.mapping.reverse(grammar, string)
            gt2 = al.systems.dsge.mapping.reverse(grammar, dt)
            gt3 = al.systems.dsge.mapping.reverse(grammar, string, parameters)
            gt4 = al.systems.dsge.mapping.reverse(grammar, string, parameters, False)
            gt5, dt5 = al.systems.dsge.mapping.reverse(
                grammar, string, parameters, True
            )
            gt6 = al.systems.dsge.mapping.reverse(
                grammar, phenotype_or_derivation_tree=string
            )
            gt7 = al.systems.dsge.mapping.reverse(
                grammar, phenotype_or_derivation_tree=dt
            )
            gt8 = al.systems.dsge.mapping.reverse(
                grammar, phenotype_or_derivation_tree=string, parameters=parameters
            )
            gt9 = al.systems.dsge.mapping.reverse(
                grammar, phenotype_or_derivation_tree=dt, parameters=parameters
            )
            gt10 = al.systems.dsge.mapping.reverse(
                grammar,
                phenotype_or_derivation_tree=string,
                parameters=parameters,
                return_derivation_tree=False,
            )
            gt11, dt11 = al.systems.dsge.mapping.reverse(
                grammar,
                phenotype_or_derivation_tree=dt,
                parameters=parameters,
                return_derivation_tree=True,
            )
            for gt in (gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10, gt11):
                # Check if reverse mapping resulted in a valid genotype
                check_genotype(gt)
                # Check if genotype allows to reproduce the original string via forward mapping
                string_from_fwd_map = grammar.generate_string("dsge", gt)
                assert string_from_fwd_map == string


def test_mapping_errors():
    bnf_text = "<S> ::= <S><S> | 1 | 2 | 3"
    grammar = al.Grammar(bnf_text=bnf_text)
    # Invalid input: a string that is not part of the grammar's language
    string = "4"
    with pytest.raises(al.exceptions.MappingError):
        al.systems.dsge.mapping.reverse(grammar, string)
    # Invalid input: a derivation tree with an unknown nonterminal
    dt = grammar.generate_derivation_tree()
    dt.root_node.symbol = al._grammar.data_structures.NonterminalSymbol("nonsense")
    with pytest.raises(al.exceptions.MappingError):
        al.systems.dsge.mapping.reverse(grammar, dt)
    # Invalid input: a derivation tree with an unknown derivation (no corresponding rule)
    dt = grammar.generate_derivation_tree()
    dt.leaf_nodes()[0].symbol = al._grammar.data_structures.TerminalSymbol("nonsense")
    with pytest.raises(al.exceptions.MappingError):
        al.systems.dsge.mapping.reverse(grammar, dt)


def test_mapping_error_messages():
    bnf_text = """
    <start> ::= <float>
    <float> ::= <first>.<second>
    <first> ::= 0 | 1 | 2
    <second> ::= <digit><second> | <digit>
    <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    for verbose in (True, False):
        # ok
        genotype = [[0], [0], [1], [0, 0, 1], [2, 5, 9]]
        phenotype = al.systems.dsge.mapping.forward(grammar, genotype)
        assert phenotype == "1.259"

        # error 1: missing gene
        genotype = [[0], [0], [1], [0, 0, 1]]
        phenotype_unfinished = al.systems.dsge.mapping.forward(
            grammar, genotype, verbose=verbose, raise_errors=False
        )
        assert "<" in phenotype_unfinished and ">" in phenotype_unfinished
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.MappingError,
            "The provided DSGE genotype is invalid, because the length of the genotype (4) "
            "does not fit to the number of nonterminal symbols (5).",
        )
        genotype = [[0]]
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.MappingError,
            "The provided DSGE genotype is invalid, because the length of the genotype (1) "
            "does not fit to the number of nonterminal symbols (5).",
        )
        genotype = [[0], [0], [1], [0, 0], [2, 5, 9], [0]]
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.MappingError,
            "The provided DSGE genotype is invalid, because the length of the genotype (6) "
            "does not fit to the number of nonterminal symbols (5).",
        )
        genotype = []
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.GenotypeError,
            "The given data could not be interpreted as a DSGE genotype. "
            "It needs to be a non-empty tuple of tuples of integers. "
            "Given data: ()",
        )

        # error 2: missing integer
        genotype = [[0], [0], [1], [0, 0], [2, 5, 9]]
        phenotype_unfinished = al.systems.dsge.mapping.forward(
            grammar, genotype, verbose=verbose, raise_errors=False
        )
        assert "<" in phenotype_unfinished and ">" in phenotype_unfinished
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.MappingError,
            "The provided DSGE genotype is invalid, because gene number 3 "
            "(which corresponds to nonterminal symbol <second>) does not "
            "contain enough integers to complete the mapping.",
        )
        genotype = [[0], [0], [1], [0, 0, 1], [2, 5]]
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.MappingError,
            "The provided DSGE genotype is invalid, because gene number 4 "
            "(which corresponds to nonterminal symbol <digit>) does not "
            "contain enough integers to complete the mapping.",
        )

        # error 3: invalid integer
        genotype = [[0], [0], [1], [0, 0], [2, 51, 9]]
        phenotype_unfinished = al.systems.dsge.mapping.forward(
            grammar, genotype, verbose=verbose, raise_errors=False
        )
        assert "<" in phenotype_unfinished and ">" in phenotype_unfinished
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.MappingError,
            "The provided DSGE genotype is invalid, because gene number 4 "
            "(which corresponds to nonterminal symbol <digit>) contains "
            "the integer 51 that can not be used to select a production "
            "out of 10 available ones.",
        )
        genotype = [[0], [0], [1], [0, 13], [2, 5, 9]]
        shared.emits_exception(
            lambda: al.systems.dsge.mapping.forward(
                grammar, genotype, verbose=verbose  # noqa: B023
            ),
            al.exceptions.MappingError,
            "The provided DSGE genotype is invalid, because gene number 3 "
            "(which corresponds to nonterminal symbol <second>) contains "
            "the integer 13 that can not be used to select a production "
            "out of 2 available ones.",
        )


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
        gt = [[0], [0], [0], [0], [0]]
        string = grammar.generate_string("dsge", gt, verbose=vb)
        assert string == "0x+("
        # Parameter: max_expansions
        for me in range(20):
            params = dict(max_wraps=None, max_expansions=me)
            if me < 5:
                with pytest.raises(al.exceptions.MappingError):
                    grammar.generate_string("dsge", gt, params, verbose=vb)
                sentential_form = grammar.generate_string(
                    "dsge", gt, params, verbose=vb, raise_errors=False
                )
                assert "<" in sentential_form
                assert ">" in sentential_form
            else:
                string = grammar.generate_string("dsge", gt, params, verbose=vb)
                assert "<" not in string
                assert ">" not in string


def test_mapping_forward_by_hand():
    bnf_text = """
    <A> ::= <B><C><D>
    <B> ::= 7 | 8 | 9
    <C> ::= x | y
    <D> ::= R | S
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_genotype_phenotype_map = [
        ([[0], [0], [0], [0]], "7xR"),
        ([[0], [0], [0], [1]], "7xS"),
        ([[0], [0], [1], [0]], "7yR"),
        ([[0], [0], [1], [1]], "7yS"),
        ([[0], [1], [0], [0]], "8xR"),
        ([[0], [1], [0], [1]], "8xS"),
        ([[0], [1], [1], [0]], "8yR"),
        ([[0], [1], [1], [1]], "8yS"),
        ([[0], [2], [0], [0]], "9xR"),
        ([[0], [2], [0], [1]], "9xS"),
        ([[0], [2], [1], [0]], "9yR"),
        ([[0], [2], [1], [1]], "9yS"),
    ]
    for genotype, expected_phenotype in expected_genotype_phenotype_map:
        phenotype = grammar.generate_string(method="dsge", genotype=genotype)
        assert phenotype == expected_phenotype
        grammar.generate_derivation_tree(method="dsge", genotype=genotype)


def test_mapping_forward_and_reverse_automated():
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | - | * | /
    <v> ::= <d> | <v><d>
    <d> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for _ in range(100):
        string1 = grammar.generate_string()
        # Reverse map: string1 -> genotype
        genotype = al.systems.dsge.mapping.reverse(grammar, string1)
        # Forward map: genotype -> string2
        string2 = al.systems.dsge.mapping.forward(grammar, genotype)
        assert string1 == string2


def test_mapping_forward_against_paper_2017_example():
    # References
    # - https://doi.org/10.1145/3071178.3071286 p. 359
    bnf_text = """
    <start> ::= <float>
    <float> ::= <first>.<second>
    <first> ::= 0 | 1 | 2
    <second> ::= <digit><second> | <digit>
    <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    # Genotype 1
    genotype = [[0], [0], [1], [0, 0, 1], [2, 5, 9]]
    phenotype = grammar.generate_string(method="dsge", genotype=genotype)
    assert phenotype == "1.259"
    # Genotype 2 (after mutating a single integer)
    genotype = [[0], [0], [2], [0, 0, 1], [2, 5, 9]]
    phenotype = grammar.generate_string(method="dsge", genotype=genotype)
    assert phenotype == "2.259"


def test_mapping_forward_against_python_reference_implementation():
    # Caution: The reference implementation reads terminals in such a
    # way that empty spaces are added if it is surrounded by nonterminals.
    # For this reason, only grammars were used that do not contain such
    # whitespaces in order to avoid confusions related to reading BNF.

    # Caution: The reference implementation contains post-processing of
    # phenotypes for some of the grammars shipped with it, e.g. {:
    # and :} are used to mark indents of Python code and there are
    # protected mathematical operations.
    def nt_to_str(sym):
        return "<{}>".format(sym)

    directory = os.path.join(IN_DIR, "mappings", "dsge_reduced")
    filepaths = [
        os.path.join(directory, filename) for filename in os.listdir(directory)
    ]
    assert len(filepaths) == 20
    for filepath in sorted(filepaths):
        print(filepath)
        # Read data from JSON file
        with open(filepath) as file_handle:
            data = json.load(file_handle)
        bnf_text = data["grammar"]["bnf"]
        start_symbol = data["grammar"]["start_symbol"]
        nonterminals = data["grammar"]["nonterminals"]
        terminals = data["grammar"]["terminals"]
        gen_phe_map = data["genotype_to_phenotype_mappings"]
        # Create grammar
        grammar = al.Grammar(bnf_text=bnf_text)
        assert nt_to_str(grammar.start_symbol) == start_symbol
        assert list(nt_to_str(nt) for nt in grammar.nonterminal_symbols) == nonterminals
        assert set(str(ts) for ts in grammar.terminal_symbols) == set(terminals)
        # Check if each genotype is mapped to the same phenotype as in
        # reference implementation. Also, check reverse mapping to see
        # if the same genotype is reproduced (from tree!)
        for gen, phe in gen_phe_map.items():
            genotype = eval(gen)

            # Default implementation (fast)
            param = dict(max_expansions=100000)
            phe_calc_fast, dt = al.systems.dsge.mapping.forward(
                grammar, genotype, return_derivation_tree=True, parameters=param
            )
            assert phe == phe_calc_fast

            # Fast implementation
            _, nt_to_gene, nt_to_cnt, _, _ = grammar._lookup_or_calc(
                "dsge", "maps", al.systems.dsge._cached_calculations.maps, grammar
            )
            nt_to_cnt = nt_to_cnt.copy()
            dt = al.systems.dsge.mapping._forward_fast(
                grammar, genotype, None, nt_to_gene, nt_to_cnt, True
            )
            phe_calc_slow = dt.string()
            assert phe == phe_calc_slow

            # Slow implementation
            _, nt_to_gene, nt_to_cnt, _, _ = grammar._lookup_or_calc(
                "dsge", "maps", al.systems.dsge._cached_calculations.maps, grammar
            )
            nt_to_cnt = nt_to_cnt.copy()
            dt = al.systems.dsge.mapping._forward_slow(
                grammar, genotype, None, nt_to_gene, nt_to_cnt, True, False
            )
            phe_calc_slow = dt.string()
            assert phe == phe_calc_slow

            # Reverse
            genotype_rev = al.systems.dsge.mapping.reverse(grammar, dt)
            genotype_ori = al.systems.dsge.representation.Genotype(genotype)
            assert genotype_ori == genotype_rev
            assert genotype_ori.data == genotype_rev.data
