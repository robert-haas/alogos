import copy
import itertools
import json
import math
import os

import pytest
import shared

import alogos as al


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, "in")


# Shared


def check_genotype(gt):
    assert isinstance(gt, al.systems.pige.representation.Genotype)
    assert isinstance(gt.data, tuple)
    assert all(isinstance(codon, int) for codon in gt.data)
    assert len(gt) == len(gt.data) > 0


def check_phenotype(phenotype):
    if phenotype is not None:
        assert isinstance(phenotype, str)
        assert len(phenotype) > 0


def check_fitness(fitness):
    assert isinstance(fitness, float)


def check_individual(ind):
    assert isinstance(ind, al.systems.pige.representation.Individual)
    check_genotype(ind.genotype)
    check_phenotype(ind.phenotype)
    check_fitness(ind.fitness)


def check_population(pop):
    assert isinstance(pop, al.systems.pige.representation.Population)
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
        "(0,)",
        "(42,)",
        "(0, 8, 15)",
        [0],
        [42],
        [0, 8, 15],
        "[0]",
        "[42]",
        "[0, 8, 15]",
    )
    for data in data_variants:
        gt = al.systems.pige.representation.Genotype(data)
        check_genotype(gt)
        # Printing
        assert isinstance(str(gt), str)
        assert isinstance(repr(gt), str)
        assert repr(gt).startswith("<piGE genotype at ")
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
        gt = al.systems.pige.representation.Genotype((1, 2, 3, 4, 5, 6))
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
            al.systems.pige.representation.Genotype(data)


def test_representation_individual():
    data_variants = (
        [],
        ["gt"],
        ["gt", "phe"],
        ["gt", "phe", "fit"],
        ["gt", "phe", "fit", "det"],
    )
    for data in data_variants:
        ind = al.systems.pige.representation.Individual(*data)
        # Member types
        assert ind.genotype is None if len(data) < 1 else data[0]
        assert ind.phenotype is None if len(data) < 2 else data[1]
        assert math.isnan(ind.fitness) if len(data) < 3 else data[2]
        assert isinstance(ind.details, dict) if len(data) < 4 else data[3]
        # Printing
        assert isinstance(str(ind), str)
        assert isinstance(repr(ind), str)
        assert str(ind).startswith("piGE individual:")
        assert repr(ind).startswith("<piGE individual object at ")
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
    ind1 = al.systems.pige.representation.Individual(fitness=1)
    ind2 = al.systems.pige.representation.Individual(fitness=2)
    assert ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 2: number and NaN
    ind1 = al.systems.pige.representation.Individual(fitness=1)
    ind2 = al.systems.pige.representation.Individual(fitness=float("nan"))
    assert ind1.less_than(ind2, "min")
    assert not ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert not ind2.greater_than(ind1, "max")
    # - Case 3: NaN and number
    ind1 = al.systems.pige.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.pige.representation.Individual(fitness=2)
    assert not ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert not ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 4: NaN and NaN
    ind1 = al.systems.pige.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.pige.representation.Individual(fitness=float("nan"))
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
        [al.systems.pige.representation.Individual("gt1")],
        [
            al.systems.pige.representation.Individual("gt1"),
            al.systems.pige.representation.Individual("gt2"),
        ],
    )
    for data in data_variants:
        pop = al.systems.pige.representation.Population(data)
        # Member types
        assert isinstance(pop.individuals, list)
        # Printing
        assert isinstance(str(pop), str)
        assert isinstance(repr(pop), str)
        assert str(pop).startswith("piGE population:")
        assert repr(pop).startswith("<piGE population at")
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
            al.systems.pige.representation.Individual("gt3"),
            al.systems.pige.representation.Individual("gt4"),
            al.systems.pige.representation.Individual("gt5"),
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
        assert isinstance(pop2, al.systems.pige.representation.Population)
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
            al.systems.pige.representation.Population(data)


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
        "(0, 1, 2)",
        [0, 1, 2],
        "[0, 1, 2]",
        "[0,1,2]",
    ]
    for gt in valid_genotypes:
        parameters = dict(init_ind_given_genotype=gt)
        ind = al.systems.pige.init_individual.given_genotype(grammar, parameters)
        check_individual(ind)
        assert ind.genotype.data == tuple(eval(str(gt)))
    # Parameter: init_ind_given_genotype not valid
    invalid_genotypes = [None, False, True, (), [], "", "abc", 3, 3.14]
    for gt in invalid_genotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_genotype=gt)
            al.systems.pige.init_individual.given_genotype(grammar, parameters)
    # Parameter: init_ind_given_genotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_individual.given_genotype(grammar)

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
        ind = al.systems.pige.init_individual.given_derivation_tree(grammar, parameters)
        check_individual(ind)
        ind_dt = ind.details["derivation_tree"]
        assert isinstance(ind_dt, al._grammar.data_structures.DerivationTree)
        assert ind_dt == dt
    # Parameter: init_ind_given_derivation_tree not valid
    invalid_derivation_trees = [None, False, True, "", "abc", 3, 3.14, (0, 1, 2)]
    for dt in invalid_derivation_trees:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_derivation_tree=dt)
            al.systems.pige.init_individual.given_derivation_tree(grammar, parameters)
    # Parameter: init_ind_given_derivation_tree not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_individual.given_derivation_tree(grammar)

    # Method: given_phenotype
    valid_phenotypes = ["11110000", "1111000011110000"]
    for phe in valid_phenotypes:
        parameters = dict(init_ind_given_phenotype=phe)
        ind = al.systems.pige.init_individual.given_phenotype(grammar, parameters)
        check_individual(ind)
        assert ind.phenotype == phe
    # Parameter: init_ind_given_phenotype not valid
    invalid_phenotypes = [None, False, True, "", "abc", 3, 3.14, (0, 1, 2)]
    for phe in invalid_phenotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_phenotype=phe)
            al.systems.pige.init_individual.given_phenotype(grammar, parameters)
    # Parameter: init_ind_given_phenotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_individual.given_phenotype(grammar)

    # Method: random_genotype
    for _ in range(shared.NUM_REPETITIONS):
        ind = al.systems.pige.init_individual.random_genotype(grammar)
        check_individual(ind)
    # Parameter: genotype_length
    parameters = dict(genotype_length=21)
    ind = al.systems.pige.init_individual.random_genotype(grammar, parameters)
    check_individual(ind)
    assert len(ind.genotype) == len(ind.genotype.data) == 21
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.pige.init_individual.random_genotype(grammar, parameters)
    # Parameter: codon_size
    parameters = dict(codon_size=1)
    ind = al.systems.pige.init_individual.random_genotype(grammar, parameters)
    check_individual(ind)
    assert all(codon in (0, 1) for codon in ind.genotype.data)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(codon_size=0)
        al.systems.pige.init_individual.random_genotype(grammar, parameters)

    # Method: random_valid_genotype
    for _ in range(shared.NUM_REPETITIONS):
        ind = al.systems.pige.init_individual.random_valid_genotype(grammar)
        check_individual(ind)
    # Parameter: genotype_length
    parameters = dict(genotype_length=21)
    ind = al.systems.pige.init_individual.random_valid_genotype(grammar, parameters)
    check_individual(ind)
    assert len(ind.genotype) == len(ind.genotype.data) == 21
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.pige.init_individual.random_valid_genotype(grammar, parameters)
    # Parameter: codon_size
    for cs, possible_vals in [
        (1, [0, 1]),
        (2, [0, 1, 2, 3]),
        (3, [0, 1, 2, 3, 4, 5, 6, 7]),
    ]:
        for _ in range(shared.NUM_REPETITIONS):
            parameters = dict(codon_size=cs, genotype_length=3000)
            ind = al.systems.pige.init_individual.random_genotype(grammar, parameters)
            check_individual(ind)
            assert all(codon in possible_vals for codon in ind.genotype.data)
            assert max(ind.genotype.data) == max(
                possible_vals
            )  # can fail, very low probability
            assert min(ind.genotype.data) == min(
                possible_vals
            )  # can fail, very low probability
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(codon_size=0)
        al.systems.pige.init_individual.random_valid_genotype(grammar, parameters)
    # Parameter: init_ind_random_valid_genotype_max_tries
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_ind_random_valid_genotype_max_tries=0)
        al.systems.pige.init_individual.random_valid_genotype(grammar, parameters)

    # Method: gp_grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.pige.init_individual.gp_grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_grow_max_depth
    al.systems.pige.init_individual.gp_grow_tree(
        grammar, dict(init_ind_gp_grow_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.pige.init_individual.gp_grow_tree(
            grammar, dict(init_ind_gp_grow_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_individual.gp_grow_tree(
            grammar, dict(init_ind_gp_grow_max_depth="nonsense")
        )

    # Method: gp_full_tree
    for _ in range(num_repetitions):
        ind = al.systems.pige.init_individual.gp_full_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_full_max_depth
    al.systems.pige.init_individual.gp_full_tree(
        grammar, dict(init_ind_gp_full_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.pige.init_individual.gp_full_tree(
            grammar, dict(init_ind_gp_full_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_individual.gp_full_tree(
            grammar, dict(init_ind_gp_full_max_depth="nonsense")
        )

    # Method: pi_grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.pige.init_individual.pi_grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_grow_max_depth
    al.systems.pige.init_individual.pi_grow_tree(
        grammar, dict(init_ind_pi_grow_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.pige.init_individual.pi_grow_tree(
            grammar, dict(init_ind_pi_grow_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_individual.pi_grow_tree(
            grammar, dict(init_ind_pi_grow_max_depth="nonsense")
        )

    # Method: ptc2
    for _ in range(num_repetitions):
        ind = al.systems.pige.init_individual.ptc2_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_ptc2_max_expansions
    al.systems.pige.init_individual.ptc2_tree(
        grammar, dict(init_ind_ptc2_max_expansions=0)
    )
    for _ in range(num_repetitions):
        al.systems.pige.init_individual.ptc2_tree(
            grammar, dict(init_ind_ptc2_max_expansions=100)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_individual.ptc2_tree(
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
        [[0]],
        [[1]],
        [[2, 5, 7, 11, 13], [2, 4, 6, 8], [1, 2, 3, 4, 5, 6, 7]],
        ["[0]"],
        ["[1]"],
        ["[2, 5, 7, 11, 13]", "[2, 4, 6, 8]", "[1, 2, 3, 4, 5, 6, 7]"],
        [[0], "[1]"],
        [[2, 5, 7, 11, 13], "[2, 4, 6, 8]", [1, 2, 3, 4, 5, 6, 7]],
    ]
    for gts in valid_genotype_collections:
        parameters = dict(init_pop_given_genotypes=gts)
        pop = al.systems.pige.init_population.given_genotypes(grammar, parameters)
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
            al.systems.pige.init_population.given_genotypes(grammar, parameters)
    # Parameter: init_genotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_population.given_genotypes(grammar)

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
        pop = al.systems.pige.init_population.given_derivation_trees(
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
            al.systems.pige.init_population.given_derivation_trees(grammar, parameters)
    # Parameter: init_pop_given_derivation_trees not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_population.given_derivation_trees(grammar)

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
        pop = al.systems.pige.init_population.given_phenotypes(grammar, parameters)
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
            al.systems.pige.init_population.given_derivation_trees(grammar, parameters)
    # Parameter: init_pop_given_phenotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.pige.init_population.given_phenotypes(grammar)

    # Method: random_genotypes
    n = 10
    for _ in range(n):
        pop = al.systems.pige.init_population.random_genotypes(grammar)
        check_population(pop)
        assert len(pop) == al.systems.pige.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.pige.init_population.random_genotypes(grammar, parameters)
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
            pop = al.systems.pige.init_population.random_genotypes(grammar, params)
            check_population(pop)
            assert len(pop) == 10
            if unique_gen or unique_phe:
                params["init_pop_size"] = 30
                params["genotype_length"] = 2
                params["codon_size"] = 2
                with pytest.raises(al.exceptions.InitializationError):
                    al.systems.pige.init_population.random_genotypes(grammar, params)
    # Parameter: init_pop_unique_max_tries
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_unique_max_tries=0)
        al.systems.pige.init_population.random_genotypes(grammar, parameters)
    # Parameter: genotype_length
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(genotype_length=0)
        al.systems.pige.init_population.random_genotypes(grammar, parameters)

    # Method: gp_rhh (=GP's ramped half and half)
    for _ in range(n):
        pop = al.systems.pige.init_population.gp_rhh(grammar)
        check_population(pop)
        assert len(pop) == al.systems.pige.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.pige.init_population.gp_rhh(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_gp_rhh_start_depth, init_pop_gp_rhh_end_depth
    parameters = dict(init_pop_gp_rhh_start_depth=3, init_pop_gp_rhh_end_depth=4)
    pop = al.systems.pige.init_population.gp_rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_gp_rhh_start_depth=5, init_pop_gp_rhh_end_depth=3)
        pop = al.systems.pige.init_population.gp_rhh(grammar, parameters)

    # Method: pi_rhh (=position-independent ramped half and half)
    for _ in range(n):
        pop = al.systems.pige.init_population.pi_rhh(grammar)
        check_population(pop)
        assert len(pop) == al.systems.pige.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.pige.init_population.pi_rhh(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_pi_rhh_start_depth, init_pop_pi_rhh_end_depth
    parameters = dict(init_pop_pi_rhh_start_depth=3, init_pop_pi_rhh_end_depth=4)
    pop = al.systems.pige.init_population.pi_rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_pi_rhh_start_depth=5, init_pop_pi_rhh_end_depth=3)
        pop = al.systems.pige.init_population.pi_rhh(grammar, parameters)

    # Method: ptc2 (=probabilistic tree creation 2)
    for _ in range(n):
        pop = al.systems.pige.init_population.ptc2(grammar)
        check_population(pop)
        assert len(pop) == al.systems.pige.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.pige.init_population.ptc2(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_ptc2_start_expansions, init_pop_ptc2_end_expansions
    parameters = dict(
        init_pop_ptc2_start_expansions=10, init_pop_ptc2_end_expansions=50
    )
    pop = al.systems.pige.init_population.ptc2(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(
            init_pop_ptc2_start_expansions=50, init_pop_ptc2_end_expansions=10
        )
        pop = al.systems.pige.init_population.ptc2(grammar, parameters)


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
        "(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        al.systems.pige.representation.Genotype((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        al.systems.pige.representation.Genotype("(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)"),
        al.systems.pige.representation.Genotype([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        al.systems.pige.representation.Genotype("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"),
    )

    # Mutation (guaranteed to change the genotype due to parameter choice)
    methods = (
        al.systems.pige.mutation.int_replacement_by_probability,
        al.systems.pige.mutation.int_replacement_by_count,
    )
    params = dict(
        mutation_int_replacement_probability=1.0, mutation_int_replacement_count=2
    )
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

    # Mutation (guaranteed to -not- change the genotype due to different parameter choice)
    params = dict(
        mutation_int_replacement_probability=0.0, mutation_int_replacement_count=0
    )
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
    bnf_text = "<bit> ::= 1 | 0"
    grammar = al.Grammar(bnf_text=bnf_text)

    # Mutation
    for _ in range(50):
        for genotype in (
            [-1],
            [-1, -1],
            [-1, -1, -1],
            [-1, -1, -1, -1],
            [-1] * 50,
            [-1] * 100,
        ):
            # Parameter: mutation_int_replacement_count
            for mutation_int_replacement_count in range(10):
                parameters = dict(
                    mutation_int_replacement_count=mutation_int_replacement_count,
                    codon_size=8,
                )
                gt_copy = copy.copy(genotype)
                gt_mut = al.systems.pige.mutation.int_replacement_by_count(
                    grammar, gt_copy, parameters
                )
                # Check expected number of int flips for different cases
                num_changed_codons = sum(codon != -1 for codon in gt_mut.data)
                if mutation_int_replacement_count == 0:
                    assert gt_mut.data == tuple([-1] * len(genotype))
                elif mutation_int_replacement_count >= len(genotype):
                    assert num_changed_codons == len(genotype)
                else:
                    assert num_changed_codons == mutation_int_replacement_count


# Crossover


def test_crossover_api():
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
        al.systems.pige.representation.Genotype(str([6, 7, 8] * 10)),
    )

    # Crossover
    methods = (al.systems.pige.crossover.two_point_length_preserving,)

    def perform_checks(gt1, gt2, gt3, gt4):
        if not isinstance(gt1, al.systems.pige.representation.Genotype):
            gt1 = al.systems.pige.representation.Genotype(copy.copy(gt1))
        if not isinstance(gt2, al.systems.pige.representation.Genotype):
            gt2 = al.systems.pige.representation.Genotype(copy.copy(gt2))
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
    gt_valid = [1, 2, 3, 4, 5]
    for gt_invalid in [None, False, True, "", 0, 1, 3.14, "101"]:
        al.systems.pige.crossover.two_point_length_preserving(
            grammar, gt_valid, gt_valid
        )
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.pige.crossover.two_point_length_preserving(
                grammar, gt_valid, gt_invalid
            )
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.pige.crossover.two_point_length_preserving(
                grammar, gt_invalid, gt_valid
            )
        with pytest.raises(al.exceptions.GenotypeError):
            al.systems.pige.crossover.two_point_length_preserving(
                grammar, gt_invalid, gt_invalid
            )
    # - too short genotype
    gt1 = [0]
    gt2 = [1]
    with pytest.raises(al.exceptions.OperatorError):
        al.systems.pige.crossover.two_point_length_preserving(grammar, gt1, gt2)

    # - genotypes with different length
    gt1 = [0, 1, 2, 3]
    gt2 = [0, 1, 2, 3, 4]
    with pytest.raises(al.exceptions.OperatorError):
        al.systems.pige.crossover.two_point_length_preserving(grammar, gt1, gt2)


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
        al.systems.pige.representation.Genotype(str([6, 7, 8, 9] * 2)),
    )

    # Neighborhood
    for gt in genotypes:
        phe = al.systems.pige.mapping.forward(gr, gt)
        # Default
        nh1 = al.systems.pige.neighborhood.int_replacement(gr, gt)
        nh2 = al.systems.pige.neighborhood.int_replacement(gr, genotype=gt)
        nh3 = al.systems.pige.neighborhood.int_replacement(grammar=gr, genotype=gt)
        nh4 = al.systems.pige.neighborhood.int_replacement(gr, gt, dict())
        nh5 = al.systems.pige.neighborhood.int_replacement(gr, gt, parameters=dict())
        nh6 = al.systems.pige.neighborhood.int_replacement(
            gr, genotype=gt, parameters=dict()
        )
        nh7 = al.systems.pige.neighborhood.int_replacement(
            grammar=gr, genotype=gt, parameters=dict()
        )
        assert nh1 == nh2 == nh3 == nh4 == nh5 == nh6 == nh7
        for new_gt in nh1:
            check_genotype(new_gt)
            new_phe = al.systems.pige.mapping.forward(gr, new_gt)
            assert new_phe != phe


@pytest.mark.parametrize(
    "bnf, genotype, phenotype",
    [
        (shared.BNF5, [0], "1"),
        (shared.BNF5, [1], "2"),
        (shared.BNF5, [2], "3"),
        (shared.BNF5, [3], "4"),
        (shared.BNF5, [4], "5"),
        (shared.BNF6, [0, 0, 0, 0], "1"),
        (shared.BNF6, [0, 0, 0, 1], "2"),
        (shared.BNF6, [0, 1, 0, 0], "3"),
        (shared.BNF6, [0, 1, 0, 1], "4"),
        (shared.BNF6, [0, 2], "5"),
        (shared.BNF7, [0, 0, 0, 0, 0, 0], "ac1"),
        (shared.BNF7, [0, 1, 0, 1, 0, 1], "bf8"),
        (shared.BNF7, [0, 0, 0, 1, 0, 1], "ad4"),
        (shared.BNF9, [0], "a"),
        (shared.BNF9, [0, 0, 0, 1, 0, 1, 0, 2], "bc"),
        (shared.BNF9, [0, 1], "22"),
        (shared.BNF9, [0, 1, 0, 0, 0, 2], "3"),
        (shared.BNF9, [0, 1, 0, 1, 0, 0, 0, 2], "13"),
    ],
)
def test_neighborhood_reachability_in_finite_languages(bnf, genotype, phenotype):
    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.pige.representation.Genotype(genotype)

    # Phenotype
    parameters = dict(max_wraps=50, max_expansions=1000)
    phe = al.systems.pige.mapping.forward(grammar, gt, parameters)
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
                nbrs = al.systems.pige.neighborhood.int_replacement(grammar, gen, param)
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                try:
                    phe = al.systems.pige.mapping.forward(grammar, gen, param)
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
        (shared.BNF10, (222, 133, 219, 174), "1x", ("2x", "3x", "4y", "5y", "6y", "7")),
        (shared.BNF11, (231, 201, 233, 95), "1", ("2", "3", "4", "22", "33", "44")),
        (
            shared.BNF12,
            (254, 80, 233, 1),
            "131",
            ("242", "2332", "22422", "21312", "223322"),
        ),
    ],
)
def test_neighborhood_reachability_in_infinite_languages(
    bnf, genotype, phenotype, strings_given
):
    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.pige.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.pige.mapping.forward(grammar, gt)
    assert phe == phenotype

    # Neighborhood
    strings_given = set(strings_given)
    params = [
        dict(),  # required time depends on the default parameters (stop criteria values)
        dict(max_expansions=10),
        dict(max_expansions=10, neighborhood_max_size=3),
        dict(max_expansions=10, stack_mode="start"),
        dict(max_expansions=10, stack_mode="end"),
        dict(max_expansions=10, stack_mode="inplace"),
        dict(max_wraps=2),
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
                nbrs = al.systems.pige.neighborhood.int_replacement(grammar, gen, param)
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                # Genotype management
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                # Phenotype generation
                try:
                    phe = al.systems.pige.mapping.forward(grammar, gen, param)
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
    dt = grammar.parse_string("1a1a")
    gt = al.systems.pige.mapping.reverse(grammar, dt)

    # Neighborhood in different distances when changing only terminals
    # - distance 1
    parameters = dict(neighborhood_only_terminals=True)
    nbrs_gt = al.systems.pige.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.pige.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1a1b", "1a2a", "1b1a", "2a1a"}

    # - distance 2
    parameters = dict(neighborhood_distance=2, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.pige.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.pige.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1a2b", "1b1b", "1b2a", "2a1b", "2a2a", "2b1a"}

    # - distance 3
    parameters = dict(neighborhood_distance=3, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.pige.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.pige.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1b2b", "2a2b", "2b1b", "2b2a"}

    # - distance 4
    parameters = dict(neighborhood_distance=4, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.pige.neighborhood.int_replacement(grammar, gt, parameters)
    nbrs = [al.systems.pige.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"2b2b"}

    # - distance 5 and greater
    for dist in range(5, 20):
        parameters = dict(neighborhood_distance=dist, neighborhood_only_terminals=True)
        nbrs_gt = al.systems.pige.neighborhood.int_replacement(grammar, gt, parameters)
        nbrs = [al.systems.pige.mapping.forward(grammar, gt) for gt in nbrs_gt]
        assert nbrs == []


@pytest.mark.parametrize(
    "bnf, gt, phe, phe_neighbors",
    [
        (shared.BNF1, [0], "0", ("1", "2")),
        (shared.BNF1, [0, 0], "0", ("1", "2")),
        (shared.BNF1, [1], "1", ("0", "2")),
        (shared.BNF1, [1, 1], "1", ("0", "2")),
        (shared.BNF1, [2], "2", ("0", "1")),
        (shared.BNF1, [3], "0", ("1", "2")),
        (shared.BNF1, [4], "1", ("0", "2")),
        (shared.BNF1, [5], "2", ("0", "1")),
        (shared.BNF2, [0], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF2, [0, 0], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF2, [0, 0, 0], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF2, [1], "1b", ("0b", "2b", "1a", "1c")),
        (shared.BNF2, [1, 1], "1b", ("0b", "2b", "1a", "1c")),
        (shared.BNF2, [1, 1, 1], "1b", ("0b", "2b", "1a", "1c")),
        (shared.BNF2, [2], "2c", ("0c", "1c", "2a", "2b")),
        (shared.BNF2, [2, 2], "2c", ("0c", "1c", "2a", "2b")),
        (shared.BNF2, [2, 2, 2], "2c", ("0c", "1c", "2a", "2b")),
        (shared.BNF2, [3], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF2, [3, 6], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF3, [0], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF3, [0, 0], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF3, [0, 0, 0], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF3, [1], "1b", ("0b", "2b", "1a", "1c")),
        (shared.BNF3, [1, 1], "1b", ("0b", "2b", "1a", "1c")),
        (shared.BNF3, [1, 1, 1], "1b", ("0b", "2b", "1a", "1c")),
        (shared.BNF3, [2], "2c", ("0c", "1c", "2a", "2b")),
        (shared.BNF3, [2, 2], "2c", ("0c", "1c", "2a", "2b")),
        (shared.BNF3, [2, 2, 2], "2c", ("0c", "1c", "2a", "2b")),
        (shared.BNF3, [3], "0a", ("1a", "2a", "0b", "0c")),
        (shared.BNF3, [3, 6], "0a", ("1a", "2a", "0b", "0c")),
        (
            shared.BNF4,
            [0],
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
            [1],
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
            [0, 0, 0, 11, 0, 2, 0, 13, 0, 15, 0, 4, 0, 17, 0, 19, 0, 6],
            "10110110",
            (
                "00110110",
                "11110110",
                "10010110",
                "10100110",
                "10111110",
                "10110010",
                "10110100",
                "10110111",
            ),
        ),
    ],
)
def test_neighborhood_parameter_max_size(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    parameters = dict(max_wraps=50, max_expansions=1000)
    assert phe == al.systems.pige.mapping.forward(gr, gt, parameters)

    # Neighborhood
    nbrs = al.systems.pige.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [
        al.systems.pige.mapping.forward(gr, nbr_gt, parameters) for nbr_gt in nbrs
    ]
    assert set(nbrs_phe) == set(phe_neighbors)

    # Parameter: neighborhood_max_size
    parameters = dict(neighborhood_max_size=None, max_wraps=50, max_expansions=1000)
    nbrs = al.systems.pige.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [
        al.systems.pige.mapping.forward(gr, nbr_gt, parameters) for nbr_gt in nbrs
    ]
    assert set(nbrs_phe) == set(phe_neighbors)

    for max_size in range(1, 5):
        parameters = dict(
            neighborhood_max_size=max_size, max_wraps=50, max_expansions=1000
        )
        nbrs_phe = set()
        for _ in range(100):
            nbrs = al.systems.pige.neighborhood.int_replacement(gr, gt, parameters)
            assert len(nbrs) <= max_size
            for nbr_gt in nbrs:
                nbr_phe = al.systems.pige.mapping.forward(gr, nbr_gt, parameters)
                assert nbr_phe in phe_neighbors
                nbrs_phe.add(nbr_phe)
        assert nbrs_phe == set(phe_neighbors)


@pytest.mark.parametrize(
    "bnf, gt, phe, phe_neighbors",
    [
        (shared.BNF5, [0], "1", ("2", "3", "4", "5")),
        (shared.BNF5, [1], "2", ("1", "3", "4", "5")),
        (shared.BNF5, [2], "3", ("1", "2", "4", "5")),
        (shared.BNF5, [3], "4", ("1", "2", "3", "5")),
        (shared.BNF5, [4], "5", ("1", "2", "3", "4")),
        (shared.BNF6, [0, 0, 0, 0], "1", ("2", "5")),
        (shared.BNF6, [0, 0, 0, 1], "2", ("1", "5")),
        (shared.BNF6, [0, 1, 0, 0], "3", ("4", "5")),
        (shared.BNF6, [0, 1, 0, 1], "4", ("3", "5")),
        (shared.BNF6, [0, 2], "5", ()),
        (shared.BNF7, [0, 0, 0, 0, 0, 0], "ac1", ("be5", "ad3", "ac2")),
        (shared.BNF7, [0, 1, 0, 1, 0, 1], "bf8", ("ad4", "be6", "bf7")),
        (shared.BNF7, [0, 0, 0, 1, 0, 1], "ad4", ("bf8", "ac2", "ad3")),
        (
            shared.BNF8,
            [0, 0],
            "<S><S><S><S>",
            ("a0g", "1g", "a0ga0g", "1g1g1g<S>", "a0ga0g<A>0<B>", "1g1g1<B><S><S>"),
        ),
        (shared.BNF8, [0, 1], "t", ("t0g", "1c", "a")),
        (shared.BNF8, [0, 2, 0, 0, 0, 1], "a0c", ("1g", "t0c", "a0g")),
        (shared.BNF8, [0, 2, 1, 1, 0, 0], "a0c", ("1c", "t0c", "a0g")),
    ],
)
def test_neighborhood_parameter_only_terminals(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    parameters = dict(neighborhood_only_terminals=True, max_wraps=2)
    assert phe == al.systems.pige.mapping.forward(
        gr, gt, parameters, raise_errors=False
    )

    # Neighborhood
    nbrs = al.systems.pige.neighborhood.int_replacement(gr, gt, parameters)
    nbrs_phe = [
        al.systems.pige.mapping.forward(gr, nbr_gt, parameters, raise_errors=False)
        for nbr_gt in nbrs
    ]
    assert set(nbrs_phe) == set(phe_neighbors)


@pytest.mark.parametrize(
    "bnf, gt, phe, phe_neighbors",
    [
        (shared.BNF7, [0, 0, 0, 0, 0, 0], "ac1", ("be5", "ad3", "ac2")),
        (shared.BNF7, [0, 1, 0, 1, 0, 1], "bf8", ("ad4", "be6", "bf7")),
        (shared.BNF7, [0, 0, 0, 1, 0, 1], "ad4", ("bf8", "ac2", "ad3")),
    ],
)
def test_neighborhood_parameter_stack_mode(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    parameters = dict(neighborhood_only_terminals=True, max_wraps=2)
    assert phe == al.systems.pige.mapping.forward(
        gr, gt, parameters, raise_errors=False
    )

    # Neighborhood
    for sm in ("start", "end", "inplace"):
        parameters["stack_mode"] = sm
        nbrs = al.systems.pige.neighborhood.int_replacement(gr, gt, parameters)
        nbrs_phe = [
            al.systems.pige.mapping.forward(gr, nbr_gt, parameters, raise_errors=False)
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

    # Genotypic data of five types:
    # 1) tuple of int 2) string thereof 3) list of int 4) string thereof 5) Genotype object
    tup = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    data_variants = (
        tup,
        str(tup),
        list(tup),
        str(list(tup)),
        al.systems.pige.representation.Genotype(tup),
    )

    # Forward mapping
    params = dict(
        codon_size=4,
        max_wraps=0,
        max_expansions=3,
        stack_mode="inplace",
    )
    kwargs = dict(
        verbose=False,
        raise_errors=False,
        return_derivation_tree=False,
    )
    for data in data_variants:
        for sm in ("start", "end", "inplace"):
            params["stack_mode"] = sm
            for vb in (True, False):
                kwargs["verbose"] = vb

                # Method of Grammar class
                string1 = grammar.generate_string("pige", data, params, **kwargs)
                string2 = grammar.generate_string(
                    "pige", data, parameters=params, **kwargs
                )
                string3 = grammar.generate_string(
                    method="pige", genotype=data, parameters=params, **kwargs
                )
                assert string1
                assert string1 == string2 == string3

                # Method of DerivationTree class
                dt1 = grammar.generate_derivation_tree("pige", data, params, **kwargs)
                dt2 = grammar.generate_derivation_tree(
                    "pige", data, parameters=params, **kwargs
                )
                dt3 = grammar.generate_derivation_tree(
                    method="pige", genotype=data, parameters=params, **kwargs
                )
                assert string1 == dt1.string() == dt2.string() == dt3.string()

                # Functions in mapping module
                string4 = al.systems.pige.mapping.forward(
                    grammar, data, params, **kwargs
                )
                string5 = al.systems.pige.mapping.forward(
                    grammar, data, parameters=params, **kwargs
                )
                string6 = al.systems.pige.mapping.forward(
                    grammar=grammar, genotype=data, parameters=params, **kwargs
                )
                assert string1 == string4 == string5 == string6

                kwargs["return_derivation_tree"] = True
                phe, dt4 = al.systems.pige.mapping.forward(
                    grammar, data, params, **kwargs
                )
                phe, dt5 = al.systems.pige.mapping.forward(
                    grammar, data, parameters=params, **kwargs
                )
                phe, dt6 = al.systems.pige.mapping.forward(
                    grammar=grammar, genotype=data, parameters=params, **kwargs
                )
                kwargs["return_derivation_tree"] = False
                assert string1 == dt4.string() == dt5.string() == dt6.string()

                # Same with errors when reaching wrap or expansion limit
                kwargs["raise_errors"] = True
                with pytest.raises(al.exceptions.MappingError):
                    grammar.generate_string(
                        method="pige", genotype=data, parameters=params, **kwargs
                    )
                with pytest.raises(al.exceptions.MappingError):
                    grammar.generate_derivation_tree(
                        method="pige", genotype=data, parameters=params, **kwargs
                    )
                with pytest.raises(al.exceptions.MappingError):
                    al.systems.pige.mapping.forward(
                        grammar, genotype=data, parameters=params, **kwargs
                    )
                with pytest.raises(al.exceptions.MappingError):
                    al.systems.pige.mapping.forward(
                        grammar, genotype=data, parameters=params, **kwargs
                    )
                kwargs["raise_errors"] = False


def test_mapping_reverse_api():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Random derivation trees and corresponding strings
    random_dts, random_strings = [], []
    for _ in range(shared.NUM_REPETITIONS):
        dt = grammar.generate_derivation_tree()
        string = dt.string()
        assert isinstance(string, str)
        assert isinstance(dt, al._grammar.data_structures.DerivationTree)
        random_dts.append(dt)
        random_strings.append(string)
    for _ in range(shared.NUM_REPETITIONS):
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
        derivation_order="leftmost",
        stack_mode="start",
    )
    for rc in (True, False):
        for do in ("leftmost", "rightmost", "random"):
            for sm in ("start", "end", "inplace"):
                parameters["codon_randomization"] = rc
                parameters["derivation_order"] = do
                parameters["stack_mode"] = sm
                # Functions in mapping module
                p1 = dict()
                p2 = dict(codon_size=4)
                p3 = dict(codon_randomization=False)
                p4 = dict(codon_size=8, codon_randomization=False)
                p5 = dict(codon_size=5, codon_randomization=True)
                for parameters in (p1, p2, p3, p4, p5):
                    for dt, string in zip(random_dts, random_strings):
                        # With parameters
                        gt1 = al.systems.pige.mapping.reverse(
                            grammar, string, parameters
                        )
                        gt2 = al.systems.pige.mapping.reverse(
                            grammar, string, parameters, False
                        )
                        gt3, dt3 = al.systems.pige.mapping.reverse(
                            grammar, string, parameters, True
                        )
                        gt4 = al.systems.pige.mapping.reverse(
                            grammar,
                            phenotype_or_derivation_tree=string,
                            parameters=parameters,
                        )
                        gt5 = al.systems.pige.mapping.reverse(
                            grammar,
                            phenotype_or_derivation_tree=dt,
                            parameters=parameters,
                        )
                        gt6 = al.systems.pige.mapping.reverse(
                            grammar,
                            phenotype_or_derivation_tree=string,
                            parameters=parameters,
                            return_derivation_tree=False,
                        )
                        gt7, dt11 = al.systems.pige.mapping.reverse(
                            grammar,
                            phenotype_or_derivation_tree=dt,
                            parameters=parameters,
                            return_derivation_tree=True,
                        )
                        for gt in (gt1, gt2, gt3, gt4, gt5, gt6, gt7):
                            # Check if reverse mapping resulted in a valid genotype
                            check_genotype(gt)
                            # Check if genotype allows to reproduce the original string
                            # via forward mapping
                            string_from_fwd_map = grammar.generate_string(
                                "pige", gt, parameters
                            )
                            assert string_from_fwd_map == string
                        # Without parameters
                        gt1 = al.systems.pige.mapping.reverse(grammar, string)
                        gt2 = al.systems.pige.mapping.reverse(grammar, dt)
                        gt3 = al.systems.pige.mapping.reverse(
                            grammar, phenotype_or_derivation_tree=string
                        )
                        gt4 = al.systems.pige.mapping.reverse(
                            grammar, phenotype_or_derivation_tree=dt
                        )
                        for gt in (gt1, gt2, gt3, gt4):
                            # Check if reverse mapping resulted in a valid genotype
                            check_genotype(gt)
                            # Check if genotype allows to reproduce the original string
                            # via forward mapping
                            string_from_fwd_map = grammar.generate_string("pige", gt)
                            assert string_from_fwd_map == string


def test_mapping_errors():
    bnf_text = "<S> ::= <S><S> | 1 | 2 | 3"
    grammar = al.Grammar(bnf_text=bnf_text)
    # Invalid input: a string that is not part of the grammar's language
    string = "4"
    with pytest.raises(al.exceptions.MappingError):
        al.systems.pige.mapping.reverse(grammar, string)
    # Invalid input: a derivation tree with an unknown nonterminal
    dt = grammar.generate_derivation_tree()
    dt.root_node.symbol = al._grammar.data_structures.NonterminalSymbol("nonsense")
    with pytest.raises(al.exceptions.MappingError):
        al.systems.pige.mapping.reverse(grammar, dt)
    # Invalid input: a derivation tree with an unknown derivation (no corresponding rule)
    dt = grammar.generate_derivation_tree()
    dt.leaf_nodes()[0].symbol = al._grammar.data_structures.TerminalSymbol("nonsense")
    with pytest.raises(al.exceptions.MappingError):
        al.systems.pige.mapping.reverse(grammar, dt)
    # Parameter: codon_size
    string = "111222333"
    parameters = dict(codon_size=1)
    with pytest.raises(al.exceptions.MappingError):
        al.systems.pige.mapping.reverse(grammar, string, parameters)
    parameters = dict(codon_size=2)
    gt = al.systems.pige.mapping.reverse(grammar, string, parameters)
    assert string == al.systems.pige.mapping.forward(grammar, gt)
    # Parameter: codon_size
    bnf_text = """<S> ::= <A><A><A><A><A><A><A><A><A><A><A><A><A><A><A><A><A><A>
    <A> ::= 1 |
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    string = "1"
    parameters = dict(codon_size=1, derivation_order="rightmost")
    with pytest.raises(al.exceptions.MappingError):
        al.systems.pige.mapping.reverse(grammar, string, parameters)


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
        string = grammar.generate_string("pige", [0, 0, 0, 0, 0, 0, 0, 0], verbose=vb)
        assert string == "0x+("
        # Parameter: max_wraps
        for mw in range(10):
            params = dict(max_wraps=mw, max_expansions=None)
            if mw < 8:
                with pytest.raises(al.exceptions.MappingError):
                    grammar.generate_string("pige", [0], params, verbose=vb)
                sentential_form = grammar.generate_string(
                    "pige", [0], params, verbose=vb, raise_errors=False
                )
                assert "<" in sentential_form
                assert ">" in sentential_form
            else:
                string = grammar.generate_string("pige", [0], params, verbose=vb)
                assert "<" not in string
                assert ">" not in string
        # Parameter: max_expansions
        for me in range(20):
            params = dict(max_wraps=None, max_expansions=me)
            if me < 5:
                with pytest.raises(al.exceptions.MappingError):
                    grammar.generate_string("pige", [0], params, verbose=vb)
                sentential_form = grammar.generate_string(
                    "pige", [0], params, verbose=vb, raise_errors=False
                )
                assert "<" in sentential_form
                assert ">" in sentential_form
            else:
                string = grammar.generate_string("pige", [0], params, verbose=vb)
                assert "<" not in string
                assert ">" not in string
        # Parameter: max_wraps and max_expansions
        params = dict(max_wraps=None, max_expansions=None)
        with pytest.raises(al.exceptions.MappingError):
            grammar.generate_string("pige", [0], params, verbose=vb)


def test_mapping_reverse_randomized():
    bnf_text = """
    <S> ::= <S><S> | <A> | <B> | a | b
    <A> ::= 1 | 2 | 3
    <B> ::= X | Y
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for string1 in ("a1X1a", "123XYba", "Yb", "3a2b1X2Y3a", "ababa1", "X1Y3Xb"):
        randomized = set()
        nonrandomized = set()
        for rc in (True, False):
            for _ in range(20):
                parameters = dict(codon_randomization=rc)
                genotype = al.systems.pige.mapping.reverse(grammar, string1, parameters)
                string2 = al.systems.pige.mapping.forward(grammar, genotype, parameters)
                assert string1 == string2
                if rc:
                    randomized.add(str(genotype))
                else:
                    nonrandomized.add(str(genotype))
        assert len(randomized) > 1
        assert len(nonrandomized) == 1


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
                genotype = al.systems.pige.mapping.reverse(grammar, string1, parameters)
                # Forward map: genotype -> string2
                string2 = al.systems.pige.mapping.forward(grammar, genotype, parameters)
                assert string1 == string2


def test_mapping_forward_and_reverse_by_hand1():
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | - | * | /
    <v> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for string1 in (
        "1+1",
        "9-4",
        "7*5-3",
        "9*8/7+6-5",
        "3+4/9-1*8",
        "1+2+3+4+5-6-7*8/9",
    ):
        for cs in (4, 6, 9):
            for do in ("leftmost", "rightmost", "random"):
                for sm in ("start", "end", "inplace"):
                    for rc in (True, False):
                        parameters = dict(
                            codon_size=cs,
                            derivation_order=do,
                            codon_randomization=rc,
                            stack_mode=sm,
                        )
                        # Reverse map: string1 -> genotype
                        genotype = al.systems.pige.mapping.reverse(
                            grammar, string1, parameters
                        )
                        # Forward map: genotype -> string2
                        string2 = al.systems.pige.mapping.forward(
                            grammar, genotype, parameters
                        )
                        assert string2 == string1


def test_mapping_forward_and_reverse_by_hand2():
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | - | * | /
    <v> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    string1 = "3+4/9-1*8"
    modes = ["start", "end", "inplace"]
    cnt_all = 0
    cnt_non_matches = 0
    for _ in range(100):
        for m1 in modes:
            for m2 in modes:
                # Reverse map with mode1
                parameters = dict(stack_mode=m1, derivation_order="random")
                genotype = al.systems.pige.mapping.reverse(grammar, string1, parameters)
                # Forward map with mode2
                parameters = dict(stack_mode=m2)
                string2 = al.systems.pige.mapping.forward(
                    grammar, genotype, parameters, raise_errors=False
                )
                # If modes are equal, strings should be equal
                if m1 == m2:
                    assert string1 == string2
                # If modes are different, strings should be different most of the time
                else:
                    cnt_all += 1
                    if string1 != string2:
                        cnt_non_matches += 1
    assert cnt_non_matches / cnt_all > 0.5


def test_mapping_forward_and_reverse_by_hand3():
    # Test scheme: Use all argument combinations to reverse map
    bnf_text = """
    <e> ::= ( <e> <o> <e> ) | <v>
    <o> ::= + | - | * | /
    <v> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    string = "((1+2)*3)"
    for _ in range(50):
        # Deterministic combinations (only if grammar is unambiguous => only one possible tree)
        genotypes = []
        genotypes_unique = set()
        for order in ["leftmost", "rightmost"]:
            for mode in ["start", "end", "inplace"]:
                parameters = dict(
                    codon_randomization=False,
                    derivation_order=order,
                    stack_mode=mode,
                )
                gen = al.systems.pige.mapping.reverse(grammar, string, parameters)
                genotypes.append(gen)
                genotypes_unique.add(str(gen))
        assert [gt.data for gt in genotypes] == [
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2),
            (0, 0, 0, 0, 2, 1, 4, 0, 2, 0, 2, 1, 2, 1, 0, 2, 0, 1, 0, 2),
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2),
            (0, 0, 2, 1, 0, 2, 1, 2, 0, 0, 2, 1, 0, 1, 1, 0, 0, 1, 0, 0),
            (0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2, 1, 1, 0, 0, 1, 0, 0),
            (0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2, 1, 1, 0, 0, 1, 0, 0),
        ]
        # There are 6 cases, 2 of which are not unique because
        # - order "leftmost"  => mode "start" and "inplace" are same
        # - order "rightmost" => mode "end" and "inplace" are same
        assert len(genotypes_unique) == 4

        # Nondeterministic combinations
        for order in ["random"]:
            for mode in ["start", "end", "inplace"]:
                parameters = dict(derivation_order=order, stack_mode=mode)
                al.systems.pige.mapping.reverse(grammar, string, parameters)


def test_mapping_forward_against_paper_2004_example():
    # References
    # - Paper 2004: https://doi.org/10.1007/978-3-540-24855-2_70
    bnf_text = """
    <expr> ::= <expr> <op> <expr>
             | <var>
      <op> ::= +
             | -
             | *
             | /
     <var> ::= x
             | y
    """
    # Caution: / was added to <op> (error in paper)
    grammar = al.Grammar(bnf_text=bnf_text)

    genotype = [
        23,
        88,
        9,
        102,
        20,
        11,
        5,
        18,
        16,
        6,
        27,
        3,
        12,
        4,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]
    parameters = dict(stack_mode="start", max_expansions=7)
    phenotype = grammar.generate_string(
        "pige", genotype, parameters, raise_errors=False
    )
    assert phenotype == "x*x<op><expr>"

    genotype = [
        23,
        88,
        11,
        102,
        20,
        11,
        5,
        18,
        16,
        6,
        27,
        3,
        12,
        4,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]
    parameters = dict(stack_mode="start", max_expansions=7)
    phenotype = grammar.generate_string(
        "pige", genotype, parameters, raise_errors=False
    )
    assert phenotype == "<expr><op>x*x"


def test_mapping_forward_against_paper_2010_example():
    # References
    # - Paper 2010: https://doi.org/10.1109/CEC.2010.5586204
    # - Paper 2011: https://doi.org/10.1007/978-3-642-20407-4_25 (same example reused)
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | *
    <v> ::= 0.5 | 5
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    genotype = [12, 8, 3, 11, 7, 6, 11, 8, 4, 3, 3, 11, 15, 7, 9, 8, 10, 3, 7, 4]

    # Evidence of new nonterminals being placed at end of stack as in figure
    parameters = dict(stack_mode="end")
    phenotype = grammar.generate_string("pige", genotype, parameters)
    assert phenotype == "0.5*0.5*0.5"

    # Evidence of new nonterminals NOT being placed at original position or start of stack
    phenotype = grammar.generate_string("pige", genotype, raise_errors=False)
    assert phenotype != "0.5*0.5*0.5"
    parameters = dict(stack_mode="inplace")
    phenotype = grammar.generate_string(
        "pige", genotype, parameters, raise_errors=False
    )
    assert phenotype != "0.5*0.5*0.5"
    parameters = dict(stack_mode="start")
    phenotype = grammar.generate_string(
        "pige", genotype, parameters, raise_errors=False
    )
    assert phenotype != "0.5*0.5*0.5"


def test_mapping_forward_against_paper_handbook_2018_example():
    # References
    # - Handbook 2018: https://doi.org/10.1007/978-3-319-78717-6 (p. 82)
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | -
    <v> ::= X | Y
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    genotype = [2, 12, 7, 9, 3, 15, 23, 1, 11, 4, 6, 13, 2, 7, 8, 3, 35, 19, 2, 6]
    phenotype = grammar.generate_string(method="pige", genotype=genotype)
    assert phenotype == "X-Y-Y"


@pytest.mark.skip(
    reason="Deviation, unclear why, perhaps wrapping on bit- versus integer-level."
)
def test_pige_implementation_against_inofficial_java_implementation_with_whge():
    directory = os.path.join(IN_DIR, "mappings", "evolved-ge-minified", "pige")
    filepaths = [
        os.path.join(directory, filename) for filename in os.listdir(directory)
    ]
    assert 10 < len(filepaths) < 1000
    for filepath in sorted(filepaths):
        # Read JSON file
        with open(filepath) as file_handle:
            data = json.load(file_handle)
        # Extract relevant data
        bnf_text = data["grammar"]
        max_wraps = int(data["parameters"]["max_wraps"])
        codon_size = int(data["parameters"]["codon_size"])
        gen_phe_map = data["genotype_to_phenotype_mappings"]
        # Create grammar
        grammar = al.Grammar(bnf_text=bnf_text)
        # Check if each genotype is mapped by Python to the same phenotype as by Java
        for gen, phe in gen_phe_map.items():
            try:
                phe_calc = al.systems.pige.mapping.forward(
                    genotype=gen,
                    grammar=grammar,
                    max_wraps=max_wraps,
                    codon_size=codon_size,
                )
            except Exception:
                phe_calc = "MappingException"
            assert phe_calc == phe
