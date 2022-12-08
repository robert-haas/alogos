import copy
import itertools
import math
import os

import pytest
import shared

import alogos as al


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, "in")


# Shared


def check_genotype(gt):
    assert isinstance(gt, al.systems.cfggpst.representation.Genotype)
    assert isinstance(gt.data, tuple)
    assert isinstance(gt.data[0], tuple)
    assert isinstance(gt.data[1], tuple)
    for item in gt.data[0]:
        assert isinstance(item, int)
    for item in gt.data[1]:
        assert isinstance(item, int)
    assert len(gt.data) == 2
    assert len(gt.data[0]) > 0
    assert len(gt.data[1]) > 0


def check_phenotype(phe):
    if phe is not None:
        assert isinstance(phe, str)
        assert len(phe) > 0


def check_fitness(fitness):
    assert isinstance(fitness, float)


def check_individual(ind):
    assert isinstance(ind, al.systems.cfggpst.representation.Individual)
    check_genotype(ind.genotype)
    check_phenotype(ind.phenotype)
    check_fitness(ind.fitness)


def check_population(pop):
    assert isinstance(pop, al.systems.cfggpst.representation.Population)
    assert len(pop) > 0
    for ind in pop:
        check_individual(ind)


# Representation


def test_representation_genotype():
    # Genotypic data of five types:
    # 1) derivation tree 2) tuple serialization of a derivation tree 3) string thereof
    # 4) list instead of tuple 5) string thereof
    bnf_text = """
    <S> ::= <A> | <B>
    <A> ::= a b | b a
    <B> ::= c d | d c | c d c
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    dt1 = grammar.parse_string("ab")
    dt2 = grammar.parse_string("dc")
    data_variants = (
        dt1,
        dt2,
        ((0, 1, 4, 3), (1, 2, 0, 0)),
        ((0, 2, 6, 5), (1, 2, 0, 0)),
        "((0,2,5,6),(1,2,0,0))",
        "((0,2,6,5),(1,2,0,0))",
        [[0, 1, 4, 3], [1, 2, 0, 0]],
        [[0, 2, 6, 5], [1, 2, 0, 0]],
        "[[0,2,5,6],[1,2,0,0]]",
        "[[0,2,6,5],[1,2,0,0]]",
    )
    for data in data_variants:
        gt = al.systems.cfggpst.representation.Genotype(data)
        check_genotype(gt)
        # Printing
        assert isinstance(str(gt), str)
        assert isinstance(repr(gt), str)
        assert repr(gt).startswith("<CFG-GP-ST genotype at ")
        p1 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p1, False)
        assert p1.string == str(gt)
        p2 = shared.MockPrettyPrinter()
        gt._repr_pretty_(p2, True)
        assert p2.string == "..."
        # Length
        assert len(gt) > 0
        if isinstance(data, tuple):
            assert len(gt) == len(data[0])
        elif isinstance(data, al._grammar.data_structures.DerivationTree):
            assert len(gt) == data.num_nodes()
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
        gt = al.systems.cfggpst.representation.Genotype(
            ((0, 2, 5, 6, 5), (1, 3, 0, 0, 0))
        )
        assert gt != gt2 == gt3 == gt4
        assert len(gt) != len(gt2) == len(gt3) == len(gt4)
        # Usage as key (genotypes that are equal are treated as the same entry)
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
            al.systems.cfggpst.representation.Genotype(data)


def test_representation_individual():
    data_variants = (
        [],
        ["gt"],
        ["gt", "phe"],
        ["gt", "phe", "fit"],
        ["gt", "phe", "fit", "det"],
    )
    for data in data_variants:
        ind = al.systems.cfggpst.representation.Individual(*data)
        # Member types
        assert ind.genotype is None if len(data) < 1 else data[0]
        assert ind.phenotype is None if len(data) < 2 else data[1]
        assert math.isnan(ind.fitness) if len(data) < 3 else data[2]
        assert isinstance(ind.details, dict) if len(data) < 4 else data[3]
        # Printing
        assert isinstance(str(ind), str)
        assert isinstance(repr(ind), str)
        assert str(ind).startswith("CFG-GP-ST individual:")
        assert repr(ind).startswith("<CFG-GP-ST individual object at ")
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
    ind1 = al.systems.cfggpst.representation.Individual(fitness=1)
    ind2 = al.systems.cfggpst.representation.Individual(fitness=2)
    assert ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 2: number and NaN
    ind1 = al.systems.cfggpst.representation.Individual(fitness=1)
    ind2 = al.systems.cfggpst.representation.Individual(fitness=float("nan"))
    assert ind1.less_than(ind2, "min")
    assert not ind1.less_than(ind2, "max")
    assert ind2.greater_than(ind1, "min")
    assert not ind2.greater_than(ind1, "max")
    # - Case 3: NaN and number
    ind1 = al.systems.cfggpst.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.cfggpst.representation.Individual(fitness=2)
    assert not ind1.less_than(ind2, "min")
    assert ind1.less_than(ind2, "max")
    assert not ind2.greater_than(ind1, "min")
    assert ind2.greater_than(ind1, "max")
    # - Case 4: NaN and NaN
    ind1 = al.systems.cfggpst.representation.Individual(fitness=float("nan"))
    ind2 = al.systems.cfggpst.representation.Individual(fitness=float("nan"))
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
        [al.systems.cfggpst.representation.Individual("gt1")],
        [
            al.systems.cfggpst.representation.Individual("gt1"),
            al.systems.cfggpst.representation.Individual("gt2"),
        ],
    )
    for data in data_variants:
        pop = al.systems.cfggpst.representation.Population(data)
        # Member types
        assert isinstance(pop.individuals, list)
        # Printing
        assert isinstance(str(pop), str)
        assert isinstance(repr(pop), str)
        assert str(pop).startswith("CFG-GP-ST population:")
        assert repr(pop).startswith("<CFG-GP-ST population at")
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
            al.systems.cfggpst.representation.Individual("gt3"),
            al.systems.cfggpst.representation.Individual("gt4"),
            al.systems.cfggpst.representation.Individual("gt5"),
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
        assert isinstance(pop2, al.systems.cfggpst.representation.Population)
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
            al.systems.cfggpst.representation.Population(data)


# Initialization


def test_initialize_individual():
    # Number of repetitions for methods with randomness
    num_repetitions = 20

    # Grammar
    bnf_text = """
    <S> ::= <A> | <B> | <S><S>
    <A> ::= a b | b a
    <B> ::= c d | d c | c d c
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Method: given_genotype
    dt1 = grammar.parse_string("ab")
    dt2 = grammar.parse_string("cdc")
    valid_genotypes = (
        dt1,
        dt2,
        ((0, 1, 4, 3), (1, 2, 0, 0)),
        ((0, 2, 6, 5), (1, 2, 0, 0)),
        "((0,2,5,6),(1,2,0,0))",
        "((0,2,5,6,5),(1,3,0,0,0))",
    )
    for gt in valid_genotypes:
        parameters = dict(init_ind_given_genotype=gt)
        ind = al.systems.cfggpst.init_individual.given_genotype(grammar, parameters)
        check_individual(ind)
        if isinstance(gt, al._grammar.data_structures.DerivationTree):
            assert ind.genotype.data == gt.to_tuple()
        elif isinstance(gt, str):
            assert str(ind.genotype) == gt
        else:
            assert ind.genotype.data == gt
    # Parameter: init_ind_given_genotype not valid
    invalid_genotypes = [None, False, True, "", "abc", 3, 3.14]
    for gt in invalid_genotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_genotype=gt)
            al.systems.cfggpst.init_individual.given_genotype(grammar, parameters)
    # Parameter: init_ind_given_genotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_individual.given_genotype(grammar)

    # Method: given_derivation_tree
    valid_derivation_trees = [
        grammar.generate_derivation_tree("ge", "[1, 2, 3, 4]"),
        grammar.generate_derivation_tree("ge", [4, 3, 2, 1]),
    ]
    for dt in valid_derivation_trees:
        parameters = dict(init_ind_given_derivation_tree=dt)
        ind = al.systems.cfggpst.init_individual.given_derivation_tree(
            grammar, parameters
        )
        check_individual(ind)
        ind_dt = ind.details["derivation_tree"]
        assert isinstance(ind_dt, al._grammar.data_structures.DerivationTree)
        assert ind_dt == dt
    # Parameter: init_ind_given_derivation_tree not valid
    invalid_derivation_trees = [None, False, True, "", "abc", 3, 3.14, (0, 1, 2)]
    for dt in invalid_derivation_trees:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_derivation_tree=dt)
            al.systems.cfggpst.init_individual.given_derivation_tree(
                grammar, parameters
            )
    # Parameter: init_ind_given_derivation_tree not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_individual.given_derivation_tree(grammar)

    # Method: given_phenotype
    valid_phenotypes = ["ab", "cd", "cdc"]
    for phe in valid_phenotypes:
        parameters = dict(init_ind_given_phenotype=phe)
        ind = al.systems.cfggpst.init_individual.given_phenotype(grammar, parameters)
        check_individual(ind)
        assert ind.phenotype == phe
    # Parameter: init_ind_given_phenotype not valid
    invalid_phenotypes = [None, False, True, "", "abc", 3, 3.14, (0, 1, 2)]
    for phe in invalid_phenotypes:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_ind_given_phenotype=phe)
            al.systems.cfggpst.init_individual.given_phenotype(grammar, parameters)
    # Parameter: init_ind_given_phenotype not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_individual.given_phenotype(grammar)

    # Method: random_genotype
    for _ in range(num_repetitions):
        ind = al.systems.cfggpst.init_individual.random_genotype(grammar)
        check_individual(ind)

    # Method: gp_grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.cfggpst.init_individual.gp_grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_grow_max_depth
    al.systems.cfggpst.init_individual.gp_grow_tree(
        grammar, dict(init_ind_gp_grow_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.cfggpst.init_individual.gp_grow_tree(
            grammar, dict(init_ind_gp_grow_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_individual.gp_grow_tree(
            grammar, dict(init_ind_gp_grow_max_depth="nonsense")
        )

    # Method: gp_full_tree
    for _ in range(num_repetitions):
        ind = al.systems.cfggpst.init_individual.gp_full_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_full_max_depth
    al.systems.cfggpst.init_individual.gp_full_tree(
        grammar, dict(init_ind_gp_full_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.cfggpst.init_individual.gp_full_tree(
            grammar, dict(init_ind_gp_full_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_individual.gp_full_tree(
            grammar, dict(init_ind_gp_full_max_depth="nonsense")
        )

    # Method: pi_grow_tree
    for _ in range(num_repetitions):
        ind = al.systems.cfggpst.init_individual.pi_grow_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_gp_grow_max_depth
    al.systems.cfggpst.init_individual.pi_grow_tree(
        grammar, dict(init_ind_pi_grow_max_depth=0)
    )
    for _ in range(num_repetitions):
        al.systems.cfggpst.init_individual.pi_grow_tree(
            grammar, dict(init_ind_pi_grow_max_depth=5)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_individual.pi_grow_tree(
            grammar, dict(init_ind_pi_grow_max_depth="nonsense")
        )

    # Method: ptc2
    for _ in range(num_repetitions):
        ind = al.systems.cfggpst.init_individual.ptc2_tree(grammar)
        check_individual(ind)
    # Parameter: init_ind_ptc2_max_expansions
    al.systems.cfggpst.init_individual.ptc2_tree(
        grammar, dict(init_ind_ptc2_max_expansions=0)
    )
    for _ in range(num_repetitions):
        al.systems.cfggpst.init_individual.ptc2_tree(
            grammar, dict(init_ind_ptc2_max_expansions=100)
        )
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_individual.ptc2_tree(
            grammar, dict(init_ind_ptc2_max_expansions="nonsense")
        )


def test_initialize_population():
    # Grammar
    bnf_text = """
    <S> ::= <A> | <B> | <C><C>
    <A> ::= a b | b a
    <B> ::= c d | d c | c d c
    <C> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Method: given_genotypes
    dt1 = grammar.parse_string("ab")
    dt2 = grammar.parse_string("dc")
    valid_genotype_collections = [
        [
            dt1,
        ],
        [
            ((0, 2, 6, 7, 6), (1, 3, 0, 0, 0)),
        ],
        [
            dt1,
            dt2,
        ],
        [
            "((0,1,5,4),(1,2,0,0))",
            ((0, 2, 6, 7, 6), (1, 3, 0, 0, 0)),
        ],
        [
            dt1,
            "((0,2,6,7),(1,2,0,0))",
        ],
        [
            ((0, 2, 7, 6), (1, 2, 0, 0)),
            dt2,
        ],
    ]
    for gts in valid_genotype_collections:
        parameters = dict(init_pop_given_genotypes=gts)
        pop = al.systems.cfggpst.init_population.given_genotypes(grammar, parameters)
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
            al.systems.cfggpst.init_population.given_genotypes(grammar, parameters)
    # Parameter: init_pop_given_genotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_population.given_genotypes(grammar)

    # Method: given_derivation_trees
    valid_derivation_tree_collections = [
        [
            grammar.generate_derivation_tree("ge", [0, 7, 11]),
            grammar.generate_derivation_tree("ge", "[1, 2, 3, 4]"),
        ],
    ]
    for dts in valid_derivation_tree_collections:
        parameters = dict(init_pop_given_derivation_trees=dts)
        pop = al.systems.cfggpst.init_population.given_derivation_trees(
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
        [3.14],
        [[0, 1], 1],
        [1, "[0, 1]"],
    ]
    for dts in invalid_derivation_tree_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_derivation_trees=dts)
            al.systems.cfggpst.init_population.given_derivation_trees(
                grammar, parameters
            )
    # Parameter: init_pop_given_derivation_trees not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_population.given_derivation_trees(grammar)

    # Method: given_phenotypes
    valid_phenotype_collections = [
        ["ab", "cd"],
        ["ab", "cd", "cdc", "ba", "dc"],
    ]
    for pts in valid_phenotype_collections:
        parameters = dict(init_pop_given_phenotypes=pts)
        pop = al.systems.cfggpst.init_population.given_phenotypes(grammar, parameters)
        check_population(pop)
        assert len(pop) == len(pts)
    # Parameter: init_pop_given_phenotypes not valid
    invalid_phenotype_collections = [
        None,
        [],
        [None],
        ["aba", "cd"],
        ["ab", "dcd"],
    ]
    for pts in invalid_phenotype_collections:
        with pytest.raises(al.exceptions.InitializationError):
            parameters = dict(init_pop_given_phenotypes=pts)
            al.systems.cfggpst.init_population.given_derivation_trees(
                grammar, parameters
            )
    # Parameter: init_pop_given_phenotypes not provided
    with pytest.raises(al.exceptions.InitializationError):
        al.systems.cfggpst.init_population.given_phenotypes(grammar)

    # Method: random_genotypes
    n = 10
    for _ in range(n):
        pop = al.systems.cfggpst.init_population.random_genotypes(grammar)
        check_population(pop)
        assert len(pop) == al.systems.cfggpst.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.cfggpst.init_population.random_genotypes(
                grammar, parameters
            )
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
            pop = al.systems.cfggpst.init_population.random_genotypes(grammar, params)
            check_population(pop)
            assert len(pop) == 10
            if unique_gen or unique_phe:
                params["init_pop_size"] = 1000
                with pytest.raises(al.exceptions.InitializationError):
                    al.systems.cfggpst.init_population.random_genotypes(grammar, params)
    # Parameter: init_pop_unique_max_tries
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_unique_max_tries=0)
        al.systems.cfggpst.init_population.random_genotypes(grammar, parameters)

    # Method: gp_rhh (=GP's ramped half and half)
    for _ in range(n):
        pop = al.systems.cfggpst.init_population.gp_rhh(grammar)
        check_population(pop)
        assert len(pop) == al.systems.cfggpst.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.cfggpst.init_population.gp_rhh(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_gp_rhh_start_depth, init_pop_gp_rhh_end_depth
    parameters = dict(init_pop_gp_rhh_start_depth=3, init_pop_gp_rhh_end_depth=4)
    pop = al.systems.cfggpst.init_population.gp_rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_gp_rhh_start_depth=5, init_pop_gp_rhh_end_depth=3)
        pop = al.systems.cfggpst.init_population.gp_rhh(grammar, parameters)

    # Method: pi_rhh (=position-independent ramped half and half)
    for _ in range(n):
        pop = al.systems.cfggpst.init_population.pi_rhh(grammar)
        check_population(pop)
        assert len(pop) == al.systems.cfggpst.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.cfggpst.init_population.pi_rhh(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_pi_rhh_start_depth, init_pop_pi_rhh_end_depth
    parameters = dict(init_pop_pi_rhh_start_depth=3, init_pop_pi_rhh_end_depth=4)
    pop = al.systems.cfggpst.init_population.pi_rhh(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(init_pop_pi_rhh_start_depth=5, init_pop_pi_rhh_end_depth=3)
        pop = al.systems.cfggpst.init_population.pi_rhh(grammar, parameters)

    # Method: ptc2 (=probabilistic tree creation 2)
    for _ in range(n):
        pop = al.systems.cfggpst.init_population.ptc2(grammar)
        check_population(pop)
        assert len(pop) == al.systems.cfggpst.default_parameters.init_pop_size
        # Parameters: init_pop_size
        for chosen_pop_size in (1, 2, 5, 12, 13, 22, 27):
            parameters = dict(init_pop_size=chosen_pop_size)
            pop = al.systems.cfggpst.init_population.ptc2(grammar, parameters)
            check_population(pop)
            assert len(pop) == chosen_pop_size
    # Parameters: init_pop_ptc2_start_expansions, init_pop_ptc2_end_expansions
    parameters = dict(
        init_pop_ptc2_start_expansions=10, init_pop_ptc2_end_expansions=50
    )
    pop = al.systems.cfggpst.init_population.ptc2(grammar, parameters)
    check_population(pop)
    with pytest.raises(al.exceptions.InitializationError):
        parameters = dict(
            init_pop_ptc2_start_expansions=50, init_pop_ptc2_end_expansions=10
        )
        pop = al.systems.cfggpst.init_population.ptc2(grammar, parameters)


# Mutation


def test_mutation1():
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
    # 1) derivation tree, 2) serialized derivation tree 3) string thereof, 4) Genotype class
    dt = grammar.parse_string("a+")
    s = dt.to_tuple()
    genotypes = [
        dt,
        s,
        str(s),
        al.systems.cfggpst.representation.Genotype(dt),
        al.systems.cfggpst.representation.Genotype(s),
        al.systems.cfggpst.representation.Genotype(str(s)),
    ]

    # Mutation
    for gt in genotypes:
        # Method: subtree_replacement
        strings = set()
        for _ in range(2000):
            # Without parameters (=using defaults)
            gt_mut = al.systems.cfggpst.mutation.subtree_replacement(grammar, gt)
            assert isinstance(gt_mut, al.systems.cfggpst.representation.Genotype)
            string = al.systems.cfggpst.mapping.forward(grammar, gt_mut)
            assert isinstance(string, str)
            strings.add(string)
        assert len(strings) == len(language)


def test_mutation2():
    # Grammar
    bnf_text = """
    <S> ::= <A><B><A>
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
    assert len(language) == 4 * 4 * 4

    # Genotypes of four types:
    # 1) derivation tree, 2) serialized derivation tree 3) string thereof, 4) Genotype class
    dt = grammar.parse_string("a+a")

    # Mutation
    # Method: subtree_replacement
    strings = set()
    for _ in range(20000):
        # Without parameters (=using defaults)
        gt_mut = al.systems.cfggpst.mutation.subtree_replacement(grammar, dt)
        assert isinstance(gt_mut, al.systems.cfggpst.representation.Genotype)
        string = al.systems.cfggpst.mapping.forward(grammar, gt_mut)
        assert isinstance(string, str)
        strings.add(string)
    assert len(strings) == len(language)


def test_mutation3():
    # Grammar
    bnf_text = """
    <prog> ::= <nibble> | <crumb> <crumb> | <bit> <crumb> <bit>
    <nibble> ::= <crumb> <bit> <bit> | <bit> <crumb> <bit> | <bit> <bit> <bit> <bit>
    <crumb> ::= <bit> <bit> | <bit> 0 | <bit> 1 | 0 <bit> | 1 <bit> | 0 0 | 0 1 | 1 0 | 1 1
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Language
    language = grammar.generate_language()
    assert len(language) == 2**4

    # Genotypes of four types:
    # 1) derivation tree, 2) serialized derivation tree 3) string thereof, 4) Genotype class
    dt = grammar.parse_string("0000")
    s = dt.to_tuple()
    genotypes = [
        dt,
        s,
        str(s),
        al.systems.cfggpst.representation.Genotype(dt),
        al.systems.cfggpst.representation.Genotype(s),
        al.systems.cfggpst.representation.Genotype(str(s)),
    ]

    # Mutation
    for gt in genotypes:
        # Method: subtree_replacement
        strings = set()
        for _ in range(2000):
            # Without parameters (=using defaults)
            gt_mut = al.systems.cfggpst.mutation.subtree_replacement(grammar, gt)
            string = al.systems.cfggpst.mapping.forward(grammar, gt_mut)
            strings.add(string)
        assert len(strings) == len(language)


def test_mutation4():
    # Grammar
    bnf_texts = [
        """
        <S> ::= AA | AG | AT | AC | GA | GG | GT | GC | TA | TG | TT | TC | CA | CG | CT | CC
        """,
        """
        <hello> ::= <grammar>
        <grammar> ::= AA | AG | AT | AC | GA | GG | GT | GC | TA | TG | TT | TC | CA | CG | CT | CC
        """,
        """
        <Pair> ::= <Base> <Base>
        <Base> ::= A | G | C | T
        """,
        """
        <Pair> ::= <Base1> <Base2>
        <Base1> ::= <X> | G | <Y> | T
        <Base2> ::= T | <Y> | G | <X>
        <X> ::= A
        <Y> ::= C
        """,
        """
        <Pair> ::= <SomeSubset> | GG | GT | GC | <OtherSubsetWithOverlap> | CA | CG | CT | CC
        <SomeSubset> ::= AA | AG | AT | AC | GA
        <OtherSubsetWithOverlap> ::= GC | TA | TG | TT | TC | CA
        """,
        """
        <S> ::= <X> | <Y> | <Z>C
        <X> ::= AA | <XX> | GA | GG | GT | GC
        <XX> ::= AG | AT | AC
        <Y> ::= TA | TG | TT | TC | CA | CG | CT
        <Z> ::= <ZZ>
        <ZZ> ::= C
        """,
        """
        <S> ::= A<X> | G<Y> | <Z>
        <X> ::= A | G | C | T
        <Y> ::= T | C | G | A
        <Z> ::= TA | TG | TT | TC | CA | CG | CT | CC
        """,
        """
        <Hierarchy> ::= <X> | <Y>
        <X> ::= <X1> | <X2>
        <X1> ::= <X11> | <X12>
        <X2> ::= <X21> | <X22>
        <X11> ::= AA | AG
        <X12> ::= AT | AC
        <X21> ::= GA | GG
        <X22> ::= GT | GC
        <Y> ::= <Y2> | <Y3>
        <Y2> ::= TA | TG | TT | TC
        <Y3> ::= CA | CG | CT | CC
        """,
        """
        <Empty> ::= <X><E> | <E><Y>
        <X> ::= <X1> | <X2>
        <X1> ::= <X11><E> | <X12>
        <X2> ::= <X21> | <E><X22>
        <X11> ::= AA | AG<E>
        <X12> ::= AT | AC
        <X21> ::= GA<E> | GG
        <X22> ::= GT | GC
        <Y> ::= <Y2> | <E><Y3>
        <Y2> ::= TA | TG<E> | TT | TC
        <Y3> ::= CA | CG | C<E>T | CC
        <E> ::=
        """,
    ]

    language = set(
        [
            "AA",
            "AG",
            "AT",
            "AC",
            "GA",
            "GG",
            "GT",
            "GC",
            "TA",
            "TG",
            "TT",
            "TC",
            "CA",
            "CG",
            "CT",
            "CC",
        ]
    )

    for bnf in bnf_texts:
        # Grammar
        grammar = al.Grammar(bnf_text=bnf)

        # Genotype
        gt = grammar.parse_string("AA")

        # Mutation
        # Method: subtree_replacement
        strings = set()
        for _ in range(1000):
            # Without parameters (=using defaults)
            gt_mut = al.systems.cfggpst.mutation.subtree_replacement(grammar, gt)
            string = al.systems.cfggpst.mapping.forward(grammar, gt_mut)
            strings.add(string)
        assert strings == language


def test_mutation_parameter_max_nodes():
    # Grammar
    bnf_text = """
    <S> ::= <S> 1 | <S> 0 |
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotype
    gt = al.systems.cfggpst.representation.Genotype(grammar.parse_string("000"))

    # Mutations
    strings = set()
    parameters = dict(max_nodes=5)
    for _ in range(30000):
        gt_mut = al.systems.cfggpst.mutation.subtree_replacement(
            grammar, gt, parameters=parameters
        )
        phe_mut = al.systems.cfggpst.mapping.forward(grammar, gt_mut)
        strings.add(phe_mut)
    assert len(strings) == 2**4 - 1
    assert all(len(s) <= 3 for s in strings)

    strings = set()
    parameters = dict(max_nodes=6)
    for _ in range(30000):
        gt_mut = al.systems.cfggpst.mutation.subtree_replacement(
            grammar, gt, parameters=parameters
        )
        phe_mut = al.systems.cfggpst.mapping.forward(grammar, gt_mut)
        strings.add(phe_mut)
    assert len(strings) == 2**5 - 1
    assert all(len(s) <= 4 for s in strings)


# Crossover


def test_crossover_api():
    # Grammar
    bnf_text = """
    <trytes> ::= <tryte> | <tryte> <tryte> | <tryte> <tryte> <tryte>
    <tryte> ::= <trit> <trit> <trit> <trit> <trit> <trit> <trit> <trit>
    <trit> ::= 0 | 1 | 2
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes of three types:
    # 1) derivation tree, 2) serialized derivation tree 3) string thereof
    dt1 = grammar.parse_string("00000000")
    genotypes = (
        dt1,
        (
            (0, 1, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4),
            (1, 8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
        ),
        "((0,1,2,5,2,5,2,5,2,5,2,5,2,5,2,5,2,5),(1,8,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0))",
    )

    # Crossover
    def perform_checks(gt1, gt2, gt3, gt4):
        if not isinstance(gt1, al.systems.cfggpst.representation.Genotype):
            gt1 = al.systems.cfggpst.representation.Genotype(copy.copy(gt1))
        if not isinstance(gt2, al.systems.cfggpst.representation.Genotype):
            gt2 = al.systems.cfggpst.representation.Genotype(copy.copy(gt2))
        assert gt1 == gt1
        assert gt1 != gt2
        assert gt3 != gt4
        # Case 1: Subtrees of the root nodes were swapped
        if gt4 == gt1 or gt3 == gt2:
            assert gt3 == gt2
            assert gt4 == gt1
        # Case 2: Subtrees of other nodes were swapped
        else:
            assert gt3 != gt1
            assert gt3 != gt2
            assert gt4 != gt2

    method = al.systems.cfggpst.crossover.subtree_exchange
    params = dict()
    for _ in range(50):
        for two_genotypes in itertools.combinations(genotypes, 2):
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


def test_crossover_minimal_example():
    # Grammar
    bnf_text = """
    <S> ::= <A> <B> <C>
    <A> ::= 1 | <AA>
    <AA> ::= 2
    <B> ::= <BB> | <BBB>
    <BB> ::= x
    <BBB> ::= x | y | <BBBB>
    <BBBB> ::= x | y | y | x
    <C> ::= <CC> | <CCC> | -
    <CC> ::= +
    <CCC> ::= - | <CCCC>
    <CCCC> ::= +
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes
    gt1 = al.systems.cfggpst.representation.Genotype(grammar.parse_string("1x+"))
    gt2 = al.systems.cfggpst.representation.Genotype(grammar.parse_string("2y-"))

    # Crossover
    results = set()
    for _ in range(2000):
        gt3, gt4 = al.systems.cfggpst.crossover.subtree_exchange(
            grammar, copy.copy(gt1), copy.copy(gt2)
        )
        s3 = al.systems.cfggpst.mapping.forward(grammar, gt3.data)
        s4 = al.systems.cfggpst.mapping.forward(grammar, gt4.data)
        results.add(s3)
        results.add(s4)
    assert results == set(
        [
            "1x+",
            "2y-",  # swap S
            "2x+",
            "1y-",  # swap A
            "1y+",
            "2x-",  # swap B
            "1x-",
            "2y+",  # swap C
        ]
    )


def test_crossover_parameter_max_nodes():
    bnf_text = """
    <S> ::= 1 <S> | <S> 0 |
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypes
    gt1 = al.systems.cfggpst.representation.Genotype(grammar.parse_string("1110"))
    gt2 = al.systems.cfggpst.representation.Genotype(grammar.parse_string("1000"))

    def generate_all_combinations(max_nodes):
        parameters = dict(max_nodes=max_nodes)
        strings = set()
        for _ in range(2000):
            gt3, gt4 = al.systems.cfggpst.crossover.subtree_exchange(
                grammar, copy.copy(gt1), copy.copy(gt2), parameters
            )
            s3 = al.systems.cfggpst.mapping.forward(grammar, gt3.data)
            s4 = al.systems.cfggpst.mapping.forward(grammar, gt4.data)
            strings.add(s3)
            strings.add(s4)
        return strings

    assert generate_all_combinations(max_nodes=9) == set(["1110", "1000"])
    assert generate_all_combinations(max_nodes=10) == set(["1000", "1100", "1110"])
    assert generate_all_combinations(max_nodes=12) == set(
        [
            "000",
            "100",
            "1000",
            "10000",
            "110",
            "1100",
            "11000",
            "111",
            "1110",
            "11100",
            "11110",
        ]
    )


def test_crossover_fails():
    # Grammar
    bnf_text = "<bit> ::= 1 | 0"
    grammar = al.Grammar(bnf_text=bnf_text)

    # Crossover
    method = al.systems.cfggpst.crossover.subtree_exchange
    # - invalid genotype types
    gt_valid = grammar.parse_string("1")
    for gt_invalid in [None, False, True, "", 0, 1, 3.14, "101"]:
        method(grammar, gt_valid, gt_valid)
        with pytest.raises(al.exceptions.GenotypeError):
            method(grammar, gt_valid, gt_invalid)
        with pytest.raises(al.exceptions.GenotypeError):
            method(grammar, gt_invalid, gt_valid)
        with pytest.raises(al.exceptions.GenotypeError):
            method(grammar, gt_invalid, gt_invalid)


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

    # Genotypes of three types:
    # 1) derivation tree, 2) serialized dt, 3) string thereof, 4) Genotype class
    dt1 = gr.parse_string("3")
    dt2 = gr.parse_string("b")
    dt3 = gr.parse_string("Z")
    genotypes = (
        dt1,
        dt2.to_tuple(),
        str(dt2.to_tuple()),
        al.systems.cfggpst.representation.Genotype(dt3),
    )

    # Neighborhood
    method = al.systems.cfggpst.neighborhood.subtree_replacement
    for gt in genotypes:
        phe = al.systems.cfggpst.mapping.forward(gr, gt)
        # Default
        nh1 = method(gr, gt)
        nh2 = method(gr, genotype=gt)
        nh3 = method(grammar=gr, genotype=gt)
        nh4 = method(gr, gt, dict())
        nh5 = method(gr, gt, parameters=dict())
        nh6 = method(gr, genotype=gt, parameters=dict())
        nh7 = method(grammar=gr, genotype=gt, parameters=dict())
        assert nh1 == nh2 == nh3 == nh4 == nh5 == nh6 == nh7
        for new_gt in nh1:
            check_genotype(new_gt)
            new_phe = al.systems.cfggpst.mapping.forward(gr, new_gt)
            assert new_phe != phe


@pytest.mark.parametrize(
    "bnf, genotype, phenotype",
    [
        (shared.BNF5, ((0, 1), (1, 0)), "1"),
        (shared.BNF5, ((0, 2), (1, 0)), "2"),
        (shared.BNF5, ((0, 3), (1, 0)), "3"),
        (shared.BNF5, ((0, 4), (1, 0)), "4"),
        (shared.BNF5, ((0, 5), (1, 0)), "5"),
        (shared.BNF6, ((0, 1, 4), (1, 1, 0)), "1"),
        (shared.BNF6, ((0, 1, 5), (1, 1, 0)), "2"),
        (shared.BNF6, ((0, 2, 6), (1, 1, 0)), "3"),
        (shared.BNF6, ((0, 2, 7), (1, 1, 0)), "4"),
        (shared.BNF6, ((0, 3), (1, 0)), "5"),
        (shared.BNF7, ((0, 7, 1, 9, 3, 13), (2, 0, 2, 0, 1, 0)), "ac1"),
        (shared.BNF7, ((0, 8, 2, 12, 6, 20), (2, 0, 2, 0, 1, 0)), "bf8"),
        (shared.BNF7, ((0, 7, 1, 10, 4, 16), (2, 0, 2, 0, 1, 0)), "ad4"),
        (shared.BNF9, ((0, 2, 4, 8), (1, 1, 1, 0)), "a"),
        (shared.BNF9, ((0, 2, 4, 9, 4, 10), (1, 2, 1, 0, 1, 0)), "bc"),
        (shared.BNF9, ((0, 1, 3, 6, 3, 6), (1, 2, 1, 0, 1, 0)), "22"),
        (shared.BNF9, ((0, 1, 3, 7), (1, 1, 1, 0)), "3"),
        (shared.BNF9, ((0, 1, 3, 5, 3, 7), (1, 2, 1, 0, 1, 0)), "13"),
    ],
)
def test_neighborhood_reachability_in_finite_languages(bnf, genotype, phenotype):
    # Grammar
    grammar = al.Grammar(bnf_text=bnf)

    # Genotype
    gt = al.systems.cfggpst.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.cfggpst.mapping.forward(grammar, gt)
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
                nbrs = al.systems.cfggpst.neighborhood.subtree_replacement(
                    grammar, gen, param
                )
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                try:
                    phe = al.systems.cfggpst.mapping.forward(grammar, gen, param)
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
        (
            shared.BNF10,
            ((0, 1, 6, 4), (2, 1, 0, 0)),
            "1x",
            ("2x", "3x", "4y", "5y", "6y", "7"),
        ),
        (shared.BNF11, ((0, 1, 4), (1, 1, 0)), "1", ("2", "3", "4", "22", "33", "44")),
        (
            shared.BNF12,
            ((0, 4, 1, 6, 4), (3, 0, 1, 0, 0)),
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
    gt = al.systems.cfggpst.representation.Genotype(genotype)

    # Phenotype
    phe = al.systems.cfggpst.mapping.forward(grammar, gt)
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
                nbrs = al.systems.cfggpst.neighborhood.subtree_replacement(
                    grammar, gen, param
                )
                if "neighborhood_max_size" in param:
                    assert len(nbrs) <= param["neighborhood_max_size"]
                genotypes_seen.add(gen)
                genotypes_nbrs = genotypes_nbrs.union(nbrs)
                phe = al.systems.cfggpst.mapping.forward(grammar, gen, param)
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
    gt = al.systems.cfggpst.mapping.reverse(grammar, dt)

    # Neighborhood in different distances when changing only terminals
    # - distance 1
    parameters = dict(neighborhood_only_terminals=True)
    nbrs_gt = al.systems.cfggpst.neighborhood.subtree_replacement(
        grammar, gt, parameters
    )
    nbrs = [al.systems.cfggpst.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1a1b", "1a2a", "1b1a", "2a1a"}

    # - distance 2
    parameters = dict(neighborhood_distance=2, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.cfggpst.neighborhood.subtree_replacement(
        grammar, gt, parameters
    )
    nbrs = [al.systems.cfggpst.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1a2b", "1b1b", "1b2a", "2a1b", "2a2a", "2b1a"}

    # - distance 3
    parameters = dict(neighborhood_distance=3, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.cfggpst.neighborhood.subtree_replacement(
        grammar, gt, parameters
    )
    nbrs = [al.systems.cfggpst.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"1b2b", "2a2b", "2b1b", "2b2a"}

    # - distance 4
    parameters = dict(neighborhood_distance=4, neighborhood_only_terminals=True)
    nbrs_gt = al.systems.cfggpst.neighborhood.subtree_replacement(
        grammar, gt, parameters
    )
    nbrs = [al.systems.cfggpst.mapping.forward(grammar, gt) for gt in nbrs_gt]
    assert set(nbrs) == {"2b2b"}

    # - distance 5 and greater
    for dist in range(5, 20):
        parameters = dict(neighborhood_distance=dist, neighborhood_only_terminals=True)
        nbrs_gt = al.systems.cfggpst.neighborhood.subtree_replacement(
            grammar, gt, parameters
        )
        nbrs = [al.systems.cfggpst.mapping.forward(grammar, gt) for gt in nbrs_gt]
        assert nbrs == []


@pytest.mark.parametrize(
    "bnf, gt, phe, phe_neighbors",
    [
        (shared.BNF1, ((0, 1), (1, 0)), "0", ("1", "2")),
        (shared.BNF1, ((0, 2), (1, 0)), "1", ("0", "2")),
        (shared.BNF1, ((0, 3), (1, 0)), "2", ("0", "1")),
        (
            shared.BNF2,
            ((0, 1, 3, 2, 6), (2, 1, 0, 1, 0)),
            "0a",
            ("1a", "2a", "0b", "0c"),
        ),
        (
            shared.BNF2,
            ((0, 1, 4, 2, 7), (2, 1, 0, 1, 0)),
            "1b",
            ("0b", "2b", "1a", "1c"),
        ),
        (
            shared.BNF2,
            ((0, 1, 5, 2, 8), (2, 1, 0, 1, 0)),
            "2c",
            ("0c", "1c", "2a", "2b"),
        ),
        (
            shared.BNF2,
            ((0, 1, 3, 2, 6), (2, 1, 0, 1, 0)),
            "0a",
            ("1a", "2a", "0b", "0c"),
        ),
        (
            shared.BNF2,
            ((0, 1, 3, 2, 7), (2, 1, 0, 1, 0)),
            "0b",
            ("1b", "2b", "0a", "0c"),
        ),
        (
            shared.BNF2,
            ((0, 1, 4, 2, 8), (2, 1, 0, 1, 0)),
            "1c",
            ("0c", "2c", "1a", "1b"),
        ),
        (
            shared.BNF3,
            ((0, 1, 2, 7, 3, 4, 5, 6, 10), (2, 1, 1, 0, 1, 1, 1, 1, 0)),
            "0a",
            ("1a", "2a", "0b", "0c"),
        ),
        (
            shared.BNF3,
            ((0, 1, 2, 8, 3, 4, 5, 6, 11), (2, 1, 1, 0, 1, 1, 1, 1, 0)),
            "1b",
            ("0b", "2b", "1a", "1c"),
        ),
        (
            shared.BNF3,
            ((0, 1, 2, 9, 3, 4, 5, 6, 12), (2, 1, 1, 0, 1, 1, 1, 1, 0)),
            "2c",
            ("0c", "1c", "2a", "2b"),
        ),
        (
            shared.BNF3,
            ((0, 1, 2, 7, 3, 4, 5, 6, 10), (2, 1, 1, 0, 1, 1, 1, 1, 0)),
            "0a",
            ("1a", "2a", "0b", "0c"),
        ),
        (
            shared.BNF3,
            ((0, 1, 2, 7, 3, 4, 5, 6, 11), (2, 1, 1, 0, 1, 1, 1, 1, 0)),
            "0b",
            ("1b", "2b", "0a", "0c"),
        ),
        (
            shared.BNF3,
            ((0, 1, 2, 8, 3, 4, 5, 6, 12), (2, 1, 1, 0, 1, 1, 1, 1, 0)),
            "1c",
            ("0c", "2c", "1a", "1b"),
        ),
        (
            shared.BNF4,
            (
                (0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2),
                (8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            ),
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
            (
                (0, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
                (8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            ),
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
            (
                (0, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3),
                (8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            ),
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
            (
                (0, 1, 2, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 2, 1, 3),
                (8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            ),
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
    assert phe == al.systems.cfggpst.mapping.forward(gr, gt)

    # Neighborhood
    nbrs = al.systems.cfggpst.neighborhood.subtree_replacement(gr, gt)
    nbrs_phe = [al.systems.cfggpst.mapping.forward(gr, nbr_gt) for nbr_gt in nbrs]
    assert set(nbrs_phe) == set(phe_neighbors)

    # Parameter: neighborhood_max_size
    parameters = dict(neighborhood_max_size=None)
    nbrs = al.systems.cfggpst.neighborhood.subtree_replacement(gr, gt, parameters)
    nbrs_phe = [al.systems.cfggpst.mapping.forward(gr, nbr_gt) for nbr_gt in nbrs]
    assert set(nbrs_phe) == set(phe_neighbors)

    for max_size in range(1, 5):
        parameters = dict(neighborhood_max_size=max_size)
        nbrs_phe = set()
        for _ in range(100):
            nbrs = al.systems.cfggpst.neighborhood.subtree_replacement(
                gr, gt, parameters
            )
            assert len(nbrs) <= max_size
            for nbr_gt in nbrs:
                nbr_phe = al.systems.cfggpst.mapping.forward(gr, nbr_gt)
                assert nbr_phe in phe_neighbors
                nbrs_phe.add(nbr_phe)
        assert nbrs_phe == set(phe_neighbors)


@pytest.mark.parametrize(
    "bnf, gt, phe, phe_neighbors",
    [
        (shared.BNF5, ((0, 1), (1, 0)), "1", ("2", "3", "4", "5")),
        (shared.BNF5, ((0, 2), (1, 0)), "2", ("1", "3", "4", "5")),
        (shared.BNF5, ((0, 3), (1, 0)), "3", ("1", "2", "4", "5")),
        (shared.BNF5, ((0, 4), (1, 0)), "4", ("1", "2", "3", "5")),
        (shared.BNF5, ((0, 5), (1, 0)), "5", ("1", "2", "3", "4")),
        (shared.BNF6, ((0, 1, 4), (1, 1, 0)), "1", ("2", "5")),
        (shared.BNF6, ((0, 1, 5), (1, 1, 0)), "2", ("1", "5")),
        (shared.BNF6, ((0, 2, 6), (1, 1, 0)), "3", ("4", "5")),
        (shared.BNF6, ((0, 2, 7), (1, 1, 0)), "4", ("3", "5")),
        (shared.BNF6, ((0, 3), (1, 0)), "5", ()),
        (
            shared.BNF7,
            ((0, 7, 1, 9, 3, 13), (2, 0, 2, 0, 1, 0)),
            "ac1",
            ("be5", "ad3", "ac2"),
        ),
        (
            shared.BNF7,
            ((0, 8, 2, 12, 6, 20), (2, 0, 2, 0, 1, 0)),
            "bf8",
            ("ac1", "be5", "bf7"),
        ),  # ge differs
        (
            shared.BNF7,
            ((0, 7, 1, 10, 4, 16), (2, 0, 2, 0, 1, 0)),
            "ad4",
            ("be5", "ac1", "ad3"),
        ),  # ge differs
        (shared.BNF8, ((0, 1, 6), (1, 1, 0)), "t", ("a0g", "1g", "a")),  # ge differs
        (
            shared.BNF8,
            ((0, 1, 5, 3, 2, 8), (3, 1, 0, 0, 1, 0)),
            "a0c",
            ("1g", "t0c", "a0g"),
        ),
    ],
)
def test_neighborhood_parameter_only_terminals(bnf, gt, phe, phe_neighbors):
    # Grammar
    gr = al.Grammar(bnf_text=bnf)

    # Forward mapping
    parameters = dict(neighborhood_only_terminals=True, max_wraps=2)
    assert phe == al.systems.cfggpst.mapping.forward(
        gr, gt, parameters, raise_errors=False
    )

    # Neighborhood
    nbrs = al.systems.cfggpst.neighborhood.subtree_replacement(gr, gt, parameters)
    nbrs_phe = [
        al.systems.cfggpst.mapping.forward(gr, nbr_gt, parameters, raise_errors=False)
        for nbr_gt in nbrs
    ]
    assert set(nbrs_phe) == set(phe_neighbors)


def test_neighborhood_internal_errors():
    # Grammar
    bnf_text = """
    <S> ::= <A> <B>
    <A> ::= 1 | 2
    <B> ::= 3 | 4
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Derivation tree
    dt = grammar.parse_string("13")

    # Neighborhood errors
    # - Missing NT error
    with pytest.raises(al.exceptions.MappingError):
        dt_defect = dt.copy()
        dt_defect.root_node.symbol.text = "Nonsense"
        al.systems.cfggpst.neighborhood._get_choices_per_position(
            grammar, dt_defect, only_terminals=False
        )

    # - Missing RHS error
    with pytest.raises(al.exceptions.MappingError):
        dt_defect = dt.copy()
        dt_defect.root_node.children[0].children[0].symbol.text = "Nonsense"
        al.systems.cfggpst.neighborhood._get_choices_per_position(
            grammar, dt_defect, only_terminals=False
        )


# Mapping


def test_mapping_forward_api():
    # Grammar
    bnf_text = """
    <bytes> ::= <byte> | <byte> <byte> | <byte> <byte> <byte>
    <byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Genotypic data of six types:
    # 1) derivation tree 2) tuple serialization of a derivation tree 3) string thereof
    # 4) list instead of tuple 5) string thereof 6) Genotype object
    dt = grammar.parse_string("11110000")
    tup = (
        (0, 1, 2, 4, 2, 4, 2, 4, 2, 4, 2, 3, 2, 3, 2, 3, 2, 3),
        (1, 8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
    )
    data_variants = (
        dt,
        tup,
        str(tup),
        list(tup),
        str(list(tup)),
        al.systems.cfggpst.representation.Genotype(dt),
    )

    # Forward mapping
    kwargs = dict(
        verbose=False,
        raise_errors=False,
        return_derivation_tree=False,
    )
    for data in data_variants:
        for vb in (True, False):
            kwargs["verbose"] = vb
            for me in (None, 3, 10, 15, 300):
                parameters = dict(max_expansions=me)

                # Method of Grammar class
                string1 = grammar.generate_string("cfggpst", data, parameters, **kwargs)
                string2 = grammar.generate_string(
                    "cfggpst", data, parameters=parameters, **kwargs
                )
                string3 = grammar.generate_string(
                    method="cfggpst", genotype=data, parameters=parameters, **kwargs
                )
                assert string1
                assert string1 == string2 == string3

                # Method of DerivationTree class
                dt1 = grammar.generate_derivation_tree(
                    "cfggpst", data, parameters, **kwargs
                )
                dt2 = grammar.generate_derivation_tree(
                    "cfggpst", data, parameters=parameters, **kwargs
                )
                dt3 = grammar.generate_derivation_tree(
                    method="cfggpst", genotype=data, parameters=parameters, **kwargs
                )
                assert string1 == dt1.string() == dt2.string() == dt3.string()

                # Functions in mapping module
                string4 = al.systems.cfggpst.mapping.forward(
                    grammar, data, parameters, **kwargs
                )
                string5 = al.systems.cfggpst.mapping.forward(
                    grammar, data, parameters=parameters, **kwargs
                )
                string6 = al.systems.cfggpst.mapping.forward(
                    grammar=grammar, genotype=data, parameters=parameters, **kwargs
                )
                assert string1 == string4 == string5 == string6

                kwargs["return_derivation_tree"] = True
                phe, dt4 = al.systems.cfggpst.mapping.forward(
                    grammar, data, parameters, **kwargs
                )
                phe, dt5 = al.systems.cfggpst.mapping.forward(
                    grammar, data, parameters=parameters, **kwargs
                )
                phe, dt6 = al.systems.cfggpst.mapping.forward(
                    grammar=grammar, genotype=data, parameters=parameters, **kwargs
                )
                kwargs["return_derivation_tree"] = False
                assert string1 == dt4.string() == dt5.string() == dt6.string()


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
    for parameters in (p1,):
        for dt, string in zip(random_dts, random_strings):
            gt1 = al.systems.cfggpst.mapping.reverse(grammar, string)
            gt2 = al.systems.cfggpst.mapping.reverse(grammar, dt)
            gt3 = al.systems.cfggpst.mapping.reverse(grammar, string, parameters)
            gt4 = al.systems.cfggpst.mapping.reverse(grammar, string, parameters, False)
            gt5, dt5 = al.systems.cfggpst.mapping.reverse(
                grammar, string, parameters, True
            )
            gt6 = al.systems.cfggpst.mapping.reverse(
                grammar, phenotype_or_derivation_tree=string
            )
            gt7 = al.systems.cfggpst.mapping.reverse(
                grammar, phenotype_or_derivation_tree=dt
            )
            gt8 = al.systems.cfggpst.mapping.reverse(
                grammar, phenotype_or_derivation_tree=string, parameters=parameters
            )
            gt9 = al.systems.cfggpst.mapping.reverse(
                grammar, phenotype_or_derivation_tree=dt, parameters=parameters
            )
            gt10 = al.systems.cfggpst.mapping.reverse(
                grammar,
                phenotype_or_derivation_tree=string,
                parameters=parameters,
                return_derivation_tree=False,
            )
            gt11, dt11 = al.systems.cfggpst.mapping.reverse(
                grammar,
                phenotype_or_derivation_tree=dt,
                parameters=parameters,
                return_derivation_tree=True,
            )
            for gt in (gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10, gt11):
                # Check if reverse mapping resulted in a valid genotype
                check_genotype(gt)
                # Check if genotype allows to reproduce the original string via forward mapping
                string_from_fwd_map = grammar.generate_string("cfggpst", gt)
                assert string_from_fwd_map == string


def test_mapping_errors():
    bnf_text = "<S> ::= <S><S> | 1 | 2 | 3"
    grammar = al.Grammar(bnf_text=bnf_text)
    # Invalid input: a string that is not part of the grammar's language
    string = "4"
    with pytest.raises(al.exceptions.MappingError):
        al.systems.cfggpst.mapping.reverse(grammar, string)
    # Invalid input: a derivation tree with an unknown nonterminal
    dt = grammar.generate_derivation_tree()
    dt.root_node.symbol = al._grammar.data_structures.NonterminalSymbol("nonsense")
    with pytest.raises(al.exceptions.MappingError):
        al.systems.cfggpst.mapping.reverse(grammar, dt)
    # Invalid input: a derivation tree with an unknown derivation (no corresponding rule)
    dt = grammar.generate_derivation_tree()
    dt.leaf_nodes()[0].symbol = al._grammar.data_structures.TerminalSymbol("nonsense")
    with pytest.raises(al.exceptions.MappingError):
        al.systems.cfggpst.mapping.reverse(grammar, dt)


def test_mapping_forward_by_hand():
    bnf_text = """
    <A> ::= <B><C>
    <B> ::= 0 | 1
    <C> ::= x | y
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_genotype_phenotype_map = [
        (((0, 1, 3, 2, 5), (2, 1, 0, 1, 0)), "0x"),
        ("((0,1,4,2,5),(2,1,0,1,0))", "1x"),
        ("((0,1,3,2,6),(2,1,0,1,0))", "0y"),
        (((0, 1, 4, 2, 6), (2, 1, 0, 1, 0)), "1y"),
    ]
    for genotype, expected_phenotype in expected_genotype_phenotype_map:
        phenotype = grammar.generate_string(method="cfggpst", genotype=genotype)
        assert phenotype == expected_phenotype
        dt = grammar.generate_derivation_tree(method="cfggpst", genotype=genotype)
        assert dt.string() == expected_phenotype


def test_mapping_forward_and_reverse_automated():
    bnf_text = """
    <data> ::= <byte> | <data><byte>
    <byte> ::= <bit><bit><bit><bit><bit><bit><bit><bit>
    <bit> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    for _ in range(1000):
        string1 = grammar.generate_string()
        # Reverse map: string1 -> genotype
        genotype = al.systems.cfggpst.mapping.reverse(grammar, string1)
        # Forward map: genotype -> string2
        string2 = al.systems.cfggpst.mapping.forward(grammar, genotype)
        assert string1 == string2


def test_mapping_forward_and_reverse_by_hand1():
    bnf_text = """
    <e> ::= <e> <o> <e> | <v>
    <o> ::= + | - | * | /
    <v> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    parameters = dict()
    for string1 in (
        "1+1",
        "9-4",
        "7*5-3",
        "9*8/7+6-5",
        "3+4/9-1*8",
        "1+2+3+4+5-6-7*8/9",
    ):
        # Reverse map: string1 -> genotype
        genotype = al.systems.cfggpst.mapping.reverse(grammar, string1, parameters)
        # Forward map: genotype -> string2
        string2 = al.systems.cfggpst.mapping.forward(grammar, genotype, parameters)
        assert string2 == string1
