import pytest

import alogos as al


BNF_TEXT = """
<S> ::= <A> | <B> | <S><S> | <S> x | <S> y
<A> ::= 1 | 2
<B> ::= a | b
"""


def test_default_method():
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    dt = grammar.generate_derivation_tree()
    assert isinstance(dt, al._grammar.data_structures.DerivationTree)
    derivation = grammar.generate_derivation()
    assert isinstance(derivation, str)
    string = grammar.generate_string()
    assert isinstance(string, str)


@pytest.mark.parametrize(
    "method, genotype",
    [
        (
            "cfggp",
            '[["S",1,2],["S",1,1],["A",1,1],["1",0,0],["S",1,1],["B",1,1],["a",0,0]]',
        ),
        ("cfggpst", ((0, 0, 1, 5, 0, 2, 7), (2, 1, 1, 0, 1, 1, 0))),
        ("dsge", ((2, 0, 1), (0,), (0,))),
        ("ge", [2, 0, 0, 1, 0]),
        ("pige", [208, 147, 122, 220, 210, 196, 85, 16, 152, 154]),
        ("whge", "11001000101011000110"),
    ],
)
def test_deterministic_methods(capsys, method, genotype):
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    dt = grammar.generate_derivation_tree(method, genotype=genotype)
    derivation = grammar.generate_derivation(method, genotype=genotype, newline=False)
    string = grammar.generate_string(method, genotype=genotype)
    assert dt.string() == string == "1a"
    assert (
        dt.derivation(newline=False)
        == derivation
        == "<S> => <S><S> => <A><S> => 1<S> => 1<B> => 1a"
    )

    # Verbose
    captured = capsys.readouterr()
    assert len(captured.out) == 0
    grammar.generate_derivation_tree(method, genotype=genotype, verbose=True)
    captured = capsys.readouterr()
    assert len(captured.out) > 0

    # Missing input data
    with pytest.raises(TypeError):
        grammar.generate_string(method)


@pytest.mark.parametrize(
    "method",
    [
        "uniform",
        "weighted",
        "grow_one_branch_to_max_depth",
        "grow_all_branches_within_max_depth",
        "grow_all_branches_to_max_depth",
        "ptc2",
    ],
)
def test_probabilistic_methods(method):
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    dt = grammar.generate_derivation_tree(method)
    assert isinstance(dt, al._grammar.data_structures.DerivationTree)
    derivation = grammar.generate_derivation()
    assert isinstance(derivation, str)
    string = grammar.generate_string()
    assert isinstance(string, str)


def test_uniform_kwargs():
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    for _ in range(1000):
        try:
            dt = grammar.generate_derivation_tree("uniform", max_expansions=2)
            break
        except al.exceptions.MappingError:
            pass
    assert isinstance(dt, al._grammar.data_structures.DerivationTree)

    for me in (0, 1):
        with pytest.raises(al.exceptions.MappingError):
            grammar.generate_derivation_tree("uniform", max_expansions=me)


def test_weighted_kwargs():
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    dt = grammar.generate_derivation_tree("weighted", reduction_factor=0.01)
    assert isinstance(dt, al._grammar.data_structures.DerivationTree)

    for me in (0, 1):
        with pytest.raises(al.exceptions.MappingError):
            grammar.generate_derivation_tree("weighted", max_expansions=me)

    with pytest.raises(al.exceptions.MappingError):
        for _ in range(100):
            grammar.generate_derivation_tree("weighted", reduction_factor=1000.0)


def test_grow_one_branch_to_max_depth():
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    for desired_depth in range(2, 30):
        dt = grammar.generate_derivation_tree(
            "grow_one_branch_to_max_depth", max_depth=desired_depth
        )
        assert dt.depth() == desired_depth  # exact


def test_grow_all_branches_to_max_depth():
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    for desired_depth in range(2, 30):
        dt = grammar.generate_derivation_tree(
            "grow_all_branches_to_max_depth", max_depth=desired_depth
        )
        assert dt.depth() == desired_depth  # exact


def test_grow_all_branches_within_max_depth():
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    for desired_depth in range(2, 30):
        dt = grammar.generate_derivation_tree(
            "grow_all_branches_within_max_depth", max_depth=desired_depth
        )
        assert dt.depth() <= desired_depth  # smaller or equal


def test_ptc2():
    grammar = al.Grammar(bnf_text=BNF_TEXT)
    for desired_max_expansions in range(5, 30):
        dt = grammar.generate_derivation_tree(
            "ptc2", max_expansions=desired_max_expansions
        )
        assert (
            dt.num_expansions() <= desired_max_expansions * 1.5
        )  # smaller or equal with some room
