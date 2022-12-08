import pytest
import shared

import alogos as al


def test_comparison_operators():
    bnf_text = """
    <S> ::= <A> <B>
    <A> ::= 1 | 2
    <B> ::= a | b
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    dt1 = grammar.generate_derivation_tree("ge", genotype=[0])
    dt2 = grammar.generate_derivation_tree("ge", genotype=[1])
    assert dt1 == dt1
    assert dt1 != dt2
    for nonsense in (4, "a", (1, 2, 3), None, str):
        assert not dt1 == nonsense
        assert not nonsense == dt1
        assert dt2 != nonsense
        assert nonsense != dt2


def test_parsing_with_different_algorithms_and_inspect_derivation_trees_closely():
    bnf_text = """
    <S> ::= 0 <S> | 1 <S> |
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = ["0", "1", "0101", "11011"]
    for string in expected_strings:
        # Recognize
        assert grammar.recognize_string(string)
        assert grammar.recognize_string(string, parser="lalr")
        assert grammar.recognize_string(string, parser="earley")

        # Parse
        tree0 = grammar.parse_string(string)
        tree1 = grammar.parse_string(string, parser="lalr")
        tree2 = grammar.parse_string(string, parser="earley")

        # Read string from tree
        s1 = tree0.string()
        s2 = tree0.string()
        s3 = tree0.string()
        assert "" != s1 == s2 == s3 != "nonsense"

        # Read derivation from tree
        for order in ["leftmost", "rightmost", "random"]:
            for nl in [True, False]:
                d0 = tree0.derivation(derivation_order=order, newline=nl)
                d1 = tree1.derivation(derivation_order=order, newline=nl)
                d2 = tree2.derivation(derivation_order=order, newline=nl)
                assert d0
                assert isinstance(d0, str)
                assert d0 == d1 == d2

        shared.check_tree(tree0)
        shared.check_tree(tree1)
        shared.check_tree(tree2)


def test_parsing_with_ambiguity_and_multiple_trees1():
    # https://en.wikipedia.org/wiki/Context-free_grammar#Derivations_and_syntax_trees
    bnf_text = """
    <S> ::= <S> + <S>
    <S> ::= 1
    <S> ::= a
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    string = "a+1+a"
    with pytest.raises(al.exceptions.ParserError):
        grammar.parse_string(string, parser="lalr", get_multiple_trees=True)

    trees = grammar.parse_string(string, parser="earley", get_multiple_trees=True)
    assert len(trees) == 2
    for tree in trees:
        assert string == tree.string()

    trees = grammar.parse_string(string, get_multiple_trees=True, max_num_trees=1)
    assert len(trees) == 1

    trees = grammar.parse_string(string, get_multiple_trees=True, max_num_trees=7)
    assert len(trees) == 2


def test_parsing_with_ambiguity_and_multiple_trees2():
    bnf_text = """
    <S> ::= 0 | 1 | <S> <S>
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    for string in ["111", "01001"]:
        trees = grammar.parse_string(string, get_multiple_trees=True)
        assert len(trees) > 1

        known_derivations = set()
        for tree in trees:
            assert tree.string() == string
            derivation = tree.derivation(derivation_order="leftmost")
            assert derivation not in known_derivations
            known_derivations.add(derivation)


def test_parsing_with_ambiguity_and_multiple_trees3():
    # https://en.wikipedia.org/wiki/Ambiguous_grammar
    bnf_text = "<A> ::= <A> + <A> | <A> - <A> | a"
    grammar = al.Grammar(bnf_text=bnf_text)

    for string in ["a+a+a", "a+a+a+a", "a+a+a+a+a"]:
        trees = grammar.parse_string(string, get_multiple_trees=True)
        assert len(trees) in [2, 5, 14]

        known_derivations = set()
        for tree in trees:
            assert tree.string() == string
            derivation = tree.derivation(derivation_order="leftmost")
            assert derivation not in known_derivations
            known_derivations.add(derivation)


def test_parsing_with_recursive_grammar():
    # In an earlier Lark version there was a problem with Earley parser and some recursive rules
    ebnf_text = """
    S = "(" S ")"
    S = "." S
    S = S "."
    S = S S
    S = ""
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    tree = grammar.parse_string(".(())")
    assert len(str(tree)) > 5


def test_parsing_with_empty_string_grammar():
    bnf_text = """
    <S> ::=
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    tree = grammar.parse_string("")
    assert tree

    ebnf_text = """
    S =
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    tree = grammar.parse_string("")
    assert tree


def test_parsing_argument_errors():
    bnf_text = """
    <S> ::= 0 | 1 | <S> <S>
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Individual arguments
    with pytest.raises(ValueError):
        grammar.parse_string("111", parser="")
    with pytest.raises(ValueError):
        grammar.parse_string("111", parser="nonsense")
    with pytest.raises(TypeError):
        grammar.parse_string("111", parser=False)

    # Multiple parse trees
    trees = grammar.parse_string("111", parser="earley", get_multiple_trees=True)
    assert len(trees) > 1
    with pytest.raises(al.exceptions.ParserError):
        grammar.parse_string("111", parser="lalr", get_multiple_trees=True)


def test_parsing_error_due_to_grammar_form_and_parser_choice_1():
    # Parser not being compatible with form of a grammar
    bnf_text = """
    <S> ::= 0 | 1 | <B> <S> 1 <S> <A>
    <A> ::=
    <B> ::=
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    grammar.parse_string("010")
    with pytest.raises(al.exceptions.ParserError):
        grammar.parse_string("010", parser="lalr")  # fails after parser generation


def test_parsing_error_due_to_grammar_form_and_parser_choice_2():
    # Example of a reduce/reduce conflict from https://en.wikipedia.org/wiki/LALR_parser
    bnf_text = """
    <S> ::= a <E> c
          | a <F> d
          | b <F> c
          | b <E> d
    <E> ::= e
    <F> ::= e
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    grammar.parse_string("aec")
    with pytest.raises(al.exceptions.ParserError):
        grammar.parse_string("aec", parser="lalr")  # fails during parser generation


def test_parsing_error_due_to_tree_label():
    # This test covers a special case that was at some point reachable via a Lark bug

    # Grammar
    bnf_text = """
    <S> ::= <S> + <S>
    <S> ::= 1
    <S> ::= a
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    grammar.parse_string("1+a+1", get_multiple_trees=True)

    # Cached objects
    nt_map_fwd, nt_map_rev = grammar._cache["lark"]["nt_maps"]
    max_num_trees = 1000
    lark_parser = grammar._cache["lark"]["earley1"][0]

    # Internal functions
    lark_trees = lark_parser.parse("1+a+1")
    lark_trees.data = "nonsense"
    with pytest.raises(al.exceptions.ParserError):
        al._grammar.parsing.parsing_with_lark._ambig_lark_tree_to_dts(
            grammar, lark_trees, nt_map_fwd, nt_map_rev, max_num_trees
        )


def test_derivation_tree_argument_errors():
    with pytest.raises(TypeError):
        al._grammar.data_structures.DerivationTree()


def test_derivation_tree_visualization():
    # Grammar
    grammar = al.Grammar(bnf_text="<S> ::= 1 <S> 1 | 0 <S> 0 | 0 | 1")

    # Trees
    dt1 = grammar.parse_string("00100")
    dt2 = grammar.generate_derivation_tree(
        "ge", genotype=[0, 0, 2], raise_errors=False, parameters=dict(max_expansions=1)
    )

    # Create and check figures
    for dt in (dt1, dt2):
        fig1 = dt.plot()
        shared.check_figure(fig1)

        fig2 = dt.plot(
            show_node_indices=True,
            layout_engine="neato",
            fontname="Arial",
            fontsize=14,
            shape_nt="circle",
            shape_unexpanded_nt="box",
            shape_t="diamond",
            fontcolor_nt="red",
            fontcolor_unexpanded_nt="blue",
            fontcolor_t="green",
            fillcolor_nt="yellow",
            fillcolor_unexpanded_nt="black",
            fillcolor_t="white",
        )
        shared.check_figure(fig2)

    # Display
    fig = dt.plot()
    fig.display(inline=False)
    fig.display(inline=True)
