import copy
import os

import pytest
import shared

import alogos as al


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, "in")


def test_grammar_symbols():
    # Creation
    bnf = """
    <S> ::= <A>
    <A> ::= A B
    """
    gr = al.Grammar(bnf_text=bnf)
    nt_A = gr.nonterminal_symbols[1]
    t_A = gr.terminal_symbols[0]
    t_B = gr.terminal_symbols[1]

    # Attributes
    assert nt_A.text == "A"
    assert t_A.text == "A"
    assert t_B.text == "B"

    # Copying
    nt_A_copy = nt_A.copy()
    t_A_copy = t_A.copy()
    t_B_copy = t_B.copy()

    assert nt_A_copy.text == "A"
    assert t_A_copy.text == "A"
    assert t_B_copy.text == "B"

    assert nt_A_copy == nt_A != t_A
    assert t_A_copy == t_A != nt_A
    assert t_B_copy == t_B != t_A

    nt_A_copy.text = "random"
    t_A_copy.text = "nonsense"
    assert nt_A_copy.text == "random"
    assert t_A_copy.text == "nonsense"
    assert nt_A.text == "A"
    assert t_A.text == "A"

    # Representations
    assert str(nt_A) == "A"
    assert str(t_A) == "A"
    assert str(t_B) == "B"
    assert str(nt_A_copy) == "random"
    assert str(t_A_copy) == "nonsense"
    assert str(t_B_copy) == "B"

    # Comparison operators
    assert nt_A == nt_A
    assert t_A == t_A
    assert t_A != nt_A
    assert nt_A != t_A
    assert t_A != t_B
    assert t_B != t_A
    assert t_A < t_B
    assert t_B > t_A
    assert t_A <= t_B
    assert t_B >= t_B

    # Hash
    assert nt_A.text == t_A.text
    assert len(dict(nt_A="some value", t_A="another value")) == 2
    assert len(set([nt_A, t_A, t_B, nt_A_copy, t_A_copy, t_B_copy] * 2)) == 5


def test_grammar_nodes():
    gr = al.Grammar(bnf_text="<S> ::= 1 2")
    dt = gr.parse_string("12")
    n0 = dt.root_node
    n1 = dt.root_node.children[0]

    # Representations
    s1 = str(n1)
    s2 = repr(n1)
    assert isinstance(s1, str)
    assert s1 == "1"
    assert isinstance(s2, str)
    assert s2.startswith("<Node object at ") and s2.endswith(">")

    # Copying
    n0_c0 = n0.copy()
    n0_c1 = copy.copy(n0)
    n0_c2 = copy.deepcopy(n0)
    assert n0_c0.symbol == n0_c1.symbol == n0_c2.symbol

    # Methods
    assert n0.contains_nonterminal() is True
    assert n0.contains_terminal() is False
    assert n0.contains_unexpanded_nonterminal() is False
    n0.children = None
    assert n0.contains_unexpanded_nonterminal() is True


def test_grammar_visualization():
    # Grammar
    grammar = al.Grammar(bnf_text="<S> ::= 0 | 1")

    # Create figure
    fig = grammar.plot()
    assert isinstance(
        fig, al._grammar.visualization.grammar_with_railroad.GrammarFigure
    )

    # Representations
    s1 = str(fig)
    s2 = repr(fig)
    s3 = fig._repr_html_()
    s4 = fig.html_text
    s5 = fig.html_text_standalone
    s6 = fig.html_text_partial
    for s in (s1, s2, s3, s4, s5, s6):
        assert s
        assert isinstance(s, str)
    assert s2.startswith("<GrammarFigure object at") and s2.endswith(">")
    assert "<" in s3 and ">" in s3

    # File export
    used_filepath = fig.export_html("test")
    assert isinstance(used_filepath, str)
    assert used_filepath.endswith("html")
    os.remove(used_filepath)

    # Display
    fig.display(inline=False)
    fig.display(inline=True)


def test_grammar_comparison_operators():
    gr1 = al.Grammar(bnf_text="<S> ::= 1 2 | 3 4")
    gr2 = al.Grammar(bnf_text="<S> ::= 1 2 | 3")
    # eq
    assert gr1 == gr1
    assert gr2 == gr2
    assert not gr1 == gr2
    assert not gr1 == 4
    assert not 4 == gr1
    assert not gr2 == "a"
    # ne
    assert not gr1 != gr1
    assert not gr2 != gr2
    assert gr1 != gr2
    assert gr1 != 4
    assert 4 != gr1


def test_grammar_errors():
    with pytest.raises(TypeError):
        al.Grammar(bnf_text=1)
    with pytest.raises(TypeError):
        al.Grammar(ebnf_text=42)

    with pytest.raises(FileNotFoundError):
        al.Grammar(bnf_file="")
    with pytest.raises(FileNotFoundError):
        al.Grammar(ebnf_file="")

    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(bnf_text="")
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(ebnf_text="")
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(bnf_file=os.path.join(IN_DIR, "empty.txt"))
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(ebnf_file=os.path.join(IN_DIR, "empty.txt"))


def test_warnings():
    for state in (None, "off", "on"):
        # Turn warnings on or off
        if state == "on":
            al.warnings.turn_on()
        elif state == "off":
            al.warnings.turn_off()

        # Redundant production rules
        bnf = """
        <S> ::= 1 | 2
        <S> ::= 1
        <S> ::= 2
        <S> ::= 3
        """
        message = """Problematic grammar specification: Some production rules are redundant. This package can deal with it, but in general it is not recommended. In particular, it introduces bias in random string generation. Following rules are contained more than one time:
  <S> ::= 1
  <S> ::= 2"""

        def func():
            al.Grammar(bnf_text=bnf)  # noqa: B023

        if state == "off":
            shared.emits_no_warning(func)
        else:
            shared.emits_warning(func, al.warnings.GrammarWarning, message)

        # Reused symbol in nonterminals and terminals
        bnf = """<S> ::= S
        """
        message = """Problematic grammar specification: The sets of nonterminal and terminal symbols are not disjoint, as required by the mathematical definition of a grammar. This package can deal with it, but in general it is not recommended. Following symbols appear in both sets:
  S as nonterminal <S> and terminal \"S\""""

        def func():
            al.Grammar(bnf_text=bnf)  # noqa: B023

        if state == "off":
            shared.emits_no_warning(func)
        else:
            shared.emits_warning(func, al.warnings.GrammarWarning, message)

        # More than one grammar specification
        bnf = "<S> ::= 1"
        ebnf = 'S = "1"'
        message = """More than one grammar specification was provided. Only the first one is used in following order of precedence: bnf_text > bnf_file > ebnf_text > ebnf_file."""

        def func():
            al.Grammar(bnf_text=bnf, ebnf_text=ebnf)  # noqa: B023

        if state == "off":
            shared.emits_no_warning(func)
        else:
            shared.emits_warning(func, al.warnings.GrammarWarning, message)

        # Stop of language generation
        bnf = "<S> ::= <S> 1 | 1"
        message = """Language generation stopped due to reaching max_steps=20000, but it did not produce all possible strings yet. To explore it further, the max_steps parameter can be increased."""
        gr = al.Grammar(bnf_text=bnf)

        def func():
            gr.generate_language()  # noqa: B023

        if state == "off":
            shared.emits_no_warning(func)
        else:
            shared.emits_warning(func, al.warnings.GrammarWarning, message)


def test_warning_for_more_than_one_grammar_specification():
    # BNF
    bnf = "<S> ::= 1 | 2 |"
    shared.emits_no_warning(lambda: al.Grammar(bnf_text=bnf))
    grammar = al.Grammar(bnf_text=bnf)
    shared.check_grammar(grammar, ["", "1", "2"], ["12", "3"])
    # EBNF
    ebnf = 'S = "1" | "2" | ""'
    shared.emits_no_warning(lambda: al.Grammar(ebnf_text=ebnf))
    grammar = al.Grammar(ebnf_text=ebnf)
    shared.check_grammar(grammar, ["", "1", "2"], ["12", "3"])
    # Both
    expected_message = (
        "More than one grammar specification was provided. "
        "Only the first one is used in following order of precedence: "
        "bnf_text > bnf_file > ebnf_text > ebnf_file."
    )
    shared.emits_warning(
        lambda: al.Grammar(bnf_text=bnf, bnf_file="nonsense"),
        al.warnings.GrammarWarning,
        expected_message,
    )
    shared.emits_warning(
        lambda: al.Grammar(bnf_text=bnf, ebnf_text=ebnf),
        al.warnings.GrammarWarning,
        expected_message,
    )
    shared.emits_warning(
        lambda: al.Grammar(bnf_text=bnf, ebnf_file="nons"),
        al.warnings.GrammarWarning,
        expected_message,
    )
    shared.emits_warning(
        lambda: al.Grammar(ebnf_text=ebnf, ebnf_file="nons"),
        al.warnings.GrammarWarning,
        expected_message,
    )

    expected_message = (
        'Could not read a grammar from file "nons". The file does not exist.'
    )
    shared.emits_exception(
        lambda: al.Grammar(bnf_file="nons", ebnf_text=ebnf),
        FileNotFoundError,
        expected_message,
    )
    shared.emits_exception(
        lambda: al.Grammar(bnf_file="nons", ebnf_file="nonsense"),
        FileNotFoundError,
        expected_message,
    )


def test_grammar_file_reading_errors():
    gr = al.Grammar()
    gr._read_file(os.path.join(IN_DIR, "fileformats", "simple.txt"))
    with pytest.raises(ValueError):
        gr._read_file(os.path.join(IN_DIR, "fileformats", "simple.bin"))
    with pytest.raises(FileNotFoundError):
        gr._read_file("hopefully_non-existing_filepath")


def test_grammar_validity():
    bnf_text = """
    <S> ::= 0 | 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    check_validity = al._grammar.parsing.io_with_regex._check_grammar_validity
    check_validity(grammar)

    # No production rules
    grammar2 = grammar.copy()
    for item in grammar.production_rules:
        del grammar2.production_rules[item]
    shared.emits_exception(
        lambda: check_validity(grammar2),
        al.exceptions.GrammarError,
        "The set of production rules is empty.",
    )

    # No nonterminal symbols
    grammar2 = grammar.copy()
    for item in grammar.nonterminal_symbols:
        grammar2.nonterminal_symbols.remove(item)
    shared.emits_exception(
        lambda: check_validity(grammar2),
        al.exceptions.GrammarError,
        "The set of nonterminal symbols is empty.",
    )

    # No terminal symbols
    grammar2 = grammar.copy()
    for item in grammar.terminal_symbols:
        grammar2.terminal_symbols.remove(item)
    shared.emits_exception(
        lambda: check_validity(grammar2),
        al.exceptions.GrammarError,
        "The set of terminal symbols is empty.",
    )


def test_grammar_chomsky_normal_form():
    # TODO: too small to be a reasonable test
    bnf_text = """
    <S> ::= <T_A> A
    <T_A> ::= 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    grammar_cnf = grammar._to_cnf()
    assert grammar.generate_language() == grammar_cnf.generate_language()
