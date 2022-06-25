import copy
import inspect
import os
import random

import ordered_set
import pytest

import alogos as al

import shared


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, 'in')


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
    assert nt_A.text == 'A'
    assert t_A.text == 'A'
    assert t_B.text == 'B'

    # Copying
    nt_A_copy = nt_A.copy()
    t_A_copy = t_A.copy()
    t_B_copy = t_B.copy()

    assert nt_A_copy.text == 'A'
    assert t_A_copy.text == 'A'
    assert t_B_copy.text == 'B'

    assert nt_A_copy == nt_A != t_A
    assert t_A_copy == t_A != nt_A
    assert t_B_copy == t_B != t_A

    nt_A_copy.text = 'random'
    t_A_copy.text = 'nonsense'
    assert nt_A_copy.text == 'random'
    assert t_A_copy.text == 'nonsense'
    assert nt_A.text == 'A'
    assert t_A.text == 'A'

    # Representations
    assert str(nt_A) == 'A'
    assert str(t_A) == 'A'
    assert str(t_B) == 'B'
    assert str(nt_A_copy) == 'random'
    assert str(t_A_copy) == 'nonsense'
    assert str(t_B_copy) == 'B'

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
    assert len(dict(nt_A='some value', t_A='another value')) == 2
    assert len(set([nt_A, t_A, t_B, nt_A_copy, t_A_copy, t_B_copy]*2)) == 5


def test_grammar_nodes():
    gr = al.Grammar(bnf_text='<S> ::= 1 2')
    dt = gr.parse_string('12')
    n0 = dt.root_node
    n1 = dt.root_node.children[0]
    n2 = dt.root_node.children[1]

    # Representations
    s1 = str(n1)
    s2 = repr(n1)
    assert isinstance(s1, str)
    assert s1 == '1'
    assert isinstance(s2, str)
    assert s2.startswith('<Node object at ') and s2.endswith('>')

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
    grammar = al.Grammar(bnf_text='<S> ::= 0 | 1')

    # Create figure
    fig = grammar.plot()
    assert isinstance(fig, al._grammar.visualization.grammar_with_railroad.GrammarFigure)

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
    assert s2.startswith('<GrammarFigure object at') and s2.endswith('>')
    assert '<' in s3 and '>' in s3

    # File export
    used_filepath = fig.export_html('test')
    assert isinstance(used_filepath, str)
    assert used_filepath.endswith('html')
    os.remove(used_filepath)

    # Display
    fig.display(inline=False)
    fig.display(inline=True)


def test_grammar_comparison_operators():
    gr1 = al.Grammar(bnf_text='<S> ::= 1 2 | 3 4')
    gr2 = al.Grammar(bnf_text='<S> ::= 1 2 | 3')
    # eq
    assert gr1 == gr1
    assert gr2 == gr2
    assert not gr1 == gr2
    # ne
    assert not gr1 != gr1
    assert not gr2 != gr2
    assert gr1 != gr2


def test_grammar_file_reading_errors():
    gr = al.Grammar()
    gr._read_file(os.path.join(IN_DIR, 'fileformats', 'simple.txt'))
    with pytest.raises(Exception):
        gr._read_file(os.path.join(IN_DIR, 'fileformats', 'simple.bin'))
    with pytest.raises(FileNotFoundError):
        gr._read_file('hopefully_non-existing_filepath')


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
    shared.emits_error(
        lambda: check_validity(grammar2),
        al.exceptions.GrammarError,
        'The set of production rules is empty.')

    # No nonterminal symbols
    grammar2 = grammar.copy()
    for item in grammar.nonterminal_symbols:
        grammar2.nonterminal_symbols.remove(item)
    shared.emits_error(
        lambda: check_validity(grammar2),
        al.exceptions.GrammarError,
        'The set of nonterminal symbols is empty.')

    # No terminal symbols
    grammar2 = grammar.copy()
    for item in grammar.terminal_symbols:
        grammar2.terminal_symbols.remove(item)
    shared.emits_error(
        lambda: check_validity(grammar2),
        al.exceptions.GrammarError,
        'The set of terminal symbols is empty.')


def test_grammar_chomsky_normal_form():
    # TODO: too small to be a reasonable test
    bnf_text = """
    <S> ::= <T_A> A
    <T_A> ::= 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    grammar_cnf = grammar.to_cnf()
    assert grammar.generate_language() == grammar_cnf.generate_language()


def test_warning_for_both_bnf_and_ebnf_provided(caplog):
    # BNF
    bnf = "<S> ::= 1 | 2 |"
    shared.emits_no_warning(lambda: al.Grammar(bnf_text=bnf), caplog)
    grammar = al.Grammar(bnf_text=bnf)
    shared.check_grammar(grammar, ['', '1', '2'], ['12', '3'])
    # EBNF
    ebnf = 'S = "1" | "2" | ""'
    shared.emits_no_warning(lambda: al.Grammar(ebnf_text=ebnf), caplog)
    grammar = al.Grammar(ebnf_text=ebnf)
    shared.check_grammar(grammar, ['', '1', '2'], ['12', '3'])
    # Both
    expected_message = (
        'More than one grammar specification was provided. '
        'Only the first one is used in following order of precedence: '
        'bnf_text > bnf_file > ebnf_text > ebnf_file.')
    shared.emits_warning(lambda: al.Grammar(bnf_text=bnf, bnf_file='nonsense'), caplog, expected_message)
    shared.emits_warning(lambda: al.Grammar(bnf_text=bnf, ebnf_text=ebnf), caplog, expected_message)
    shared.emits_warning(lambda: al.Grammar(bnf_text=bnf, ebnf_file='nons'), caplog, expected_message)
    shared.emits_warning(lambda: al.Grammar(ebnf_text=ebnf, ebnf_file='nons'), caplog, expected_message)

    expected_message = 'Could not read a grammar from file "nons". The file does not exist.'
    shared.emits_error(lambda: al.Grammar(bnf_file='nons', ebnf_text=ebnf),
                FileNotFoundError, expected_message)
    shared.emits_error(lambda: al.Grammar(bnf_file='nons', ebnf_file='nonsense'),
                FileNotFoundError, expected_message)
