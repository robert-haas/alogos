import copy
import inspect
import os
import random

import ordered_set
import pytest

import alogos as al


# Number of repetitions for methods with randomness

NUM_REPETITIONS = 20


# Helper functions

def emits_warning(func, caplog, expected_message):
    if isinstance(expected_message, str):
        expected_message = [expected_message]
    caplog.clear()
    func()
    if len(caplog.records) == 0:
        raise ValueError('The expected warning was not emitted: {}'.format(expected_message))
    for message, record in zip(expected_message, caplog.records):
        assert message in record.message


def emits_no_warning(func, caplog):
    caplog.clear()
    func()
    assert len(caplog.records) == 0


def emits_error(function, error_type, expected_message=None):
    # https://docs.pytest.org/en/latest/assert.html#assertions-about-expected-exceptions
    with pytest.raises(error_type) as excp:
        function()
    if expected_message:
        given_message = str(excp.value)
        assert expected_message == given_message


def prints_to_stdout(func, capsys, message=None, partial_message=None, start_message=None,
                     end_message=None):
    capsys.readouterr()
    func()
    out, err = capsys.readouterr()
    if message is not None:
        assert message == out
    if partial_message is not None:
        assert partial_message in out
    if start_message is not None:
        assert out.startswith(start_message)
    if end_message is not None:
        assert out.endswith(end_message)


def prints_nothing_to_stdout(func, capsys):
    capsys.readouterr()
    func()
    out, err = capsys.readouterr()
    assert out == ''


def get_path_of_this_file():
    return os.path.abspath(inspect.getsourcefile(lambda _: None))


def conv_keys_to_str(given_dict):
    return {str(key): val for key, val in given_dict.items()}


class MockPrettyPrinter():
    def __init__(self):
        self.string = ''

    def text(self, s):
        self.string += s


def call_repr_pretty(obj, cycle):
    mock_printer = MockPrettyPrinter()
    obj._repr_pretty_(mock_printer, cycle=cycle)
    return mock_printer.string


def filter_list_unique(items):
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


# Grammar checks

GRAMMAR = al._grammar.data_structures.Grammar(bnf_text='<S> ::= 0 | 1')
TREE = GRAMMAR.parse_string('0')


def check_grammar(grammar, pos_examples, neg_examples, language=None, max_steps=None):
    # Type
    assert isinstance(grammar, al._grammar.data_structures.Grammar)

    # Attributes
    assert isinstance(grammar.nonterminal_symbols, ordered_set.OrderedSet)
    assert isinstance(grammar.terminal_symbols, ordered_set.OrderedSet)
    assert isinstance(grammar.production_rules, dict)
    assert isinstance(grammar.start_symbol, al._grammar.data_structures.NonterminalSymbol)
    assert not hasattr(grammar, '_cache') or  isinstance(grammar._cache, dict)

    assert set(grammar.production_rules) == set(grammar.nonterminal_symbols)
    for lhs, multiple_rhs in grammar.production_rules.items():
        for rhs in multiple_rhs:
            for sym in rhs:
                assert isinstance(sym, al._grammar.data_structures.Symbol)
                if sym not in grammar.nonterminal_symbols:
                    assert sym in grammar.terminal_symbols
                if sym not in grammar.terminal_symbols:
                    assert sym in grammar.nonterminal_symbols

    # Copying
    gr2 = grammar.copy()
    gr3 = copy.copy(grammar)
    gr4 = copy.deepcopy(grammar)

    # Reset
    gr5 = grammar.copy()
    gr5._set_empty_state()
    assert len(grammar._cache) == 0
    assert len(gr5.production_rules) == 0
    assert len(gr5.nonterminal_symbols) == 0
    assert len(gr5.terminal_symbols) == 0

    # Representations
    for gr in (grammar, gr2, gr3, gr4, gr5):
        s1 = str(gr)
        s2 = repr(gr)
        s3 = gr._repr_html_()
        s4 = call_repr_pretty(gr, cycle=True)
        s5 = call_repr_pretty(gr, cycle=False)
        for s in (s1, s2, s3, s4, s5):
            assert s and isinstance(s, str)
        assert s2.startswith('<Grammar object at') and s2.endswith('>')

    # Equality
    assert grammar == gr2 == gr3 == gr4 != gr5
    assert not grammar != gr2
    for other in (GRAMMAR, gr5, 5, 3.14, 'a', None, tuple(), set()):
        assert grammar != other
        assert other != grammar
        assert not grammar == other
        assert not other == grammar

    # Hashing
    assert isinstance(hash(grammar), int)
    assert hash(grammar) == hash(gr2) == hash(gr3) == hash(gr4) != hash(GRAMMAR)

    gr_set = set()
    gr_set.add(grammar)
    gr_set.add(gr2)
    gr_set.add(gr3)
    gr_set.add(gr4)
    assert len(gr_set) == 1

    gr_dict = dict()
    gr_dict[grammar] = 'a'
    gr_dict[gr2] = 'b'
    gr_dict[gr3] = 'c'
    gr_dict[gr4] = 'd'
    assert len(gr_dict) == 1

    # Reading and writing
    # - Duplicate the grammar from BNF output to see if grammar -> BNF -> grammar holds
    try:
        # BNF text
        grammar_bnf = grammar.to_bnf_text()
        assert grammar_bnf
        assert isinstance(grammar_bnf, str)
        # BNF file
        grammar.to_bnf_file('test.bnf')
        with open('test.bnf') as f:
            assert grammar_bnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al.Grammar()
        grammar2.from_bnf_text(grammar_bnf)
        grammar3 = al.Grammar(bnf_file='test.bnf')
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile('test.bnf'):
            os.remove('test.bnf')

    # - Duplicate the grammar from BNF v2 output to see if grammar -> BNF -> grammar holds
    try:
        kwargs = dict(
            start_terminal_symbol='"', end_terminal_symbol='"',
            start_terminal_symbol2="'", end_terminal_symbol2="'")
        # BNF text
        grammar_bnf = grammar.to_bnf_text(**kwargs)
        assert grammar_bnf
        assert isinstance(grammar_bnf, str)
        # BNF file
        grammar.to_bnf_file('test.bnf', **kwargs)
        with open('test.bnf') as f:
            assert grammar_bnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al.Grammar()
        grammar2.from_bnf_text(grammar_bnf, **kwargs)
        grammar3 = al.Grammar(bnf_file='test.bnf', **kwargs)
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile('test.bnf'):
            os.remove('test.bnf')

    # - Duplicate the grammar from EBNF output to see if grammar -> EBNF -> grammar holds
    try:
        # EBNF text
        grammar_ebnf = grammar.to_ebnf_text()
        assert grammar_ebnf
        assert isinstance(grammar_ebnf, str)
        # EBNF file
        grammar.to_ebnf_file('test.ebnf')
        with open('test.ebnf') as f:
            assert grammar_ebnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al._grammar.data_structures.Grammar(ebnf_file='test.ebnf')
        grammar3 = al._grammar.data_structures.Grammar()
        grammar3.from_ebnf_file('test.ebnf')
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile('test.ebnf'):
            os.remove('test.ebnf')

    # - Duplicate the grammar from EBNF v2 output to see if grammar -> EBNF -> grammar holds
    try:
        kwargs = dict(start_nonterminal_symbol='<', end_nonterminal_symbol='>')
        # EBNF text
        grammar_ebnf = grammar.to_ebnf_text(**kwargs)
        assert grammar_ebnf
        assert isinstance(grammar_ebnf, str)
        # EBNF file
        grammar.to_ebnf_file('test.ebnf', **kwargs)
        with open('test.ebnf') as f:
            assert grammar_ebnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al._grammar.data_structures.Grammar(ebnf_file='test.ebnf', **kwargs)
        grammar3 = al._grammar.data_structures.Grammar()
        grammar3.from_ebnf_file('test.ebnf', **kwargs)
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile('test.ebnf'):
            os.remove('test.ebnf')

    # Visualization
    # - in separate test for reasons of performance

    # Normal forms
    # - CNF
    assert isinstance(grammar.is_cnf(), bool)
    gr_cnf = grammar.to_cnf()
    assert gr_cnf.is_cnf()
    if grammar.is_cnf():
        grammar == gr_cnf
    else:
        grammar != gr_cnf
    # - GNF
    with pytest.raises(NotImplementedError):
        assert not grammar.is_gnf()  # TODO: implement it!
    with pytest.raises(NotImplementedError):
        gr_gnf = grammar.to_gnf()
        assert gr_gnf.is_gnf()
    # - BCNF
    assert isinstance(grammar.is_bcnf(), bool)
    gr_bcnf = grammar.to_bcnf()
    assert gr_bcnf.is_bcnf()
    if grammar.is_bcnf():
        grammar == gr_bcnf
    else:
        grammar != gr_bcnf

    # Derivation tree generation
    grammars = [grammar, gr_cnf, gr_bcnf]
    for i in range(30):
        # Generate a random string (with the grammar or a random normal form)
        gr = random.choice(grammars)
        dt = gr.generate_derivation_tree(reduction_factor=0.2)
        string = dt.string()
        assert isinstance(string, str)
        # Check if it is recognized (by the grammar and all its normal forms)
        grammar.recognize_string(string)
        gr_cnf.recognize_string(string)
        gr_bcnf.recognize_string(string)

    # String generation
    grammars = [grammar, gr_cnf, gr_bcnf]
    for i in range(30):
        # Generate a random string (with the grammar or a random normal form)
        gr = random.choice(grammars)
        string = gr.generate_string(reduction_factor=0.2)
        assert isinstance(string, str)
        # Check if it is recognized (by the grammar and all its normal forms)
        grammar.recognize_string(string)
        gr_cnf.recognize_string(string)
        gr_bcnf.recognize_string(string)

    # Language generation: only up to max_steps if provided, necessary for infinite languages
    if language:
        if max_steps:
            strings = grammar.generate_language(max_steps)
        else:
            strings = grammar.generate_language()
        assert isinstance(strings, list)
        assert len(strings) > 0
        assert len(strings) == len(set(strings))
        assert sorted(strings) == sorted(language)
        for string in strings:
            assert isinstance(string, str)
            grammar.recognize_string(string)
            gr_cnf.recognize_string(string)
            gr_bcnf.recognize_string(string)

    # Parsing of strings in the grammar's language
    for string in pos_examples:
        # Recognize
        assert grammar.recognize_string(string)
        # Parse
        derivation_tree = grammar.parse_string(string)
        # Check
        assert derivation_tree.string() == string
        assert ''.join(str(x) for x in derivation_tree.tokens()) == string

    # Parsing of strings that are NOT in the grammar's language
    for string in neg_examples:
        assert not grammar.recognize_string(string)
        assert not gr_cnf.recognize_string(string)
        assert not gr_bcnf.recognize_string(string)
        with pytest.raises(al.exceptions.ParserError):
            grammar.parse_string(string)
        with pytest.raises(al.exceptions.ParserError):
            gr_cnf.parse_string(string)
        with pytest.raises(al.exceptions.ParserError):
            gr_bcnf.parse_string(string)


def check_language(grammar, expected_strings, max_steps=None):
    generated_strings = grammar.generate_language(max_steps=max_steps)
    assert isinstance(generated_strings, list)

    # 1) All expected strings were generated
    for expected_string in expected_strings:
        # Language
        assert expected_string in generated_strings

        # Parsing
        assert grammar.recognize_string(expected_string)
        tree = grammar.parse_string(expected_string)
        assert expected_string == tree.string()

    # 2) All generated strings were expected
    assert set(generated_strings) == set(expected_strings)


def check_tree(dt):
    # Attributes
    assert isinstance(dt.grammar, al._grammar.data_structures.Grammar)
    assert isinstance(dt.root_node, al._grammar.data_structures.Node)
    assert not hasattr(dt, '_cache') or  isinstance(dt._cache, dict)

    # Copying
    dt2 = dt.copy()
    dt3 = copy.copy(dt)
    dt4 = copy.deepcopy(dt)

    # Representations
    class MP():
        def __init__(self):
            self.string = ''

        def text(self, s):
            self.string += s

    s1 = str(dt)
    s2 = repr(dt2)
    s3 = dt3._repr_html_()
    s4 = dt4.to_bracket_notation()
    s5 = dt4.to_tree_notation()
    s6 = call_repr_pretty(dt3, cycle=True)
    s7 = call_repr_pretty(dt4, cycle=False)
    for s in (s1, s2, s3, s4, s5, s6, s7):
        assert s and isinstance(s, str)
    assert s2.startswith('<DerivationTree object at') and s2.endswith('>')
    assert s3.startswith('<')
    for sym in ('<', '>', '(', ')'):
        assert sym in s4
        assert sym not in s5
    for sym in (' ', os.linesep):
        assert sym not in s4
        assert sym in s5

    # Equality
    assert dt == dt2
    assert not dt != dt2
    for other in (TREE, 5, 3.14, 'a', None, tuple(), set()):
        assert dt != other
        assert other != dt
        assert not dt == other
        assert not other == dt

    # Independence of copied objects observed by inequality
    dt5 = dt.copy()
    assert dt == dt5
    dt5.root_node.symbol.text = 'nonsense'
    assert dt != dt5

    # Hashing
    assert isinstance(hash(dt), int)
    assert hash(dt) == hash(dt2) == hash(dt3) == hash(dt4) != hash(TREE)

    tree_set = set()
    tree_set.add(dt)
    tree_set.add(dt2)
    tree_set.add(dt3)
    assert len(tree_set) == 1

    tree_dict = dict()
    tree_dict[dt] = 'a'
    tree_dict[dt2] = 'b'
    tree_dict[dt3] = 'c'
    assert len(tree_dict) == 1

    # Serialization
    data = dt.to_tuple()
    assert isinstance(data, tuple)
    assert isinstance(data[0], tuple)
    assert isinstance(data[1], tuple)

    dt_new = al._grammar.data_structures.DerivationTree(dt.grammar)
    dt_new.from_tuple(data)
    assert dt_new == dt

    # Visualization
    # - in separate test for reasons of performance

    # Traversal
    nodes_dfs1 = dt.nodes()
    nodes_dfs2 = dt.nodes(order='dfs')
    nodes_bfs = dt.nodes(order='bfs')
    assert nodes_dfs1 and isinstance(nodes_dfs1, list)
    assert nodes_dfs2 and isinstance(nodes_dfs2, list)
    assert nodes_bfs and isinstance(nodes_bfs, list)
    assert nodes_dfs1 == nodes_dfs2
    for node in nodes_dfs1 + nodes_dfs2 + nodes_bfs:
        assert isinstance(node, al._grammar.data_structures.Node)

    # Check if all nodes are fully expanded (no nonterminal as leaf node)
    is_exp = dt.is_completely_expanded()
    assert isinstance(is_exp, bool)
    assert is_exp
    dt_ue = dt.copy()
    dt_ue.root_node.children = None
    is_exp = dt_ue.is_completely_expanded()
    assert isinstance(is_exp, bool)
    assert not is_exp

    # Read leaf nodes from tree
    leaf_nodes = dt.leaf_nodes()
    assert leaf_nodes
    assert isinstance(leaf_nodes, list)
    assert isinstance(leaf_nodes[0].symbol.text, str)

    # Read internal nodes
    internal_nodes = dt.internal_nodes()
    assert internal_nodes
    assert isinstance(internal_nodes, list)
    assert isinstance(internal_nodes[0].symbol.text, str)
    assert all(isinstance(node.symbol, al._grammar.data_structures.NonterminalSymbol)
               for node in internal_nodes)

    # Read tokens from leaves of the tree
    tokens = dt.tokens()
    assert tokens
    assert isinstance(tokens, list)
    assert isinstance(tokens[0].text, str)

    # Read string from leaves of the tree
    string = dt.string()
    assert string
    assert isinstance(string, str)
    assert string == ''.join(node.symbol.text for node in leaf_nodes)
    string = dt.string(separator='-')
    assert string
    assert isinstance(string, str)
    assert string == '-'.join(node.symbol.text for node in leaf_nodes)

    # Read derivation from tree
    string = dt.derivation()
    assert string
    assert isinstance(string, str)
    for do in ('leftmost', 'rightmost', 'random'):
        for sl in (True, False):
            string = dt.derivation(derivation_order=do, separate_lines=sl)
            assert string
            assert isinstance(string, str)

    # Count expansions
    num_exp = dt.num_expansions()
    assert num_exp > 1 and isinstance(num_exp, int)

    # Count nodes
    num_nodes = dt.num_nodes()
    assert num_nodes > 1 and num_nodes > num_exp and isinstance(num_nodes, int)

    # Get depth
    depth = dt.depth()
    assert depth > 1 and isinstance(depth, int)

    # Compare depth
    is_deeper_than_0 = dt._is_deeper_than(0)
    assert is_deeper_than_0 and isinstance(is_deeper_than_0, bool)
    is_deeper_than_100 = dt._is_deeper_than(100)
    assert not is_deeper_than_100 and isinstance(is_deeper_than_100, bool)
    is_deeper_than_depth = dt._is_deeper_than(depth)
    assert not is_deeper_than_depth and isinstance(is_deeper_than_depth, bool)
    is_deeper_than_depth_min_1 = dt._is_deeper_than(depth-1)
    assert is_deeper_than_depth_min_1 and isinstance(is_deeper_than_depth_min_1, bool)


def check_figure(fig):
    # Type
    assert isinstance(fig, al._grammar.visualization.tree_with_graphviz.DerivationTreeFigure)

    # Representations
    s1 = str(fig)
    s2 = repr(fig)
    s3 = fig._repr_html_()
    s4 = fig.html_text
    s5 = fig.html_text_standalone
    s6 = fig.html_text_partial
    s7 = fig.svg_text
    for s in (s1, s2, s3, s4, s5, s6, s7):
        assert s
        assert isinstance(s, str)
    assert s2.startswith('<DerivationTreeFigure object at') and s2.endswith('>')
    assert '<' in s3 and '>' in s3
    assert '<' in s3 and '>' in s4
    assert '<' in s3 and '>' in s5
    assert '<' in s3 and '>' in s6
    assert '<' in s3 and '>' in s7

    # File export
    for fileformat in ('dot', 'eps', 'gv', 'html', 'pdf', 'png', 'ps', 'svg'):
        method_name = 'export_{}'.format(fileformat)
        method = getattr(fig, method_name)
        used_filepath = method('test')
        assert isinstance(used_filepath, str)
        assert used_filepath.endswith(fileformat)
        os.remove(used_filepath)


def test_tree_visualization():
    # Grammar
    grammar = al.Grammar(bnf_text='<S> ::= 1 <S> 1 | 0 <S> 0 | 0 | 1')

    # Trees
    dt1 = grammar.parse_string('00100')
    dt2 = grammar.generate_derivation_tree(
        'ge', [0, 0, 2], raise_errors=False, parameters=dict(max_expansions=1))

    # Create and check figures
    for dt in (dt1, dt2):
        fig1 = dt.plot()
        check_figure(fig1)

        fig2 = dt.plot(
            show_node_indices=True,
            layout_engine='neato',
            fontname='Arial',
            fontsize=14,
            shape_nt='circle',
            shape_unexpanded_nt='box',
            shape_t='diamond',
            fontcolor_nt='red',
            fontcolor_unexpanded_nt='blue',
            fontcolor_t='green',
            fillcolor_nt='yellow',
            fillcolor_unexpanded_nt='black',
            fillcolor_t='white')
        check_figure(fig2)
    
    # Display
    fig = dt.plot()
    fig.display(inline=False)
    fig.display(inline=True)



# Grammars for neighborhood tests

BNF1 = """
<S> ::= 0 | 1 | 2
"""

BNF2 = """
<S> ::= <A> <B>
<A> ::= 0 | 1 | 2
<B> ::= a | b | c
"""

BNF3 = """
<S> ::= <A> <B>
<A> ::= <AA>
<AA> ::= 0 | 1 | 2
<B> ::= <BB>
<BB> ::= <BBB>
<BBB> ::= <BBBB>
<BBBB> ::= a | b | c
"""

BNF4 = """
<byte> ::= <bit> <bit> <bit> <bit> <bit> <bit> <bit> <bit>
<bit> ::= 0 | 1
"""

BNF5 = """
<S> ::= 1 | 2 | 3 | 4 | 5
"""

BNF6 = """
<S> ::= <T> | <U> | 5
<T> ::= 1 | 2
<U> ::= 3 | 4
"""

BNF7 = """
<S> ::= a <A> | b <B>
<A> ::= c <C> | d <D>
<B> ::= e <E> | f <F>
<C> ::= 1 | 2
<D> ::= 3 | 4
<E> ::= 5 | 6
<F> ::= 7 | 8
"""

BNF8 = """
<S> ::= <S> <S> | <A> | <A> 0 <B> | 1 <B>
<A> ::= a | t
<B> ::= g | c
"""

BNF9 = """
<S> ::= <text> | <number>
<number> ::= <digit> | <digit> <digit>
<text> ::= <char> | <char> <char>
<digit> ::= 1 | 2 | 3
<char> ::= a | b | c
"""

BNF10 = """
<S> ::= <S><S> | <A> x | <B> y | <C>
<A> ::= 1 | 2 | 3
<B> ::= 4 | 5 | 6
<C> ::= <C><C> | 7
"""

BNF11 = """
<S> ::= <A> | <B> | <C>
<A> ::= <B><B> | 1
<B> ::= <C><C> | 2
<C> ::= 3 | 4
"""

BNF12 = """
<S> ::= 1 <A> 1 | 2 <B> 2
<A> ::= <S> <S> | 3
<B> ::= 4 | <A> <A> | <C>
<C> ::= <S>
"""
