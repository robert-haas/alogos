import copy
import inspect
import os
import random
import time
import warnings
from collections.abc import Callable

import ordered_set
import pytest

import alogos as al


# Number of repetitions for methods with randomness

NUM_REPETITIONS = 20


# Helper functions


def emits_warning(func, expected_type, expected_message):
    # https://docs.python.org/3/library/warnings.html#testing-warnings
    with warnings.catch_warnings(record=True) as w:
        func()

        if len(w) == 0:
            raise ValueError("No warning was issued")

        warning = w[-1]
        found_type = warning.category
        if not issubclass(found_type, expected_type):
            raise TypeError(
                "The wrong type of warning was issued.\nExpected: {}\nFound: {}".format(
                    expected_type, found_type
                )
            )

        found_message = str(warning.message)
        if expected_message not in found_message:
            raise ValueError(
                "The wrong warning message was issued.\nExpected: {}\nFound: {}".format(
                    repr(expected_message), repr(found_message)
                )
            )


def emits_no_warning(func):
    with warnings.catch_warnings(record=True) as w:
        func()
        if len(w):
            warning = w[-1]
            found_type = warning.category
            found_message = str(warning.message)
            raise ValueError(
                "At least one warning was issued:\nType: {}\nMessage: {}".format(
                    found_type, found_message
                )
            )


def emits_exception(function, error_type, expected_message=None):
    # https://docs.pytest.org/en/latest/assert.html#assertions-about-expected-exceptions
    with pytest.raises(error_type) as excp:
        function()
    if expected_message:
        given_message = str(excp.value)
        assert expected_message == given_message


def prints_to_stdout(
    func,
    capsys,
    message=None,
    partial_message=None,
    start_message=None,
    end_message=None,
):
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
    assert out == ""


def get_path_of_this_file():
    return os.path.abspath(inspect.getsourcefile(lambda _: None))


def conv_keys_to_str(given_dict):
    return {str(key): val for key, val in given_dict.items()}


class MockPrettyPrinter:
    def __init__(self):
        self.string = ""

    def text(self, s):
        self.string += s


def call_repr_pretty(obj, cycle):
    mock_printer = MockPrettyPrinter()
    obj._repr_pretty_(mock_printer, cycle=cycle)
    return mock_printer.string


def filter_list_unique(items):
    # Keeps the original order in constrast to list(set(items))
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


# Grammar checks

GRAMMAR = al._grammar.data_structures.Grammar(bnf_text="<S> ::= 0 | 1")
TREE = GRAMMAR.parse_string("0")


def check_grammar(grammar, pos_examples, neg_examples, language=None, max_steps=None):
    # Type
    assert isinstance(grammar, al._grammar.data_structures.Grammar)

    # Attributes
    assert isinstance(grammar.nonterminal_symbols, ordered_set.OrderedSet)
    assert isinstance(grammar.terminal_symbols, ordered_set.OrderedSet)
    assert isinstance(grammar.production_rules, dict)
    assert isinstance(
        grammar.start_symbol, al._grammar.data_structures.NonterminalSymbol
    )
    assert not hasattr(grammar, "_cache") or isinstance(grammar._cache, dict)

    assert set(grammar.production_rules) == set(grammar.nonterminal_symbols)
    for multiple_rhs in grammar.production_rules.values():
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
        assert s2.startswith("<Grammar object at") and s2.endswith(">")

    # Equality
    assert grammar == gr2 == gr3 == gr4 != gr5
    assert not grammar != gr2
    for other in (GRAMMAR, gr5, 5, 3.14, "a", None, tuple(), set()):
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
    gr_dict[grammar] = "a"
    gr_dict[gr2] = "b"
    gr_dict[gr3] = "c"
    gr_dict[gr4] = "d"
    assert len(gr_dict) == 1

    # Reading and writing
    # - Duplicate the grammar from BNF output to see if grammar -> BNF -> grammar holds
    try:
        # BNF text
        grammar_bnf = grammar.to_bnf_text()
        assert grammar_bnf
        assert isinstance(grammar_bnf, str)
        # BNF file
        grammar.to_bnf_file("exported_grammar.bnf")
        with open("exported_grammar.bnf") as f:
            assert grammar_bnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al.Grammar()
        grammar2.from_bnf_text(grammar_bnf)
        grammar3 = al.Grammar(bnf_file="exported_grammar.bnf")
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile("exported_grammar.bnf"):
            os.remove("exported_grammar.bnf")

    # - Duplicate the grammar from BNF v2 output to see if grammar -> BNF -> grammar holds
    try:
        kwargs = dict(
            start_terminal_symbol='"',
            end_terminal_symbol='"',
            start_terminal_symbol2="'",
            end_terminal_symbol2="'",
        )
        # BNF text
        grammar_bnf = grammar.to_bnf_text(**kwargs)
        assert grammar_bnf
        assert isinstance(grammar_bnf, str)
        # BNF file
        grammar.to_bnf_file("exported_grammar.bnf", **kwargs)
        with open("exported_grammar.bnf") as f:
            assert grammar_bnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al.Grammar()
        grammar2.from_bnf_text(grammar_bnf, **kwargs)
        grammar3 = al.Grammar(bnf_file="exported_grammar.bnf", **kwargs)
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile("exported_grammar.bnf"):
            os.remove("exported_grammar.bnf")

    # - Duplicate the grammar from EBNF output to see if grammar -> EBNF -> grammar holds
    try:
        # EBNF text
        grammar_ebnf = grammar.to_ebnf_text()
        assert grammar_ebnf
        assert isinstance(grammar_ebnf, str)
        # EBNF file
        grammar.to_ebnf_file("exported_grammar.ebnf")
        with open("exported_grammar.ebnf") as f:
            assert grammar_ebnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al._grammar.data_structures.Grammar(
            ebnf_file="exported_grammar.ebnf"
        )
        grammar3 = al._grammar.data_structures.Grammar()
        grammar3.from_ebnf_file("exported_grammar.ebnf")
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile("exported_grammar.ebnf"):
            os.remove("exported_grammar.ebnf")

    # - Duplicate the grammar from EBNF v2 output to see if grammar -> EBNF -> grammar holds
    try:
        kwargs = dict(start_nonterminal_symbol="<", end_nonterminal_symbol=">")
        # EBNF text
        grammar_ebnf = grammar.to_ebnf_text(**kwargs)
        assert grammar_ebnf
        assert isinstance(grammar_ebnf, str)
        # EBNF file
        grammar.to_ebnf_file("exported_grammar.ebnf", **kwargs)
        with open("exported_grammar.ebnf") as f:
            assert grammar_ebnf == f.read()
    except al.exceptions.GrammarError:
        # If export failed with correct error, ignore it, the user has to choose suitable marks
        pass
    else:
        # If export worked, read and compare
        grammar2 = al._grammar.data_structures.Grammar(
            ebnf_file="exported_grammar.ebnf", **kwargs
        )
        grammar3 = al._grammar.data_structures.Grammar()
        grammar3.from_ebnf_file("exported_grammar.ebnf", **kwargs)
        assert grammar == grammar2 == grammar3
    finally:
        # Cleanup
        if os.path.isfile("exported_grammar.ebnf"):
            os.remove("exported_grammar.ebnf")

    # Visualization
    # - in separate test for reasons of performance

    # Normal forms
    # - CNF
    assert isinstance(grammar._is_cnf(), bool)
    gr_cnf = grammar._to_cnf()
    assert gr_cnf._is_cnf()
    if grammar._is_cnf():
        grammar == gr_cnf  # noqa: B015
    else:
        grammar != gr_cnf  # noqa: B015
    # - GNF
    with pytest.raises(NotImplementedError):
        assert not grammar._is_gnf()
    with pytest.raises(NotImplementedError):
        gr_gnf = grammar._to_gnf()
        assert gr_gnf._is_gnf()
    # - BCF
    assert isinstance(grammar._is_bcf(), bool)
    gr_bcf = grammar._to_bcf()
    assert gr_bcf._is_bcf()
    if grammar._is_bcf():
        grammar == gr_bcf  # noqa: B015
    else:
        grammar != gr_bcf  # noqa: B015

    # Derivation tree generation
    grammars = [grammar, gr_cnf, gr_bcf]
    for _ in range(30):
        # Generate a random string (with the grammar or a random normal form)
        gr = random.choice(grammars)
        dt = gr.generate_derivation_tree(reduction_factor=0.2)
        string = dt.string()
        assert isinstance(string, str)
        # Check if it is recognized (by the grammar and all its normal forms)
        grammar.recognize_string(string)
        gr_cnf.recognize_string(string)
        gr_bcf.recognize_string(string)

    # String generation
    grammars = [grammar, gr_cnf, gr_bcf]
    for _ in range(30):
        # Generate a random string (with the grammar or a random normal form)
        gr = random.choice(grammars)
        string = gr.generate_string(reduction_factor=0.2)
        assert isinstance(string, str)
        # Check if it is recognized (by the grammar and all its normal forms)
        grammar.recognize_string(string)
        gr_cnf.recognize_string(string)
        gr_bcf.recognize_string(string)

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
            gr_bcf.recognize_string(string)

    # Parsing of strings in the grammar's language
    for string in pos_examples:
        # Recognize
        assert grammar.recognize_string(string)
        # Parse
        derivation_tree = grammar.parse_string(string)
        # Check
        assert derivation_tree.string() == string
        assert "".join(str(x) for x in derivation_tree.tokens()) == string

    # Parsing of strings that are NOT in the grammar's language
    for string in neg_examples:
        assert not grammar.recognize_string(string)
        assert not gr_cnf.recognize_string(string)
        assert not gr_bcf.recognize_string(string)
        with pytest.raises(al.exceptions.ParserError):
            grammar.parse_string(string)
        with pytest.raises(al.exceptions.ParserError):
            gr_cnf.parse_string(string)
        with pytest.raises(al.exceptions.ParserError):
            gr_bcf.parse_string(string)


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
    assert not hasattr(dt, "_cache") or isinstance(dt._cache, dict)

    # Copying
    dt2 = dt.copy()
    dt3 = copy.copy(dt)
    dt4 = copy.deepcopy(dt)

    # Representations
    class MP:
        def __init__(self):
            self.string = ""

        def text(self, s):
            self.string += s

    s1 = str(dt)
    s2 = repr(dt2)
    s3 = dt3._repr_html_()
    s4 = dt4.to_parenthesis_notation()
    s5 = dt4.to_tree_notation()
    s6 = call_repr_pretty(dt3, cycle=True)
    s7 = call_repr_pretty(dt4, cycle=False)
    for s in (s1, s2, s3, s4, s5, s6, s7):
        assert s and isinstance(s, str)
    assert s2.startswith("<DerivationTree object at") and s2.endswith(">")
    assert s3.startswith("<")
    for sym in ("<", ">", "(", ")"):
        assert sym in s4
    for sym in ("<", ">"):
        assert sym in s5
    for sym in (" ", os.linesep):
        assert sym not in s4
        assert sym in s5

    # Equality
    assert dt == dt2
    assert not dt != dt2
    for other in (TREE, 5, 3.14, "a", None, tuple(), set()):
        assert dt != other
        assert other != dt
        assert not dt == other
        assert not other == dt

    # Independence of copied objects observed by inequality
    dt5 = dt.copy()
    assert dt == dt5
    dt5.root_node.symbol.text = "nonsense"
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
    tree_dict[dt] = "a"
    tree_dict[dt2] = "b"
    tree_dict[dt3] = "c"
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
    nodes_dfs2 = dt.nodes(order="dfs")
    nodes_bfs = dt.nodes(order="bfs")
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
    assert all(
        isinstance(node.symbol, al._grammar.data_structures.NonterminalSymbol)
        for node in internal_nodes
    )

    # Read tokens from leaves of the tree
    tokens = dt.tokens()
    assert tokens
    assert isinstance(tokens, list)
    assert isinstance(tokens[0].text, str)

    # Read string from leaves of the tree
    string = dt.string()
    assert string
    assert isinstance(string, str)
    assert string == "".join(node.symbol.text for node in leaf_nodes)
    string = dt.string(separator="-")
    assert string
    assert isinstance(string, str)
    assert string == "-".join(node.symbol.text for node in leaf_nodes)

    # Read derivation from tree
    string = dt.derivation()
    assert string
    assert isinstance(string, str)
    for do in ("leftmost", "rightmost", "random"):
        for nl in (True, False):
            string = dt.derivation(derivation_order=do, newline=nl)
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
    is_deeper_than_depth_min_1 = dt._is_deeper_than(depth - 1)
    assert is_deeper_than_depth_min_1 and isinstance(is_deeper_than_depth_min_1, bool)


def check_figure(fig):
    # Type
    assert isinstance(
        fig, al._grammar.visualization.tree_with_graphviz.DerivationTreeFigure
    )

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
    assert s2.startswith("<DerivationTreeFigure object at") and s2.endswith(">")
    assert "<" in s3 and ">" in s3
    assert "<" in s3 and ">" in s4
    assert "<" in s3 and ">" in s5
    assert "<" in s3 and ">" in s6
    assert "<" in s3 and ">" in s7

    # File export
    for fileformat in ("dot", "eps", "gv", "html", "pdf", "png", "ps", "svg"):
        method_name = "export_{}".format(fileformat)
        method = getattr(fig, method_name)
        used_filepath = method("test")
        assert isinstance(used_filepath, str)
        assert used_filepath.endswith(fileformat)
        os.remove(used_filepath)


# Evolutionary algorithm


def check_ea_algorithm(ea):
    # Type
    assert isinstance(ea, al.EvolutionaryAlgorithm)

    # Attributes
    assert isinstance(ea.parameters, al._utilities.parametrization.ParameterCollection)
    assert isinstance(ea.parameters.grammar, al.Grammar)
    assert isinstance(ea.parameters.objective_function, Callable)
    assert isinstance(ea.parameters.objective, str)
    assert ea.parameters.objective in ("min", "max")
    assert not isinstance(ea.parameters.system, str)
    assert isinstance(ea.parameters.evaluator, Callable)

    assert isinstance(ea.state, al._optimization.ea.state.State)
    if ea.state.generation == 0:
        assert ea.state.best_individual is None
    else:
        assert isinstance(
            ea.state.best_individual, ea.parameters.system.representation.Individual
        )

    if ea.parameters.database_on:
        assert isinstance(ea.database, al._optimization.ea.database.Database)
    else:
        assert ea.database is None

    # Methods
    assert isinstance(ea.is_stop_criterion_met, Callable)
    assert isinstance(ea.reset, Callable)
    assert isinstance(ea.run, Callable)
    assert isinstance(ea.step, Callable)

    # Representations
    s1 = repr(ea)
    s2 = str(ea)
    s3 = call_repr_pretty(ea, cycle=True)
    s4 = call_repr_pretty(ea, cycle=False)
    s5 = repr(ea.parameters)
    s6 = str(ea.parameters)
    s7 = call_repr_pretty(ea.parameters, cycle=True)
    s8 = call_repr_pretty(ea.parameters, cycle=False)
    s9 = repr(ea.state)
    s10 = str(ea.state)
    s11 = call_repr_pretty(ea.state, cycle=True)
    s12 = call_repr_pretty(ea.state, cycle=False)
    if ea.parameters.database_on:
        s13 = repr(ea.database)
        s14 = str(ea.database)
        s15 = call_repr_pretty(ea.database, cycle=True)
        s16 = call_repr_pretty(ea.database, cycle=False)
    else:
        s13, s14, s15, s16 = s9, s10, s11, s12
    for string in (
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
        s14,
        s15,
        s16,
    ):
        assert isinstance(string, str)
        assert string
    assert s1.startswith("<EvolutionaryAlgorithm object at") and s1.endswith(">")
    assert s5.startswith("<ParameterCollection object at") and s5.endswith(">")
    assert s9.startswith("<EvolutionaryAlgorithmState object at") and s9.endswith(">")
    if ea.parameters.database_on:
        assert s13.startswith(
            "<EvolutionaryAlgorithmDatabase object at"
        ) and s13.endswith(">")

    # Database
    if ea.database is not None:
        check_ea_database(ea)

    # State
    has_run = ea.state.generation > 0
    check_ea_state(ea, has_run)


def check_ea_state(ea, has_run):
    if has_run:
        assert ea.state.generation > 0
        assert ea.state.num_gen_to_phe_evaluations > 0
        assert ea.state.num_phe_to_fit_evaluations > 0
        assert ea.state.best_individual is not None
        assert ea.state.max_individual is not None
        assert ea.state.min_individual is not None
        assert ea.state.population is not None
    else:
        assert ea.state.generation == 0
        assert ea.state.num_gen_to_phe_evaluations == 0
        assert ea.state.num_phe_to_fit_evaluations == 0
        assert ea.state.best_individual is None
        assert ea.state.max_individual is None
        assert ea.state.min_individual is None
        assert ea.state.population is None


def check_ea_database(ea):
    db = ea.database
    system = db._deserializer._system
    for gen_range in [None, 0, (0, 1), [0, 1], (0, None), (None, 0), [None, None]]:
        for only_main in (False, True):
            # Counts
            num_gen = db.num_generations()
            num_ind = db.num_individuals(gen_range, only_main)
            num_gt = db.num_genotypes(gen_range, only_main)
            num_phe = db.num_phenotypes(gen_range, only_main)
            num_fit = db.num_fitnesses(gen_range, only_main)
            num_det = db.num_details(gen_range, only_main)
            num_gen_phe = db.num_gen_to_phe_evaluations()
            num_phe_fit = db.num_phe_to_fit_evaluations()
            counts = (
                num_gen,
                num_ind,
                num_gt,
                num_phe,
                num_fit,
                num_det,
                num_gen_phe,
                num_phe_fit,
            )
            for val in counts:
                assert isinstance(val, int)
                if num_gen == 0:
                    assert val == 0
                else:
                    assert val > 0

            # Generation
            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db.generation_first()
                with pytest.raises(al.exceptions.DatabaseError):
                    db.generation_last()
            else:
                gen_first = db.generation_first()
                gen_last = db.generation_last()
                if gen_first is not None or gen_last is not None:
                    assert gen_last >= gen_first
                    for val in (gen_first, gen_last):
                        assert isinstance(val, int)
                        assert val >= 0

            # Individual
            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db._individual_max_id(gen_range, only_main)
            else:
                ind_max_id = db._individual_max_id(gen_range, only_main)
                for val in (ind_max_id,):
                    if val is not None:
                        assert isinstance(val, int)
                        assert val >= 0

            ind = db.individuals(gen_range, only_main)
            some_fitness = 0.0 if len(ind) == 0 else ind[0].fitness
            ind_exact = db.individuals_with_given_fitness(
                some_fitness, gen_range, only_main
            )
            ind_min = db.individuals_with_min_fitness(gen_range, only_main)
            ind_max = db.individuals_with_max_fitness(gen_range, only_main)
            ind_low = db.individuals_with_low_fitness(
                generation_range=gen_range, only_main=only_main
            )
            ind_low2 = db.individuals_with_low_fitness(2, gen_range, only_main)
            ind_high = db.individuals_with_high_fitness(
                generation_range=gen_range, only_main=only_main
            )
            ind_high2 = db.individuals_with_high_fitness(2, gen_range, only_main)
            vals = (
                ind,
                ind_exact,
                ind_min,
                ind_max,
                ind_low,
                ind_low2,
                ind_high,
                ind_high2,
            )
            for val in vals:
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                    assert all(
                        isinstance(ind, system.representation.Individual) for ind in val
                    )

            # Population
            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db.population_size_min()
                with pytest.raises(al.exceptions.DatabaseError):
                    db.population_size_max()
            else:
                pop_size_min = db.population_size_min()
                pop_size_max = db.population_size_max()
                for val in (pop_size_min, pop_size_max):
                    if val is not None:
                        assert isinstance(val, int)
                        assert val >= 1

            # Genotype
            gt = db.genotypes(gen_range, only_main)
            gt_exact = db.genotypes_with_given_fitness(
                some_fitness, gen_range, only_main
            )
            gt_min = db.genotypes_with_min_fitness(gen_range, only_main)
            gt_max = db.genotypes_with_max_fitness(gen_range, only_main)
            for val in (gt, gt_exact, gt_min, gt_max):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                    assert all(
                        isinstance(gt, system.representation.Genotype) for gt in val
                    )

            # Phenotype
            phe = db.phenotypes(gen_range, only_main)
            phe_exact = db.phenotypes_with_given_fitness(
                some_fitness, gen_range, only_main
            )
            phe_min = db.phenotypes_with_min_fitness(gen_range, only_main)
            phe_max = db.phenotypes_with_max_fitness(gen_range, only_main)
            for val in (phe, phe_exact, phe_min, phe_max):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                    assert all(isinstance(phe, str) for phe in val)

            # Details
            det = db.details(gen_range, only_main)
            det_exact = db.details_with_given_fitness(
                some_fitness, gen_range, only_main
            )
            det_min = db.details_with_min_fitness(gen_range, only_main)
            det_max = db.details_with_max_fitness(gen_range, only_main)
            for val in (det, det_exact, det_min, det_max):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0

            # Fitness
            fitnesses = db.fitnesses(gen_range, only_main)
            assert isinstance(fitnesses, list)
            if num_gen == 0:
                assert len(fitnesses) == 0
            else:
                assert len(fitnesses) > 0
                assert all(isinstance(val, float) for val in fitnesses)

            if num_gen == 0:
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_min(gen_range, only_main)
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_max(gen_range, only_main)
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_min_after_num_evals(2)
                with pytest.raises(al.exceptions.DatabaseError):
                    db.fitness_max_after_num_evals(2)
            else:
                fit_min = db.fitness_min(gen_range, only_main)
                fit_max = db.fitness_max(gen_range, only_main)
                fit_min2 = db.fitness_min_after_num_evals(2)
                fit_max2 = db.fitness_max_after_num_evals(2)
                for val in (fit_min, fit_max, fit_min2, fit_max2):
                    assert isinstance(val, float)
                    assert val == val  # not NaN

            n = 17
            # Genotype-phenotype evaluations
            gt_phe_map1 = db.gen_to_phe_evaluations()
            gt_phe_map2 = db.gen_to_phe_evaluations(n)
            assert gt_phe_map1[:n] == gt_phe_map2
            for val in (gt_phe_map1, gt_phe_map2):
                assert isinstance(val, list)
                if num_gen == 0:
                    assert len(val) == 0
                else:
                    assert len(val) > 0
                # Type of genotypes
                assert all(
                    isinstance(row[0], system.representation.Genotype) for row in val
                )
                # Type of phenotypes
                assert all(isinstance(row[1], str) for row in val)

            # Phenotye-fitness evaluations
            for wd in (True, False):
                phe_fit_map1 = db.phe_to_fit_evaluations(with_details=wd)
                phe_fit_map2 = db.phe_to_fit_evaluations(n, wd)
                assert phe_fit_map1[:n] == phe_fit_map2
                for val in (phe_fit_map1, phe_fit_map2):
                    assert isinstance(val, list)
                    if num_gen == 0:
                        assert len(val) == 0
                    else:
                        assert len(val) > 0
                    # Length of a row depends on with_details being True or False
                    if wd:
                        assert all(len(row) == 3 for row in val)
                    else:
                        assert all(len(row) == 2 for row in val)
                    # Type of phenotypes
                    assert all(isinstance(row[0], str) for row in val)
                    # Type of fitnesses
                    assert all(isinstance(row[1], float) for row in val)

            # Connections between some of these queries
            assert len(set(gt)) == num_gt
            assert len(set(phe)) == num_phe
            assert len([str(val) for val in det]) == num_det
            assert len(set(fitnesses)) == num_fit

            for ind, gt, phe in zip(ind_min, gt_min, phe_min):
                assert ind.genotype == gt
                assert ind.phenotype == phe
                assert ind.fitness == fit_min
            for ind, gt, phe in zip(ind_max, gt_max, phe_max):
                assert ind.genotype == gt
                assert ind.phenotype == phe
                assert ind.fitness == fit_max

            # Checks against some values determined from to_list and to_dataframe
            if random.choice(
                [True, False]
            ):  # random order of calling to_list and to_dataframe
                data = ea.database.to_list(gen_range, only_main)
                df = ea.database.to_dataframe(gen_range, only_main)
            else:
                df = ea.database.to_dataframe(gen_range, only_main)
                data = ea.database.to_list(gen_range, only_main)

            if gen_range is None and len(df) > 0:
                assert (
                    len(set([row[2] for row in data]))
                    == df["generation"].nunique()
                    == num_gen
                )
                assert data[0][2] == min(df["generation"]) == db.generation_first()
                assert data[-1][2] == max(df["generation"]) == db.generation_last()

            assert len(data) == len(df) == num_ind
            assert (
                len(set([row[4] for row in data])) == df["genotype"].nunique() == num_gt
            )
            assert (
                len(set([row[5] for row in data]))
                == df["phenotype"].nunique()
                == num_phe
            )
            assert df["fitness"].nunique(dropna=False) == num_fit
            assert df["details"].nunique(dropna=False) == num_det

            assert all(isinstance(pi, list) for pi in df["parent_ids"])
            assert all(isinstance(gn, int) for gn in df["generation"])
            known_labels = ("main", "parent_selection", "crossover", "mutation")
            assert all(label in known_labels for label in df["label"])


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


# Grammars and objective functions for evolutionary algorithm tests

BNF_FLOAT = """
<number> ::= <sign><digit>.<digits>
<sign> ::= +|-
<digits> ::= <digit><digit><digit>
<digit> ::= 0|1|2|3|4|5|6|7|8|9
"""

GRAMMAR_FLOAT = al.Grammar(bnf_text=BNF_FLOAT)


def OBJ_FUN_FLOAT(string):
    x = float(string)
    return abs(x - 0.4321)


def OBJ_FUN_FLOAT_DETAILS(string):
    x = float(string)
    return abs(x - 0.4321), x  # returns string interpretation as details


BNF_TUPLE = """
<tuple> ::= (<number>, <number>)
<number> ::= <sign><digit>.<digits>
<sign> ::= +|-
<digits> ::= <digit><digit><digit>
<digit> ::= 0|1|2|3|4|5|6|7|8|9
"""

GRAMMAR_TUPLE = al.Grammar(bnf_text=BNF_TUPLE)


def OBJ_FUN_TUPLE(string):
    x, y = eval(string)
    z = (x + 0.1234) ** 2 + (y + 0.6123) ** 2
    return z


def OBJ_FUN_TUPLE_SLOW(string):
    # Artificial delay for performance comparisons of serial versus parallel evaluation
    time.sleep(0.01)
    x, y = eval(string)
    z = (x + 0.1234) ** 2 + (y + 0.6123) ** 2
    return z
