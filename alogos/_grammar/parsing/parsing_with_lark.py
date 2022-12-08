import lark as _lark

from ... import exceptions as _exceptions
from ..._utilities import argument_processing as _ap
from ..._utilities.operating_system import NEWLINE as _NEWLINE
from .. import data_structures as _data_structures


# Shortcuts
_NT = _data_structures.NonterminalSymbol
_T = _data_structures.TerminalSymbol
_BACKSLASH = "\\"  # Caution: r'\' is not a valid string literal


def parse_string(grammar, string, parser, get_multiple_trees, max_num_trees=None):
    """Parse a string with Lark and construct a derivation tree of this package.

    The derivation tree is built with the data structure defined in this package,
    not with the parse tree object provided by Lark. The reason is that the
    derivation tree is subsequently used for various tasks such as different
    traversals, extracting strings or derivations and visualization. Therefore it
    desirable to have it decoupled from the parser library and that it comes
    with methods for the mentioned tasks.

    References
    ----------
    - https://github.com/lark-parser/lark
    - https://github.com/lark-parser/lark/blob/master/lark/exceptions.py

    """
    # Argument processing
    string = _ap.str_arg("string", string)
    parser = _ap.str_arg("parser", parser, vals=["earley", "lalr"])
    get_multiple_trees = _ap.bool_arg(
        "get_multiple_trees", get_multiple_trees, default=False
    )
    max_num_trees = _ap.int_arg(
        "max_num_trees", max_num_trees, default=1_000_000_000_000
    )
    if get_multiple_trees and parser != "earley":
        _exceptions.raise_lark_parser_mult_error()

    # Caching: Create the Lark parser once and reuse it in subsequent calls
    parser_id = (
        parser if parser == "lalr" else "{}{}".format(parser, int(get_multiple_trees))
    )
    lark_parser, nt_map_fwd, nt_map_rev = grammar._lookup_or_calc(
        "lark",
        parser_id,
        _calc_lark_parser_and_nt_maps,
        grammar,
        parser,
        get_multiple_trees,
    )

    # Parsing
    try:
        tree_s = lark_parser.parse(string)
    except Exception as excp:
        _exceptions._raise_parser_string_error(excp)

    # Tree conversion
    if get_multiple_trees:
        tree_s = _ambig_lark_tree_to_dts(
            grammar, tree_s, nt_map_fwd, nt_map_rev, max_num_trees
        )
    else:
        tree_s = _lark_tree_to_dt(grammar, tree_s, nt_map_rev)
    return tree_s


# Result conversion


def _lark_tree_to_dt(grammar, lark_tree, nt_map_rev):
    """Convert a parse tree of lark-parser into a derivation tree of this package."""

    def get_label(lark_node):
        if isinstance(lark_node, _lark.Tree):
            return _NT(nt_map_rev[lark_node.data].text)
        return _T(lark_node.value)

    # Generate the derivation tree
    dt = _data_structures.DerivationTree(grammar)
    dt.root_node = _data_structures.Node(grammar.start_symbol)

    # Traverse the Lark tree via dfs and generate the go derivation tree node by node
    lark_stack = [lark_tree]
    ori_stack = [dt.root_node]
    while lark_stack:
        lark_node = lark_stack.pop()
        ori_node = ori_stack.pop()
        if isinstance(lark_node, _lark.Tree):
            lark_child_nodes = lark_node.children
            child_symbols = [get_label(child) for child in lark_child_nodes]
            if not child_symbols:
                # Empty string is not a node in the Lark tree and hence needs special treatment
                empty_string = _T("")
                child_symbols = [empty_string]
                dt._expand(ori_node, child_symbols)
                continue
            ori_child_nodes = dt._expand(ori_node, child_symbols)
            lark_stack.extend(lark_child_nodes)
            ori_stack.extend(ori_child_nodes)
    return dt


def _ambig_lark_tree_to_dts(grammar, lark_tree, nt_map_fwd, nt_map_rev, max_num_trees):
    """Convert an ambiguous parse tree of lark-parser into a list of derivation trees."""
    # Get all trees from the forest provided by Lark
    stack = [lark_tree]
    lark_start_symbol = nt_map_fwd[grammar.start_symbol]
    valid_trees = []
    while stack:
        candidate_tree = stack.pop()
        node_label = candidate_tree.data
        # "_ambig" in a node means that multiple trees are present in the child nodes
        if node_label == "_ambig":
            new_candidate_trees = candidate_tree.children
            stack.extend(new_candidate_trees)
        elif node_label == lark_start_symbol:
            valid_trees.append(candidate_tree)
            # Restrict number of trees
            if len(valid_trees) >= max_num_trees:
                break
        else:
            _exceptions._raise_parser_node_error(node_label)

    # Convert Lark trees to derivation trees of this package
    dts = [_lark_tree_to_dt(grammar, tree, nt_map_rev) for tree in valid_trees]
    return dts


# Cached calculations


def _calc_lark_parser_and_nt_maps(grammar, parser, get_multiple_trees):
    """Create all Lark objects in a single function.

    This means that whenever a parser is generated, the same nonterminal map and
    grammar are generated again. This redundant calculations are accepted in
    order to improve lookup speed.

    """
    # Nonterminal maps
    nt_map_fwd, nt_map_rev = grammar._lookup_or_calc(
        "lark", "nt_maps", _calc_nonterminal_maps, grammar
    )

    # Lark grammar
    lark_grammar = grammar._lookup_or_calc(
        "lark", "grammar", _calc_lark_grammar, grammar, nt_map_fwd
    )

    # Lark parser: different parsers are created from the same nonterminal maps and Lark grammar
    lark_parser = _calc_lark_parser(
        grammar, parser, nt_map_fwd, lark_grammar, get_multiple_trees
    )
    return lark_parser, nt_map_fwd, nt_map_rev


def _calc_nonterminal_maps(grammar):
    """Create a mapping between nonterminals of the original grammar and simplified symbols."""
    nt_map_fwd = {
        nt: "nt{}".format(i) for i, nt in enumerate(grammar.nonterminal_symbols)
    }
    nt_map_rev = {val: key for key, val in nt_map_fwd.items()}
    return nt_map_fwd, nt_map_rev


def _calc_lark_grammar(grammar, nt_map_fwd):
    """Convert the grammar object into an EBNF-like text that Lark is guaranteed to recognize.

    References
    ----------
    - https://lark-parser.readthedocs.io/en/latest/grammar

    """
    parts = []
    for lhs, rhs_multiple in grammar.production_rules.items():
        # Add a new lhs
        parts.append(_NEWLINE)
        used_text = nt_map_fwd[lhs]
        parts.append(used_text + ": ")

        # Add one or several rhs to this new lhs. If several, they are separted by | as "or" symbol
        for rhs in rhs_multiple:
            for symbol in rhs:
                if isinstance(symbol, _NT):
                    parts.append(nt_map_fwd[symbol] + " ")
                else:
                    parts.append(_to_lark_terminal(symbol.text) + " ")
            parts.append("| ")
        parts.pop()  # Remove last separator symbol
    parts.pop(0)  # Remove first newline
    lark_grammar = "".join(parts)
    return lark_grammar


def _calc_lark_parser(grammar, parser, nt_map_fwd, lark_grammar, get_multiple_trees):
    """Create a lark parser with or without ambiguity detection.

    References
    ----------
    - https://lark-parser.readthedocs.io/en/latest/classes/

    """
    # Argument processing
    specific_options = dict()
    if parser == "earley" and get_multiple_trees:
        specific_options = dict(ambiguity="explicit")

    # Create the parser
    try:
        lark_start_symbol = nt_map_fwd[grammar.start_symbol]
        lark_parser = _lark.Lark(
            grammar=lark_grammar,
            parser=parser,
            start=lark_start_symbol,
            keep_all_tokens=True,
            **specific_options,
        )
    except Exception as excp:
        _exceptions._raise_parser_creation_error(excp)
    return lark_parser


def _to_lark_terminal(symbol):
    """Prevent problems with certain characters inside the text of a terminal.

    References
    ----------
    - https://stackoverflow.com/questions/647769/why-cant-pythons-raw-string-literals-end-with-a-single-backslash

    """
    if symbol:
        # A non-even number of backslashes at the end of a symbol would escape the last quote
        if symbol.endswith(_BACKSLASH):
            num_trailing_backslashes = len(symbol) - len(symbol.rstrip(_BACKSLASH))
            uneven_num_trailing_backslashes = num_trailing_backslashes % 2 == 1
            if uneven_num_trailing_backslashes:
                symbol = symbol + _BACKSLASH

        # A double quote symbol would end the terminal too early and lead to downstream problems
        if '"' in symbol:
            symbol = symbol.replace('"', r"\"")

        # For some reason a newline symbol is not allowed and needs to be replaced with its
        # escape sequence representation
        if "\n" in symbol:
            symbol = symbol.replace("\n", r"\n")

        # Enclose the safe symbol in double quotes for Lark
        lark_terminal_symbol = '"{}"'.format(symbol)
    else:
        lark_terminal_symbol = ""
    return lark_terminal_symbol
