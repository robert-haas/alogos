"""Forward and reverse mapping functions for CFG-GP."""

import copy as _copy

from ... import _grammar
from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import default_parameters as _dp
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_GT = _representation.Genotype
_DT = _grammar.data_structures.DerivationTree
_ND = _grammar.data_structures.Node
_NT = _grammar.data_structures.NonterminalSymbol
_T = _grammar.data_structures.TerminalSymbol


# Forward mapping
def forward(
    grammar,
    genotype,
    parameters=None,
    raise_errors=True,
    return_derivation_tree=False,
    verbose=False,
):
    """Map a CFG-GP genotype to a string phenotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype` or data that can be converted to it
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``max_expansions``  (`int`): Maximum number of nonterminal
          expansions allowed in the derivation created by the mapping
          process.
    raise_errors : `bool`, optional
        Possible values:

        - `True`: A mapping error will be raised if a derivation is
          not finished within a limit provided in the parameters.
        - `False`: A partial derivation is allowed. In this case, the
          returned string will contain unexpanded nonterminal symbols.
          Therefore it is not a valid phenotype, i.e. not a string of
          the grammar's language but a so-called sentential form.
    return_derivation_tree : `bool`, optional
        If `True`, not only the phenotype is returned but additionally
        also the derivation tree.
    verbose : `bool`, optional
        If `True`, output about steps of the mapping process is printed.

    Returns
    -------
    phenotype : `str`
        If ``return_derivation_tree`` is `False`, which is the default.
    (phenotype, derivation_tree) : `tuple` with two elements of type `str` and `~alogos._grammar.data_structures.DerivationTree`
        If ``return_derivation_tree`` is `True`.

    Raises
    ------
    MappingError
        If ``raise_errors`` is `True` and the mapping process can not
        generate a full derivation before reaching a limit provided in
        the parameters.

    """
    # Parameter extraction
    me = _get_given_or_default("max_expansions", parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Mapping
    if verbose:
        dt = _forward_slow(grammar, genotype, me, raise_errors, verbose)
    else:
        dt = _forward_fast(grammar, genotype, me, raise_errors)
    phe = dt.string()

    # Conditional return
    if return_derivation_tree:
        return phe, dt
    return phe


def _forward_fast(gr, gt, me, re):
    """Calculate the genotype-to-phenotype map of CFG-GP in a fast way."""
    dt = gt.data

    # Limit check
    if me is not None and me < dt.num_expansions():
        if re:
            _exceptions.raise_max_expansion_error(me)
        dt = _copy_tree_truncated(gr, dt, me)
    return dt


def _forward_slow(grammar, genotype, max_expansions, raise_errors, verbose):
    """Calculate the genotype-to-phenotype map of CFG-GP in a slow way.

    This is a readable implementation of the mapping process, which
    also allows to print output about the steps it involves.
    It served as basis for the faster, minified implementation in this
    module and may be helpful in understanding, replicating or modifying
    the algorithm.

    """
    dt = genotype.data

    if verbose:
        header = "Start reading the phenotype directly from the genotype"
        print(header)
        print("=" * len(header))
    terminals = []
    for i, nd in enumerate(dt.nodes()):
        sym_text = nd.symbol.text
        if verbose:
            sym_type = "nonterminal" if isinstance(nd.symbol, _NT) else "terminal"
            num_children = len(nd.children)
            text = '- Entry {}: Symbol "{}" is a {} with {} children'.format(
                i, sym_text, sym_type, num_children
            )
            print(text)
        if isinstance(nd.symbol, _T):
            terminals.append(sym_text)
    phe = "".join(terminals)
    if verbose:
        print()
        header = "End of reading"
        print(header)
        print("=" * len(header))
        print("- Collected terminals in order of discovery: {}".format(terminals))
        print()
        print("String: {}".format(phe))

    # Limit check
    if max_expansions is not None and max_expansions < dt.num_expansions():
        if raise_errors:
            _exceptions.raise_max_expansion_error(max_expansions)
        dt = _copy_tree_truncated(grammar, dt, max_expansions)
    return dt


def _copy_tree_truncated(grammar, dt_ori, max_expansions):
    """Create a copy of a derivation tree that contains only a part of the nodes."""

    def copy_node_shallow(nd):
        return _ND(_copy.copy(nd.symbol))

    dt_new = _DT(grammar)
    dt_new.root_node = copy_node_shallow(dt_ori.root_node)
    stack = [(dt_ori.root_node, dt_new.root_node)]
    num_expansions = 0
    while stack:
        if num_expansions >= max_expansions:
            break
        node_given, node_new = stack.pop(0)
        if node_given.children:
            num_expansions += 1
            node_new.children = [copy_node_shallow(ch) for ch in node_given.children]
            stack = [
                (c1, c2) for c1, c2 in zip(node_given.children, node_new.children)
            ] + stack
    return dt_new


# Reverse mapping
def reverse(
    grammar, phenotype_or_derivation_tree, parameters=None, return_derivation_tree=False
):
    """Map a string phenotype (or derivation tree) to a CFG-GP genotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    phenotype_or_derivation_tree : `str` or `~alogos._grammar.data_structures.DerivationTree`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        No keyword-value pairs are considered by this function.
        This argument is only available to have a consistent interface.
    return_derivation_tree : `bool`, optional
        If `True`, not only the genotype is returned but additionally
        also the derivation tree.

    Returns
    -------
    genotype : `~.representation.Genotype`
        If ``return_derivation_tree`` is `False`, which is the default.
    (genotype, derivation_tree) : `tuple` with two elements of type `~.representation.Genotype` and `~alogos._grammar.data_structures.DerivationTree`
        If ``return_derivation_tree`` is `True`.

    Raises
    ------
    MappingError
        If the reverse mapping fails because the string does not belong
        to the grammar's language or the derivation tree does not
        represent a valid derivation.

    """
    # Prepare data structures
    dt = _shared.mapping.get_derivation_tree(grammar, phenotype_or_derivation_tree)

    # Check integrity
    known_sym = set(grammar.nonterminal_symbols).union(grammar.terminal_symbols)
    for node in dt.nodes():
        if node.symbol not in known_sym:
            _exceptions.raise_missing_nt_error(node)

    # Finalize data structures
    gt = _GT(dt)

    # Conditional return
    if return_derivation_tree:
        return gt, dt
    return gt
