import copy as _copy

from . import default_parameters as _dp
from . import representation as _representation
from .. import _shared
from ... import _grammar
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from ... import exceptions as _exceptions


# Shortcuts for brevity and minor speedup
_GT = _representation.Genotype
_DT = _grammar.data_structures.DerivationTree
_ND = _grammar.data_structures.Node


# Forward mapping
def forward(grammar, genotype, parameters=None,
            raise_errors=True, return_derivation_tree=False, verbose=False):
    """Map a CFG-GP genotype to a string phenotype."""
    # Parameter extraction
    me = _get_given_or_default('max_expansions', parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Transformation
    dt = genotype.data
    if me is not None and me < dt.num_expansions():
        if raise_errors:
            _exceptions.raise_max_expansion_error(me)
        dt = _copy_tree_truncated(grammar, dt, me)
    phe = dt.string()

    # Conditional return
    if return_derivation_tree:
        return phe, dt
    return phe


def _copy_tree_truncated(grammar, dt_ori, max_expansions):
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
            stack = [(c1, c2) for c1, c2 in zip(node_given.children, node_new.children)] + stack
    return dt_new


# Reverse mapping
def reverse(grammar, phenotype_or_derivation_tree, parameters=None,
            return_derivation_tree=False):
    """Map a string phenotype (or derivation tree) to a CFG-GP genotype."""
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
