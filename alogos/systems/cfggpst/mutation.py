"""Mutation functions for CFG-GP-ST."""

import random as _random

from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import _cached_calculations
from . import default_parameters as _dp
from . import representation as _representation


# Shortcuts for minor speedup
_GT = _representation.Genotype
_fe = _representation._find_subtree_end
_rc = _random.choice
_ri = _random.randint
_rs = _random.shuffle


def subtree_replacement(grammar, genotype, parameters=None):
    """Change a randomly chosen node by attaching a randomly generated subtree.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``max_nodes`` (`int`) : Maximum number of nodes in a tree.

    Returns
    -------
    genotype : `~.representation.Genotype`
        Mutated genotype.

    Notes
    -----
    The limiation of the tree size via `max_nodes` is only a soft
    constraint, because the tree branches are grown randomly and
    independently from each other. To finish each branch it can be
    necessary to go beyond the node limit, because it is checked when
    opening a branch, but only considering existing nodes and not all
    nodes still required for each other unfinished branch. It is
    possible to make it a hard constraint, but would require more
    computation and memory, as well as likely not improving the
    sampling of the search space.

    References
    ----------
    - `Grammatically-based Genetic Programming (1995)
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.2091>`__

        - "Mutation applies to a single program. A program is selected
          for mutation, and one non-terminal is randomly selected as the
          site for mutation. The tree below this non-terminal is
          deleted, and a new tree randomly generated from the grammar
          using this non-terminal as a starting point. The tree is
          limited in total depth by the current maximum allowable
          program depth (MAX-TREE-DEPTH), in an operation similar to
          creating the initial population."

    """
    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Parameter extraction
    mn = _get_given_or_default("max_nodes", parameters, _dp)

    # Mutation
    # - Random choice of a nonterminal to mutate
    s1, c1 = genotype.data
    a1 = _rc([i for i, c in enumerate(c1) if c])  # choose idx of a random nonterminal
    b1 = _fe(a1, c1) + 1  # find idx of last symbol in its subtree
    # - Grow a random new subtree
    s2, c2 = _grow_random_subtree(grammar, s1[a1], mn - len(s1) + (b1 - a1))
    # - New genotype: replace the subtree of the chosen nonterminal with the new subtree
    data = (s1[:a1] + s2 + s1[b1:], c1[:a1] + c2 + c1[b1:])
    return _GT(data)


def _grow_random_subtree(grammar, sym, max_nodes):
    # Caching
    imn = grammar._lookup_or_calc(
        "cfggpst",
        "idx_min_nodes",
        _shared._cached_calculations.idx_min_nodes_to_terminals,
        grammar,
    )
    ipr = grammar._lookup_or_calc(
        "cfggpst",
        "idx_production_rules",
        _cached_calculations.idx_production_rules,
        grammar,
    )

    # Construction of a tree in form of two lists
    def flatten(seq):
        # quick and dirty list flattening: [[1, 2], [3]] => [1, 2, 3]
        return sum(seq, [])

    def trav(sy):
        nonlocal n
        n += 1  # counter for each node added to the tree
        try:
            # 1) Symbol is a nonterminal, therefore requires expansion and will have >0 children
            # Choose rule: randomly from those rules that do not lead over the wanted num nodes
            rhs = _rc(_filter_rules(sy, ipr[sy], imn[sy], max_nodes - n))
            # Expand the nonterminal with the rhs of the chosen rule
            m = len(rhs)
            r = list(range(m))
            _rs(r)
            ret = [None] * m
            for i in r:
                # Recursive call for each child in random order, return in original order (dfs)
                ret[i] = trav(rhs[i])
            symbols = [sy] + flatten(x[0] for x in ret)
            counts = [m] + flatten(x[1] for x in ret)
        except KeyError:
            # 2) Symbol is a terminal, therefore requires no expansion and will have 0 children
            symbols = [sy]
            counts = [0]
        return symbols, counts

    n = -1
    symbols, counts = trav(sym)
    return tuple(symbols), tuple(counts)


def _filter_rules(sym, rules, mn, limit):
    # Try to choose rules that do not lead the tree to grow beyond max_nodes
    r = [r for r, n in zip(rules, mn) if n <= limit]
    # If not possible, choose those rules that share the lowest number of added nodes
    if not r:
        v = min(mn)
        r = [r for r, n in zip(rules, mn) if n == v]
    return r
