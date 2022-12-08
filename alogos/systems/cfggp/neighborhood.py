"""Neighborhood functions to generate nearby genotypes for CFG-GP."""

from ... import _grammar
from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import default_parameters as _default_parameters
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_NT = _grammar.data_structures.NonterminalSymbol
_T = _grammar.data_structures.TerminalSymbol


def subtree_replacement(grammar, genotype, parameters=None):
    """Systematically change a chosen number of nodes.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``neighborhood_distance`` (`int`) : The distance from the
          original genotype to a new genotype in terms of replaced
          subtrees.
        - ``neighborhood_max_size`` (`int`) : Maximum number of
          neighbor genotypes to generate.
        - ``neighborhood_only_terminals`` (`bool`) : If `True`, only
          replace nodes with terminals in them.

    Returns
    -------
    neighbors : `list` of `~.representation.Genotype` objects

    """
    # Parameter extraction
    distance = _get_given_or_default(
        "neighborhood_distance", parameters, _default_parameters
    )
    max_size = _get_given_or_default(
        "neighborhood_max_size", parameters, _default_parameters
    )
    only_t = _get_given_or_default(
        "neighborhood_only_terminals", parameters, _default_parameters
    )

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Get alternative choices per position by going through the productions in the tree
    dt = genotype.data
    nodes, choices = _get_choices_per_position(grammar, dt, only_t)
    num_choices_per_pos = [len(x) for x in choices]

    # Generate combinations of choices
    combinations = _shared.neighborhood.generate_combinations(
        num_choices_per_pos, distance, max_size
    )

    # Neighborhood construction
    if distance == 1:
        nbrs = _generate_nbrs_fast_for_d1(grammar, dt, nodes, choices, combinations)
    else:
        nbrs = _generate_nbrs_slow(grammar, dt, nodes, choices, combinations)
    return nbrs


def _generate_nbrs_fast_for_d1(grammar, dt, nodes, choices, combinations):
    """Quickly generate a neighborhood with distance 1."""
    nbrs = []
    for comb in combinations:
        # Semantics of comb: 0 means no change, >0 points to a certain alternative choice
        for idx, val in enumerate(comb):
            if val > 0:
                node = nodes[idx]
                rhs_idx = choices[idx][val - 1]
                rhs = grammar.production_rules[node.symbol][rhs_idx]
                # Create a new subtree
                node_new = _grow_deterministic_subtree(grammar, node, rhs)
                # Swap old node with the new one
                node.children, node_new.children = node_new.children, node.children
                # Copy the new tree
                dt_new = dt.copy()
                # Restore the original tree
                node.children, node_new.children = node_new.children, node.children
                break
        nbrs.append(_representation.Genotype(dt_new))
    return nbrs


def _generate_nbrs_slow(grammar, dt, nodes, choices, combinations):
    """Slowly generate a neighborhood with distance >1."""
    nbrs = set()
    for comb in combinations:
        # Semantics of comb: 0 means no change, >0 points to a certain alternative choice
        swaps = []
        for idx, val in enumerate(comb):
            if val > 0:
                node = nodes[idx]
                rhs_idx = choices[idx][val - 1]
                rhs = grammar.production_rules[node.symbol][rhs_idx]
                # Create a new subtree
                node_new = _grow_deterministic_subtree(grammar, node, rhs)
                swaps.append((node, node_new))
        # Swap all old nodes with the new ones
        for old, new in swaps:
            old.children, new.children = new.children, old.children
        # Copy the new tree
        dt_new = dt.copy()
        # Restore the original tree
        for old, new in swaps:
            old.children, new.children = new.children, old.children
        nbrs.add(_representation.Genotype(dt_new))
    return list(nbrs)


def _get_choices_per_position(grammar, dt, only_terminals):
    """Get all productions found when traversing the tree in leftmost order."""
    nodes = []
    choices = []
    root = dt.root_node
    stack = [root]
    while stack:
        # 1) Choose nonterminal -> Leftmost
        chosen_nt_node = stack.pop(0)

        # 2) Choose rule -> Deduce it from tree
        try:
            rules = grammar.production_rules[chosen_nt_node.symbol]
            num_rules = len(rules)
        except Exception:
            _exceptions.raise_missing_nt_error(chosen_nt_node)
        try:
            chosen_rule = [node.symbol for node in chosen_nt_node.children]
            chosen_rule_idx = rules.index(chosen_rule)
            if only_terminals:
                other_rule_indices = [
                    idx
                    for idx, rule in enumerate(rules)
                    if idx != chosen_rule_idx
                    and any(isinstance(sym, _T) for sym in rule)
                ]
            else:
                other_rule_indices = [
                    idx for idx in range(num_rules) if idx != chosen_rule_idx
                ]
        except ValueError:
            _exceptions.raise_missing_rhs_error(chosen_nt_node, chosen_rule)

        # 3) Expand the chosen nonterminal with rhs of the chosen rule -> Follow the expansion
        new_nt_nodes = [
            node for node in chosen_nt_node.children if isinstance(node.symbol, _NT)
        ]
        stack = new_nt_nodes + stack

        # Store the observed decisions
        nodes.append(chosen_nt_node)
        choices.append(other_rule_indices)
    return nodes, choices


def _grow_deterministic_subtree(grammar, node, rhs):
    # Cache lookup or one-time calculation
    min_depths = grammar._lookup_or_calc(
        "shared",
        "min_depths",
        _shared._cached_calculations.min_depths_to_terminals,
        grammar,
    )

    # Create a new node to grow the subtree from
    node = node.copy()

    # First expansion: Use the chosen rhs
    node.children = [_grammar.data_structures.Node(sym) for sym in rhs]

    # Follow-up expansions: Grow by applying the first rule with shortest path to terminals
    def grow(node):
        nt = node.symbol
        rules = grammar.production_rules[nt]
        depths_per_rule = min_depths[nt]
        first_min_pos = depths_per_rule.index(min(depths_per_rule))
        chosen_rule = rules[first_min_pos]  # first rule with min depth
        new_nodes = [_grammar.data_structures.Node(sym) for sym in chosen_rule]
        node.children = new_nodes
        for nd in new_nodes:
            if nd.contains_nonterminal():
                grow(nd)

    for nd in node.children:
        if nd.contains_nonterminal():
            grow(nd)
    return node
