"""Mutation functions for CFG-GP."""

import random as _random

from ..._grammar import data_structures as _data_structures
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .._shared import _cached_calculations
from . import default_parameters as _dp
from . import representation as _representation


def subtree_replacement(grammar, genotype, parameters=None):
    """Change a randomly chosen node by attaching a randomly generated subtree.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``max_depth`` (`int`) : Maximum tree depth.

    Returns
    -------
    genotype : `~.representation.Genotype`
        Mutated genotype.

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
    # Parameter extraction
    max_depth = _get_given_or_default("max_depth", parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Mutation
    # - Get all internal nodes and their depths
    dt = genotype.data
    nodes_and_depths = []
    stack = [(dt.root_node, 0)]
    while stack:
        node, depth = stack.pop()
        if node.children:
            nodes_and_depths.append((node, depth))
            stack = stack + [(node, depth + 1) for node in node.children]
    # - Randomly select a node for mutation
    node, depth = _random.choice(nodes_and_depths)
    # - Replace the node's subtree with a randomly generated new one
    node.children = []
    _grow_random_subtree(grammar, max_depth, start_depth=depth, root_node=node)
    return genotype


def _grow_random_subtree(grammar, max_depth, start_depth, root_node):
    # Caching
    min_depths = grammar._lookup_or_calc(
        "shared", "min_depths", _cached_calculations.min_depths_to_terminals, grammar
    )

    # Random tree construction
    dt = _data_structures.DerivationTree(grammar)
    if root_node is not None:
        dt.root_node = root_node
    stack = [(dt.root_node, start_depth)]
    while stack:
        # 1) Choose nonterminal: leftmost
        chosen_nt_node, depth = stack.pop(0)
        # 2) Choose rule: randomly from those that do not lead over the wanted depth
        rules = grammar.production_rules[chosen_nt_node.symbol]
        rules = _filter_rules_for_grow(
            chosen_nt_node.symbol, rules, depth, max_depth, min_depths
        )
        chosen_rule_idx = _random.randint(0, len(rules) - 1)
        chosen_rule = rules[chosen_rule_idx]
        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [
            (node, depth + 1)
            for node in new_nodes
            if isinstance(node.symbol, _data_structures.NonterminalSymbol)
        ]
        stack = new_nt_nodes + stack
    return dt


def _filter_rules_for_grow(nt, rules, current_depth, max_depth, min_depths):
    # Try to choose rules that do not lead the tree to grow beyond max_depth
    depths_per_rule = min_depths[nt]
    used_rules = []
    for rule, rule_depth in zip(rules, depths_per_rule):
        if (current_depth + rule_depth) <= max_depth:
            used_rules.append(rule)
    # If not possible, choose those rules that share the lowest depth
    if not used_rules:
        min_depth = min(depths_per_rule)
        used_rules = [r for r, s in zip(rules, depths_per_rule) if s == min_depth]
    return used_rules
