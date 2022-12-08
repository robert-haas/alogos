"""Shared functions for generating derivation trees."""

import copy as _copy
import itertools as _itertools
import random as _random

from ... import exceptions as _exceptions
from ..._grammar import data_structures as _data_structures
from . import _cached_calculations


def uniform(grammar, max_expansions=10_000):
    """Generate a derivation tree by choosing uniformly random rules.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    max_expansions : `int`
        Maximum number of expansions of nonterminal symbols.
        This is a limit value. Reaching it leads to raising an error.

    Returns
    -------
    derivation_tree : `~alogos._grammar.data_structures.DerivationTree`

    Raises
    ------
    `~alogos.exceptions.MappingError`
        If the maximum number of expansions is reached.

    Notes
    -----
    Each possible rule has the same probability of being selected
    throughout the entire derivation [1]_.

    This is a problem if there are recursive rules that tend to
    introduce more symbols than they remove. It is unlikely to
    generate a finished derivation tree in this case.

    References
    ----------
    .. [1] `Generating random sentences from a context free grammar
       <https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-grammar>`__

    """
    # Generate the derivation tree
    dt = _data_structures.DerivationTree(grammar)
    stack = [dt.root_node]
    expansion_counter = 0
    while stack:
        # Check max expansions limit
        if expansion_counter >= max_expansions:
            _exceptions.raise_max_expansion_error(max_expansions)

        # 1) Choose nonterminal: leftmost for simplicity
        chosen_nt_idx = 0
        chosen_nt_node = stack.pop(chosen_nt_idx)

        # 2) Choose rule: randomly
        rules = grammar.production_rules[chosen_nt_node.symbol]
        if len(rules) > 1:
            chosen_rule_idx = _random.randint(0, len(rules) - 1)
        else:
            chosen_rule_idx = 0
        chosen_rule = rules[chosen_rule_idx]

        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [
            node
            for node in new_nodes
            if isinstance(node.symbol, _data_structures.NonterminalSymbol)
        ]
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
        expansion_counter += 1
    return dt


def weighted(grammar, max_expansions=10_000, reduction_factor=0.96):
    """Generate a derivation tree by choosing weighted random rules.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    max_expansions : `int`, optional
        Maximum number of expansions of nonterminal symbols.
        This is a limit value. Reaching it leads to raising an error.
    reduction_factor : `float`, optional
        Factor by which the weight of a rule is multiplied to
        reduce its probability of being selected again within
        the same branch. Its value should be between 0.0 and 1.0.

    Returns
    -------
    derivation_tree : `~alogos._grammar.data_structures.DerivationTree`

    Raises
    ------
    `~alogos.exceptions.MappingError`
        If the maximum number of expansions is reached.

    Notes
    -----
    Each rule gets an initial weight of 1.0 that influences its
    chance of being selected in the next expansion of a
    nonterminal. Every time a rule is chosen within a branch of the
    tree, its weight gets reduced (only in that branch) by multiplying
    it with a provided factor [2]_.

    This limits the chance of a single rule
    being chosen over and over again in the same branch, which leads
    to a better chance of generating a finished derivation tree
    within a given maximum number of expansions, especially in the
    presence of recursive rules.

    References
    ----------
    .. [2] `Generating random sentences from a context free grammar
       <https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-grammar>`__

    """

    def weighted_choice(lhs, weights):
        rhs_list = grammar.production_rules[lhs]
        weight_list = weights[lhs]
        chosen_rule_cumulative_weight = sum(weight_list) * _random.random()
        chosen_rule_idx = 0
        for cumulative_weight in _itertools.accumulate(weight_list):
            if cumulative_weight >= chosen_rule_cumulative_weight:
                break
            chosen_rule_idx += 1
        else:
            chosen_rule_idx = len(rhs_list) - 1
        chosen_rule = rhs_list[chosen_rule_idx]
        new_weights = _copy.deepcopy(weights)
        new_weights[lhs][chosen_rule_idx] *= reduction_factor
        return chosen_rule_idx, chosen_rule, new_weights

    # Generate the derivation tree
    initial_weights = {
        lhs: [1.0 for rhs in rules] for lhs, rules in grammar.production_rules.items()
    }
    dt = _data_structures.DerivationTree(grammar)
    stack = [(dt.root_node, initial_weights)]
    expansion_counter = 0
    while stack:
        # Check max expansions limit
        if expansion_counter >= max_expansions:
            _exceptions.raise_max_expansion_error(max_expansions)

        # 1) Choose nonterminal: leftmost
        chosen_nt_idx = 0
        chosen_nt_node, weights = stack.pop(chosen_nt_idx)

        # 2) Choose rule: randomly with shrinking weights for rules used repeatetly in this subtree
        lhs = chosen_nt_node.symbol
        rules = grammar.production_rules[lhs]
        if len(rules) > 1:
            chosen_rule_idx, chosen_rule, weights = weighted_choice(lhs, weights)
        else:
            chosen_rule_idx = 0
            chosen_rule = rules[chosen_rule_idx]

        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [
            (node, weights)
            for node in new_nodes
            if isinstance(node.symbol, _data_structures.NonterminalSymbol)
        ]
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
        expansion_counter += 1
    return dt


def ptc2(grammar, max_expansions=100):
    """Create a derivation tree with Nicolau's PTC2 variant.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    max_expansions : `int`, optional
        Desired maximum number of nonterminal expansions.
        This is a target value. Missing it slightly below or above
        does not raise an error.

    Returns
    -------
    derivation_tree : `~alogos._grammar.data_structures.DerivationTree`

    """
    # Caching
    is_recursive = grammar._lookup_or_calc(
        "shared", "is_recursive", _cached_calculations.is_recursive, grammar
    )
    min_expansions = grammar._lookup_or_calc(
        "shared",
        "min_expansions",
        _cached_calculations.min_expansions_to_terminals,
        grammar,
    )
    min_expansions_per_symbol = {sym: min(vals) for sym, vals in min_expansions.items()}

    # Random tree construction
    dt = _data_structures.DerivationTree(grammar)
    stack = [dt.root_node]
    expansions = 0
    while stack:
        # 1) Choose nonterminal: random
        chosen_nt_idx = _random.choice(range(len(stack)))
        chosen_nt_node = stack.pop(chosen_nt_idx)
        # 2) Choose rule: randomly from those that do not lead over the wanted expansions
        rules = grammar.production_rules[chosen_nt_node.symbol]
        # Check if max_expansions was reached
        rules = _filter_rules_for_ptc2(
            chosen_nt_node.symbol,
            rules,
            expansions,
            max_expansions,
            is_recursive,
            min_expansions,
            min_expansions_per_symbol,
            stack,
        )
        chosen_rule_idx = _random.randint(0, len(rules) - 1)
        chosen_rule = rules[chosen_rule_idx]
        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [
            node
            for node in new_nodes
            if isinstance(node.symbol, _data_structures.NonterminalSymbol)
        ]
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
        expansions += 1
    return dt


def grow_one_branch_to_max_depth(grammar, max_depth=20):
    """Randomly grow a tree and try to reach a maximum depth in at least one branch.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    max_depth : `int`, optional
        Desired maximum depth of the tree.
        This is a target value. Missing it slightly below or above
        does not raise an error.

    Returns
    -------
    derivation_tree : `~alogos._grammar.data_structures.DerivationTree`

    """
    # Caching
    is_recursive = grammar._lookup_or_calc(
        "shared", "is_recursive", _cached_calculations.is_recursive, grammar
    )
    min_depths = grammar._lookup_or_calc(
        "shared", "min_depths", _cached_calculations.min_depths_to_terminals, grammar
    )

    # Random tree construction
    max_depth_reached = False
    last_recursive_symbol = False
    dt = _data_structures.DerivationTree(grammar)
    stack = [(dt.root_node, 0)]
    while stack:
        # 1) Choose nonterminal: random
        chosen_nt_idx = _random.choice(range(len(stack)))
        chosen_nt_node, depth = stack.pop(chosen_nt_idx)
        # 2) Choose rule: randomly from those that do not lead over the wanted depth
        rules = grammar.production_rules[chosen_nt_node.symbol]
        # Check if max_depth was reached once
        if depth + 1 >= max_depth:
            max_depth_reached = True
        # Check if no recursive nonterminal remains on the stack currently
        last_recursive_symbol = not any(
            any(is_recursive[node.symbol]) for node, depth in stack
        )
        # If necessary, use "full" to expand the last recursive nonterminal towards max_depth
        if last_recursive_symbol and not max_depth_reached:
            rules = _filter_rules_for_full(
                chosen_nt_node.symbol, rules, depth, max_depth, min_depths, is_recursive
            )
        # Otherwise, use "grow" as usual
        else:
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
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
    return dt


def grow_all_branches_within_max_depth(grammar, max_depth=20):
    """Randomly grow a tree and try to stay below a maximum depth in all branches.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    max_depth : `int`, optional
        Desired maximum depth of the tree.
        This is a target value. Missing it slightly below or above
        does not raise an error.

    Returns
    -------
    derivation_tree : `~alogos._grammar.data_structures.DerivationTree`

    """
    # Caching
    min_depths = grammar._lookup_or_calc(
        "shared", "min_depths", _cached_calculations.min_depths_to_terminals, grammar
    )

    # Random tree construction
    dt = _data_structures.DerivationTree(grammar)
    stack = [(dt.root_node, 0)]
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


def grow_all_branches_to_max_depth(grammar, max_depth=20):
    """Randomly grow a tree and try to reach a maximum depth in all branches.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    max_depth : `int`, optional
        Desired maximum depth of the tree.
        This is a target value. Missing it slightly below or above
        does not raise an error.

    Returns
    -------
    derivation_tree : `~alogos._grammar.data_structures.DerivationTree`

    """
    # Caching
    is_recursive = grammar._lookup_or_calc(
        "shared", "is_recursive", _cached_calculations.is_recursive, grammar
    )
    min_depths = grammar._lookup_or_calc(
        "shared", "min_depths", _cached_calculations.min_depths_to_terminals, grammar
    )

    # Random tree construction
    dt = _data_structures.DerivationTree(grammar)
    stack = [(dt.root_node, 0)]
    while stack:
        # 1) Choose nonterminal: leftmost
        chosen_nt_node, depth = stack.pop(0)
        # 2) Choose rule: randomly from recursive ones, if they do not lead over the wanted depth
        rules = grammar.production_rules[chosen_nt_node.symbol]
        rules = _filter_rules_for_full(
            chosen_nt_node.symbol, rules, depth, max_depth, min_depths, is_recursive
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
    """Filter rules depending on current tree depth and min remaining depth required by each rule.

    References
    ----------
    - 2017, Nicolau: `Understanding grammatical evolution:
      initialisation <https://doi.org/10.1007/s10710-017-9309-9>`__

        - "only productions whose minimum depths lead to a branch depth
          less than or equal to the (ramped) maximum depth specified
          are chosen"

        - "SI can occasionally generate deeper trees than requested,
          when non-recursive productions exist that require deeper
          sub-trees to terminate than recursive productions. Thus the
          specified maximum derivation tree depth is a soft constraint."

    """
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


def _filter_rules_for_full(
    nt, rules, current_depth, max_depth, min_depths, is_recursive
):
    """Filter rules depending on current tree depth and min remaining depth required by each rule.

    References
    ----------
    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__

        - "only productions whose minimum depths lead to a branch depth
          less than or equal to the (ramped) maximum depth specified
          are chosen"

        - "When using Full, only recursive productions are chosen (if
          possible)"

        - "depending on the grammar used, not all can reach the desired
          depth, even when using the Full method"

    """
    # Try to choose recursive rules that do not lead the tree to grow beyond max_depth
    depths_per_rule = min_depths[nt]
    rec_per_rule = is_recursive[nt]
    used_rules = []
    for rule, rule_depth, rec in zip(rules, depths_per_rule, rec_per_rule):
        if rec and (current_depth + rule_depth) <= max_depth:
            used_rules.append(rule)
    # If not possible, try the same with non-recursive rules
    if not used_rules:
        for rule, rule_depth in zip(rules, depths_per_rule):
            if (current_depth + rule_depth) <= max_depth:
                used_rules.append(rule)
    # If not possible, choose those rules that share the lowest depth
    if not used_rules:
        min_depth = min(depths_per_rule)
        used_rules = [r for r, s in zip(rules, depths_per_rule) if s == min_depth]
    return used_rules


def _filter_rules_for_ptc2(
    nt,
    rules,
    current_expansions,
    max_expansions,
    is_recursive,
    min_expansions,
    min_expansions_per_symbol,
    stack,
):
    """Filter rules based on number of expansions.

    Used:

    - Maximum number of expansions that are desired.
    - Current number of expansions.
    - Minimum number of expansions required by each rule to reach
      a sequence consisting only of terminals.

    References
    ----------
    - 2000, Luke: `Two Fast Tree-Creation Algorithms for Genetic
      Programming <https://doi.org/10.1109/4235.873237>`__

        - "With PTC2, the user provides a probability distribution of
          requested tree sizes. PTC2 guarantees that, once it has
          picked a random tree size from this distribution, it will
          generate and return a tree of that size or slightly larger."

    - 2010, Harper: `GE, explosive grammars and the lasting legacy of
      bad initialisation <https://doi.org/10.1109/CEC.2010.5586336>`__

        - "PTC2 is the second algorithm introduced by Luke and
          guarantees that once a random tree size has been picked it
          will return a tree of that size or slightly larger. [...]
          In essence the algorithm keeps track of all the current
          non-terminals in the parse tree and chooses which one to
          expand randomly. This is repeated until the requisite number
          of expansions has been carried out. If the algorithm is
          called in a ramped way (i.e. starting with a low number of
          expansions, say 20, and increasing until say 240) then a
          large number of trees of different size and shapes will be
          generated."

    - 2017, Nicolau: `Understanding grammatical evolution:
      initialisation <https://doi.org/10.1007/s10710-017-9309-9>`__

        - "A refined version of Luke’s and Harper’s PTC2 is used in
          this study. As with SI, grammar productions can also be
          labelled in terms of the minimum number of expansions
          required for termination"

        - "recursive productions will be chosen only if they will not
          exceed the specified number of expansions while also taking
          into account the minimum number of expansions required to map
          all outstanding (not fully mapped) branches"

        - "unlike Luke’s and Harper’s implementations, no maximum tree
          depth is employed in this PTC2 version."

    """
    # Calculate how many of the remaining expansions can be consumed by this branch
    expansions_for_other_branches = sum(
        min_expansions_per_symbol[node.symbol] for node in stack
    )
    free_expansions = (
        max_expansions - current_expansions - expansions_for_other_branches
    )

    # Try to choose recursive rules that do not lead the branch to grow beyond the free expansions
    expansions_per_rule = min_expansions[nt]
    rec_per_rule = is_recursive[nt]
    used_rules = []
    for rule, rule_expansions, rec in zip(rules, expansions_per_rule, rec_per_rule):
        if rec and rule_expansions <= free_expansions:
            used_rules.append(rule)
    # If not possible, use all non-recursive rules
    if not used_rules:
        used_rules = [rule for rule, rec in zip(rules, rec_per_rule) if not rec]
    # If also not possible, use all rules
    if not used_rules:
        used_rules = rules
    return used_rules
