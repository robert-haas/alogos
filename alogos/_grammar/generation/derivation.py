import copy as _copy
import random as _random
import itertools as _itertools

from ... import exceptions as _exceptions
from .. import data_structures as _data_structures


def generate_derivation_simple(grammar, max_expansions=1000,
                               verbose=False, raise_errors=True, return_derivation_tree=False):
    """Generate a derivation in a random and simple way.

    References
    ----------
    - https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-grammar

    """
    # Generate the derivation tree
    dt = _data_structures.DerivationTree(grammar)
    stack = [dt.root_node]
    expansion_counter = 0
    while stack:
        # Check max expansions limit
        if expansion_counter >= max_expansions:
            if raise_errors:
                _exceptions.raise_max_expansion_error(max_expansions)
            break

        # Remember for optional report
        if verbose:
            ori_stack = stack[:]

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
        new_nt_nodes = [node for node in new_nodes
                        if isinstance(node.symbol, _data_structures.NonterminalSymbol)]
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
        expansion_counter += 1

    # Conditional return
    string = dt.string()
    if return_derivation_tree:
        return string, dt
    return string


def generate_derivation_weighted(grammar, max_expansions=1_000_000, reduction_factor=0.961,
                                 verbose=False, raise_errors=True, return_derivation_tree=False):
    """Generate a derivation in a random and weighted way.

    Adapting the probabilities of choosing certain rules for expansion
    ensures a high chance of convergence.

    References
    ----------
    - https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-grammar

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
    initial_weights = {lhs: [1.0 for rhs in rules]
                       for lhs, rules in grammar.production_rules.items()}
    dt = _data_structures.DerivationTree(grammar)
    stack = [(dt.root_node, initial_weights)]
    expansion_counter = 0
    while stack:
        # Check max expansions limit
        if expansion_counter >= max_expansions:
            if raise_errors:
                _exceptions.raise_max_expansion_error(max_expansions)
            break

        # Remember for optional report
        if verbose:
            ori_stack = [node for node, weight in stack]

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
        new_nt_nodes = [(node, weights) for node in new_nodes
                        if isinstance(node.symbol, _data_structures.NonterminalSymbol)]
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
        expansion_counter += 1

    # Conditional return
    string = dt.string()
    if return_derivation_tree:
        return string, dt
    return string
