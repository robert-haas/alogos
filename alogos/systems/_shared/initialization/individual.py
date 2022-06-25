import random as _random

from ...._grammar import data_structures as _data_structures
from ...._utilities.parametrization import get_given_or_default as _get_given_or_default
from .... import exceptions as _exceptions
from .. import _cached_calculations


def given_genotype(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a given genotype."""
    # Parameter extraction
    gt = _get_given_or_default('init_ind_given_genotype', parameters, default_parameters)

    # Argument processing
    if not isinstance(gt, representation.Genotype):
        try:
            if gt is None:
                _exceptions.raise_no_genotype_error()
            gt = representation.Genotype(gt)
        except Exception:
            _exceptions.raise_init_ind_from_gt_error(gt)

    # Transformation
    try:
        phe, dt = mapping.forward(grammar, gt, parameters, return_derivation_tree=True)
    except _exceptions.MappingError:
        phe = None
        dt = None
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))


def given_derivation_tree(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a given derivation tree."""
    # Parameter extraction
    dt = _get_given_or_default('init_ind_given_derivation_tree', parameters, default_parameters)

    # Transformation
    try:
        if dt is None:
            _exceptions.raise_no_derivation_tree_error()
        gt = mapping.reverse(grammar, dt, parameters)
    except Exception:
        _exceptions.raise_init_ind_from_dt_error(dt)
    phe = dt.string()
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))


def given_phenotype(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a given phenotype."""
    # Parameter extraction
    phe = _get_given_or_default('init_ind_given_phenotype', parameters, default_parameters)

    # Transformation
    try:
        if phe is None:
            _exceptions.raise_no_phenotype_error()
        gt, dt = mapping.reverse(grammar, phe, parameters, return_derivation_tree=True)
    except Exception:
        _exceptions.raise_init_ind_from_phe_error(phe)
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))


def random_genotype(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a random genotype."""
    # Parameter extraction
    gl = _get_given_or_default('genotype_length', parameters, default_parameters)
    cs = _get_given_or_default('codon_size', parameters, default_parameters)

    # Transformation
    try:
        assert cs > 0
        max_int = 2 ** cs - 1
        random_genotype = [_random.randint(0, max_int) for _ in range(gl)]
    except Exception:
        _exceptions.raise_init_ind_rand_gt_error()

    if parameters is None:
        parameters = dict()
    parameters['init_ind_given_genotype'] = random_genotype
    return given_genotype(grammar, parameters, default_parameters, representation, mapping)


def random_valid_genotype(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a random valid genotype."""
    # Parameter extraction
    mt = _get_given_or_default(
        'init_ind_random_valid_genotype_max_tries', parameters, default_parameters)

    # Transformation
    for _ in range(mt):
        ind = random_genotype(grammar, parameters, default_parameters, representation, mapping)
        if ind.phenotype is not None:
            break
    else:
        _exceptions.raise_init_ind_valid_rand_gt_error(mt)
    return ind


def grow_tree(grammar, parameters, default_parameters, representation, mapping):
    """

    References
    ----------
    See rhh  TODO

    """
    # Parameter extraction
    md = _get_given_or_default('init_ind_grow_max_depth', parameters, default_parameters)

    try:
        # Create the tree with the "grow" strategy
        dt = _grow_tree_below_max_depth(grammar, md)

        # Map tree to genotype and phenotype
        ind = _construct_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_grow_error()
    return ind


def full_tree(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a random tree grown

    References
    ----------
    See rhh  TODO

    """
    # Parameter extraction
    md = _get_given_or_default('init_ind_full_max_depth', parameters, default_parameters)

    try:
        # Create the tree with the "full" strategy
        dt = _grow_tree_to_max_depth(grammar, md)

        # Map tree to genotype and phenotype
        ind = _construct_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_full_error()
    return ind


def pi_grow_tree(grammar, parameters, default_parameters, representation, mapping):
    """

    References
    ----------
    - 2016, Fagan et al.: `Exploring position independent initialisation in grammatical evolution
      <https://doi.org/10.1109/CEC.2016.7748331>`__

        - "PI Grow now randomly selects a new non-terminal from the queue
          and again a production is done and the resulting symbols are added
          to the queue in the position from which the parent node was removed."

        - "As the tree is being expanded, the algorithm checks to see if the
          current symbol is the last recursive symbol remaining in the queue.
          If the depth limit hasn’t been reached, and PI Grow currently has
          the last recursive symbol to expand, Pi Grow will only pick a production
          that results in recursive symbols. This process guarantees that at
          least one branch will reach the specified depth limit.

    """
    # Parameter extraction
    md = _get_given_or_default('init_ind_grow_max_depth', parameters, default_parameters)

    try:
        # Create the tree with the "pi grow" strategy, mix of "grow" + "full" for a single branch
        dt = _grow_tree_branch_to_max_depth(grammar, md)

        # Map tree to genotype and phenotype
        ind = _construct_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_pi_grow_error()
    return ind


def ptc2_tree(grammar, parameters, default_parameters, representation, mapping):
    """Create a random tree with a variant of the PTC2 method."""
    # Parameter extraction
    me = _get_given_or_default('init_ind_ptc2_max_expansions', parameters, default_parameters)

    try:
        # Create the tree with the "PTC2" strategy
        dt = _grow_ptc2_tree(grammar, me)

        # Map tree to genotype and phenotype
        ind = _construct_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_ptc2_error()
    return ind


# Shared functions

def _grow_tree_below_max_depth(grammar, max_depth, start_depth=0, root_node=None):
    """Randomly grow a tree and try to stay below a maximum depth in all branches."""
    # Note: Used by ramped half-and-half initialization as well as CFG-GP mutation operator

    # Caching
    min_depths = grammar._lookup_or_calc(
        'shared', 'min_depths', _cached_calculations.min_depths_to_terminals, grammar)

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
        rules = _filter_rules_for_grow(chosen_nt_node.symbol, rules, depth, max_depth, min_depths)
        chosen_rule_idx = _random.randint(0, len(rules) - 1)
        chosen_rule = rules[chosen_rule_idx]
        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [(node, depth + 1) for node in new_nodes
                        if isinstance(node.symbol, _data_structures.NonterminalSymbol)]
        stack = new_nt_nodes + stack
    return dt


def _grow_tree_branch_to_max_depth(grammar, max_depth, start_depth=0, root_node=None):
    """Randomly grow a tree and try to reach a maximum depth in at least one branch."""
    # Caching
    is_recursive = grammar._lookup_or_calc(
        'shared', 'is_recursive', _cached_calculations.is_recursive, grammar)
    min_depths = grammar._lookup_or_calc(
        'shared', 'min_depths', _cached_calculations.min_depths_to_terminals, grammar)

    # Random tree construction
    max_depth_reached = False
    last_recursive_symbol = False
    dt = _data_structures.DerivationTree(grammar)
    if root_node is not None:
        dt.root_node = root_node
    stack = [(dt.root_node, start_depth)]
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
        last_recursive_symbol = not any(any(is_recursive[node.symbol]) for node, depth in stack)
        # If necessary, use "full" to expand the last recursive nonterminal towards max_depth
        if last_recursive_symbol and not max_depth_reached:
            rules = _filter_rules_for_full(
                chosen_nt_node.symbol, rules, depth, max_depth, min_depths, is_recursive)
        # Otherwise, use "grow" as usual
        else:
            rules = _filter_rules_for_grow(
                chosen_nt_node.symbol, rules, depth, max_depth, min_depths)
        chosen_rule_idx = _random.randint(0, len(rules) - 1)
        chosen_rule = rules[chosen_rule_idx]
        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [(node, depth+1) for node in new_nodes
                        if isinstance(node.symbol, _data_structures.NonterminalSymbol)]
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
    return dt


def _grow_tree_to_max_depth(grammar, max_depth, start_depth=0, root_node=None):
    """Randomly grow a tree and try to reach a maximum depth in all branches."""
    # Caching
    is_recursive = grammar._lookup_or_calc(
        'shared', 'is_recursive', _cached_calculations.is_recursive, grammar)
    min_depths = grammar._lookup_or_calc(
        'shared', 'min_depths', _cached_calculations.min_depths_to_terminals, grammar)

    # Random tree construction
    dt = _data_structures.DerivationTree(grammar)
    if root_node is not None:
        dt.root_node = root_node
    stack = [(dt.root_node, start_depth)]
    while stack:
        # 1) Choose nonterminal: leftmost
        chosen_nt_node, depth = stack.pop(0)
        # 2) Choose rule: randomly from recursive ones, if they do not lead over the wanted depth
        rules = grammar.production_rules[chosen_nt_node.symbol]
        rules = _filter_rules_for_full(
            chosen_nt_node.symbol, rules, depth, max_depth, min_depths, is_recursive)
        chosen_rule_idx = _random.randint(0, len(rules) - 1)
        chosen_rule = rules[chosen_rule_idx]
        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [(node, depth+1) for node in new_nodes
                        if isinstance(node.symbol, _data_structures.NonterminalSymbol)]
        stack = new_nt_nodes + stack
    return dt


def _grow_ptc2_tree(grammar, max_expansions):
    # Caching
    is_recursive = grammar._lookup_or_calc(
        'shared', 'is_recursive', _cached_calculations.is_recursive, grammar)
    min_expansions = grammar._lookup_or_calc(
        'shared', 'min_expansions', _cached_calculations.min_expansions_to_terminals, grammar)
    min_expansions_per_symbol = {sym: min(vals) for sym, vals in min_expansions.items()}

    # Random tree construction
    max_exp_reached = False
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
        if expansions + 1 == max_expansions:
            max_exp_reached = True
        rules = _filter_rules_for_ptc2(
            chosen_nt_node.symbol, rules, expansions, max_expansions,
            is_recursive, min_expansions, min_expansions_per_symbol, stack)
        chosen_rule_idx = _random.randint(0, len(rules) - 1)
        chosen_rule = rules[chosen_rule_idx]
        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        new_nt_nodes = [node for node in new_nodes
                        if isinstance(node.symbol, _data_structures.NonterminalSymbol)]
        stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
        expansions += 1
    return dt


def _filter_rules_for_grow(nt, rules, current_depth, max_depth, min_depths):
    """Filter rules depending on current tree depth and min remaining depth required by each rule.

    References
    ----------
    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__

        - "only productions whose minimum depths lead to a branch depth less
          than or equal to the (ramped) maximum depth specified are chosen"

        - "SI can occasionally generate deeper trees than requested, when
          non-recursive productions exist that require deeper sub-trees to
          terminate than recursive productions. Thus the specified maximum
          derivation tree depth is a soft constraint."

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


def _filter_rules_for_full(nt, rules, current_depth, max_depth, min_depths, is_recursive):
    """Filter rules depending on current tree depth and min remaining depth required by each rule.

    References
    ----------
    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__

        - "only productions whose minimum depths lead to a branch depth less
          than or equal to the (ramped) maximum depth specified are chosen"

        - "When using Full, only recursive productions are chosen (if possible)"

        - "depending on the grammar used, not all can reach the desired depth,
          even when using the Full method"

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


def _filter_rules_for_ptc2(nt, rules, current_expansions, max_expansions, is_recursive,
                           min_expansions, min_expansions_per_symbol, stack):
    """Filter rules depending on current number of expansions and min exp required by each rule.

    References
    ----------
    - 2000, Luke: `Two Fast Tree-Creation Algorithms for Genetic Programming
      <https://doi.org/10.1109/4235.873237>`__

        - "With PTC2, the user provides a probability distribution of requested tree sizes.
          PTC2 guarantees that, once it has picked a random tree size from this distribution,
          it will generate and return a tree of that size or slightly larger."

    - 2010, Harper: `GE, explosive grammars and the lasting legacy of bad initialisation
      <https://doi.org/10.1109/CEC.2010.5586336>`__

        - "PTC2 is the second algorithm introduced by Luke and guarantees that
          once a random tree size has been picked it will return a tree of that
          size or slightly larger. [...]
          In essence the algorithm keeps track of all the current non-terminals
          in the parse tree and chooses which one to expand randomly. This is
          repeated until the requisite number of expansions has been carried out.
          If the algorithm is called in a ramped way (i.e. starting with a low
          number of expansions, say 20, and increasing until say 240) then a large
          number of trees of different size and shapes will be generated."

    - 2017, Nicolau: `Understanding grammatical evolution: initialisation
      <https://doi.org/10.1007/s10710-017-9309-9>`__

        - "A refined version of Luke’s and Harper’s PTC2 is used in this study.
          As with SI, grammar productions can also be labelled in terms of the
          minimum number of expansions required for termination"

        - "recursive productions will be chosen only if they will not exceed the specified
          number of expansions while also taking into account the minimum number
          of expansions required to map all outstanding (not fully mapped) branches"

        - "unlike Luke’s and Harper’s implementations, no maximum tree depth is
          employed in this PTC2 version."

    """
    # Calculate how many of the remaining expansions can be consumed by this branch
    expansions_for_other_branches = sum(min_expansions_per_symbol[node.symbol] for node in stack)
    free_expansions = max_expansions - current_expansions - expansions_for_other_branches

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


def _construct_ind_from_tree(grammar, dt, representation, mapping):
    """Given a derivation tree, construct an individual with genotype and phenotype from it."""
    # Genotype can be found by reverse mapping of the derivation tree, reading the decisions in it
    gt = mapping.reverse(grammar, dt)

    # Phenotype can be retrieved by reading the tree leaves, which contain the terminals
    phe = dt.string()

    # Combine data into a single Individual object
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))
