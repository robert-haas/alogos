from ... import _grammar


# Max and min integer values on a 32 bit system (2**31 - 1) before auto-switching type to long
_MAX_INT = 2147483647


def is_recursive(grammar):
    """Calculate for each production whether it contains a recursive nonterminal.

    A nonterminal is considered to be recursive if it can produce
    itself, i.e. if starting from it some derivation can be constructed
    where the nonterminal appears again.

    References
    ----------
    - `PonyGE2: src/representation/grammar.py
      <https://github.com/PonyGE/PonyGE2/blob/master/src/representation/grammar.py>`__

        - ``def check_recursion(self, cur_symbol, seen)`` solves the same task
          in a different way with recursive function calls

    Examples
    --------
    - ``S => SS => 1S => 12`` means that ``S`` is trivially recursive,
      because there is a rule that replaces ``S`` directly with a
      sequence of symbols that contains ``S`` again.
    - ``S => A => BB => 5B => 5S => 51`` means ``S`` is non-trivially
      recursive, because ``S`` can be replaced indirectly (via other
      nonterminals and their rules) with a sequence of symbols that
      contains ``S`` again.

    """
    # Compute for each nonterminal whether it can produce itself
    def can_nt_produce_itself(nt):
        """Check if a nonterminal can produce itself.

        This is found out by following each derived nonterminal once.

        """
        observed = set([nt])
        stack = [nt]
        while stack:
            symbol = stack.pop()
            for rhs in grammar.production_rules[symbol]:
                for new_symbol in rhs:
                    if isinstance(
                        new_symbol, _grammar.data_structures.NonterminalSymbol
                    ):
                        if new_symbol == nt:
                            return True
                        if new_symbol not in observed:
                            stack.append(new_symbol)
                        observed.add(new_symbol)
        return False

    is_nt_recursive = {
        nt: can_nt_produce_itself(nt) for nt in grammar.nonterminal_symbols
    }

    # Compute for each production whether it contains a recursive nonterminal
    is_production_recursive = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        is_production_recursive[lhs] = []
        for rhs in rhs_multiple:
            recursive = any(is_nt_recursive.get(sym, False) for sym in rhs)
            is_production_recursive[lhs].append(recursive)
    return is_production_recursive


def min_depths_to_terminals(grammar):
    """Calculate for each production the depth required to reach only terminals.

    This information is used during a derivation in order to remain
    within a maximum depth limit.

    Algorithms that utilize it:

    - Initialization of an inidividual with "grow", "pi grow" and "full"
      methods that are part of ramped half-and-half (rhh) population initialization.

    References
    ----------
    - `PonyGE2: src/representation/grammar.py
      <https://github.com/PonyGE/PonyGE2/blob/master/src/representation/grammar.py>`__:
      Sensible initialization from GE (and other systems) uses this information to select rules

        - ``def check_depths(self)`` solves the same task with other data structures

    """
    # Set initial values: 1 for each rhs that is a single terminal, MAX_INT for others
    min_depths = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        min_depths[lhs] = [
            1 if _contains_only_terminals(rhs) else _MAX_INT for rhs in rhs_multiple
        ]

    # For each production, compute the minimal depth required to reach only terminals
    while True:
        finished = True
        for lhs, entries in min_depths.items():
            for rhs_idx in range(len(entries)):
                rhs = grammar.production_rules[lhs][rhs_idx]
                if not _contains_only_terminals(rhs):
                    max_depth = 0
                    for sym in rhs:
                        if isinstance(sym, _grammar.data_structures.NonterminalSymbol):
                            # Find the smallest min_depth of any nonterminal symbol in the rhs
                            min_depth = min(min_depths[sym])
                            # If the found min_depth is a valid value, use it and add one level
                            if min_depth < _MAX_INT:
                                min_depth += 1
                            max_depth = max(min_depth, max_depth)
                    min_depths[lhs][rhs_idx] = max_depth
                    if max_depth == _MAX_INT:
                        finished = False
        if finished:
            break
    return min_depths


def min_expansions_to_terminals(grammar):
    """Calculate for each production the number of expansions required to reach only terminals.

    This information is used during a derivation in order to remain
    within a maximum expansions limit.

    Algorithms that utilize it:

    - Initialization of an inidividual with "PTC2" method.

    """
    # Set initial values: 0 for each rhs that contains only terminals, MAX_INT for others
    min_expansions = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        min_expansions[lhs] = [
            0 if _contains_only_terminals(rhs) else _MAX_INT for rhs in rhs_multiple
        ]

    # For each production with a nonterminal, calc min num of expansions to reach only terminals
    while True:
        finished = True
        for lhs, entries in min_expansions.items():
            for rhs_idx in range(len(entries)):
                rhs = grammar.production_rules[lhs][rhs_idx]
                if not _contains_only_terminals(rhs):
                    # Calc sum of min expansions required for each nonterminal in the rhs
                    min_expansion = 0
                    for sym in rhs:
                        if isinstance(sym, _grammar.data_structures.NonterminalSymbol):
                            min_expansion += min(min_expansions[sym]) + 1
                    if min_expansion < _MAX_INT:
                        min_expansions[lhs][rhs_idx] = min_expansion
                    else:
                        # Continue if some rhs still contains the initial MAX_INT value
                        finished = False
        if finished:
            break
    return min_expansions


def min_nodes_to_terminals(grammar):
    """Calculate for each production the number of nodes required to reach only terminals."""
    # Set initial values: 1 for each rhs that is a single terminal, MAX_INT for others
    min_nodes = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        min_nodes[lhs] = [
            len(rhs) if _contains_only_terminals(rhs) else _MAX_INT
            for rhs in rhs_multiple
        ]

    # For each production, compute the minimal nodes required to reach only terminals
    while True:
        finished = True
        for lhs, entries in min_nodes.items():
            for rhs_idx in range(len(entries)):
                rhs = grammar.production_rules[lhs][rhs_idx]
                if not _contains_only_terminals(rhs):
                    # Calc sum of min nodes required for each nonterminal in the rhs
                    min_nd = 0
                    for sym in rhs:
                        if isinstance(sym, _grammar.data_structures.NonterminalSymbol):
                            min_nd += min(min_nodes[sym]) + 1
                        else:
                            min_nd += 1
                    if min_nd < _MAX_INT:
                        min_nodes[lhs][rhs_idx] = min_nd
                    else:
                        # Continue if some rhs still contains the initial MAX_INT value
                        finished = False
        if finished:
            break
    return min_nodes


def idx_min_nodes_to_terminals(grammar):
    """Create a dictionary of index to number of min nodes associations."""
    mn = min_nodes_to_terminals(grammar)
    sim = grammar._calc_sym_idx_map()
    return {sim[key]: val for key, val in mn.items()}


def _contains_only_terminals(rhs):
    """Detect if the rhs of a production contains only terminal symbols."""
    return all(isinstance(sym, _grammar.data_structures.TerminalSymbol) for sym in rhs)
