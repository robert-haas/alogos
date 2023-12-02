import itertools as _itertools

from .. import data_structures as _data_structures
from . import _shared


def is_cnf(grammar):
    """Check if the grammar is in Chomsky Normal Form (CNF)."""

    def is_in_form1(rhs):
        return len(rhs) == 1 and isinstance(rhs[0], _data_structures.TerminalSymbol)

    def is_in_form2(rhs):
        return len(rhs) == 2 and all(
            isinstance(sym, _data_structures.NonterminalSymbol) for sym in rhs
        )

    for rhs_multiple in grammar.production_rules.values():
        for rhs in rhs_multiple:
            if not is_in_form1(rhs) and not is_in_form2(rhs):
                return False
    return True


def to_cnf(grammar):
    """Convert the grammar G into Chomsky Normal Form (CNF).

    Notes
    -----
    CNF is defined unambiguously, but there are different algorithms to
    transform a grammar into a form that meets those criteria. These
    algorithms have varying time complexities and also lead to different
    growth of the grammar. Here a version is implemented where the new
    grammar's size is bounded by O(|G|^2) instead of O(2^2^|G|),
    adhering mainly to the presentation in a textbook by Elaine Rich.

    References
    ----------
    - Websites

        - https://en.wikipedia.org/wiki/Chomsky_normal_form
        - https://www.tutorialspoint.com/automata_theory/chomsky_normal_form.htm
        - https://www.geeksforgeeks.org/converting-context-free-grammar-chomsky-normal-form

    - Papers

        - Chomsky - On Certain Formal Properties of Grammars (1959)

            - An implication on pp. 149-150: "Theorem 5 asserts in
              particular that all type 2 languages can be generated by
              grammars which yield only trees with no more than two
              branches from each node."

        - Lange, Leiß - To CNF or not to CNF? An efficient yet
          presentable version of the CYK algorithm (2009)

            - pp. 5-6: "the order in which the single transformation
              steps are carried out should be: BIN→DEL→UNIT. This will
              yield a grammar of size O(|G|^2). Nevertheless, many
              textbooks choose a different order, namely DEL→UNIT→BIN
              which yields agrammar of size O(2^2^|G|)."

            - Table 3 on p. 7 gives an overview of textbooks that show
              how to transform a grammar into CNF. It indicates whether
              the transformation results in a grammar with size bound
              by O(|G|^2) or O(2^2^|G|). Based on this table, the
              textbook by Elaine Rich was chosen here as reference. The
              textbook by Hopcroft and Ullman is also mentioned because
              it is very well known and provides a detailed discussion
              of CNF, yet not a transformation with O(|G|^2) guarantee.

    - Books

        - Rich - Automata, Computability and Complexity (2007):
          pp. 171-175

            - p. 171: "There exists a straightforward four-step
              algorithm that converts a grammar [...] into a new
              grammar [...] in Chomsky normal form and
              L(G_C) = L(G) - {ε}"

                1. removeEps: removing from G all ε-rules
                2. removeUnits: removing from G all unit productions
                3. removeMixed: removing from G all rules whose
                   right-hand sides have length greater than 1 and
                   include a terminal
                4. removeLong: removing from G all rules whose
                   right-hand sides have length greater than 2

            - p. 175: "We will run removeLong as step 1 rather than as
              step 4. [...] With this change, removeEps runs in linear
              time. [...] So, if we change converttoChomsky so that it
              does step 4 first, its time complexity is O(n^2) and the
              size of the grammar that it produces is also O(n^2)."

        - Hopcroft, Ullman - Automata theory, language and
          computation (2004)

            - Preliminary simplications: pp. 261-272

                - Eliminate ε-productions
                - Eliminate unit productions
                - Eliminate useless symbols

            - Transformation to Chomsky Normal Form pp. 272-275

                - Arrange that all bodies of length 2 or more consist
                  only of variables
                - Break bodies of length 3 or more into a cascade of
                  production

    """
    # Copy the given grammar
    gr = grammar.copy()

    # Transformation
    gr = _extra_preprocessing(gr)
    gr = _remove_rules_longer_than_2(gr)  # BIN
    gr = _remove_eps_productions(gr)  # DEL
    gr = _remove_unit_productions(gr)  # UNIT
    gr = _remove_mixed_productions(gr)
    return gr


def _extra_preprocessing(grammar):
    """Prepare grammar to meet assumptions of some CNF transformations.

    Textbooks seems to have some unstated assumptions about the
    structure of the right hand sides of the grammar, which is
    established here by extra preparation:
    - The empty string ε seems to be present only in an isolated
      fashion, never aside of other strings or symbols, i.e. collapsed
      with its neighboring context whenever possible.
      Example: A -> B C | D ε | ε E ε F | ε becomes
      A -> B C | D | E F | ε
    - Nonterminals seem to never have just an ε production,
      i.e. at least one nonempty rhs. This has an implication for a
      transformation where they are found to be "nullable", namely that
      they can be "null" (=ε) but also "non-null"
      (=anything else than ε) and hence they should be absent in one
      rule variant but also present in another. This is only the case
      if there is not only A -> ε but also A -> x in the grammar,
      which is not guaranteed for a general CFG but can be ensured by
      preprocessing.
      Example: A -> x | ε remains as it is but A -> ε requires A to be
      removed from the grammar before CNF transformation

    """

    # 1) Remove nonterminals that can only be derived to an empty string
    # or list of empty strings and replace their appearances on rhs
    # with a ε terminal.
    # Examples for which this is the case (A would be removed and
    # replaced in the grammar):
    #   A -> ε
    #   A -> ε | ε
    #   A -> ε ε
    #   A -> ε ε | ε ε ε | ε ε ε ε ε
    # Reason: Without this step later removal of eps productions could
    # fail in different cases.
    def contains_only_eps(rhs):
        """Check if a rhs contains only ε symbols."""
        return len(rhs) > 0 and all(_shared.is_empty_terminal(sym) for sym in rhs)

    def remove_and_replace_nonterminal(grammar, nt_to_remove):
        new_rules = dict()
        for lhs, rhs_multiple in grammar.production_rules.items():
            # Remove it from lhs
            if lhs == nt_to_remove:
                continue
            new_rules[lhs] = []
            # Replace it on rhs with ε (=a newly created terminal
            # with string '')
            for rhs in rhs_multiple:
                new_rhs = [
                    sym if sym != nt_to_remove else _shared.create_empty_terminal()
                    for sym in rhs
                ]
                new_rules[lhs].append(new_rhs)
        grammar.production_rules = new_rules
        return grammar

    while True:
        newly_found = set()
        for lhs, rhs_multiple in grammar.production_rules.items():
            if len(rhs_multiple) > 0 and all(
                contains_only_eps(rhs) for rhs in rhs_multiple
            ):
                newly_found.add(lhs)
        if newly_found:
            for nt in newly_found:
                grammar = remove_and_replace_nonterminal(grammar, nt)
        else:
            break

    # 2) Remove all ε terminals that have other symbols besides them.
    # In other words, collapse empty strings with their neighbor
    # strings.
    # Examples of transformations:
    #   A -> B ε C ε  becomes  A -> B C
    #   A -> ε B      becomes  A -> B
    #   A -> ε ε      becomes  A -> ε
    #   A -> ε        remains as it is
    # Reason: Without this step later checks for nullable symbols could
    # fail in different cases.
    new_rules = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        new_rules[lhs] = []
        for rhs in rhs_multiple:
            # keep only non-ε
            new_rhs = [sym for sym in rhs if not _shared.is_empty_terminal(sym)]
            # if it was a series of ε, keep one ε
            if len(new_rhs) == 0 and len(rhs) > 0:
                new_rhs = [rhs[0]]
            new_rules[lhs].append(new_rhs)
    grammar.production_rules = new_rules

    # Repair step updates all grammar properties to fit to the new
    # production rules
    grammar = _shared.update_grammar_parts(grammar)
    return grammar


def _remove_eps_productions(grammar):
    """Remove ε productions from grammar without changing its language.

    Caution: The grammar G is modified, but its language L(G) shall
    remain the same. There is one exception: If the original grammar G
    can produce the empty string ε, the new grammar G1 will not be
    able to do so. In other words, if ε is part of the language L(G),
    it will no longer be part of the language L(G1). In set notation,
    this fact can be captured as L(G1) = L(G) - {ε}

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007): pp. 162-163
    - Hopcroft, Ullman - Automata theory, language and
      computation (2004): pp. 265-268

    """
    # First step strictly follows Rich (p. 162, step 2)
    nullable_nonterminals = _find_nullable_nonterminals(grammar)

    # Second step follows a mix of Rich (p. 162, step 3) and
    # Hopcroft (p. 266, present or absent)
    grammar = _process_nullable_nonterminals(grammar, nullable_nonterminals)

    # Third step strictly follows Rich (p. 162, step 4)
    grammar = _remove_remaining_eps(grammar)

    # Repair step updates all grammar properties to fit to the
    # new production rules
    grammar = _shared.update_grammar_parts(grammar)
    return grammar


def _find_nullable_nonterminals(grammar):
    """Find nullable nonterminals to prepare for removing ε productions.

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007):
      p. 162, step 2 of removeEps

    """
    # "2. Find the set N of nullable variables"
    nullable_nonterminals = set()

    # Condition (1): A -> ε
    def condition1(rhs):
        return len(rhs) == 1 and _shared.is_empty_terminal(rhs[0])

    # Condition (2): A -> B C D  where all rhs symbols are already
    # marked as nullable
    def condition2(rhs, nullable_nonterminals):
        return all(sym in nullable_nonterminals for sym in rhs)

    # "2.1. Set N to the set of variables that satisfy (1)"
    for lhs, rhs_multiple in grammar.production_rules.items():
        for rhs in rhs_multiple:
            if condition1(rhs):
                nullable_nonterminals.add(lhs)
                break

    # "2.2. Until an entire pass is made without adding anything to
    # N do: Evaluate all other variables with respect to (2)."
    while True:
        found_new_nullable = False
        for lhs, rhs_multiple in grammar.production_rules.items():
            for rhs in rhs_multiple:
                if condition2(rhs, nullable_nonterminals):
                    if lhs not in nullable_nonterminals:
                        nullable_nonterminals.add(lhs)
                        found_new_nullable = True
        if not found_new_nullable:
            break
    return nullable_nonterminals


def _process_nullable_nonterminals(grammar, nullable_nonterminals):
    """Add rules to cover all combinations of nullable nonterminals.

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007): pp. 162,
      step 3 of removeEps
    - Hopcroft, Ullman - Automata theory, language and
      computation (2004): p. 266

    """

    # For each nullable nonterminal, let it be present or absent
    # wherever it appears on a rhs by adding new rules (all
    # combinatorial possibilities that arise from multiple nullable nt)
    def get_all_rhs_variants(rhs, nullable_nonterminals):
        # Find symbols in rhs that are nullable by remembering their
        # positions
        nullable_positions = [
            pos for pos, sym in enumerate(rhs) if sym in nullable_nonterminals
        ]
        map_pos = {pos_in_rhs: i for i, pos_in_rhs in enumerate(nullable_positions)}

        # Create combinations
        num_all_symbols = len(rhs)
        num_nullable_symbols = len(nullable_positions)
        combinations = list(
            _itertools.product([False, True], repeat=num_nullable_symbols)
        )
        if num_nullable_symbols == num_all_symbols:
            # If all symbols are nullable, remove a combination where
            # each symbol is set to absent
            combinations = combinations[1:]

        # Insert symbols according to combinations
        all_new_rhs = []
        for combination in combinations:
            new_rhs = [
                sym
                for pos, sym in enumerate(rhs)
                if pos not in nullable_positions  # include non-nullable symbol always
                or combination[map_pos[pos]]
            ]  # include nullable symbol sometimes
            all_new_rhs.append(new_rhs)
        return all_new_rhs

    new_rules = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        new_rules[lhs] = []
        for rhs in rhs_multiple:
            some_new_rhs = get_all_rhs_variants(rhs, nullable_nonterminals)
            new_rules[lhs].extend(some_new_rhs)
    grammar.production_rules = new_rules

    # For each lhs, remove duplicate rhs
    # (that came from the construction or were present before)
    new_rules = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        new_rules[lhs] = []
        known_rhs = set()
        for rhs in rhs_multiple:
            # nonterminals and terminals are distinguishable as str
            rhs_hashable = str(rhs)
            if rhs_hashable not in known_rhs:
                new_rules[lhs].append(rhs)
                known_rhs.add(rhs_hashable)
    grammar.production_rules = new_rules
    return grammar


def _remove_remaining_eps(grammar):
    """Remove remaining productions of form A -> ε.

    Additional step: If A -> ε is removed and A has no other rhs,
    then A has to be removed from the entire grammar. If by that removal
    another nonterminal has no rhs left, that has to be removed too,
    iteratively until no removal is required anymore.

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007):
      p. 162, step 4 of removeEps

    """

    # Remove A -> ε
    def is_eps(rhs):
        return len(rhs) == 1 and _shared.is_empty_terminal(rhs[0])

    new_rules = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        new_rules[lhs] = []
        for rhs in rhs_multiple:
            if not is_eps(rhs):
                new_rules[lhs].append(rhs)
    grammar.production_rules = new_rules
    return grammar


def _remove_unit_productions(grammar):
    """Remove unit productions from the grammar.

    Definition from Rich on p. 171: "A unit production is a rule whose
    right-hand side consists of a single nonterminal symbol."

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007): p. 171
    - Hopcroft, Ullman - Automata theory, language and
      computation (2004): pp. 268-271

    """

    def to_hashable_rule(lhs, rhs):
        return "{}>{}".format(lhs, rhs)

    def is_new_unit_production(known_up, lhs, rhs):
        is_up = len(rhs) == 1 and isinstance(rhs[0], _data_structures.NonterminalSymbol)
        if is_up:
            up_hashable = to_hashable_rule(lhs, rhs)
            if up_hashable not in known_up:
                known_up.add(up_hashable)
                return True
        return False

    def remove_unit_production(known_up, grammar, lhs, rhs_to_remove):
        # 2.2. Remove [X → Y] from G
        grammar.production_rules[lhs] = [
            rhs for rhs in grammar.production_rules[lhs] if rhs != rhs_to_remove
        ]
        # 2.3. [...] For every rule Y → β [...] Add to G the rule X → β
        # unless that is a rule that has already been removed once
        for rhs in grammar.production_rules[rhs_to_remove[0]]:
            if len(rhs) == 1 and to_hashable_rule(lhs, rhs) in known_up:
                continue
            if rhs not in grammar.production_rules[lhs]:
                # add X → β (if not present already)
                grammar.production_rules[lhs].append(rhs)

    known_up = set()
    while True:
        # Search for unit productions
        new_up = []
        for lhs, rhs_multiple in grammar.production_rules.items():
            for rhs in rhs_multiple:
                if is_new_unit_production(known_up, lhs, rhs):
                    new_up.append((lhs, rhs))
                    break
        # If new unit productions were found, remove them.
        # Otherwise stop.
        if new_up:
            for lhs, rhs in new_up:
                remove_unit_production(known_up, grammar, lhs, rhs)
        else:
            break

    # Repair step updates all grammar properties to fit to the new
    # production rules
    grammar = _shared.update_grammar_parts(grammar)
    return grammar


def _remove_mixed_productions(grammar):
    """Remove mixed productions from the grammar.

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007): p. 172

    """

    # 2. Create a new nonterminal T_a for each terminal a in Σ
    def create_new_nonterminal(grammar, terminal):
        i = 0
        while True:
            text = "T_{}".format(terminal.text)
            if i > 0:
                text += "_{}".format(i)
            symbol = _data_structures.NonterminalSymbol(text)
            if symbol not in grammar.nonterminal_symbols:
                grammar.nonterminal_symbols.add(symbol)
                break
            i += 1
        return symbol

    new_nt_map = {
        sym: create_new_nonterminal(grammar, sym) for sym in grammar.terminal_symbols
    }

    # 3. Modify each rule in G whose right-hand side has length greater
    # than 1 and that contains a terminal symbol by substituting T_a for
    # each occurrence of the terminal a
    new_rules = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        new_rules[lhs] = []
        for rhs in rhs_multiple:
            if len(rhs) > 1:
                rhs_new = [
                    new_nt_map[sym]
                    if isinstance(sym, _data_structures.TerminalSymbol)
                    else sym
                    for sym in rhs
                ]
            else:
                rhs_new = rhs
            new_rules[lhs].append(rhs_new)
    grammar.production_rules = new_rules

    # 4. Add to G, for each T_a, the rule T_a → a
    for terminal, nonterminal in new_nt_map.items():
        grammar.production_rules[nonterminal] = [[terminal]]

    # Repair step updates all grammar properties to fit to the
    # new production rules
    grammar = _shared.update_grammar_parts(grammar)
    return grammar


def _remove_rules_longer_than_2(grammar):
    """Remove long rules from the grammar.

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007): p. 173
    - Hopcroft, Ullman - Automata theory, language and
      computation (2004): p. 273

    """

    # 2. For each rule [...] n > 2, create new nonterminals
    def create_and_add_new_rules(new_rules, lhs, rhs):
        ns = rhs
        ms = [
            _shared.create_new_nonterminal(grammar, prefix="M_")
            for _ in range(len(ns) - 2)
        ]
        # Start: A → N_1 M_2
        last_nt = lhs
        # Mid: M_2 → N_2 M_3, M_3 → N_3 M_4, ...
        for n, m in zip(ns, ms):
            rhs = [n, m]
            if last_nt in new_rules:
                new_rules[last_nt].append(rhs)
            else:
                new_rules[last_nt] = [rhs]
            last_nt = m
        # End: M_n-1 → N_n-1 N_n
        rhs = [ns[-2], ns[-1]]
        new_rules[last_nt] = [rhs]

    new_rules = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        new_rules[lhs] = []
        for rhs in rhs_multiple:
            if len(rhs) > 2:
                create_and_add_new_rules(new_rules, lhs, rhs)
            else:
                new_rules[lhs].append(rhs)
    grammar.production_rules = new_rules

    # Repair step updates all grammar properties to fit to the
    # new production rules
    grammar = _shared.update_grammar_parts(grammar)
    return grammar
