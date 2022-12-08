"""Forward mapping function for WHGE.

Note that a reverse function for WHGE is supposedly
not possible in full generality.

"""

from ... import _grammar
from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from . import _cached_calculations
from . import default_parameters as _dp
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_GT = _representation.Genotype
_DT = _grammar.data_structures.DerivationTree
_NT = _grammar.data_structures.NonterminalSymbol


def _fe(me, re):
    if re:
        _exceptions.raise_max_expansion_error(me)


# Forward mapping
def forward(
    grammar,
    genotype,
    parameters=None,
    raise_errors=True,
    return_derivation_tree=False,
    verbose=False,
):
    """Map a WHGE genotype to a string phenotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype` or data that can be converted to it
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``max_expansions``  (`int`): Maximum number of nonterminal
          expansions allowed in the derivation created by the mapping
          process.
        - ``max_depth``  (`int`): Maximum tree depth. This is considered
          only in calculating the expressive power of different rules.
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
    md = _get_given_or_default("max_depth", parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Cache look-up
    sd = grammar._lookup_or_calc(
        "whge", "sd", _cached_calculations.shortest_distances, grammar
    )
    ep = grammar._lookup_or_calc(
        "whge", "ep_d{}".format(md), _cached_calculations.expressive_powers, grammar, md
    )

    # Mapping
    if verbose:
        dt = _forward_slow(grammar, genotype.data, me, sd, ep, raise_errors, verbose)
    else:
        dt = _forward_fast(grammar, genotype.data, me, sd, ep, raise_errors)

    # Conditional return
    phe = dt.string()
    if return_derivation_tree:
        return phe, dt
    return phe


def _forward_fast(gr, gt, me, sd, ep, re):
    """Calculate the genotype-to-phenotype map of WHGE in a fast way."""
    dt = _DT(gr)
    x = dt._expand
    a1 = gt.count()
    st = [(dt.root_node, gt)]
    p = st.pop
    a = st.append
    w = gr.production_rules
    ec = 0
    while st:
        if me is not None and ec >= me:
            _fe(me, re)
            break
        nd, gt = p()
        sy = nd.symbol
        n1 = gt.count()
        b = len(gt)
        rs = w[sy]
        nr = len(rs)
        if nr == 1:
            r = 0
        elif b >= nr:
            lb = b // nr
            la = lb + 1
            m = b % nr
            gs = []
            s = 0
            for i in range(nr):
                e = s + (la if i < m else lb)
                gs.append(gt[s:e])
                s = e
            v = []
            d = -1.0
            for j, g in enumerate(gs):
                c = g.count() / len(g)
                if c == d:
                    v.append(j)
                elif c > d:
                    d = c
                    v = [j]
            r = v[n1 % len(v)]
        else:
            c = n1 if b > 0 else a1
            v = sd[sy]
            r = v[c % len(v)]
        z = x(nd, rs[r])
        u = len(z)
        ec += 1
        if u > b:
            f = [gt[0:0] for _ in z]
        else:
            e = [ep[n.symbol] if isinstance(n.symbol, _NT) else 0 for n in z]
            t = sum(e)
            if t > 0.0:
                y = [int(h / t * b) for h in e]
            else:
                y = [0 for _ in e]
            q = sum(y)
            if q > 0:
                ms = int(b / q)
                nr = b - ms * q
                y = [x * ms for x in y]
            else:
                nr = b - q
            for c in range(nr):
                y[c % u] += 1
            f = []
            s = 0
            for n in y:
                e = s + n
                f.append(gt[s:e])
                s = e
            for k in range(u):
                if len(f[k]) == b:
                    f[k] = f[k][:-1]
                    break
        for i in range(u):
            if isinstance(z[i].symbol, _NT):
                a((z[i], f[i]))
    return dt


def _forward_slow(
    grammar,
    genotype,
    max_expansions,
    shortest_distances,
    expressive_powers,
    raise_errors,
    verbose,
):
    """Calculate the genotype-to-phenotype map of WHGE in a slow way.

    This is a readable implementation of the mapping process, which
    also allows to print output about the steps it involves.
    It served as basis for the faster, minified implementation in this
    module and may be helpful in understanding, replicating or modifying
    the algorithm.

    """
    derivation_tree = _grammar.data_structures.DerivationTree(grammar)
    stack = [(derivation_tree.root_node, genotype)]
    ones_in_full_genotype = genotype.count()
    expansion_counter = 0
    if verbose:
        header = "Start of the derivation"
        print(header)
        print("=" * len(header))
        print(
            "- Get start symbol of the grammar: <{}>".format(grammar.start_symbol.text)
        )
        print()
        print("Sentential form: {}".format(derivation_tree.string()))
        print()
    while stack:
        # Check stop conditions (not part of paper, but of all other mappings)
        if max_expansions is not None and expansion_counter >= max_expansions:
            if verbose:
                header = "Stop condition fulfilled"
                print(header)
                print("=" * len(header))
                print(
                    "- The maximum number of expansions is reached: {}".format(
                        max_expansions
                    )
                )
                print()
            if raise_errors:
                _exceptions.raise_max_expansion_error(max_expansions)
            break

        # 1) Choose nonterminal: WHGE is order-independent, simply use leftmost
        chosen_nt_idx = 0
        chosen_nt_node, current_genotype = stack.pop(chosen_nt_idx)
        current_symbol = chosen_nt_node.symbol
        ones_in_current_genotype = current_genotype.count()
        if verbose:
            header = "Expansion {}".format(expansion_counter + 1)
            print(header)
            print("=" * len(header))
            print("- Choice of nonterminal to expand")
            print("    Index: {} by always using leftmost".format(chosen_nt_idx))
            print("    Symbol s: <{}>".format(chosen_nt_node.symbol.text))
            print(
                '    Genotype substring g\': "{}"'.format(
                    _bitarray_to_str(current_genotype)
                )
            )

        # 2) Choose rule: WHGE decides with full information of a sub-genotype, not a single codon
        rules = _rules_for(grammar, current_symbol)
        num_bits = len(current_genotype)
        num_rules = len(rules)
        if verbose:
            print("- Choice of rule to apply")
            print("    Calculation steps")
            print("      R_s = RulesFor(s)")
            operator = ">=" if num_bits >= num_rules else "<"
            print(
                "      Number of bits |g'|: {} {} Number of rules |R_s|: {}".format(
                    num_bits, operator, num_rules
                )
            )
        if num_rules == 1:
            chosen_rule_idx = 0
            if verbose:
                print(
                    "    Index: {} by using the only available rule without "
                    "considering the genotype substring".format(chosen_rule_idx)
                )
        elif num_bits >= num_rules:
            new_genotypes_for_rule_choice = _split_for_rule(current_genotype, rules)
            if verbose:
                print("        g\"_i = SplitForRule(g', |R_s|)")
                for i, gen in enumerate(new_genotypes_for_rule_choice):
                    print(
                        '          g"_{}: "{}" with {} bits'.format(
                            i, _bitarray_to_str(gen), len(gen)
                        )
                    )
                print('        LargestCardIndex(g"_i)')
            chosen_rule_idx = _largest_card_index(
                new_genotypes_for_rule_choice, ones_in_current_genotype
            )
            if verbose:
                print(
                    "    Index: {} by calculation with SplitForRule and "
                    "LargestCardIndex".format(chosen_rule_idx)
                )
        else:
            if verbose:
                print("        ShortestRuleIndex(R_s)")
            chosen_rule_idx = _shortest_rule_index(
                current_genotype,
                current_symbol,
                shortest_distances,
                ones_in_current_genotype,
                ones_in_full_genotype,
            )
            if verbose:
                print(
                    "    Index: {} by calculation with ShortestRuleIndex".format(
                        chosen_rule_idx
                    )
                )

        # 3) Expand the chosen nonterminal (1) with the rhs of the chosen rule (2)
        if verbose:
            print("- Application of rule to nonterminal")
            print("    ApplyRule(R_s, i={})".format(chosen_rule_idx))
        new_nodes = _apply_rule(rules, chosen_rule_idx, derivation_tree, chosen_nt_node)
        expansion_counter += 1

        # 4) Calculate sub-genotypes belonging to each new nonterminal and terminal symbol
        if verbose:
            print("- Distribution of bits to new child nodes")
            print("    g'_i = SplitForChildren()")
        new_genotypes_for_children = _split_for_children(
            current_genotype, new_nodes, expressive_powers
        )
        for i, child_genotype in enumerate(new_genotypes_for_children):
            if child_genotype == current_genotype:
                new_genotypes_for_children[i] = _drop_trailing_bit(child_genotype)
                if verbose:
                    print(
                        '      g\'_{}: "{}" by DropTrailingBit("{}")'.format(
                            i,
                            _bitarray_to_str(new_genotypes_for_children[i]),
                            _bitarray_to_str(child_genotype),
                        )
                    )
            else:
                if verbose:
                    print(
                        '      g\'_{}: "{}"'.format(i, _bitarray_to_str(child_genotype))
                    )

        # 5) Add new nodes that contain a nonterminal and genotypes to the stack
        stack = _append_children(new_nodes, new_genotypes_for_children, stack)
        if verbose:
            print()
            print("Sentential form: {}".format(derivation_tree.string()))
            print()
    header = "End of the derivation"
    if verbose:
        print(header)
        print("=" * len(header))
        print()
        name = (
            "String" if derivation_tree.is_completely_expanded() else "Sentential form"
        )
        print("{}: {}".format(name, derivation_tree.string()))
    return derivation_tree


# Functions adhering closely to pseudocode in 2018 paper


def _rules_for(grammar, symbol):
    return grammar.production_rules[symbol]


def _split_for_rule(genotype, rules):
    num_rules = len(rules)
    num_bits = len(genotype)
    l_g = num_bits // num_rules  # equivalent to float division followed by floor
    first_n = num_bits % num_rules
    length1 = l_g + 1
    length2 = l_g
    new_genotypes = []
    start = 0
    for i in range(num_rules):
        length = length1 if i < first_n else length2
        end = start + length
        sub_genotype = genotype[start:end]
        new_genotypes.append(sub_genotype)
        start = end
    return new_genotypes


def _largest_card_index(genotypes, ones_in_current_genotype):
    # Caution: card == largest_card should not be used to compare float
    # values but is used in the Java reference implementation and
    # therefore duplicated here to deliver identical behavior (perhaps
    # only on identical machines!)
    # Cleaner (and a bit slower) would be: abs(card - largest_card) < EPS
    indices = []
    largest_card = -1.0
    for i, genotype in enumerate(genotypes):
        num_bits = len(genotype)
        num_1 = genotype.count()
        card = num_1 / num_bits
        if card == largest_card:
            indices.append(i)
        elif card > largest_card:
            largest_card = card
            indices = [i]
    num_options = len(indices)
    if num_options == 1:
        chosen_rule_index = indices[0]
        print(
            '          g"_{} has highest relative cardinality {}'.format(
                chosen_rule_index, largest_card
            )
        )
    else:
        # Comment in original Java code: for avoiding choosing always
        # the 1st option in case of tie, choose depending on count of
        # 1s in genotype
        chosen_option = ones_in_current_genotype % num_options
        chosen_rule_index = indices[chosen_option]
        genotypes_str = ", ".join('g"_{}'.format(i) for i in indices)
        print(
            "          {} share the same highest relative cardinality {}".format(
                genotypes_str, largest_card
            )
        )
        print(
            '          Selected g"_{} by {} % {} = {}'.format(
                chosen_rule_index,
                ones_in_current_genotype,
                num_options,
                chosen_option,
            )
        )
    return chosen_rule_index


def _shortest_rule_index(
    genotype,
    symbol,
    shortest_distances,
    ones_in_genotype,
    ones_in_full_genotype,
):
    count = ones_in_genotype if len(genotype) > 0 else ones_in_full_genotype
    indices = shortest_distances[symbol]
    num_options = len(indices)
    index = indices[count % num_options]
    if num_options == 1:
        print(
            "          Rule {} has the lowest number of steps to reach a "
            "terminal sequence.".format(index)
        )
    else:
        print(
            "          Rule {} share the same lowest number of steps to reach a "
            "terminal sequence.".format(", ".join(str(ind) for ind in indices))
        )
        print(
            "          Selected rule {} by "
            "{} % {} = {}".format(index, count, num_options, index)
        )
    return index


def _apply_rule(rules, chosen_rule_idx, derivation_tree, chosen_nt_node):
    chosen_rule = rules[chosen_rule_idx]
    new_nodes = derivation_tree._expand(chosen_nt_node, chosen_rule)
    rhs = "".join(
        sym.text
        if isinstance(sym, _grammar.data_structures.TerminalSymbol)
        else "<{}>".format(sym.text)
        for sym in chosen_rule
    )
    print("      Substitution: <{}> -> {}".format(chosen_nt_node.symbol.text, rhs))
    return new_nodes


def _drop_trailing_bit(genotype):
    return genotype[:-1]


def _split_for_children(genotype, child_nodes, expressive_powers):
    if len(child_nodes) > len(genotype):
        genotypes = [genotype[0:0] for _ in child_nodes]
    else:
        lengths = _weighted_partitioning(len(genotype), child_nodes, expressive_powers)
        genotypes = []
        last_length = 0
        for length in lengths:
            new_length = last_length + length
            genotype_part = genotype[last_length:new_length]
            genotypes.append(genotype_part)
            last_length = new_length
    return genotypes


def _weighted_partitioning(length, child_nodes, expressive_powers):
    """Calculate the length of each sub-genotype (=non-overlapping parts of genotype)."""
    # Design choice: Don't store 0 values for terminals, because it is
    # irrelevant information and omitting it may give a minor speedup
    # (and prevents NT and T name clashes if using str)
    log_ep_values = [
        expressive_powers[node.symbol] if node.contains_nonterminal() else 0
        for node in child_nodes
    ]
    log_ep_sum = sum(log_ep_values)
    if log_ep_sum > 0:
        lengths = [
            int(log_ep_val / log_ep_sum * length) for log_ep_val in log_ep_values
        ]
    else:
        lengths = [0 for _ in log_ep_values]

    # Distribute remaining bits
    sum_lengths = sum(lengths)
    if sum_lengths > 0:
        min_size = int(length / sum_lengths)
        num_remaining_bits = length - min_size * sum_lengths
        lengths = [n * min_size for n in lengths]
    else:
        num_remaining_bits = length - sum(lengths)
    num_substrings = len(child_nodes)
    for cnt in range(num_remaining_bits):
        position_j = (
            cnt % num_substrings
        )  # Paper has +1 here because indexing starts with 1
        lengths[position_j] += 1
    return lengths


def _append_children(new_nodes, new_genotypes, stack):
    new_stack_entries = [
        (node, genotype)
        for node, genotype in zip(new_nodes, new_genotypes)
        if node.contains_nonterminal()
    ]
    stack = new_stack_entries + stack
    return stack


# Helper functions for printing


def _bitarray_to_str(arr):
    return "".join("1" if bit else "0" for bit in arr.tolist())
