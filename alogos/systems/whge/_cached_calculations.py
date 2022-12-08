"""Cached calculations of Weighted Hierarchical Grammatical Evolution (WHGE)."""

import math as _math

from ... import _grammar
from .. import _shared


def shortest_distances(grammar):
    """Calculate for each nonterminal the minimum distance to reach only terminals.

    Distance is not clear, because it could mean number of expansions but
    actually means tree depth.

    References
    ----------
    - Bartoli, Castelli, Medvet in 2018: `Weighted Hierarchical Grammatical Evolution
      <https://doi.org/10.1109/TCYB.2018.2876563>`__

        - p. 4: "set i equal to the index of the production rule in R s which leads to a sequence
          of terminals in the lowest number of derivations from s [ShortestRuleIndex()]"

    - Software implementation

        - `HierarchicalMapper.java
          <https://github.com/ericmedvet/evolved-ge/blob/master/src/main/java/it/units/malelab/ege/ge/mapper/HierarchicalMapper.java>`__

            - data structure: ``optionJumpsToTerminalMap = new LinkedHashMap<>();``
            - algorithm: two loops, the first to set a default, the second to actually compute it
            - final value: shortestOptionIndexesMap stores the indices of the shortest rules
              of each nonterminal

    """
    # Compute distances
    option_jumps_to_terminal = grammar._lookup_or_calc(
        "shared",
        "min_depths",
        _shared._cached_calculations.min_depths_to_terminals,
        grammar,
    )

    # Compute positions of minimal values instead of having repeated distance comparisons later
    max_int = 2147483647
    shortest_option_indices = dict()
    for lhs_symbol in grammar.production_rules:
        min_jumps = max_int
        for i in range(len(option_jumps_to_terminal[lhs_symbol])):
            local_jumps = option_jumps_to_terminal[lhs_symbol][i]
            if local_jumps < min_jumps:
                min_jumps = local_jumps
        indices = []
        for i in range(len(option_jumps_to_terminal[lhs_symbol])):
            if option_jumps_to_terminal[lhs_symbol][i] == min_jumps:
                indices.append(i)
        shortest_option_indices[lhs_symbol] = indices
    return shortest_option_indices


def expressive_powers(grammar, max_depth):
    """Pre-calculate the expressive power of each non-terminal symbol in the grammar.

    What is actually calculated and stored is not ``expressive_power`` itself but
    ``ceil(log2(expressive_power))`` in order to further decrease the amount of
    repetitive computations later.

    References
    ----------
    - Bartoli, Castelli, Medvet in 2018: `Weighted Hierarchical Grammatical Evolution
      <https://doi.org/10.1109/TCYB.2018.2876563>`__

        - p. 4: "We quantify the expressive power e_s of a symbol s with the number of
          different (partial) derivation trees with which can be generated from
          s (e_s = 1 for terminal symbols). [...]
          if a derivation tree still contains nonterminals at depth n_d,
          we do not further derive them and count the resulting partial derivation trees
          without further deriving them."

        - p. 4: Pseudocode - Algorithm 2:

            - There seem to be two small errors

                1) The floor function ⌊ ⌋ should actually be the ceiling function ⌈ ⌉
                   and the length l should be outside of it and multiplied after rounding.
                   This has the effect that if log2 of an expressive power is between 0 and 1,
                   it is rounded up to 1 and therefore the nonterminal has some weight instead
                   of none.
                   The reference code correctly uses the ceiling function to calculate
                   the weights.
                   See also https://en.wikipedia.org/wiki/Floor_and_ceiling_functions
                2) The index in the while loop needs to start with 1 instead of 0.
                   If code starts counting from 0, then j = 1 + (c mod n) becomes
                   j = c mod n or otherwise j may become too large and cause an IndexError.

    - Software implementation

        - `WeightedHierarchicalMapper.java
          <https://github.com/ericmedvet/evolved-ge/blob/master/src/main/java/it/units/malelab/ege/ge/mapper/WeightedHierarchicalMapper.java`__

            - For each nonterminal, the expressive power is calculated and stored
              in a HashMap as ceil(log2(expressive_power)):

                - data structure: weightsMap = new HashMap<>();
                - algorithm: a method called for each nonterminal with the name
                  countOptions(T symbol, int level, int maxLevel)
                - final value: int bits = (int) Math.ceil(Math.log10(options) / Math.log10(2d));

    """

    def count_options(grammar, symbol, level, max_level):
        if isinstance(symbol, _grammar.data_structures.TerminalSymbol):
            return 1
        rules = grammar.production_rules[symbol]
        if level >= max_level:
            return len(rules)
        count = 0
        for rule in rules:
            for rhs_symbol in rule:
                count += count_options(grammar, rhs_symbol, level + 1, max_level)
        return count

    ep_map = dict()
    for nt_symbol in grammar.nonterminal_symbols:
        value = count_options(grammar, nt_symbol, level=0, max_level=max_depth)
        ep_map[nt_symbol] = _math.ceil(_math.log2(value))
    return ep_map
