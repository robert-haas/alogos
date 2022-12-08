"""Neighborhood functions to generate nearby genotypes for piGE."""

from ... import _grammar
from ..._utilities import argument_processing as _ap
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import default_parameters as _default_parameters
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_NT = _grammar.data_structures.NonterminalSymbol
_T = _grammar.data_structures.TerminalSymbol


def int_replacement(grammar, genotype, parameters=None):
    """Systematically change a chosen number of int codons.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``neighborhood_distance`` (`int`) : The distance from the
          original genotype to a new genotype in terms of replaced
          int codons.
        - ``neighborhood_max_size`` (`int`) : Maximum number of
          neighbor genotypes to generate.
        - ``neighborhood_only_terminals`` (`bool`) : If `True`, only
          replace codons that represent terminals.
        - ``max_expansions`` (`bool`) : Maximum number of nonterminal
          expansions used in a derivation.
        - ``max_wraps`` (`bool`) : Maximum number of wraps of the
          genotype that can be used in a derivation.
        - ``stack_mode`` (`str`) : Mode by which new nonterminals are
          added to the stack of all nonterminals that were not yet
          expanded.

          Possible values:

          - ``"start"``: At the beginning of the stack.
          - ``"end"``: At the end of the stack.
          - ``"inplace"``: At the position where the nonterminal was
            located that was rewritten and led to the new nonterminal.

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
    max_expansions = _get_given_or_default(
        "max_expansions", parameters, _default_parameters
    )
    max_wraps = _get_given_or_default("max_wraps", parameters, _default_parameters)
    stack_mode = _get_given_or_default("stack_mode", parameters, _default_parameters)

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Get alternative choices per position by going through a forward mapping
    choices = _get_choices_per_position(
        grammar, genotype.data, max_expansions, max_wraps, stack_mode, only_t
    )
    num_choices_per_pos = [len(x) for x in choices]

    # Generate combinations of choices
    combinations = _shared.neighborhood.generate_combinations(
        num_choices_per_pos, distance, max_size
    )

    # Construct neighborhood genotypes from combinations
    nbrs = []
    n = len(genotype)
    gt_unrolled = [genotype.data[i % n] for i in range(len(choices))]
    for comb in combinations:
        # Semantics of comb: 0 means no change, >0 points to a certain alternative choice
        data = [
            choices[i][val - 1] if val > 0 else gt_unrolled[i]
            for i, val in enumerate(comb)
        ]
        nbrs.append(_representation.Genotype(data))
    return nbrs


def _get_choices_per_position(
    grammar, codons, max_expansions, max_wraps, stack_mode, only_terminals
):
    """Determine alternative choices for each position in the genotype."""
    # Argument processing
    _ap.str_arg("stack_mode", stack_mode, vals=("start", "end", "inplace"))

    # Create derivation tree
    dt = _grammar.data_structures.DerivationTree(grammar)
    stack = [dt.root_node]
    num_codons = len(codons)
    codon_counter = 0
    expansion_counter = 0
    choices = []
    while stack:
        # Check stop conditions
        if max_wraps is not None and (codon_counter // num_codons) > max_wraps:
            break
        if max_expansions is not None and expansion_counter >= max_expansions:
            break

        # 1) Choose nonterminal: piGE uses an "order codon" to select one from all unexpanded nt
        order_codon = codons[codon_counter % num_codons]
        codon_counter += 1
        num_nonterminals = len(stack)
        chosen_nt_idx = order_codon % num_nonterminals
        chosen_nt_node = stack.pop(chosen_nt_idx)

        # 2) Choose rule: piGE uses a "content codon" to select a rule from 1 to n avaiable ones
        content_codon = codons[codon_counter % num_codons]
        codon_counter += 1
        rules = grammar.production_rules[chosen_nt_node.symbol]
        num_rules = len(rules)
        chosen_rule_idx = content_codon % num_rules
        chosen_rule = rules[chosen_rule_idx]

        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        expansion_counter += 1

        # 4) Add new nodes that contain a nonterminal to the stack
        new_nt_nodes = [node for node in new_nodes if isinstance(node.symbol, _NT)]
        if stack_mode == "start":
            # Alternative 1: Insert new nodes at start, suggested by partial example in 2004 paper
            stack = new_nt_nodes + stack
        elif stack_mode == "end":
            # Alternative 2: Insert new nodes at end, suggested by full example in 2010 paper
            # shown explicitly in Fig. 3
            stack = stack + new_nt_nodes
        elif stack_mode == "inplace":
            # Alternative 3: Insert new nodes at position of the expanded nt, suggested by
            # sentential forms during a derivation and by handbook in 2018
            stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes

        # Remember alternative choices for the current production rule
        if only_terminals:
            other_rule_indices = [
                idx
                for idx, rule in enumerate(rules)
                if idx != chosen_rule_idx and any(isinstance(sym, _T) for sym in rule)
            ]
        else:
            other_rule_indices = [
                idx for idx in range(num_rules) if idx != chosen_rule_idx
            ]
        choices.append([])  # order codon: fixed, use no alternatives
        choices.append(other_rule_indices)  # content codon: changable, use alternatives
    return choices
