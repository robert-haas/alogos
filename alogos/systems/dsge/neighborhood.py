"""Neighborhood functions to generate nearby genotypes for DSGE."""

from ... import _grammar
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import _cached_calculations
from . import default_parameters as _default_parameters
from . import repair as _repair
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_T = _grammar.data_structures.TerminalSymbol


def int_replacement(grammar, genotype, parameters=None):
    """Change systematically chosen integers."""
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
    max_depth = _get_given_or_default("max_depth", parameters, _default_parameters)
    repair_parameters = parameters.copy() if parameters else dict()
    repair_parameters["repair_with_random_choices"] = False

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Get alternative choices per position by going through a forward mapping
    choices = _get_choices_per_position(grammar, genotype, max_depth, only_t)
    num_choices_per_pos = [len(x) for x in choices]

    # Generate combinations
    combinations = _shared.neighborhood.generate_combinations(
        num_choices_per_pos, distance, max_size
    )

    # Construct neighborhood genotypes from combinations
    index_map = {}
    cnt = 0
    for gi, gene in enumerate(genotype.data):
        for ci in range(len(gene)):
            index_map[cnt] = (gi, ci)
            cnt += 1

    nbrs = set()
    for comb in combinations:
        data = [[codon for codon in gene] for gene in genotype.data]
        for idx, val in enumerate(comb):
            if val != 0:
                gi, ci = index_map[idx]
                data[gi][ci] = choices[idx][val - 1]
        gt = _repair.fix_genotype(grammar, data, parameters=repair_parameters)
        nbrs.add(gt)
    return list(nbrs)


def _get_choices_per_position(grammar, genotype, max_depth, only_terminals):
    """Determine alternative choices for each position in the genotype."""
    # Cache look-up
    non_recursive_rhs = grammar._lookup_or_calc(
        "dsge", "non_recursive_rhs", _cached_calculations.non_recursive_rhs, grammar
    )
    (
        gene_to_nt,
        nt_to_gene,
        nt_to_cnt,
        nt_to_num_options,
        mutable_genes,
    ) = grammar._lookup_or_calc("dsge", "maps", _cached_calculations.maps, grammar)
    current_depth = _cached_calculations.get_tree_depth(
        grammar, genotype, nt_to_gene, nt_to_cnt
    )

    # Get choices
    genes = genotype.data
    choices = []
    for gene_idx in range(len(genes)):
        nt = gene_to_nt[gene_idx]
        for codon in genes[gene_idx]:
            options = _cached_calculations.get_all_valid_codons(
                nt, current_depth, max_depth, nt_to_num_options, non_recursive_rhs
            )
            if only_terminals:
                rhs = grammar.production_rules[nt]
                other_rule_indices = [
                    opt
                    for opt in options
                    if opt != codon and any(isinstance(sym, _T) for sym in rhs[opt])
                ]
            else:
                other_rule_indices = [opt for opt in options if opt != codon]
            choices.append(other_rule_indices)
    return choices
