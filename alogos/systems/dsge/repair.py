"""Repair function to correct modified genotypes for DSGE."""

from ... import _grammar
from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from . import _cached_calculations
from . import default_parameters as _dp
from . import representation as _representation


_DT = _grammar.data_structures.DerivationTree
_NT = _grammar.data_structures.NonterminalSymbol


def fix_genotype(grammar, genotype, parameters=None, raise_errors=True):
    """Repair a genotype after modification by other operators.

    Parts of the genotype repair procedure:

    - The genes and codons guide a derivation, which is constructed here
      step-by-step to see how far it works and correct where it fails.
    - Repair 1: If the derivation is not finished yet, but every codon
      of a gene was used, all choices following afterwards are made by
      adding random codons.
    - Repair 2: If the derivation comes to a point where an encountered
      codon value is invalid, it is replaced by a random valid codon.
    - Repair 3: If the derivation ends before all codons of a gene were
      used, each unused codon following afterwards is deleted.
    - If the derivation reaches max depth in a branch, only
      non-recursive options are used from there on. This can mean that
      existing codons have to be replaced if they stand for recursive
      options, or new random codons have to be added which stand for
      recursive options.

    Notes
    -----
    The maximum tree-depth restriction is enforced in following way:

    - It is considered in each branch of the tree separately, instead
      of looking at the depth of the entire tree.
    - It is enforced as a soft limit and therefore a tree can become a
      bit deeper than the max depth value. Whenever a branch reaches
      the max depth, recursive rules are no longer used in expansions,
      but the branch can still grow a bit by applying non-recursive
      rules.

    The maximum expansion limit, which is not present in the original
    authors' conception, is used in following way when the user
    provides it:

    - It is enforced as a hard limit, a tree can not have more
      expansions in it.
    - If it is hit, a repair error is raised.

    """
    # Parameter extraction
    max_depth = _get_given_or_default("max_depth", parameters, _dp)
    max_expansions = _get_given_or_default("max_expansions", parameters, _dp)
    repair_with_random_choices = _get_given_or_default(
        "repair_with_random_choices", parameters, _dp
    )

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

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Transformation: Go through the derivation, add or remove codons when necessary
    nt_to_cnt = nt_to_cnt.copy()
    data = list(list(gene) for gene in genotype.data)
    derivation_tree = _DT(grammar)
    current_depth = 0
    expansion_counter = 0
    stack = [(derivation_tree.root_node, current_depth)]
    while stack:
        # Check stop conditions
        if max_expansions is not None and expansion_counter >= max_expansions:
            if raise_errors:
                _exceptions.raise_dsge_genotype_repair_error(max_expansions)
            break

        # 1) Choose nonterminal: DSGE uses the leftmost, unexpanded nonterminal
        chosen_nt_idx = 0
        chosen_nt_node, current_depth = stack.pop(chosen_nt_idx)
        chosen_nt = chosen_nt_node.symbol
        gene_idx = nt_to_gene[chosen_nt]
        # 2) Choose rule: DSGE decides by the next integer in the gene of the nonterminal
        rules = grammar.production_rules[chosen_nt]
        cnt = nt_to_cnt[chosen_nt]
        try:
            # Codon can be read from the genotype
            chosen_rule_idx = data[gene_idx][cnt]
        except IndexError:
            # Repair type 1: Add a valid codon
            if len(rules) == 1:
                chosen_rule_idx = 0
            else:
                if repair_with_random_choices:
                    chosen_rule_idx = _cached_calculations.get_random_valid_codon(
                        chosen_nt,
                        current_depth,
                        max_depth,
                        nt_to_num_options,
                        non_recursive_rhs,
                    )
                else:
                    chosen_rule_idx = _cached_calculations.get_first_valid_codon(
                        chosen_nt,
                        current_depth,
                        max_depth,
                        nt_to_num_options,
                        non_recursive_rhs,
                    )
            data[gene_idx].append(chosen_rule_idx)
        try:
            # Codon can be used to select a rule
            chosen_rule = rules[chosen_rule_idx]
        except IndexError:
            # Repair type 2: Replace an invalid codon with a valid one
            if repair_with_random_choices:
                chosen_rule_idx = _cached_calculations.get_random_valid_codon(
                    chosen_nt,
                    current_depth,
                    max_depth,
                    nt_to_num_options,
                    non_recursive_rhs,
                )
            else:
                chosen_rule_idx = _cached_calculations.get_first_valid_codon(
                    chosen_nt,
                    current_depth,
                    max_depth,
                    nt_to_num_options,
                    non_recursive_rhs,
                )
            data[gene_idx][cnt] = chosen_rule_idx
            chosen_rule = rules[chosen_rule_idx]
        nt_to_cnt[chosen_nt] += 1

        # 3) Expand the chosen nonterminal (1) with the rhs of the chosen rule (2)
        new_nodes = derivation_tree._expand(chosen_nt_node, chosen_rule)
        expansion_counter += 1

        # 4) Add new nodes that contain a nonterminal to the stack
        new_nt_nodes = [
            (node, current_depth + 1)
            for node in new_nodes
            if isinstance(node.symbol, _NT)
        ]
        stack = new_nt_nodes + stack

    # Repair type 3: Remove codons that were not used
    for nt in grammar.nonterminal_symbols:
        gene_idx = nt_to_gene[nt]
        last_codon_idx = nt_to_cnt[nt]
        data[gene_idx] = data[gene_idx][:last_codon_idx]

    data = tuple(tuple(gene) for gene in data)
    return _representation.Genotype(data)
