import random as _random

from ... import _grammar


def maps(grammar):
    """Calculate different mappings used by DSGE at various points."""
    gene_to_nt = {}
    nt_to_gene = {}
    nt_to_cnt = {}
    nt_to_num_options = {}
    mutable_genes = []
    for i, nt in enumerate(grammar.nonterminal_symbols):
        gene_to_nt[i] = nt
        nt_to_gene[nt] = i
        nt_to_cnt[nt] = 0
        num_options = len(grammar.production_rules[nt])
        nt_to_num_options[nt] = num_options
        if num_options > 1:
            mutable_genes.append(i)
    return gene_to_nt, nt_to_gene, nt_to_cnt, nt_to_num_options, mutable_genes


def non_recursive_rhs(grammar):
    """Determine which right-hand sides of rules are not recursive."""
    non_recursive_rhs = {}
    for nt in grammar.nonterminal_symbols:
        nr_rhs = []
        for idx, rhs in enumerate(grammar.production_rules[nt]):
            for symbol in rhs:
                if symbol == nt:
                    break
            else:
                nr_rhs.append(idx)
        non_recursive_rhs[nt] = nr_rhs
    return non_recursive_rhs


# Functions that could be cached, but are not, because it provides no speedup and requires memory


def get_all_valid_codons(
    nt, current_depth, max_depth, nt_to_num_options, non_recursive_rhs
):
    """Get all valid codons depending on current and max depth."""
    # Used in repair mechanism and neighborhood generation
    if current_depth >= max_depth:
        options = non_recursive_rhs[nt]
    else:
        options = list(range(nt_to_num_options[nt]))
    return options


def get_first_valid_codon(
    nt, current_depth, max_depth, nt_to_num_options, non_recursive_rhs
):
    """Get the first valid codon depending on current and max depth."""
    # Used in deterministic repair mechanism
    options = get_all_valid_codons(
        nt, current_depth, max_depth, nt_to_num_options, non_recursive_rhs
    )
    return options[0]


def get_random_valid_codon(
    nt, current_depth, max_depth, nt_to_num_options, non_recursive_rhs
):
    """Get a random valid codons depending on current and max depth."""
    # Used in stochastic repair mechanism
    options = get_all_valid_codons(
        nt, current_depth, max_depth, nt_to_num_options, non_recursive_rhs
    )
    return _random.choice(options)


def get_tree_depth(grammar, genotype, nt_to_gene, nt_to_cnt):
    """Determine the depth of the derivation tree encoded by a genotype."""
    nt_to_cnt = nt_to_cnt.copy()
    dt = _grammar.data_structures.DerivationTree(grammar)
    depth = 0
    stack = [(dt.root_node, 0)]
    while stack:
        # 1) Choose nonterminal: DSGE uses the leftmost, unexpanded nonterminal
        chosen_nt_node, dn = stack.pop(0)
        if dn > depth:
            depth = dn
        nt = chosen_nt_node.symbol
        gene_idx = nt_to_gene[nt]
        # 2) Choose rule: DSGE decides by the next integer in the gene of the nonterminal
        rules = grammar.production_rules[nt]
        cnt = nt_to_cnt[nt]
        chosen_rule_idx = genotype.data[gene_idx][cnt]
        chosen_rule = rules[chosen_rule_idx]
        nt_to_cnt[nt] += 1
        # 3) Expand the chosen nonterminal (1) with the rhs of the chosen rule (2)
        new_nodes = dt._expand(chosen_nt_node, chosen_rule)
        # 4) Add new nodes that contain a nonterminal to the stack
        new_nt_nodes = [
            (node, dn + 1)
            for node in new_nodes
            if isinstance(node.symbol, _grammar.data_structures.NonterminalSymbol)
        ]
        stack = new_nt_nodes + stack
    return depth
