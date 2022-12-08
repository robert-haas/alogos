"""Mutation functions for DSGE."""

import random as _random

from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from . import _cached_calculations
from . import default_parameters as _dp
from . import repair as _repair
from . import representation as _representation


def int_replacement_by_probability(grammar, genotype, parameters=None):
    """Change randomly chosen integers to new random values with some restrictions.

    Restrictions on what can be mutated according to the 2018 paper:

    - A randomly chosen gene needs to have more than one option for
      expansion.
    - A randomly chosen integer needs to be used in the
      genotype-to-phenotype mapping.
    - The maximum tree-depth needs to be respected.

    References
    ----------
    - Software implementations by the authors of the approach

        - Python: `dsge
          <https://github.com/nunolourenco/dsge>`__

            - `core/sge.py
              <https://github.com/nunolourenco/dsge/blob/master/src/core/sge.py>`__:
              ``def mutate(p)`` is the implementation of the mutation
              operator

    - Papers

        - Lourenço et al. in 2018:
          `Structured Grammatical Evolution: A Dynamic Approach
          <https://doi.org/10.1007/978-3-319-78717-6_6>`__ (DSGE)

            - p. 145: "The mutation operator is restricted to integers
              that are used in the genotype to phenotype mapping and
              changes a randomly selected expansion option (encoded as
              an integer) to another valid one, constrained to the
              restrictions imposed by the maximum tree-depth. To do so,
              we first select one gene. Then, we randomly select one of
              its integers and replace it with another valid
              possibility. Note that genes where there is just one
              possibility for expansion are not considered for mutation
              purposes."

    """
    # Parameter extraction
    probability = _get_given_or_default(
        "mutation_int_replacement_probability", parameters, _dp
    )
    max_depth = _get_given_or_default("max_depth", parameters, _dp)
    repair = _get_given_or_default("repair_after_mutation", parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Cache lookup
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

    # Mutation: Randomly decide for each positions in the genotype whether it shall be modified
    data = genotype.data
    num_pos = sum(len(data[gi]) for gi in mutable_genes)
    pos = set(i for i in range(num_pos) if _random.random() < probability)
    data = _change_chosen_positions(
        data,
        pos,
        current_depth,
        max_depth,
        gene_to_nt,
        non_recursive_rhs,
        nt_to_num_options,
        mutable_genes,
    )

    # Optional repair of the new genotype
    if repair:
        genotype = _repair.fix_genotype(grammar, data, parameters)
    else:
        genotype = _representation.Genotype(data)
    return genotype


def int_replacement_by_count(grammar, genotype, parameters=None):
    """Change randomly chosen integers to new random values with some restrictions.

    Restrictions on what can be mutated according to the 2018 paper:
    - A randomly chosen gene needs to have more than one option for
      expansion.
    - A randomly chosen integer needs to be used in the
      genotype-to-phenotype mapping.
    - The maximum tree-depth needs to be respected.

    Notes
    -----
    This mutation operator is not mentioned in DSGE literature, but a
    straightforward variant of the default procedure.

    """
    # Parameter extraction
    flip_count = _get_given_or_default(
        "mutation_int_replacement_count", parameters, _dp
    )
    max_depth = _get_given_or_default("max_depth", parameters, _dp)
    repair = _get_given_or_default("repair_after_mutation", parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Cache lookup
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

    # Mutation: Choose n different positions in the genotype to flip
    data = genotype.data
    num_pos = sum(len(data[gi]) for gi in mutable_genes)
    pos = range(num_pos)
    if num_pos > flip_count:
        pos = _random.sample(pos, flip_count)
    pos = set(pos)
    data = _change_chosen_positions(
        data,
        pos,
        current_depth,
        max_depth,
        gene_to_nt,
        non_recursive_rhs,
        nt_to_num_options,
        mutable_genes,
    )

    # Optional repair of the new genotype (either here and/or later)
    if repair:
        genotype = _repair.fix_genotype(grammar, data, parameters, raise_errors=False)
    else:
        genotype = _representation.Genotype(data)
    return genotype


def _change_chosen_positions(
    data,
    pos,
    current_depth,
    max_depth,
    gene_to_nt,
    non_recursive_rhs,
    nt_to_num_options,
    mutable_genes,
):
    """Mutate the selected positions by replacing the integer with another valid option.

    Notes
    -----
    The maximum tree-depth restriction could be enforced in at least two
    different ways:

    1. If a tree has reached the maximum depth, only use non-recursive
       rules when modifying any position in the tree.

    2. If a branch of the tree reached the the maximum depth, only use
       non-recursive rules when modifying a position within this branch,
       while still allowing the use of recursive rules in other
       branches.

    This implementation uses option 1, because the original authors'
    implementation does so too. Note that this is in contrast to how the
    maximum tree-depth restriction is enforced in the initialization and
    repair procedures, where the depth of the current branch is
    considered rather then the depth of the tree.

    """
    # Define which values are valid depending on depth limit
    if (
        current_depth >= max_depth
    ):  # Note: >= is used in the original authors' implementation

        def get_new_codon(nt, codon):
            return _random.choice([x for x in non_recursive_rhs[nt] if x != codon])

    else:

        def get_new_codon(nt, codon):
            return _random.choice(
                [x for x in range(nt_to_num_options[nt]) if x != codon]
            )

    # Flip the chosen positions to new valid codons
    new = [[] for _ in data]
    cnt = 0
    for gi in mutable_genes:
        gene = data[gi]
        nt = gene_to_nt[gi]
        for codon in gene:
            if cnt in pos:
                new[gi].append(get_new_codon(nt, codon))
            else:
                new[gi].append(codon)
            cnt += 1
    return tuple(tuple(gene) for gene in new)
