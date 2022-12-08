"""Forward and reverse mapping functions for piGE."""

import random as _random

from ... import _grammar
from ... import exceptions as _exceptions
from ..._utilities import argument_processing as _ap
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import default_parameters as _dp
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_GT = _representation.Genotype
_DT = _grammar.data_structures.DerivationTree
_NT = _grammar.data_structures.NonterminalSymbol


def _fe(me, re):
    if re:
        _exceptions.raise_max_expansion_error(me)


def _fw(mw, re):
    if re:
        _exceptions.raise_max_wraps_error(mw)


# Forward mapping
def forward(
    grammar,
    genotype,
    parameters=None,
    raise_errors=True,
    return_derivation_tree=False,
    verbose=False,
):
    """Map a piGE genotype to a string phenotype.

    Beginning with the start symbol, each nonterminal symbol is expanded
    by applying a production rule. These rule applications are done
    until the sequence of symbols contains only terminals. Then the
    symbols form a sentence of the language defined by the grammar and
    hence are a valid phenotype.

    Choices to be made in the mapping process:

    1. Which nonterminal is expanded if several are present in the
       current sequence of symbols?

       While standard GE always uses the leftmost nonterminal, here the
       next nonterminal is chosen by the next codon in the data
       (aka "order codon").

    2. Which production rule is used to expand a selected nonterminal?

       In case there is more than one production rule available for the
       nonterminal, both GE and piGE use the next codon in the data to
       select one (aka "content codon").

       The formula used is: ``rule id = codon value % number of rules for the NT``.

       The data is read from left to right. If the end is reached, a
       so-called wrap is done to restart again from the left, i.e.
       codons can be used more than once. If a maximum number of wraps
       is reached, the mapping process is stopped and the individual
       has no valid phenotype and should receive the worst possible
       fitness during fitness evaluation.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype` or data that can be converted to it
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        - ``max_expansions``  (`int`): Maximum number of nonterminal
          expansions allowed in the derivation created by the mapping
          process.
        - ``max_wraps``  (`int`): Maximum number of times the genotype
          is allowed to be wrapped in order to have more codons
          available for making decisions in the mapping process.
        - ``stack_mode`` (`str`): Mode by which newly discovered symbols
          are added to the stack of all currently present symbols in
          a derivation.

          Possible values:

          - ``"start"``: Insert new nodes at the start of the stack.
          - ``"end"``: Insert new nodes at the end of the stack.
          - ``"inplace"``: Insert new nodes at the same place as the
            expanded nonterminal in the stack.
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
    mw = _get_given_or_default("max_wraps", parameters, _dp)
    sm = _get_given_or_default("stack_mode", parameters, _dp)
    if me is None and mw is None:
        mw = 0

    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Mapping
    if verbose or sm != "inplace":
        dt = _forward_slow(grammar, genotype.data, me, mw, sm, raise_errors, verbose)
    else:
        dt = _forward_fast(grammar, genotype.data, me, mw, sm, raise_errors)
    phe = dt.string()

    # Conditional return
    if return_derivation_tree:
        return phe, dt
    return phe


def _forward_fast(gr, gt, me, mw, sm, re):
    """Calculate the genotype-to-phenotype map of piGE in a fast way."""
    d = _DT(gr)
    x = d._expand
    s = [d.root_node]
    p = s.pop
    w = gr.production_rules
    g = len(gt)
    c = 0
    e = 0
    while s:
        if me is not None and e >= me:
            _fe(me, re)
            break
        if mw is not None and (c // g) > mw:
            _fw(mw, re)
            break
        o = gt[c % g]
        c += 1
        k = gt[c % g]
        c += 1
        i = o % len(s)
        f = p(i)
        r = w[f.symbol]
        s[i:i] = [n for n in x(f, r[k % len(r)]) if isinstance(n.symbol, _NT)]
        e += 1
    return d


def _forward_slow(
    grammar, genotype, max_expansions, max_wraps, stack_mode, raise_errors, verbose
):
    """Calculate the genotype-to-phenotype map of piGE in a slow way.

    This is a readable implementation of the mapping process, which
    also allows to print output about the steps it involves.
    It served as basis for the faster, minified implementation in this
    module and may be helpful in understanding, replicating or modifying
    the algorithm.

    """
    # Argument processing
    _ap.str_arg("stack_mode", stack_mode, vals=["start", "end", "inplace"])

    # Create derivation tree
    derivation_tree = _grammar.data_structures.DerivationTree(grammar)
    stack = [derivation_tree.root_node]
    num_codons = len(genotype)
    codon_counter = 0
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
        # Check stop conditions
        if max_wraps is not None and (codon_counter // num_codons) > max_wraps:
            if verbose:
                header = "Stop condition fulfilled"
                print(header)
                print("=" * len(header))
                print(
                    "- The maximum number of wraps and end of genotype are "
                    "reached: {}".format(max_wraps)
                )
                print()
            if raise_errors:
                _exceptions.raise_max_wraps_error(max_wraps)
            break
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

        # 1) Choose nonterminal: piGE uses an "order codon" to select one from all unexpanded nt
        order_codon = genotype[codon_counter % num_codons]
        codon_counter += 1
        num_nonterminals = len(stack)
        chosen_nt_idx = order_codon % num_nonterminals
        chosen_nt_node = stack.pop(chosen_nt_idx)
        if verbose:
            header = "Expansion {}".format(expansion_counter + 1)
            print(header)
            print("=" * len(header))
            print("- Choice of nonterminal to expand")
            print(
                "    Index: {} by order codon value {} % {} "
                "nonterminals".format(chosen_nt_idx, order_codon, num_nonterminals)
            )
            print("    Symbol: <{}>".format(chosen_nt_node.symbol.text))

        # 2) Choose rule: piGE uses a "content codon" to select a rule from 1 to n avaiable ones
        if verbose:
            print("- Choice of rule to apply")
        content_codon = genotype[codon_counter % num_codons]
        codon_counter += 1
        rules = grammar.production_rules[chosen_nt_node.symbol]
        num_rules = len(rules)
        chosen_rule_idx = content_codon % num_rules
        chosen_rule = rules[chosen_rule_idx]
        if verbose:
            print(
                "    Index: {} by content codon value {} % {} "
                "rules".format(chosen_rule_idx, content_codon, num_rules)
            )

        # 3) Expand the chosen nonterminal with the rhs of the chosen rule
        if verbose:
            print("- Application of rule to nonterminal")
        new_nodes = derivation_tree._expand(chosen_nt_node, chosen_rule)
        expansion_counter += 1
        if verbose:
            rhs = "".join(
                sym.text
                if isinstance(sym, _grammar.data_structures.TerminalSymbol)
                else "<{}>".format(sym.text)
                for sym in chosen_rule
            )
            print(
                "    Substitution: <{}> -> {}".format(chosen_nt_node.symbol.text, rhs)
            )

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
        if verbose:
            print()
            print("Sentential form: {}".format(derivation_tree.string()))
            print()
    if verbose:
        header = "End of the derivation"
        print(header)
        print("=" * len(header))
        print()
        name = (
            "String" if derivation_tree.is_completely_expanded() else "Sentential form"
        )
        print("{}: {}".format(name, derivation_tree.string()))
    return derivation_tree


# Reverse mapping
def reverse(
    grammar, phenotype_or_derivation_tree, parameters=None, return_derivation_tree=False
):
    """Map a string phenotype (or derivation tree) to a piGE genotype.

    This is a reversal of the mapping procedure of
    position-independent Grammatical Evolution (piGE).
    Note that many different piGE genotypes can encode the same
    derivation tree and phenotype. It is possible to return a
    deterministic piGE genotype that uses the lowest possible integer
    value for each choice of expansion, or a random piGE genotype that
    uses a random integer value within the codon size limit.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    phenotype_or_derivation_tree : `str` or `~alogos._grammar.data_structures.DerivationTree`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``codon_size``  (`int`): Codon size in bits. The number of
          different integer values a codon can assume is therefore
          determined by ``2**codon_size``.
        - ``codon_randomization``  (`bool`): If `True`, the reverse
          mapping will use the so-called "unmod" operation and therefore
          generate random integers that encode the required rule choice.
          If `False`, the lowest possible integer will be used and the
          reverse mapping becomes deterministic.
        - ``derivation_order`` (`str`): Choice of next nonterminal to
          expand, which in piGE is chosen by a codon in the genotype.

          Possible values:

          - ``"leftmost"``: Always choose the leftmost unexpanded
            nonterminal in the partial derivation.
          - ``"rightmost"``: Always choose the rightmost unexpanded
            nonterminal in the partial derivation.
          - ``"random"``: Always choose a random unexpanded
            nonterminal in the partial derivation. This makes the
            reverse mapping stochastic.
        - ``stack_mode`` (`str`): Mode by which newly discovered symbols
          are added to the stack of all currently present symbols in
          a derivation.

          Possible values:

          - ``"start"``: Insert new nodes at the start of the stack.
          - ``"end"``: Insert new nodes at the end of the stack.
          - ``"inplace"``: Insert new nodes at the same place as the
            expanded nonterminal in the stack.
    return_derivation_tree : `bool`, optional
        If `True`, not only the genotype is returned but additionally
        also the derivation tree.

    Returns
    -------
    genotype : `~.representation.Genotype`
        If ``return_derivation_tree`` is `False`, which is the default.
    (genotype, derivation_tree) : `tuple` with two elements of type `~.representation.Genotype` and `~alogos._grammar.data_structures.DerivationTree`
        If ``return_derivation_tree`` is `True`.

    Raises
    ------
    MappingError
        If the reverse mapping fails because the string does not belong
        to the grammar's language or the derivation tree does not
        represent a valid derivation.

    """
    # Parameter extraction
    codon_size = _get_given_or_default("codon_size", parameters, _dp)
    codon_randomization = _get_given_or_default("codon_randomization", parameters, _dp)
    derivation_order = _get_given_or_default("derivation_order", parameters, _dp)
    stack_mode = _get_given_or_default("stack_mode", parameters, _dp)

    # Argument processing
    _ap.str_arg(
        "derivation_order", derivation_order, vals=("leftmost", "rightmost", "random")
    )
    _ap.str_arg("stack_mode", stack_mode, vals=("start", "end", "inplace"))
    max_int = 2**codon_size - 1

    # Preparation of data structures
    dt = _shared.mapping.get_derivation_tree(grammar, phenotype_or_derivation_tree)
    gt = []

    # Trace all decisions contained in the given derivation tree
    root = dt.root_node
    stack = [root]  # stack to ensure stack_mode
    stack_tlo = [
        root
    ]  # stack to ensure derivation_order (tlo = nodes in tree leaf order)
    while stack:
        # 1) Choose nonterminal: piGE decides via next "order codon" -> Choose it arbitrarily
        if derivation_order == "leftmost":
            chosen_nt_idx_tlo = 0
        elif derivation_order == "rightmost":
            chosen_nt_idx_tlo = len(stack_tlo) - 1
        elif derivation_order == "random":
            chosen_nt_idx_tlo = _random.randint(0, len(stack_tlo) - 1)
        chosen_nt_node = stack_tlo.pop(chosen_nt_idx_tlo)
        chosen_nt_idx = stack.index(chosen_nt_node)
        chosen_nt_idx_stored = chosen_nt_idx
        chosen_nt_node = stack.pop(chosen_nt_idx)
        if chosen_nt_idx > max_int:
            _exceptions.raise_limited_codon_size_error(chosen_nt_idx, max_int)
        if codon_randomization:
            options = range(chosen_nt_idx, max_int + 1, len(stack) + 1)
            chosen_nt_idx_stored = _random.choice(options)

        # 2) Choose rule: piGE decides via next "content codon" -> Deduce it from tree
        try:
            rules = grammar.production_rules[chosen_nt_node.symbol]
        except Exception:
            _exceptions.raise_missing_nt_error(chosen_nt_node)
        try:
            chosen_rule = [node.symbol for node in chosen_nt_node.children]
            chosen_rule_idx = rules.index(chosen_rule)
        except ValueError:
            _exceptions.raise_missing_rhs_error(chosen_nt_node, chosen_rule)
        if chosen_rule_idx > max_int:
            _exceptions.raise_limited_codon_size_error(chosen_rule_idx, max_int)
        if codon_randomization:
            options = range(chosen_rule_idx, max_int + 1, len(rules))
            chosen_rule_idx = _random.choice(options)

        # 3) Expand the chosen nonterminal with the rhs of the chosen rule -> Follow the expansion
        new_nt_nodes = [
            node for node in chosen_nt_node.children if isinstance(node.symbol, _NT)
        ]
        if stack_mode == "start":
            stack = new_nt_nodes + stack
        elif stack_mode == "end":
            stack = stack + new_nt_nodes
        elif stack_mode == "inplace":
            stack[chosen_nt_idx:chosen_nt_idx] = new_nt_nodes
        stack_tlo[chosen_nt_idx_tlo:chosen_nt_idx_tlo] = new_nt_nodes

        # Store the observed decisions, even if there was only one option
        gt.append(chosen_nt_idx_stored)
        gt.append(chosen_rule_idx)

    # Finalization of data structures
    gt = _GT(gt)

    # Conditional return
    if return_derivation_tree:
        return gt, dt
    return gt
