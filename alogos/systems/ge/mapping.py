"""Forward and reverse mapping functions for GE."""

import random as _random

from ... import _grammar
from ... import exceptions as _exceptions
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
    """Map a GE genotype to a string phenotype.

    It converts a genotype (=originally a list of integers) to a
    derivation tree (=linked nodes that each contain a nonterminal
    or terminal symbol), which can be further converted to a phenotype
    (=a string of the grammar's language).

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
    if me is None and mw is None:
        mw = 0

    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Mapping
    if verbose:
        dt = _forward_slow(grammar, genotype.data, me, mw, raise_errors, verbose)
    else:
        dt = _forward_fast(grammar, genotype.data, me, mw, raise_errors)
    phe = dt.string()

    # Conditional return
    if return_derivation_tree:
        return phe, dt
    return phe


def _forward_fast(gr, gt, me, mw, re):
    """Calculate the genotype-to-phenotype map of GE in a fast way."""
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
        f = p(0)
        r = w[f.symbol]
        if len(r) == 1:
            h = r[0]
        else:
            h = r[gt[c % g] % len(r)]
            c += 1
        s[0:0] = [n for n in x(f, h) if isinstance(n.symbol, _NT)]
        e += 1
    return d


def _forward_slow(grammar, genotype, max_expansions, max_wraps, raise_errors, verbose):
    """Calculate the genotype-to-phenotype map of GE in a slow way.

    This is a readable implementation of the mapping process, which
    also allows to print output about the steps it involves.
    It served as basis for the faster, minified implementation in this
    module and may be helpful in understanding, replicating or modifying
    the algorithm.

    """
    # Note: Using deque from collections as stack did not increase
    # performance due to the necessity of using reverse() with it
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
        if max_wraps is not None and codon_counter // num_codons > max_wraps:
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

        # 1) Choose nonterminal: GE uses the leftmost, unexpanded nonterminal
        chosen_nt_idx = 0
        chosen_nt_node = stack.pop(chosen_nt_idx)
        if verbose:
            header = "Expansion {}".format(expansion_counter + 1)
            print(header)
            print("=" * len(header))
            print("- Choice of nonterminal to expand")
            print("    Index: {} by always using leftmost".format(chosen_nt_idx))
            print("    Symbol: <{}>".format(chosen_nt_node.symbol.text))

        # 2) Choose rule: GE decides by the next "codon" in the genotype
        if verbose:
            print("- Choice of rule to apply")
        rules = grammar.production_rules[chosen_nt_node.symbol]
        num_rules = len(rules)
        if num_rules == 1:
            chosen_rule_idx = 0
            if verbose:
                print(
                    "    Index: {} by using the only available rule without "
                    "consuming a codon".format(chosen_rule_idx)
                )
        else:
            content_codon = genotype[codon_counter % num_codons]
            codon_counter += 1
            chosen_rule_idx = content_codon % num_rules
            if verbose:
                print(
                    "    Index: {} by codon value {} % {} "
                    "rules".format(chosen_rule_idx, content_codon, num_rules)
                )
        chosen_rule = rules[chosen_rule_idx]

        # 3) Expand the chosen nonterminal (1) with the rhs of the chosen rule (2)
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
        stack = new_nt_nodes + stack
        if verbose:
            print()
            print("Sentential form: {}".format(derivation_tree.string()))
            print()
    if verbose:
        header = "End of the derivation"
        print(header)
        print("=" * len(header))
        complete = derivation_tree.is_completely_expanded()
        if complete:
            message = (
                "The derivation is finished. The result contains only terminal symbols."
            )
            name = "String"
        else:
            message = (
                "The derivation is not finished. The result contains unexpanded "
                "nonterminal symbols."
            )
            name = "Sentential form"
        print("- Completeness check: {}".format(message))
        print()
        print("{}: {}".format(name, derivation_tree.string()))
    return derivation_tree


# Reverse mapping
def reverse(
    grammar, phenotype_or_derivation_tree, parameters=None, return_derivation_tree=False
):
    """Map a string phenotype (or derivation tree) to a GE genotype.

    This is a reversal of the mapping procedure of
    Grammatical Evolution (GE). Note that many different GE genotypes
    can encode the same derivation tree and phenotype. It is possible
    to return a deterministic GE genotype that uses the lowest possible
    integer value for each choice of expansion, or a random GE genotype
    that uses a random integer value within the codon size limit.
    The latter is describes in literature as using an "unmod" operation
    to find a random codon value which is valid, in the sense that after
    applying the "mod" operation to it the result is the lowest possible
    integer.

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

    References
    ----------
    - `Ryan, O'Neill, Collins - Handbook of Grammatical Evolution (2018)
      <https://doi.org/10.1007/978-3-319-78717-6>`__

        - p. 12: "Unmod produces the actual codons that will be used and
          essentially performs the opposite operation to mod, returning
          a number that, when divided by the number of choices available
          for the particular non-terminal, will return the choice made."

    """
    # Parameter extraction
    codon_size = _get_given_or_default("codon_size", parameters, _dp)
    codon_randomization = _get_given_or_default("codon_randomization", parameters, _dp)

    # Argument processing
    max_int = 2**codon_size - 1

    # Preparation of data structures
    dt = _shared.mapping.get_derivation_tree(grammar, phenotype_or_derivation_tree)
    gt = []

    # Trace all decisions contained in the given derivation tree
    stack = [dt.root_node]
    while stack:
        # 1) Choose nonterminal: GE uses leftmost -> Choose it accordingly
        chosen_nt_idx = 0
        chosen_nt_node = stack.pop(chosen_nt_idx)

        # 2) Choose rule: GE decides via the next codon in the genotype -> Deduce it from tree
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
        stack = new_nt_nodes + stack

        # Store the observed decision, but only if there was more than one option to choose from
        if len(rules) > 1:
            gt.append(chosen_rule_idx)

    # Finalization of data structures
    gt = _GT(gt)

    # Conditional return
    if return_derivation_tree:
        return gt, dt
    return gt
