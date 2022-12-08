"""Forward and reverse mapping functions for DSGE."""

from ... import _grammar
from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from .. import _shared
from . import _cached_calculations
from . import default_parameters as _dp
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_GT = _representation.Genotype
_DT = _grammar.data_structures.DerivationTree
_NT = _grammar.data_structures.NonterminalSymbol
_T = _grammar.data_structures.TerminalSymbol


def _fe(me, re):
    if re:
        _exceptions.raise_max_expansion_error(me)


def _ge(gi, nt, re):
    if re:
        _exceptions.raise_dsge_mapping_error2(gi, nt)


def _he(gi, sym, r, ri, re):
    if re:
        _exceptions.raise_dsge_mapping_error3(gi, sym, r, ri)


# Forward mapping
def forward(
    grammar,
    genotype,
    parameters=None,
    raise_errors=True,
    return_derivation_tree=False,
    verbose=False,
):
    """Map a DSGE genotype to a string phenotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype` or data that can be converted to it
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``max_expansions``  (`int`): Maximum number of nonterminal
          expansions allowed in the derivation created by the mapping
          process.
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

    Notes
    -----
    The 2018 paper describes that the mapping function sometimes has to
    randomly select rules during a derivation. This is necessary when
    the genotype was modified by a variation operator in such a way that
    not enough codons are available in a gene of the genotype to fully
    encode a phenotype. Here, in this implementation of DSGE, the random
    selection of rules was removed from the mapping function. Instead, a
    dedicated repair function was added to the system, which is
    responsible to add missing or remove superfluous codons of a
    genotype after it was modified by a variation operator. The
    advantage with this approach is that the mapping function becomes
    deterministic, meaning that the same genotype is always mapped to
    the same phenotype, as is the usually the case for systems inspired
    by Grammatical Evolution (GE). This eases the analysis of completed
    runs and the reuse of genotypes in future runs.

    """
    # Parameter extraction
    me = _get_given_or_default("max_expansions", parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Cache look-up
    _, nt_to_gene, nt_to_cnt, _, _ = grammar._lookup_or_calc(
        "dsge", "maps", _cached_calculations.maps, grammar
    )
    nt_to_cnt = nt_to_cnt.copy()

    # Mapping
    if raise_errors:
        if len(genotype) != len(grammar.nonterminal_symbols):
            _exceptions.raise_dsge_mapping_error1(genotype, grammar.nonterminal_symbols)
    if verbose:
        dt = _forward_slow(
            grammar, genotype.data, me, nt_to_gene, nt_to_cnt, raise_errors, verbose
        )
    else:
        dt = _forward_fast(
            grammar, genotype.data, me, nt_to_gene, nt_to_cnt, raise_errors
        )
    phe = dt.string()

    # Conditional return
    if return_derivation_tree:
        return phe, dt
    return phe


def _forward_fast(gr, gt, me, m1, m2, re):
    """Calculate the genotype-to-phenotype map of DSGE in a fast way."""
    d = _DT(gr)
    s = [d.root_node]
    e = 0
    x = d._expand
    p = s.pop
    w = gr.production_rules
    while s:
        if me is not None and e >= me:
            _fe(me, re)
            break
        f = p(0)
        n = f.symbol
        r = w[n]
        g = m1[n]
        c = m2[n]
        try:
            i = gt[g][c]
        except IndexError:
            _ge(g, n, re)
            break
        m2[n] += 1
        try:
            h = r[i]
        except IndexError:
            _he(g, n, r, i, re)
            break
        s[0:0] = [n for n in x(f, h) if isinstance(n.symbol, _NT)]
        e += 1
    return d


def _forward_slow(
    grammar, genotype, max_expansions, nt_to_gene, nt_to_cnt, raise_errors, verbose
):
    """Calculate the genotype-to-phenotype map of DSGE in a slow way.

    This is a readable implementation of the mapping process, which
    also allows to print output about the steps it involves.
    It served as basis for the faster, minified implementation in this
    module and may be helpful in understanding, replicating or modifying
    the algorithm.

    """
    derivation_tree = _DT(grammar)
    stack = [derivation_tree.root_node]
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

        # 1) Choose nonterminal: DSGE uses the leftmost, unexpanded nonterminal
        chosen_nt_idx = 0
        chosen_nt_node = stack.pop(chosen_nt_idx)
        chosen_nt = chosen_nt_node.symbol
        if verbose:
            header = "Expansion {}".format(expansion_counter + 1)
            print(header)
            print("=" * len(header))
            print("- Choice of nonterminal to expand")
            print("    Index: {} by always using leftmost".format(chosen_nt_idx))
            print("    Symbol: <{}>".format(chosen_nt.text))

        # 2) Choose rule: DSGE decides by the next integer in the gene of the nonterminal
        if verbose:
            print("- Choice of rule to apply")
        rules = grammar.production_rules[chosen_nt]
        gi = nt_to_gene[chosen_nt]
        cnt = nt_to_cnt[chosen_nt]
        try:
            chosen_rule_idx = genotype[gi][cnt]
        except IndexError:
            if raise_errors:
                _exceptions.raise_dsge_mapping_error2(gi, chosen_nt)
            break
        nt_to_cnt[chosen_nt] += 1
        if verbose:
            print(
                "    Index: {} by codon value {} from gene {} at position {}".format(
                    chosen_rule_idx, genotype[gi][cnt], gi, cnt
                )
            )
        try:
            chosen_rule = rules[chosen_rule_idx]
        except IndexError:
            if raise_errors:
                _exceptions.raise_dsge_mapping_error3(
                    gi, chosen_nt, rules, chosen_rule_idx
                )
            break

        # 3) Expand the chosen nonterminal (1) with the rhs of the chosen rule (2)
        if verbose:
            print("- Application of rule to nonterminal")
        new_nodes = derivation_tree._expand(chosen_nt_node, chosen_rule)
        expansion_counter += 1
        if verbose:
            rhs = "".join(
                sym.text if isinstance(sym, _T) else "<{}>".format(sym.text)
                for sym in chosen_rule
            )
            print("    Substitution: <{}> -> {}".format(chosen_nt.text, rhs))

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
    """Map a string phenotype (or derivation tree) to a DSGE genotype."""
    # Cache look-up
    _, nt_to_gene, _, _, _ = grammar._lookup_or_calc(
        "dsge", "maps", _cached_calculations.maps, grammar
    )

    # Preparation of data structures
    dt = _shared.mapping.get_derivation_tree(grammar, phenotype_or_derivation_tree)
    gt = [[] for _ in range(len(grammar.nonterminal_symbols))]

    # Trace all decisions contained in the given derivation tree
    stack = [dt.root_node]
    while stack:
        # 1) Choose nonterminal: DSGE uses leftmost -> Choose it accordingly
        chosen_nt_idx = 0
        chosen_nt_node = stack.pop(chosen_nt_idx)

        # 2) Choose rule: DSGE decides via next integer in NT's gene -> Deduce it from tree
        try:
            rules = grammar.production_rules[chosen_nt_node.symbol]
        except Exception:
            _exceptions.raise_missing_nt_error(chosen_nt_node)
        try:
            chosen_rule = [node.symbol for node in chosen_nt_node.children]
            chosen_rule_idx = rules.index(chosen_rule)
        except ValueError:
            _exceptions.raise_missing_rhs_error(chosen_nt_node, chosen_rule)

        # 3) Expand the chosen nonterminal with the rhs of the chosen rule -> Follow the expansion
        new_nt_nodes = [
            node for node in chosen_nt_node.children if isinstance(node.symbol, _NT)
        ]
        stack = new_nt_nodes + stack

        # Store the observed decision, even if there was only one option
        gene_idx = nt_to_gene[chosen_nt_node.symbol]
        gt[gene_idx].append(chosen_rule_idx)

    # Finalization of data structures
    gt = _GT(gt)

    # Conditional return
    if return_derivation_tree:
        return gt, dt
    return gt
