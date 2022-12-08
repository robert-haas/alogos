"""Forward and reverse mapping functions for CFG-GP-ST."""

from ... import _grammar
from ... import exceptions as _exceptions
from .. import _shared
from . import representation as _representation


# Shortcuts for brevity and minor speedup
_GT = _representation.Genotype
_DT = _grammar.data_structures.DerivationTree


# Forward mapping
def forward(
    grammar,
    genotype,
    parameters=None,
    raise_errors=True,
    return_derivation_tree=False,
    verbose=False,
):
    """Map a CFG-GP-ST genotype to a string phenotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype : `~.representation.Genotype` or data that can be converted to it
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        No keyword-value pairs are considered by this function.
        This argument is only available to have a consistent interface.
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
    # Argument processing
    if not isinstance(genotype, _GT):
        genotype = _GT(genotype)

    # Cache look-up
    ism = grammar._lookup_or_calc(
        "serialization", "idx_sym_map", grammar._calc_idx_sym_map
    )

    # Mapping
    if verbose:
        phe = _forward_slow(grammar, genotype, ism, verbose)
    else:
        phe = _forward_fast(grammar, genotype, ism)

    # Conditional return
    if return_derivation_tree:
        dt = _DT(grammar)
        dt.from_tuple(genotype.data)
        return phe, dt
    return phe


def _forward_fast(grammar, gt, ism):
    """Calculate the genotype-to-phenotype map of CFG-GP-ST in a fast way."""
    return "".join(ism[s].text for s, n in zip(*gt.data) if n == 0)


def _forward_slow(grammar, genotype, ism, verbose):
    """Calculate the genotype-to-phenotype map of CFG-GP-ST in a slow way.

    This is a readable implementation of the mapping process, which
    also allows to print output about the steps it involves.
    It served as basis for the faster, minified implementation in this
    module and may be helpful in understanding, replicating or modifying
    the algorithm.

    """
    if verbose:
        header = "Start reading the phenotype directly from the genotype"
        print(header)
        print("=" * len(header))
    terminals = []
    for i, (sym_idx, num_children) in enumerate(zip(*genotype.data)):
        sym = ism[sym_idx]
        if num_children == 0:
            sym_type = "terminal"
            terminals.append(sym.text)
        else:
            sym_type = "nonterminal"
        if verbose:
            text = (
                '- Entry {}: {} means symbol "{}", {} means {} with {} children'.format(
                    i, sym_idx, sym, num_children, sym_type, num_children
                )
            )
            print(text)
    phe = "".join(terminals)
    if verbose:
        print()
        header = "End of reading"
        print(header)
        print("=" * len(header))
        print("- Collected terminals in order of discovery: {}".format(terminals))
        print()
        print("String: {}".format(phe))
    return phe


# Reverse mapping
def reverse(
    grammar, phenotype_or_derivation_tree, parameters=None, return_derivation_tree=False
):
    """Map a string phenotype (or derivation tree) to a CFG-GP-ST genotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    phenotype_or_derivation_tree : `str` or `~alogos._grammar.data_structures.DerivationTree`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        No keyword-value pairs are considered by this function.
        This argument is only available to have a consistent interface.
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
    # Transformation
    try:
        dt = _shared.mapping.get_derivation_tree(grammar, phenotype_or_derivation_tree)
        gt = _representation.Genotype(dt.to_tuple())
    except Exception:
        _exceptions.raise_invalid_mapping_data2(phenotype_or_derivation_tree)

    # Conditional return
    if return_derivation_tree:
        return gt, dt
    return gt
