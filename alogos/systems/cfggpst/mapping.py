from . import default_parameters as _dp
from . import representation as _representation
from .. import _shared
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from ... import _grammar
from ... import exceptions as _exceptions


# Shortcuts for brevity and minor speedup
_DT = _grammar.data_structures.DerivationTree


# Forward mapping
def forward(grammar, genotype, parameters=None,
            raise_errors=True, return_derivation_tree=False, verbose=False):
    """Map a CFG-GP-ST genotype to a string phenotype."""
    # Cache look-up
    ism = grammar._lookup_or_calc('serialization', 'idx_sym_map', grammar._calc_idx_sym_map)

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Mapping
    if verbose:
        # slow
        header = 'Start reading the phenotype directly from the genotype'
        print(header)
        print('=' * len(header))
        terminals = []
        for i, (sym_idx, num_children) in enumerate(zip(*genotype.data)):
            sym = ism[sym_idx]
            if num_children == 0:
                sym_type = 'terminal'
                terminals.append(sym.text)
            else:
                sym_type = 'nonterminal'
            text = '- Entry {}: {} means symbol "{}", {} means {} with {} children'.format(
                i, sym_idx, sym, num_children, sym_type, num_children)
            print(text)
        phe = ''.join(terminals)
        print()
        header = 'End of reading'
        print(header)
        print('=' * len(header))
        print('- Collected terminals in order of discovery: {}'.format(
            terminals))
        print()
        print('String: {}'.format(phe))
    else:
        # fast
        phe = ''.join(ism[s].text for s, n in zip(*genotype.data) if n == 0)

    # Conditional return
    if return_derivation_tree:
        dt = _DT(grammar)
        dt.from_tuple(genotype.data)
        return phe, dt
    return phe


# Reverse mapping
def reverse(grammar, phenotype_or_derivation_tree, parameters=None,
            return_derivation_tree=False):
    """Map a string phenotype (or derivation tree) to a CFG-GP-ST genotype."""
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
