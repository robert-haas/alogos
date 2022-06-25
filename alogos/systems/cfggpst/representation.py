from . import mapping as _mapping
from .. import _shared
from ... import _grammar
from ... import exceptions as _exceptions


_LABEL = 'CFG-GP-ST'


class Genotype(_shared.representation.BaseGenotype):
    """Genotype for context-free grammar genetic programming on serialized trees (CFG-GP-ST)."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Initialize an unmutable genotype with data that defines its identity permanently.

        Parameters
        ----------
        data : :ref:`DerivationTree`, tuple serialization of a derivation tree, or string thereof
            In any case, the provided data is converted to a tuple serialization of a
            :ref:`DerivationTree`.

        Raises
        ------
        :py:class:`~alogos.exceptions.GenotypeError`
            Raised when the provided data can not be converted to the desired form.

        """
        # Immutable data attribute
        object.__setattr__(self, 'data', self._convert_input(data))

    def _convert_input(self, data):
        if not isinstance(data, tuple) or not data:
            try:
                if isinstance(data, str):
                    # String to tuple
                    data = eval(data)
                    if not isinstance(data, tuple):
                        data = (tuple(data[0]), tuple(data[1]))
                elif isinstance(data, list):
                    # List to tuple
                    data = (tuple(data[0]), tuple(data[1]))
                elif isinstance(data, _grammar.data_structures.DerivationTree):
                    # DerivationTree to tuple
                    data = _mapping.reverse(data.grammar, data).data
                else:
                    raise TypeError
                # Quick check
                assert isinstance(data[1][0], int)
            except Exception:
                _exceptions.raise_cfggpst_genotype_error(data)
        return data

    # Representation
    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')

    # Length
    def __len__(self):
        return len(self.data[0])


class Individual(_shared.representation.BaseIndividual):
    """Individual for context-free grammar genetic programming on serialized trees (CFG-GP-ST)."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')


class Population(_shared.representation.BasePopulation):
    """Population for context-free grammar genetic programming on serialized trees (CFG-GP-ST)."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')


def _find_subtree_end(s, cnt):
    c = cnt[s]
    while c:
        s += 1;n = cnt[s]
        if n == 0: c -= 1
        elif n > 1: c += n - 1
    return s
