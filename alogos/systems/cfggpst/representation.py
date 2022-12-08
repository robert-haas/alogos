"""Representations for CFG-GP-ST."""

from ... import _grammar
from ... import exceptions as _exceptions
from .. import _shared


_LABEL = "CFG-GP-ST"


class Genotype(_shared.representation.BaseGenotype):
    """CFG-GP-ST genotype."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Create a CFG-GP-ST genotype with immutable data.

        Parameters
        ----------
        data : `~alogos._grammar.data_structures.DerivationTree`, or `tuple` representation of it, or `str` representation of it
            If the provided data is of type `tuple` or `str` it is
            automatically converted to a tree.

        Raises
        ------
        GenotypeError
            If the provided data can not be converted to the desired
            form.

        """
        # Immutable data attribute
        object.__setattr__(self, "data", self._convert_input(data))

    def _convert_input(self, data):
        """Convert different genotype formats to a single one."""
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
                    from . import mapping as _mapping

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
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")

    # Length
    def __len__(self):
        """Calculate the length of the genotype."""
        return len(self.data[0])


class Individual(_shared.representation.BaseIndividual):
    """CFG-GP-ST individual having a CFG-GP-ST genotype."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


class Population(_shared.representation.BasePopulation):
    """CFG-GP-ST population consisting of CFG-GP-ST individuals."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


def _find_subtree_end(s, cnt):
    """Find index of the last symbol in a subtree.

    This method is used by crossover and mutation operators.

    """
    c = cnt[s]
    while c:
        s += 1
        n = cnt[s]
        if n == 0:
            c -= 1
        elif n > 1:
            c += n - 1
    return s
