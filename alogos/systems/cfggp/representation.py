"""Representations for CFG-GP."""

from ... import _grammar
from ... import exceptions as _exceptions
from .. import _shared


_DT = _grammar.data_structures.DerivationTree
_LABEL = "CFG-GP"


class Genotype(_shared.representation.BaseGenotype):
    """CFG-GP genotype."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Create a CFG-GP genotype with immutable data.

        Parameters
        ----------
        data : `~alogos._grammar.data_structures.DerivationTree`, or `str` representation of it
            If the provided data is of type `str` it is automatically
            converted to a tree.

        Raises
        ------
        GenotypeError
            If the provided data can not be converted to the desired
            form.

        """
        object.__setattr__(self, "data", self._convert_input(data))

    def _convert_input(self, data):
        """Convert different genotype formats to a single one."""
        if not isinstance(data, _DT):
            try:
                if isinstance(data, str):
                    # JSON string to DerivationTree
                    dt = _DT(grammar=None)
                    dt.from_json(data)
                    data = dt
                else:
                    raise TypeError
                # Quick check
                assert data.root_node
            except Exception:
                _exceptions.raise_cfggp_genotype_error(data)
        return data

    # Copying
    def copy(self):
        """Create a deep copy of the genotype."""
        return self.__class__(self.data.copy())

    def __copy__(self):
        """Create a deep copy of the genotype."""
        return self.__class__(self.data.copy())

    def __deepcopy__(self, memo):
        """Create a deep copy of the genotype."""
        return self.__class__(self.data.copy())

    # Representation
    def __str__(self):
        """Compute the "informal" string representation of the grammar."""
        return str(self.data.to_json())

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")

    # Length
    def __len__(self):
        """Calculate the length of the genotype."""
        return self.data.num_nodes()


class Individual(_shared.representation.BaseIndividual):
    """CFG-GP individual having a CFG-GP genotype."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


class Population(_shared.representation.BasePopulation):
    """CFG-GP population consisting of CFG-GP individuals."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")
