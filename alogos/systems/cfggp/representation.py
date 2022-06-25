from .. import _shared
from ... import _grammar
from ... import exceptions as _exceptions
from ..._utilities import argument_processing as _ap


_DT = _grammar.data_structures.DerivationTree
_LABEL = 'CFG-GP'


class Genotype(_shared.representation.BaseGenotype):
    """Genotype for context-free grammar genetic programming (CFG-GP)."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Initialize an unmutable genotype with data that defines its identity permanently.

        Parameters
        ----------
        data : :ref:`DerivationTree`, or string representation of it
            In any case, the provided data is converted to a :ref:`DerivationTree` object.

        Raises
        ------
        :py:class:`~alogos.exceptions.GenotypeError`
            Raised when the provided data can not be converted to the desired form.

        """
        object.__setattr__(self, 'data', self._convert_input(data))

    def _convert_input(self, data):
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
        return self.__class__(self.data.copy())

    def __copy__(self):
        return self.__class__(self.data.copy())

    def __deepcopy__(self, memo):
        return self.__class__(self.data.copy())

    # Representation
    def __str__(self):
        return str(self.data.to_json())

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')

    # Length
    def __len__(self):
        return self.data.num_nodes()


class Individual(_shared.representation.BaseIndividual):
    """Individual for context-free grammar genetic programming (CFG-GP)."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')


class Population(_shared.representation.BasePopulation):
    """Population for context-free grammar genetic programming (CFG-GP)."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')
