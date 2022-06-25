from .. import _shared
from ... import exceptions as _exceptions


_LABEL = 'GE'


class Genotype(_shared.representation.BaseGenotype):
    """Genotype for Grammatical Evolution (GE)."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Initialize an unmutable genotype with data that defines its identity permanently.

        Parameters
        ----------
        data : tuple of int, list of int, string representation of the former
            In any case, the provided data is converted to a tuple of int.

            Examples:

            - ``(177, 29, 113, 4, 55, 13, 220)``
            - ``[177, 29, 113, 4, 55, 13, 220]``
            - ``"(177, 29, 113, 4, 55, 13, 220)"``
            - ``"[177, 29, 113, 4, 55, 13, 220]"``

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
                    data = tuple(eval(data))
                elif isinstance(data, list):
                    # List to tuple
                    data = tuple(data)
                else:
                    raise TypeError
                # Quick check
                assert isinstance(data[0], int)
            except Exception:
                _exceptions.raise_ge_genotype_error(data)
        return data

    # Representation
    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')


class Individual(_shared.representation.BaseIndividual):
    """Individual for Grammatical Evolution (GE)."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')


class Population(_shared.representation.BasePopulation):
    """Population for Grammatical Evolution (GE)."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')
