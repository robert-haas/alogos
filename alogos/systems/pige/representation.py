"""Representations for piGE."""

from ... import exceptions as _exceptions
from .. import _shared


_LABEL = "piGE"


class Genotype(_shared.representation.BaseGenotype):
    """PiGE genotype."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Create a piGE genotype with immutable data.

        Parameters
        ----------
        data : `tuple` of `int`, or `list` of `int`, or `str` representation of one of the former options
            The provided data is converted to the first form.

        Examples
        --------
        The argument `data` could get following values of different
        types that all lead to the same result:

        - ``(177, 29, 113, 4, 55, 13, 220)``
        - ``[177, 29, 113, 4, 55, 13, 220]``
        - ``"(177, 29, 113, 4, 55, 13, 220)"``
        - ``"[177, 29, 113, 4, 55, 13, 220]"``

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
                    data = tuple(eval(data))
                elif isinstance(data, list):
                    data = tuple(data)
                else:
                    raise TypeError
                assert isinstance(data[0], int)
            except Exception:
                _exceptions.raise_pige_genotype_error(data)
        return data

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


class Individual(_shared.representation.BaseIndividual):
    """PiGE individual having a piGE genotype."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


class Population(_shared.representation.BasePopulation):
    """PiGE population consisting of piGE individuals."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")
