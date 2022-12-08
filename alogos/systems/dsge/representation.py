"""Representations for DSGE."""

from ... import exceptions as _exceptions
from .. import _shared


_LABEL = "DSGE"


class Genotype(_shared.representation.BaseGenotype):
    """DSGE genotype."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Create a DSGE genotype with immutable data.

        Parameters
        ----------
        data : `tuple` of `tuple` objects of `int`, or `list` of `list` objects of `int`, or `str` representation of one of the former options
            The provided data is converted to the first form.

        Examples
        --------
        The argument `data` could get following values of different
        types that all lead to the same result:

        - ``((177, 29), (113,), (4, 55, 13, 220))``

          Note that a tuple with just one member also requires a comma
          inside the parentheses to be recognized as tuple.
        - ``[[177, 29], [113], [4, 55, 13, 220]]``
        - ``"((177, 29), (113,), (4, 55, 13, 220))"``
        - ``"[[177, 29], [113], [4, 55, 13, 220]]"``

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
                    data = tuple(tuple(gene) for gene in eval(data))
                elif isinstance(data, list):
                    data = tuple(tuple(gene) for gene in data)
                else:
                    raise TypeError
                assert data
            except Exception:
                _exceptions.raise_dsge_genotype_error(data)
        return data

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


class Individual(_shared.representation.BaseIndividual):
    """DSGE individual having a DSGE genotype."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


class Population(_shared.representation.BasePopulation):
    """DSGE population consisting of DSGE individuals."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")
