"""Representations for WHGE."""

from bitarray import bitarray as _bitarray

from ... import exceptions as _exceptions
from .. import _shared


_LABEL = "WHGE"


class Genotype(_shared.representation.BaseGenotype):
    """WHGE genotype."""

    __slots__ = ()
    _label = _LABEL

    def __init__(self, data):
        """Create a WHGE genotype with immutable data.

        Parameters
        ----------
        data : `bitarray`, or `str` of ``"0"`` and ``"1"`` characters
            The provided data is converted to the first form.

        Examples
        --------
        The argument `data` could get following values of different
        types that all lead to the same result:

        - ``bitarray.bitarray("11110000")``

          The package containing this data structure is available on
          PyPI under the name
          `bitarray <https://pypi.org/project/bitarray/>`__

        - ``"11110000"``

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
        if not isinstance(data, _bitarray) or not data:
            try:
                if isinstance(data, str):
                    data = _bitarray(data)
                else:
                    raise TypeError
                assert data
            except Exception:
                _exceptions.raise_whge_genotype_error(data)
        return data

    # Copying
    def copy(self):
        """Create a deep copy of the genotype."""
        return Genotype(self.data.copy())

    def __copy__(self):
        """Create a deep copy of the genotype."""
        return Genotype(self.data.copy())

    def __deepcopy__(self, memo):
        """Create a deep copy of the genotype."""
        return Genotype(self.data.copy())

    # Representation
    def __str__(self):
        """Compute the "informal" string representation of the genotype."""
        return self.data.to01()

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")

    # Hashing
    def __hash__(self):
        """Calculate a hash value for this object.

        It is used for operations on hashed collections such as `set`
        and `dict`.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__hash__

        """
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self.data.tobytes())
            return self._hash


class Individual(_shared.representation.BaseIndividual):
    """WHGE individual having a WHGE genotype."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")


class Population(_shared.representation.BasePopulation):
    """WHGE population consisting of WHGE individuals."""

    __slots__ = ()
    _label = _LABEL

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        p.text(str(self) if not cycle else "...")
