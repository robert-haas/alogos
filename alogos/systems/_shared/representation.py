"""Shared representations serving as base classes for all systems."""

from copy import deepcopy as _dc

from ... import exceptions as _exceptions
from ..._utilities import argument_processing as _ap
from ..._utilities.operating_system import NEWLINE as _NEWLINE


class BaseGenotype:
    """Base genotype for all systems to define a shared structure."""

    __slots__ = ("data", "_hash")

    # Immutable data attribute: provided once at object creation and converted system-dependently
    def __setattr__(self, key, val):
        """Implement attribute assignment to ensure data is immutable.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__setattr__

        """
        if key == "data":
            _exceptions.raise_data_write_error()
        object.__setattr__(self, key, val)

    # Copying
    def copy(self):
        """Create a deep copy of the genotype."""
        return self.__class__(self.data)

    def __copy__(self):
        """Create a deep copy of the genotype."""
        return self.__class__(self.data)

    def __deepcopy__(self, memo):
        """Create a deep copy of the genotype."""
        return self.__class__(self.data)

    # Representation
    def __repr__(self):
        """Compute the "official" string representation of the genotype."""
        return "<{} genotype at {}>".format(self._label, hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the genotype."""
        return str(self.data).replace(" ", "")

    # Length
    def __len__(self):
        """Calculate the length of the genotype."""
        return len(self.data)

    # Equality
    def __eq__(self, other):
        """Compute whether two genotypes are equal."""
        if isinstance(other, self.__class__):
            return self.data == other.data
        return NotImplemented

    def __ne__(self, other):
        """Compute whether two genotypes are not equal."""
        if isinstance(other, self.__class__):
            return self.data != other.data
        return NotImplemented

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
            self._hash = hash(self.data)
            return self._hash


class BaseIndividual:
    """Base individual for all systems to define a shared structure."""

    __slots__ = ("genotype", "phenotype", "fitness", "details")

    def __init__(
        self, genotype=None, phenotype=None, fitness=float("nan"), details=None
    ):
        """Create an individual as simple container for genotype, phenotype and fitness."""
        # Argument processing
        if details is None:
            details = {}

        # Assignments
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.details = details

    # Copying: Phenotype & fitness are immutable objects (None, str, int, float) requiring no copy
    def copy(self):
        """Create a deep copy of the individual."""
        return self.__class__(
            _dc(self.genotype), self.phenotype, self.fitness, _dc(self.details)
        )

    def __copy__(self):
        """Create a deep copy of the individual."""
        return self.__class__(
            _dc(self.genotype), self.phenotype, self.fitness, _dc(self.details)
        )

    def __deepcopy__(self, memo):
        """Create a deep copy of the individual."""
        return self.__class__(
            _dc(self.genotype), self.phenotype, self.fitness, _dc(self.details)
        )

    # Representation
    def __repr__(self):
        """Compute the "official" string representation of the individual."""
        return "<{} individual object at {}>".format(self._label, hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the individual."""
        text = (
            "{lab} individual:{nl}"
            "- Genotype: {gt}{nl}"
            "- Phenotype: {phe}{nl}"
            "- Fitness: {fit}".format(
                lab=self._label,
                nl=_NEWLINE,
                gt=self.genotype,
                phe=self.phenotype,
                fit=self.fitness,
            )
        )
        return text

    # Fitness comparison: objective-dependent NaN treatment, therefore not using __lt__ and __gt__
    def less_than(self, other, objective):
        """Determine if the fitness of this individual is less than that of another.

        Parameters
        ----------
        other : `Individual`
        objective : `str`
            Possible values:

            - ``"min"`` for a minimization problem
            - ``"max"`` for a maximization problem

            It determines how ``NaN`` values are treated in the
            comparsion.

        Notes
        -----
        There is a conceptual problem with ``NaN`` values, making the
        comparison depending on the type of optimization problem being
        tackled. In case of a minimization problem, any valid `float`
        number should be considered to be smaller than ``NaN``, so that
        the individuals with ``NaN`` fitnesses loose in comparisons.
        In case of a maximization problem, it is the other way around.
        Therefore this explicit method with the argument ``objective``
        is provided instead of the special method ``__lt__`` that would
        allow individuals to be compared with the ``<`` operator but
        without any arguments.

        References
        ----------
        - https://docs.python.org/3/library/operator.html

        """
        # Argument processing: no checks for objective in ('min', 'max') to improve speed
        f1 = self.fitness
        f2 = other.fitness

        # Regular case: 0 NaN values
        if f1 == f1 and f2 == f2:
            return f1 < f2

        # Special cases: 1 or 2 NaN values
        if f1 != f1:
            if f2 != f2:
                # NaN < NaN: False
                return False
            # NaN < number: True if maximization, False if minimization
            return objective == "max"
        # number < NaN: True if minimization, False if maximization
        return objective == "min"

    def greater_than(self, other, objective):
        """Determine if the fitness of this individual is greater than that of another.

        Parameters
        ----------
        other : `Individual`
        objective : `str`
            Possible values:

            - ``"min"`` for a minimization problem
            - ``"max"`` for a maximization problem

            It determines how ``NaN`` values are treated in the
            comparsion.

        Notes
        -----
        There is a conceptual problem with ``NaN`` values, making the
        comparison depending on the type of optimization problem being
        tackled. In case of a minimization problem, any valid `float`
        number should be considered to be smaller than ``NaN``, so that
        the individuals with ``NaN`` fitnesses loose in comparisons.
        In case of a maximization problem, it is the other way around.
        Therefore this explicit method with the argument ``objective``
        is provided instead of the special method ``__gt__`` that would
        allow individuals to be compared with the ``>`` operator but
        without any arguments.

        References
        ----------
        - https://docs.python.org/3/library/operator.html

        """
        # Argument processing: no checks for objective in ('min', 'max') to improve speed
        f1 = self.fitness
        f2 = other.fitness

        # Regular case: 0 NaN values
        if f1 == f1 and f2 == f2:
            return f1 > f2

        # Special cases: 1 or 2 NaN values
        if f1 != f1:
            if f2 != f2:
                # NaN > NaN: False
                return False
            # NaN > number: True if minimization, False if maximization
            return objective == "min"
        # number > NaN: True if maximization, False if minimization
        return objective == "max"


class BasePopulation:
    """Base population for all systems to define a shared structure."""

    __slots__ = "individuals"

    def __init__(self, individuals):
        """Create a population as container for multiple individuals."""
        # Argument processing
        _ap.check_arg("individuals", individuals, types=(list, tuple))

        # Assignments
        self.individuals = individuals

    # Copying
    def copy(self):
        """Create a deep copy of the population."""
        return self.__class__([ind.copy() for ind in self.individuals])

    def __copy__(self):
        """Create a deep copy of the population."""
        return self.__class__([ind.copy() for ind in self.individuals])

    def __deepcopy__(self, memo):
        """Create a deep copy of the population."""
        return self.__class__([ind.copy() for ind in self.individuals])

    # Representation
    def __repr__(self):
        """Compute the "official" string representation of the population."""
        return "<{} population at {}>".format(self._label, hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the population."""
        text = (
            "{lab} population:{nl}"
            "- Individuals: {ind}{nl}"
            "- Unique genotypes: {gt}{nl}"
            "- Unique phenotypes: {phe}{nl}"
            "- Unique fitnesses: {fit}".format(
                lab=self._label,
                nl=_NEWLINE,
                ind=len(self),
                gt=self.num_unique_genotypes,
                phe=self.num_unique_phenotypes,
                fit=self.num_unique_fitnesses,
            )
        )
        return text

    # Access
    def __getitem__(self, key):
        """Implement index-based access for the population's individuals.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__getitem__

        """
        return self.individuals[key]

    def __setitem__(self, key, value):
        """Implement index-based modification for the population's individuals.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__setitem__

        """
        if not isinstance(value, BaseIndividual):
            _exceptions.raise_pop_assignment_error(value)
        self.individuals[key] = value

    def __delitem__(self, key):
        """Implement index-based deletion for the population's individuals.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__delitem__

        """
        del self.individuals[key]

    # Length
    def __len__(self):
        """Calculate the number of individuals in a population."""
        return len(self.individuals)

    # Iteration
    def __iter__(self):
        """Implement the return of an iterator object.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__iter__
        - https://docs.python.org/3/library/stdtypes.html#container.__iter__

        """
        return iter(self.individuals)

    # Concatenation
    def __add__(self, other):
        """Implement concatenation of populations by the plus symbol."""
        return self.__class__(self.individuals + other.individuals)

    @property
    def num_unique_genotypes(self):
        """Get the number of unique genotypes in this population."""
        return len({str(ind.genotype) for ind in self.individuals})

    @property
    def num_unique_phenotypes(self):
        """Get the number of unique phenotypes in this population."""
        return len({str(ind.phenotype) for ind in self.individuals})

    @property
    def num_unique_fitnesses(self):
        """Get the number of unique fitness values in this population."""
        return len({str(ind.fitness) for ind in self.individuals})
