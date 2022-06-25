from copy import deepcopy as _dc

from ... import exceptions as _exceptions
from ..._utilities import argument_processing as _ap
from ..._utilities.operating_system import NEWLINE as _NEWLINE


class BaseGenotype:
    """Base genotype that defines shared structure and behavior of different systems."""

    __slots__ = ('data', '_hash')

    # Immutable data attribute: provided once at object creation and converted system-dependently
    def __setattr__(self, key, val):
        if key == 'data':
            _exceptions.raise_data_write_error()
        object.__setattr__(self, key, val)

    # Copying
    def copy(self):
        return self.__class__(self.data)

    def __copy__(self):
        return self.__class__(self.data)

    def __deepcopy__(self, memo):
        return self.__class__(self.data)

    # Representation
    def __str__(self):
        return str(self.data).replace(' ', '')

    def __repr__(self):
        return '<{} genotype at {}>'.format(self._label, hex(id(self)))

    # Length
    def __len__(self):
        return len(self.data)

    # Equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.data == other.data
        return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self.data != other.data
        return True

    # Hashing
    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self.data)
            return self._hash


class BaseIndividual:
    """Base individual that defines shared structure and behavior of different systems."""

    __slots__ = ('genotype', 'phenotype', 'fitness', 'details')

    def __init__(self, genotype=None, phenotype=None, fitness=float('nan'), details=None):
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
        return self.__class__(
            _dc(self.genotype), self.phenotype, self.fitness, _dc(self.details))

    def __copy__(self):
        return self.__class__(
            _dc(self.genotype), self.phenotype, self.fitness, _dc(self.details))

    def __deepcopy__(self, memo):
        return self.__class__(
            _dc(self.genotype), self.phenotype, self.fitness, _dc(self.details))

    # Representation
    def __str__(self):
        text = (
            '{lab} individual:{nl}'
            '- Genotype: {gt}{nl}'
            '- Phenotype: {phe}{nl}'
            '- Fitness: {fit}'.format(
                lab=self._label,
                nl=_NEWLINE,
                gt=self.genotype,
                phe=self.phenotype,
                fit=self.fitness))
        return text

    def __repr__(self):
        return '<{} individual object at {}>'.format(self._label, hex(id(self)))

    # Fitness comparison: objective-dependent NaN treatment, therefore not using __lt__ and __gt__
    def less_than(self, other, objective):
        """Determine if the fitness of this individual is less than the fitness of another.

        Parameters
        ----------
        other : Individual
        objective : str
            "min" for minimization problem, "max" for maximization problem.
            It determines how ``NaN`` values are treated in the comparsion.

        Notes
        -----
        There is a conceptual problem with ``NaN`` values, making the comparison
        depending on the type of optimization problem being tackled.
        In case of a minimization problem, any valid float number should be considered
        to be smaller than NaN, so that the individuals with NaN fitnesses loose in
        comparisons. In case of a maximization problem, it is the other way around.
        Therefore this explicit method with the argument ``objective``is provided
        instead of the special method ``__lt__`` that would allow individuals to
        be compared with the ``<`` operator but without any arguments.

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
            return objective == 'max'
        # number < NaN: True if minimization, False if maximization
        return objective == 'min'

    def greater_than(self, other, objective):
        """Determine if the fitness of this individual is greater than the fitness of another.

        Parameters
        ----------
        other : Individual
        objective : str
            "min" for minimization problem, "max" for maximization problem.
            It determines how ``NaN`` values are treated in the comparsion.

        Notes
        -----
        There is a conceptual problem with ``NaN`` values, making the comparison
        depending on the type of optimization problem being tackled.
        In case of a minimization problem, any valid float number should be considered
        to be smaller than NaN, so that the individuals with NaN fitnesses loose in
        comparisons. In case of a maximization problem, it is the other way around.
        Therefore this explicit method with the argument ``objective``is provided
        instead of the special method ``__gt__`` that would allow individuals to
        be compared with the ``>`` operator but without any arguments.

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
            return objective == 'min'
        # number > NaN: True if maximization, False if minimization
        return objective == 'max'


class BasePopulation:
    """Base population that defines shared structure and behavior of different systems."""

    __slots__ = 'individuals'

    def __init__(self, individuals):
        """Create a population as simple container for multiple individuals."""
        # Argument processing
        _ap.check_arg('individuals', individuals, types=(list, tuple))

        # Assignments
        self.individuals = individuals

    # Copying
    def copy(self):
        return self.__class__([ind.copy() for ind in self.individuals])

    def __copy__(self):
        return self.__class__([ind.copy() for ind in self.individuals])

    def __deepcopy__(self, memo):
        return self.__class__([ind.copy() for ind in self.individuals])

    # Representation
    def __str__(self):
        text = (
            '{lab} population:{nl}'
            '- Individuals: {ind}{nl}'
            '- Unique genotypes: {gt}{nl}'
            '- Unique phenotypes: {phe}{nl}'
            '- Unique fitnesses: {fit}'.format(
                lab=self._label,
                nl=_NEWLINE,
                ind=len(self),
                gt=self.num_unique_genotypes,
                phe=self.num_unique_phenotypes,
                fit=self.num_unique_fitnesses))
        return text

    def __repr__(self):
        return '<{} population at {}>'.format(self._label, hex(id(self)))

    # Access
    def __getitem__(self, key):
        return self.individuals[key]

    def __setitem__(self, key, value):
        if not isinstance(value, BaseIndividual):
            _exceptions.raise_pop_assignment_error(value)
        self.individuals[key] = value

    def __delitem__(self, key):
        del self.individuals[key]

    # Length
    def __len__(self):
        return len(self.individuals)

    # Iteration
    def __iter__(self):
        return iter(self.individuals)

    # Concatenation
    def __add__(self, other):
        return self.__class__(self.individuals + other.individuals)

    @property
    def num_unique_genotypes(self):
        return len({str(ind.genotype) for ind in self.individuals})

    @property
    def num_unique_phenotypes(self):
        return len({str(ind.phenotype) for ind in self.individuals})

    @property
    def num_unique_fitnesses(self):
        return len({str(ind.fitness) for ind in self.individuals})
