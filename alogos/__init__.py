"""A library of grammatical optimization systems.

This package implements several methods from the field
of Grammar-Guided Genetic Programming (G3P).

"""

__all__ = [
    "Grammar",
    "EvolutionaryAlgorithm",
    "systems",
    "exceptions",
    "warnings",
]

__version__ = "0.2.0"

from . import exceptions, systems, warnings
from ._grammar import Grammar
from ._optimization import EvolutionaryAlgorithm


warnings.turn_on()
