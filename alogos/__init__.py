"""alogos

This package aims to provide an implementation of following
grammar-guided genetic programming (G3P) systems:

- Context-Free Grammar Genetic Programming (CFG-GP)
- Grammatical Evolution (GE)
- Position-independent Grammatical Evolution (piGE)
- Weighted Hierarchical Grammatical Evolution (WHGE)
- Dynamic Structured Grammatical Evolution (DSGE)


The structure of this package is as follows:

- :ref:`Grammar`: This class allows to define a context-free grammar
  from a string in BNF or EBNF format. It can be used to test
  whether a string is part of the grammar's language, parse a string
  to find a parse tree, generate a random derivation and string,
  generate all strings of the grammar's language, and other tasks.

- :ref:`EvolutionaryAlgorithm`: This class allows to create an evolutionary search
  object, which can be used to search for an optimal string in the language of a
  given grammar with a chosen G3P system.

- :ref:`systems`: This sub-package contains all parts of each G3P system,
  except the evolutionary search engine. As a consequence, these building blocks
  can easily be used by other metaheuristic search methods, not only by
  evolutionary algorithms.

- :ref:`exceptions`: This module contains all custom exception classes
  used in this package.

"""

__all__ = [
    'Grammar',
    'EvolutionaryAlgorithm',
    'systems',
    'exceptions',
]

__version__ = '0.1.0'

from . import exceptions
from . import systems
from ._grammar import Grammar
from ._optimization import EvolutionaryAlgorithm
