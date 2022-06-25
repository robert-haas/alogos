import random as _random

from . import default_parameters as _default_parameters
from . import representation as _representation
from ... import _grammar
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default


def subtree_exchange(grammar, genotype1, genotype2, parameters=None):
    """Randomly select nodes containing the same nonterminal in two trees and swap their subtrees.

    Notes
    -----
    The randomly selected nodes may be the root nodes of both trees, which means
    that the crossover will produce genotypes that are identical to the given genotypes.
    It would be possible to exclude the root nodes from the random node selection,
    but in case of recursive grammars the start symbol may appear in lower parts
    of the tree and then a swap between root node and lower node produces novel genotypes.

    References
    ----------
    - 1995, Whigham: `Grammatically-based Genetic Programming
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.2091>`__

        - "All terminals have at least one nonterminal above them in the
          program tree (at the very least S), so without loss of generality
          we may constrain crossover points to be located only on nonterminals.
          The crossover operation maintains legal programs of the language (as
          defined by the grammar) by ensuring that the same non-terminals are
          selected at each crossover site. The parameter MAX-TREE-DEPTH is
          used to indicate the deepest parse tree that may exist in the population.
          The crossover algorithm (see figure 3) is: [...]"
        
        - "We note that the parameter MAX-TREE-DEPTH may exclude somecrossover
          operations from being performed. In the current system, if following
          crossover either new program exceeds MAX-TREE-DEPTH the entire operation
          is aborted, and the crossover procedure recommenced from step 1."

    - 2003, Poli, McPhee:
      `General Schema Theory for Genetic Programming with Subtree-Swapping Crossover: Part II
      <https://doi.org/10.1162/106365603766646825>`__
      
        - An example where the same crossover operation is called "subtree-swapping"
          as support for the name chosen for this function.

    """
    # Parameter extraction
    max_depth = _get_given_or_default('max_depth', parameters, _default_parameters)

    # Argument processing
    if not isinstance(genotype1, _representation.Genotype):
        genotype1 = _representation.Genotype(genotype1)
    if not isinstance(genotype2, _representation.Genotype):
        genotype2 = _representation.Genotype(genotype2)

    # Crossover
    ns1 = genotype1.data.internal_nodes()
    ns2 = genotype2.data.internal_nodes()
    s1 = {n.symbol.text for n in ns1}
    s2 = {n.symbol.text for n in ns2}
    si = s1.intersection(s2)
    if si:
        # Randomly select a non-terminal in the first tree, which is also part of second tree
        n1 = _random.choice([n for n in ns1 if n.symbol.text in si])
        t1 = n1.symbol.text
        # Randomly select the same non-terminal in the second genotype
        n2 = _random.choice([n for n in ns2 if n.symbol.text == t1])
        # Swap subtrees by exchanging the list of child nodes
        n1.children, n2.children = n2.children, n1.children
        # Ensure max_depth constraint is not violated
        if genotype1.data._is_deeper_than(max_depth) or genotype2.data._is_deeper_than(max_depth):
            # If invalid, reverse the swap
            n1.children, n2.children = n2.children, n1.children
    return genotype1, genotype2
