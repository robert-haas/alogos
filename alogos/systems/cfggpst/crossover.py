"""Crossover functions for CFG-GP-ST."""

import random as _random

from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from . import default_parameters as _default_parameters
from . import representation as _representation


# Shortcuts for minor speedup
_GT = _representation.Genotype
_fe = _representation._find_subtree_end
_rc = _random.choice


def subtree_exchange(grammar, genotype1, genotype2, parameters=None):
    """Generate new CFG-GP-ST genotypes by exchanging suitable subtrees.

    Randomly select nodes containing the same nonterminal in two trees
    and swap their subtrees. Note that CFG-GP-ST uses serialized trees.
    Crossover can be done directly in the serialized representation and
    does not require a preliminary reconstructing of trees. This inplace
    modification leads to a considerable speedup.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    genotype1 : `~.representation.Genotype`
        Genotype of the first parent.
    genotype2 : `~.representation.Genotype`
        Genotype of the second parent.
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``max_nodes`` (`int`) : Represents the maximum number of nodes
          allowed inside newly generated trees

    Returns
    -------
    genotype1 : `~.representation.Genotype`
        Genotype of the first child.
    genotype2 : `~.representation.Genotype`
        Genotype of the second child.

    Notes
    -----
    The randomly selected nodes may be the root nodes of both trees,
    which means that the crossover will produce genotypes that are
    identical to the given genotypes. It would be possible to exclude
    the root nodes from the random node selection, but in case of
    recursive grammars the start symbol may appear in lower parts of
    the tree and then a swap between root node and lower node produces
    novel genotypes.

    References
    ----------
    - 1995, Whigham: `Grammatically-based Genetic Programming
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.2091>`__

        - "All terminals have at least one nonterminal above them in
          the program tree (at the very least S), so without loss of
          generality we may constrain crossover points to be located
          only on nonterminals. The crossover operation maintains legal
          programs of the language (as defined by the grammar) by
          ensuring that the same non-terminals are selected at each
          crossover site. The parameter MAX-TREE-DEPTH is used to
          indicate the deepest parse tree that may exist in the
          population. The crossover algorithm (see figure 3) is: [...]"

        - "We note that the parameter MAX-TREE-DEPTH may exclude some
          crossover operations from being performed. In the current
          system, if following crossover either new program exceeds
          MAX-TREE-DEPTH the entire operation is aborted, and the
          crossover procedure recommenced from step 1."

    - 2003, Poli, McPhee:
      `General Schema Theory for Genetic Programming with Subtree-Swapping Crossover: Part II
      <https://doi.org/10.1162/106365603766646825>`__

        - An example where the name "subtree-swapping" is used for the
          same crossover operation.

    """
    # Argument processing
    if not isinstance(genotype1, _GT):
        genotype1 = _GT(genotype1)
    if not isinstance(genotype2, _GT):
        genotype2 = _GT(genotype2)

    # Parameter extraction
    mn = _get_given_or_default("max_nodes", parameters, _default_parameters)

    # Crossover
    s1, c1 = genotype1.data
    s2, c2 = genotype2.data
    n1 = {s for s, c in zip(s1, c1) if c != 0}  # set of nonterminals in tree 1
    nb = {s for s in s2 if s in n1}  # set of nonterminals in both trees
    if nb:
        # - Randomly select a nonterminal in tree 1, which is also present in tree 2
        a1 = _rc([i for i, s in enumerate(s1) if s in nb])
        b1 = _fe(a1, c1) + 1
        # - Randomly select the same nonterminal at some position in tree 2
        sym = s1[a1]
        a2 = _rc([i for i, s in enumerate(s2) if s == sym])
        b2 = _fe(a2, c2) + 1
        # - Swap: only if max_nodes condition is not violated afterwards for both trees
        l1 = b1 - a1
        l2 = b2 - a2
        ld = l2 - l1
        if len(s1) + ld <= mn and len(s2) - ld <= mn:
            n1 = (s1[:a1] + s2[a2:b2] + s1[b1:], c1[:a1] + c2[a2:b2] + c1[b1:])
            n2 = (s2[:a2] + s1[a1:b1] + s2[b2:], c2[:a2] + c1[a1:b1] + c2[b2:])
        else:
            n1 = genotype1.data
            n2 = genotype2.data
    return _GT(n1), _GT(n2)
