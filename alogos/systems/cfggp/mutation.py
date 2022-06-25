import random as _random

from . import default_parameters as _dp
from . import representation as _representation
from .._shared.initialization.individual import _grow_tree_below_max_depth
from ... import _grammar
from ..._grammar import data_structures as _data_structures
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default


def subtree_replacement(grammar, genotype, parameters=None):
    """Change a randomly chosen node in the tree by attaching a randomly generated subtree.

    References
    ----------
    - `Grammatically-based Genetic Programming (1995)
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.2091>`__

        - "Mutation applies to a single program. A program is selected for mutation,
          and one non-terminal is randomly selected as the site for mutation.
          The tree below this non-terminal is deleted, and a new tree randomly
          generated from the grammar using this non-terminal as a starting point.
          The tree is limited in total depth by the current maximum allowable
          program depth (MAX-TREE-DEPTH), in an operation similar to creating
          the initial population."

    """
    # Parameter extraction
    count = _get_given_or_default('mutation_subtree_replacement_count', parameters, _dp)
    max_depth = _get_given_or_default('max_depth', parameters, _dp)

    # Argument processing
    if not isinstance(genotype, _representation.Genotype):
        genotype = _representation.Genotype(genotype)

    # Mutation
    # - Get all internal nodes and their depths
    dt = genotype.data
    nodes_and_depths = []
    stack = [(dt.root_node, 0)]
    while stack:
        node, depth = stack.pop()
        if node.children:
            nodes_and_depths.append((node, depth))
            stack = stack + [(node, depth + 1) for node in node.children]
    # - Randomly select a node for mutation
    node, depth = _random.choice(nodes_and_depths)
    # - Replace the node's subtree with a randomly generated new one
    node.children = []
    _grow_tree_below_max_depth(grammar, max_depth, start_depth=depth, root_node=node)
    return genotype
