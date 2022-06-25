from .. import data_structures as _data_structures
from . import _shared


def is_gnf(grammar):
    """Check if the grammar is in Greibach Normal Form (GNF).

    All rules need to be in the following form:
    - X → B a where B is a nonterminal and a is a terminal

    Tasks that are simplified by having a grammar in GNF:
    - Deciding if a given string is part of the grammar's language
    - Converting the grammar to a pushdown automaton (PDA) without ε-transitions,
      which is useful because it is guaranteed to halt

    References
    ----------
    - Rich - Automata, Computability and Complexity (2007): p. 169

    """
    raise NotImplementedError  # TODO
    return False


def to_gnf(grammar):
    """Convert the grammar to Greibach Normal Form (GNF).

    - Websites

        - https://www.tutorialspoint.com/automata_theory/greibach_normal_form.htm
        - https://www.geeksforgeeks.org/converting-context-free-grammar-greibach-normal-form
    
    - Papers
        
        - `Greibach Normal Form Transformation Revisited
          <https://doi.org/10.1006/inco.1998.2772>`__
        
        - `Bals et al.: Incremental Construction of Greibach Normal Form (2013)
          <https://doi.org/10.1109/TASE.2013.42>`__

    - Books

        - Rich - Automata, Computability and Complexity (2007): pp. 169-170, 630-635

    """
    # Copy the given grammar
    gr = grammar.copy()

    # Transformation
    raise NotImplementedError  # TODO
    
    # Repair step updates all grammar properties to fit to the new production rules
    grammar = _shared.update_grammar_parts(grammar)
    return grammar
