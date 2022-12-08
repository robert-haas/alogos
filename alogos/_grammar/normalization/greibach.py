from . import _shared


def is_gnf(grammar):
    """Check if the grammar is in Greibach Normal Form (GNF)."""
    raise NotImplementedError


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
    raise NotImplementedError

    # Repair step updates all grammar properties to fit to the new production rules
    gr = _shared.update_grammar_parts(gr)
    return gr
