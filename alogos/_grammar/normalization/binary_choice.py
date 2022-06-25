import math as _math

from .. import data_structures as _data_structures
from . import _shared


def is_bcnf(grammar):
    """Check if the grammar is in Binary Choice Normal Form (BCNF).

    All nonterminals need to have one or two productions:
    - A → X
    - A → X | Y

    Tasks that are simplified by having a grammar in BCNF:
    - Grammatical Evolution (GE) can use codons of size 1 (=1 bit, 2 possible states)
      to perform a genotype-to-phenotype mapping, i.e. there is no redundancy in the
      individual genes and the modulo rule becomes irrelevant.

    """
    for lhs, rhs_multiple in grammar.production_rules.items():
        if len(rhs_multiple) > 2:
            return False
    return True


def to_bcnf(grammar):
    """Convert the grammar G into Binary Choice Normal Form without changing its language L(G)."""
    # Copy the given grammar
    grammar = grammar.copy()

    # Transformation
    def split_and_add_rules(new_rules, grammar, lhs, rhs_multiple):
        num_symbols = len(rhs_multiple)
        if num_symbols > 2:
            num_left = _math.ceil(num_symbols / 2.0)
            nt_left = _shared.create_new_nonterminal(grammar, prefix='BL_')
            nt_right = _shared.create_new_nonterminal(grammar, prefix='BR_')
            new_rules[lhs] = [[nt_left], [nt_right]]
            # left half
            rhs_left = rhs_multiple[:num_left]
            split_and_add_rules(new_rules, grammar, nt_left, rhs_left)
            # right half
            rhs_right = rhs_multiple[num_left:]
            split_and_add_rules(new_rules, grammar, nt_right, rhs_right)
        else:
            new_rules[lhs] = rhs_multiple

    new_rules = dict()
    for lhs, rhs_multiple in grammar.production_rules.items():
        if len(rhs_multiple) > 2:
            split_and_add_rules(new_rules, grammar, lhs, rhs_multiple)
        else:
            new_rules[lhs] = rhs_multiple
    grammar.production_rules = new_rules

    # Repair step updates all grammar properties to fit to the new production rules
    grammar = _shared.update_grammar_parts(grammar)
    return grammar
