from .. import data_structures as _data_structures


def create_empty_terminal():
    """Create a terminal that represents the empty string, usually denoted as ε."""
    return _data_structures.TerminalSymbol('')


def is_empty_terminal(symbol):
    """Check if a symbol is a terminal that represents the empty string, usually denoted as ε."""
    return symbol.text == '' and isinstance(symbol, _data_structures.TerminalSymbol)


def create_new_nonterminal(grammar, prefix):
    """Create a new nonterminal that is not yet part of the grammar using a prefix and increment.

    Caution: Requires the grammar's nonterminal set to be up to date (with the production rules).

    """
    template = str(prefix) + '{}'
    for i in range(1_000_000_000_000_000):
        text = template.format(i)
        symbol = _data_structures.NonterminalSymbol(text)
        if symbol not in grammar.nonterminal_symbols:
            grammar.nonterminal_symbols.add(symbol)
            break
    return symbol


def update_grammar_parts(grammar):
    """Repair the grammar, so that after rule modifications all parts fit to each other again."""
    # Tabula rasa
    remembered_rules = grammar.production_rules
    grammar._set_empty_state()

    # Restore a consistent state
    # - Production rules
    grammar.production_rules = remembered_rules
    # - Sets of nonterminals and terminals
    for lhs in grammar.production_rules.keys():
        grammar.nonterminal_symbols.add(lhs)  # order of nonterminals is that of appearance in lhs
    for rhs_multiple in grammar.production_rules.values():
        for rhs in rhs_multiple:
            for sym in rhs:
                if isinstance(sym, _data_structures.NonterminalSymbol):
                    grammar.nonterminal_symbols.add(sym)
                else:
                    grammar.terminal_symbols.add(sym)
    # - Start symbol
    grammar.start_symbol = grammar.nonterminal_symbols[0]
    return grammar
