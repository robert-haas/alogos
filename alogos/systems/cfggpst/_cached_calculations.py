def idx_production_rules(grammar):
    """Create a dictionary of index to production rule associations."""
    sim = grammar._calc_sym_idx_map()
    ipr = {}
    for lhs, rhs_multiple in grammar.production_rules.items():
        ipr[sim[lhs]] = [[sim[sym] for sym in rhs] for rhs in rhs_multiple]
    return ipr
