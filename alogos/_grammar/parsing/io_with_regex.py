import re as _re

from ... import exceptions as _exceptions
from ... import warnings as _warnings
from ..._utilities import argument_processing as _ap
from ..._utilities.operating_system import NEWLINE as _NEWLINE
from .. import data_structures as _data_structures


# Shared regex patterns
# - Letter followed by arbitrary sequence of letters, digits, _ and -
_STANDALONE_NONTERMINAL_PATTERN = r"[a-zA-Z][-_0-9a-zA-Z]*"

# - Sequence of non-whitespace symbols
_STANDALONE_TERMINAL_PATTERN = r"[^\s]+"

# - Shortest possible sequence of symbols and newline (until first end mark is found)
_ENCLOSED_NONTERMINAL_PATTERN = r".+?"
_ENCLOSED_TERMINAL_PATTERN = r"(?:.|{newline})*?".format(newline=_NEWLINE)

# - Longest possible sequence of whitespaces including empty (space, tab, newline, linefeed, etc.)
_MAX_OPT_WHITESPACE = r"\s*"

# Suffix for naming rules derived in automatic EBNF to BNF conversion
_SHARED_SUFFIX = "_§"


# Reading BNF and EBNF


def read_bnf(grammar, bnf_text, verbose, *symbols):
    """Read grammar from text in BNF notation."""
    # Argument processing
    bnf_text = _ap.str_arg("bnf_text", bnf_text)
    verbose = _ap.bool_arg("verbose", verbose)
    (
        def_symbol,
        rule_separator_symbol,
        start_nt,
        end_nt,
        start_t1,
        end_t1,
        start_t2,
        end_t2,
    ) = _check_symbols_validity(*symbols)

    # Conversion of user-provided symbols into escaped strings for secure use in regex patterns
    esc_def_symbol = _re.escape(def_symbol)
    esc_start_nt = _re.escape(start_nt)
    esc_end_nt = _re.escape(end_nt)
    esc_start_t1 = _re.escape(start_t1)
    esc_end_t1 = _re.escape(end_t1)
    esc_start_t2 = _re.escape(start_t2)
    esc_end_t2 = _re.escape(end_t2)

    # Print arguments
    if verbose:

        def represent(text):
            return "Empty string" if text == "" else text

        heading = "Used symbols"
        print(heading)
        print("-" * len(heading))
        print()
        print("defining_symbol:", represent(def_symbol))
        print("rule_separator_symbol:", represent(rule_separator_symbol))
        print("start_nonterminal_symbol:", represent(start_nt))
        print("end_nonterminal_symbol:", represent(end_nt))
        print("start_terminal_symbol:", represent(start_t1))
        print("end_terminal_symbol:", represent(end_t1))
        print("start_terminal_symbol2:", represent(start_t2))
        print("end_terminal_symbol2:", represent(end_t2))

    # Parsing step 1: Split full text into list of lhs (=nonterminal) and rhs (=other symbols)
    # 1.a) Regex pattern preparation
    if start_nt and end_nt:
        pattern_nt = r"{}{}{}".format(
            esc_start_nt, _ENCLOSED_NONTERMINAL_PATTERN, esc_end_nt
        )
    else:
        pattern_nt = _STANDALONE_NONTERMINAL_PATTERN
    pattern_lhs_rhs = r"{newline}{ws}({nonterminal}){ws}{defsymbol}".format(
        newline=_NEWLINE,
        ws=_MAX_OPT_WHITESPACE,
        nonterminal=pattern_nt,
        defsymbol=esc_def_symbol,
    )
    re_lhs_rhs = _re.compile(pattern_lhs_rhs)
    bnf_text = _NEWLINE + bnf_text  # newline to guarantee recognition of the first lhs

    # 1.b) Parsing
    txt_list_lhs_rhs = re_lhs_rhs.split(bnf_text)
    txt_list_lhs_rhs = txt_list_lhs_rhs[1:]  # get rid of an empty first match

    # Parsing step 2: Finding the single nonterminal symbol in lhs and arbitrary symbols in rhs
    # 2.a) Helper functions
    def create_enclosed_symbol_pattern(start, mid, end):
        if start and end:
            return r"{}{}{}".format(start, mid, end)
        return None

    def crop_symbol(txt_sym, txt_start, txt_end):
        idx_start = len(txt_start)
        idx_end = len(txt_sym) - len(txt_end)
        return txt_sym[idx_start:idx_end]

    def text_is_a_nonterminal(text):
        return re_nt.fullmatch(text)

    def text_is_a_terminal(text):
        return re_t.fullmatch(text)

    def process_a_nonterminal(txt_nt_enclosed, crop=True):
        if crop:
            txt_nt = crop_symbol(txt_nt_enclosed, start_nt, end_nt)
        sym_nt = _data_structures.NonterminalSymbol(txt_nt)
        return sym_nt

    def process_a_terminal(txt_t_enclosed):
        if txt_t_enclosed.startswith(start_t1) and txt_t_enclosed.endswith(end_t1):
            txt_t = crop_symbol(txt_t_enclosed, start_t1, end_t1)
        else:
            txt_t = crop_symbol(txt_t_enclosed, start_t2, end_t2)
        sym_t = _data_structures.TerminalSymbol(txt_t)
        _add_terminal_to_grammar(grammar, sym_t)
        return sym_t

    # 2.b) Regex pattern preparation
    # Nonterminal symbols
    pattern_nt = create_enclosed_symbol_pattern(
        esc_start_nt, _ENCLOSED_NONTERMINAL_PATTERN, esc_end_nt
    )
    if pattern_nt:
        re_nt = _re.compile("({})".format(pattern_nt))
    else:
        re_nt = None
    # Terminal symbols
    pattern_t1 = create_enclosed_symbol_pattern(
        esc_start_t1, _ENCLOSED_TERMINAL_PATTERN, esc_end_t1
    )
    pattern_t2 = create_enclosed_symbol_pattern(
        esc_start_t2, _ENCLOSED_TERMINAL_PATTERN, esc_end_t2
    )
    if pattern_t1 and pattern_t2:
        re_t = _re.compile("({})".format("|".join([pattern_t1, pattern_t2])))
    elif pattern_t1:
        re_t = _re.compile("({})".format(pattern_t1))
    else:
        re_t = None
    # Right-hand sides
    pattern_nonempty = [
        pat for pat in [pattern_nt, pattern_t1, pattern_t2] if pat is not None
    ]
    pattern_rhs_splitter = "({})".format("|".join(pattern_nonempty))
    re_rhs_splitter = _re.compile(pattern_rhs_splitter)

    # 2.c) Parsing
    i = 0
    while i < len(txt_list_lhs_rhs):
        # Get next left-hand side (lhs) and multiple right-hand sides (rhs) belonging to it
        txt_lhs = txt_list_lhs_rhs[i]
        txt_rhs = txt_list_lhs_rhs[i + 1]
        i += 2
        if verbose:
            print()
            heading = "Production list {}".format(i // 2)
            print(heading)
            print("-" * len(heading))
            print()

        # Parse lhs: Get the single nonterminal
        sym_lhs_nt = process_a_nonterminal(txt_lhs, crop=True)
        _add_nonterminal_to_grammar(grammar, sym_lhs_nt)
        if grammar.start_symbol is None:
            grammar.start_symbol = sym_lhs_nt
        if verbose:
            print("Left-hand side")
            print("  Text: {}".format(txt_lhs))
            print("  Interpretation: {}".format(repr(sym_lhs_nt)))
            print()
            print("Right-hand sides:")
            print("  Text: {}".format(txt_rhs))
            print("  Interpretation:")

        # Parse rhs: Get several productions, each consisting of arbitrary symbols
        sym_list_rhs = []
        for sym_or_seq in re_rhs_splitter.split(txt_rhs):
            # Mode 1: Nonterminals and terminals are enclosed by indicator symbols
            if re_nt and re_t:
                if text_is_a_nonterminal(sym_or_seq):
                    sym_nt = process_a_nonterminal(sym_or_seq)
                    sym_list_rhs.append(sym_nt)
                elif text_is_a_terminal(sym_or_seq):
                    sym_t = process_a_terminal(sym_or_seq)
                    sym_list_rhs.append(sym_t)
                else:
                    txt_list_rhs = sym_or_seq.split(rule_separator_symbol)
                    if len(txt_list_rhs) > 1:
                        # Add a rule
                        _add_rule_to_grammar(grammar, sym_lhs_nt, sym_list_rhs, verbose)
                        sym_list_rhs = []
            # Mode 2: Only nonterminals are enclosed by indicator symbols
            elif re_nt:
                if text_is_a_nonterminal(sym_or_seq):
                    sym_nt = process_a_nonterminal(sym_or_seq)
                    sym_list_rhs.append(sym_nt)
                else:
                    txt_list_rhs = sym_or_seq.split(rule_separator_symbol)
                    for cnt, txt_rhs in enumerate(txt_list_rhs):
                        if cnt > 0:
                            # Add a rule
                            _add_rule_to_grammar(
                                grammar, sym_lhs_nt, sym_list_rhs, verbose
                            )
                            sym_list_rhs = []
                        for txt_t in txt_rhs.split():
                            sym_t = process_a_terminal(txt_t)
                            sym_list_rhs.append(sym_t)
            # Mode 3: Only terminals are enclosed by indicator symbols
            elif re_t:
                if text_is_a_terminal(sym_or_seq):
                    sym_t = process_a_terminal(sym_or_seq)
                    sym_list_rhs.append(sym_t)
                else:
                    txt_list_rhs = sym_or_seq.split(rule_separator_symbol)
                    for cnt, txt_rhs in enumerate(txt_list_rhs):
                        if cnt > 0:
                            # Add a rule
                            _add_rule_to_grammar(
                                grammar, sym_lhs_nt, sym_list_rhs, verbose
                            )
                            sym_list_rhs = []
                        for txt_nt in txt_rhs.split():
                            sym_nt = process_a_nonterminal(txt_nt)
                            sym_list_rhs.append(sym_nt)
        # Add the last rule
        _add_rule_to_grammar(grammar, sym_lhs_nt, sym_list_rhs, verbose)

    # Check if the resulting grammar is valid
    _check_grammar_validity(grammar)


def read_ebnf(grammar, ebnf_text, verbose, *symbols):
    """Read grammar from text in EBNF notation.

    References
    ----------
    - https://en.wikipedia.org/wiki/Wirth_syntax_notation
    - http://homepage.divms.uiowa.edu/~jones/compiler/gtools/

    - Conversion of EBNF to BNF notation

        - https://stackoverflow.com/questions/2466484/converting-ebnf-to-bnf
        - https://stackoverflow.com/questions/2842809/lexers-vs-parsers
          useful notes in "EBNF really doesn't add much to the power of grammars."
        - https://condor.depaul.edu/ichu/csc447/notes/wk3/BNF.pdf
          useful notes in summary on last page
        - https://stackoverflow.com/questions/20175248/ebnf-is-this-an-ll1-grammar

    """
    # Argument processing
    ebnf_text = _ap.str_arg("ebnf_text", ebnf_text)
    verbose = _ap.bool_arg("verbose", verbose)
    (
        def_symbol,
        rule_separator_symbol,
        start_nt,
        end_nt,
        start_t1,
        end_t1,
        start_t2,
        end_t2,
    ) = _check_symbols_validity(*symbols)

    # Fixed symbols
    start_group_symbol = "("
    end_group_symbol = ")"
    start_option_symbol = "["
    end_option_symbol = "]"
    start_repeat_symbol = "{"
    end_repeat_symbol = "}"
    quantifier_0_to_1_symbol = "?"
    quantifier_1_to_n_symbol = "+"
    quantifier_0_to_n_symbol = "*"

    # Conversion of symbols into escaped strings for secure use in regex patterns
    esc_defining_symbol = _re.escape(def_symbol)
    esc_rule_separator_symbol = _re.escape(rule_separator_symbol)
    esc_start_nt = _re.escape(start_nt)
    esc_end_nt = _re.escape(end_nt)
    esc_start_t1 = _re.escape(start_t1)
    esc_end_t1 = _re.escape(end_t1)
    esc_start_t2 = _re.escape(start_t2)
    esc_end_t2 = _re.escape(end_t2)

    esc_start_group_symbol = _re.escape(start_group_symbol)
    esc_end_group_symbol = _re.escape(end_group_symbol)
    esc_start_option_symbol = _re.escape(start_option_symbol)
    esc_end_option_symbol = _re.escape(end_option_symbol)
    esc_start_repeat_symbol = _re.escape(start_repeat_symbol)
    esc_end_repeat_symbol = _re.escape(end_repeat_symbol)
    esc_quantifier_0_to_1_symbol = _re.escape(quantifier_0_to_1_symbol)
    esc_quantifier_1_to_n_symbol = _re.escape(quantifier_1_to_n_symbol)
    esc_quantifier_0_to_n_symbol = _re.escape(quantifier_0_to_n_symbol)

    # Print arguments and fixed symbols
    if verbose:

        def represent(text):
            return "Empty string" if text == "" else text

        heading = "Used symbols"
        print(heading)
        print("-" * len(heading))
        print()
        print("defining_symbol:", represent(def_symbol))
        print("rule_separator_symbol:", represent(rule_separator_symbol))
        print("start_nonterminal_symbol:", represent(start_nt))
        print("end_nonterminal_symbol:", represent(end_nt))
        print("start_terminal_symbol:", represent(start_t1))
        print("end_terminal_symbol:", represent(end_t1))
        print("start_terminal_symbol2:", represent(start_t2))
        print("end_terminal_symbol2:", represent(end_t2))
        print()
        print("start_group_symbol:", represent(start_group_symbol))
        print("end_group_symbol:", represent(end_group_symbol))
        print("start_option_symbol:", represent(start_option_symbol))
        print("end_option_symbol:", represent(end_option_symbol))
        print("start_repeat_symbol:", represent(start_repeat_symbol))
        print("end_repeat_symbol:", represent(end_repeat_symbol))
        print("quantifier_0_to_1_symbol:", represent(quantifier_0_to_1_symbol))
        print("quantifier_1_to_n_symbol:", represent(quantifier_1_to_n_symbol))
        print("quantifier_0_to_n_symbol:", represent(quantifier_0_to_n_symbol))

    # Parsing step 1: Split full text into list of lhs (=nonterminal) and rhs (=other symbols)
    # 1.a) Regex pattern preparation
    if start_nt and end_nt:
        pattern_nt = r"{}{}{}".format(
            esc_start_nt, _ENCLOSED_NONTERMINAL_PATTERN, esc_end_nt
        )
    else:
        pattern_nt = _STANDALONE_NONTERMINAL_PATTERN
    pattern_lhs_rhs = r"{newline}{ws}({nonterminal}){ws}{defsymbol}".format(
        newline=_NEWLINE,
        ws=_MAX_OPT_WHITESPACE,
        nonterminal=pattern_nt,
        defsymbol=esc_defining_symbol,
    )
    re_lhs_rhs = _re.compile(pattern_lhs_rhs)
    ebnf_text = (
        _NEWLINE + ebnf_text
    )  # newline to guarantee recognition of the first lhs

    # 1.b) Parsing
    txt_list_lhs_rhs = re_lhs_rhs.split(ebnf_text)
    txt_list_lhs_rhs = txt_list_lhs_rhs[1:]  # get rid of an empty first match

    # Parsing step 2: Finding the single nonterminal symbol in lhs and arbitrary symbols in rhs
    # 2.a) Helper functions
    def crop_symbol(txt_sym, txt_start, txt_end):
        idx_start = len(txt_start)
        idx_end = len(txt_sym) - len(txt_end)
        return txt_sym[idx_start:idx_end]

    def process_a_nonterminal(txt_nt, crop=False):
        if crop:
            txt_nt = crop_symbol(txt_nt, start_nt, end_nt)
        sym_nt = _data_structures.NonterminalSymbol(txt_nt)
        return sym_nt

    def process_a_terminal(txt_t_enclosed):
        if txt_t_enclosed.startswith(start_t1) and txt_t_enclosed.endswith(end_t1):
            txt_t = crop_symbol(txt_t_enclosed, start_t1, end_t1)
        else:
            txt_t = crop_symbol(txt_t_enclosed, start_t2, end_t2)
        sym_t = _data_structures.TerminalSymbol(txt_t)
        # _add_terminal_to_grammar(grammar, sym_t)  # deferred to process_a_rule for empty string
        return sym_t

    def text_is_a_nonterminal(text):
        return re_nt.fullmatch(text)

    def text_is_a_terminal(text):
        return re_t.fullmatch(text)

    def process_a_nt_special_mix(rhs_text):
        seq = [
            process_a_nonterminal(text, crop=True)
            if text_is_a_nonterminal(text)
            else text
            for text in re_rhs_splitter.findall(rhs_text)
        ]
        return seq

    def find_an_innermost_bracket_pair(txt_list):
        found_bracket = None
        last_opening_symbol = None
        last_opening_symbol_idx = None
        for idx, text in enumerate(txt_list):
            if isinstance(text, str):
                if text in [
                    start_group_symbol,
                    start_option_symbol,
                    start_repeat_symbol,
                ]:
                    last_opening_symbol = text
                    last_opening_symbol_idx = idx
                elif (
                    text == end_group_symbol
                    and last_opening_symbol == start_group_symbol
                ):
                    found_bracket = ("group", last_opening_symbol_idx, idx)
                    break
                elif (
                    text == end_option_symbol
                    and last_opening_symbol == start_option_symbol
                ):
                    found_bracket = ("option", last_opening_symbol_idx, idx)
                    break
                elif (
                    text == end_repeat_symbol
                    and last_opening_symbol == start_repeat_symbol
                ):
                    found_bracket = ("repeat", last_opening_symbol_idx, idx)
                    break
        return found_bracket

    def find_a_quantifier(txt_list):
        found_quantifier = None
        for idx, text in enumerate(txt_list):
            if idx == 0:
                continue  # a quantifier on zeroth position is ignored
            if isinstance(text, str):
                if text == quantifier_0_to_1_symbol:
                    found_quantifier = ("0 to 1", idx)
                    break
                if text == quantifier_1_to_n_symbol:
                    found_quantifier = ("1 to n", idx)
                    break
                if text == quantifier_0_to_n_symbol:
                    found_quantifier = ("0 to n", idx)
                    break
        return found_quantifier

    def create_nt_with_derived_name(text, suffix):
        cnt = 0
        while True:
            new_text = "{text}{suffix}{count}".format(
                text=text, suffix=suffix, count=cnt
            )
            sym_nt = process_a_nonterminal(new_text)
            if sym_nt not in seen_nonterminals:
                seen_nonterminals.add(sym_nt)
                break
            cnt += 1
        return sym_nt

    def cut_out_exclusive(seq, start_idx, end_idx):
        left = seq[:start_idx]
        inner = seq[start_idx + 1 : end_idx]
        right = seq[end_idx + 1 :]
        return left, inner, right

    def cut_out_left_inclusive(seq, start_idx, end_idx):
        left = seq[:start_idx]
        inner = seq[start_idx:end_idx]
        right = seq[end_idx + 1 :]
        return left, inner, right

    def remove_brackets(rule):
        """Remove {}, () and [] from a EBNF right-hand side by creating new helper rules.

        References
        ----------
        - https://stackoverflow.com/questions/2466484/converting-ebnf-to-bnf
          Caution: {} is imprecise because {a|b|c} means (a|b|c)* and not a*|b|c

        """
        sym_lhs_nt, sym_list_rhs = rule
        helper_rules = []
        while True:
            # Find the first pair of brackets that have no other brackets within them
            bracket_hit = find_an_innermost_bracket_pair(sym_list_rhs)
            if bracket_hit is None:
                break
            bracket_type, start_idx, end_idx = bracket_hit

            # Handle {} by reducing the expression {something} to the expression (something)*
            if bracket_type == "repeat":
                sym_list_rhs[start_idx] = "("
                sym_list_rhs[end_idx] = ")"
                end_idx_incr = end_idx + 1
                sym_list_rhs[end_idx_incr:end_idx_incr] = ["*"]
                continue

            # Handle the other two types of brackets
            sym_nt_new = create_nt_with_derived_name(
                sym_lhs_nt.text, suffix=_SHARED_SUFFIX
            )
            left, sym_list_inner, right = cut_out_exclusive(
                sym_list_rhs, start_idx, end_idx
            )
            sym_list_rhs = left + [sym_nt_new] + right
            if bracket_type == "group":
                # Handle ()
                helper_rule = (sym_nt_new, sym_list_inner)
            elif bracket_type == "option":
                # Handle []
                empty = process_a_terminal("")
                helper_rule = (sym_nt_new, sym_list_inner + ["|"] + [empty])

            helper_rules.append(helper_rule)
        modified_rule = (sym_lhs_nt, sym_list_rhs)
        new_rules = [modified_rule] + helper_rules
        return new_rules

    def remove_quantifiers(rule):
        """Remove *, + and ? from a EBNF right-hand side by creating new helper rules.

        References
        ----------
        - https://stackoverflow.com/questions/2466484/converting-ebnf-to-bnf

        """
        sym_lhs_nt, list_rhs = rule
        helper_rules = []
        while True:
            # Find the first quantifier
            quantifier_hit = find_a_quantifier(list_rhs)
            if quantifier_hit is None:
                break
            quantifier_type, idx = quantifier_hit

            # Handle all three types of quantifiers
            sym_nt_new = create_nt_with_derived_name(
                sym_lhs_nt.text, suffix=_SHARED_SUFFIX
            )
            left, inner, right = cut_out_left_inclusive(list_rhs, idx - 1, idx)
            list_rhs = left + [sym_nt_new] + right
            if quantifier_type == "0 to 1":
                sym_t_empty = process_a_terminal("")
                helper_rule = (sym_nt_new, inner + ["|"] + [sym_t_empty])
            elif quantifier_type == "0 to n":
                sym_t_empty = process_a_terminal("")
                helper_rule = (sym_nt_new, [sym_nt_new] + inner + ["|"] + [sym_t_empty])
            elif quantifier_type == "1 to n":
                helper_rule = (sym_nt_new, inner + ["|"] + [sym_nt_new] + inner)
            helper_rules.append(helper_rule)
        modified_rule = (sym_lhs_nt, list_rhs)
        new_rules = [modified_rule] + helper_rules
        return new_rules

    def remove_alternation_and_nonsense(rule):
        sym_lhs_nt, list_rhs = rule
        part = []
        splitted_rules = []
        for item in list_rhs:
            if isinstance(item, str):
                if item == rule_separator_symbol:
                    rule = (sym_lhs_nt, part)
                    splitted_rules.append(rule)
                    part = []
            else:
                part.append(item)
        rule = (sym_lhs_nt, part)
        splitted_rules.append(rule)
        return splitted_rules

    def process_a_rule(rule):
        # Treat all occurrences of brackets: {}, () and []
        intermediary_rules1 = remove_brackets(rule)

        # Treat all occurrences of quantifiers: *, + and ?
        intermediary_rules2 = []
        for intermediary_rule in intermediary_rules1:
            new_rules = remove_quantifiers(intermediary_rule)
            intermediary_rules2.extend(new_rules)

        # Treat all occurrences of alternation symbols: |
        final_rules = []
        for intermediary_rule in intermediary_rules2:
            new_rules = remove_alternation_and_nonsense(intermediary_rule)
            final_rules.extend(new_rules)

        # Add all final rules to the grammar
        for lhs, rhs in final_rules:
            # Add nonterminals to grammar
            _add_nonterminal_to_grammar(grammar, lhs)
            # Add terminals to grammar here, so that all empty strings arrive at correct position
            for sym in rhs:
                if isinstance(sym, _data_structures.TerminalSymbol):
                    _add_terminal_to_grammar(grammar, sym)
            # Add rule
            _add_rule_to_grammar(grammar, lhs, rhs, verbose)

    # 2.b) Regex pattern preparation
    # Nonterminal symbols
    re_nt = _re.compile(pattern_nt)
    # Terminal symbols
    use_t1 = start_t1 and end_t1
    use_t2 = start_t2 and end_t2
    pattern_t1 = "{}{}{}".format(esc_start_t1, _ENCLOSED_TERMINAL_PATTERN, esc_end_t1)
    pattern_t2 = "{}{}{}".format(esc_start_t2, _ENCLOSED_TERMINAL_PATTERN, esc_end_t2)
    if use_t1 and use_t2:
        re_t = _re.compile("({}|{})".format(pattern_t1, pattern_t2))
    elif use_t1:
        re_t = _re.compile("({})".format(pattern_t1))
    # Special symbols
    special_symbols = [
        esc_rule_separator_symbol,
        esc_start_group_symbol,
        esc_end_group_symbol,
        esc_start_option_symbol,
        esc_end_option_symbol,
        esc_start_repeat_symbol,
        esc_end_repeat_symbol,
        esc_quantifier_0_to_1_symbol,
        esc_quantifier_1_to_n_symbol,
        esc_quantifier_0_to_n_symbol,
    ]
    pattern_special_symbols = "{}".format("|".join(special_symbols))
    # Right-hand sides
    pattern_rhs_splitter = "({}|{})".format(pattern_special_symbols, pattern_nt)
    re_rhs_splitter = _re.compile(pattern_rhs_splitter)

    # 2.c) Parsing
    seen_nonterminals = set()
    i = 0
    while i < len(txt_list_lhs_rhs):
        # Get next left-hand side (lhs) and multiple right-hand sides (rhs) belonging to it
        txt_lhs = txt_list_lhs_rhs[i]
        txt_rhs = txt_list_lhs_rhs[i + 1]
        i += 2
        if verbose:
            print()
            heading = "Production list {}".format(i // 2)
            print(heading)
            print("-" * len(heading))
            print()

        # Parse lhs: Get the single nonterminal
        sym_lhs_nt = process_a_nonterminal(txt_lhs, crop=True)
        if grammar.start_symbol is None:
            grammar.start_symbol = sym_lhs_nt
        if verbose:
            print("Left-hand side")
            print("  Text: {}".format(txt_lhs))
            print("  Interpretation: {}".format(repr(sym_lhs_nt)))
            print()
            print("Right-hand sides:")
            print("  Text: {}".format(txt_rhs))
            print("  Interpretation:")

        # Parse rhs: Get several productions, each consisting of arbitrary symbols
        t_vs_others_seq = [
            process_a_terminal(text) if text_is_a_terminal(text) else text
            for text in re_t.split(txt_rhs)
        ]
        t_nt_vs_special_sym_seq = []
        for sym_or_seq in t_vs_others_seq:
            if isinstance(sym_or_seq, _data_structures.TerminalSymbol):
                t_nt_vs_special_sym_seq.append(sym_or_seq)
            else:
                sym_list = process_a_nt_special_mix(sym_or_seq)
                t_nt_vs_special_sym_seq.extend(sym_list)

        # Process the rule consisting of lhs and rhs
        # = remove brackets, quantifiers, alternation and add the rule to the grammar
        rule = (sym_lhs_nt, t_nt_vs_special_sym_seq)
        process_a_rule(rule)

    # Delete duplicate rules that come from processing brackets and quantifiers
    try:
        _delete_duplicate_rules(grammar, only_helper_rules=True)
    except Exception:
        # If may fail for some grammars, where the subsequent validity check reports in detail
        pass

    # Check if the resulting grammar is valid
    _check_grammar_validity(grammar)


# Writing BNF and EBNF


def write_bnf(grammar, rules_on_separate_lines, *symbols):
    """Write grammar as text in BNF notation."""
    # Argument processing
    rules_on_separate_lines = _ap.bool_arg(
        "rules_on_separate_lines", rules_on_separate_lines
    )
    (
        def_symbol,
        rule_separator_symbol,
        start_nt,
        end_nt,
        start_t1,
        end_t1,
        start_t2,
        end_t2,
    ) = _check_symbols_validity(*symbols)
    use_nt = start_nt and end_nt
    use_t1 = start_t1 and end_t1
    use_t2 = start_t2 and end_t2

    # Regex pattern preparation
    re_nt_standalone = _re.compile(_STANDALONE_NONTERMINAL_PATTERN)
    re_t_standalone = _re.compile(_STANDALONE_TERMINAL_PATTERN)

    # Helper functions
    def contains_nt_indicator(text):
        return start_nt in text or end_nt in text

    def contains_t1_indicator(text):
        return start_t1 in text or end_t1 in text

    def contains_t2_indicator(text):
        return start_t2 in text or end_t2 in text

    def invalid_unenclosed_nonterminal(text):
        return not re_nt_standalone.fullmatch(text)

    def invalid_unenclosed_terminal(text):
        return not re_t_standalone.fullmatch(text)

    def process_nonterminal(text):
        if use_nt:
            if contains_nt_indicator(text):
                _exceptions.raise_write_nonterminal_error(text)
            start = start_nt
            end = end_nt
        else:
            if invalid_unenclosed_nonterminal(text):
                _exceptions.raise_write_nonterminal_error(text)
            start = ""
            end = ""
        text_with_start_and_end = "{}{}{}".format(start, text, end)
        return text_with_start_and_end

    def process_terminal(text):
        if use_t1 and use_t2:
            if contains_t1_indicator(text) and contains_t2_indicator(text):
                _exceptions.raise_write_terminal_error(text)
            if contains_t1_indicator(text):
                start = start_t2
                end = end_t2
            else:
                start = start_t1
                end = end_t1
        elif use_t1:
            if contains_t1_indicator(text):
                _exceptions.raise_write_terminal_error(text)
            start = start_t1
            end = end_t1
        else:
            if invalid_unenclosed_terminal(text):
                _exceptions.raise_write_terminal_error(text)
            start = ""
            end = ""
        text_with_start_and_end = "{}{}{}".format(start, text, end)
        return text_with_start_and_end

    # Write text
    seq = []
    for sym_lhs_nt, list_rhs in grammar.production_rules.items():
        # Left-hand side nonterminal symbol
        txt_lhs_nt = process_nonterminal(sym_lhs_nt.text)
        txt_lhs_nt_def = "{} {} ".format(txt_lhs_nt, def_symbol)
        seq.append(txt_lhs_nt_def)
        # Right-hand side sequence of symbols
        for sym_list_rhs in list_rhs:
            for sym in sym_list_rhs:
                if isinstance(sym, _data_structures.NonterminalSymbol):
                    txt_nt = process_nonterminal(sym.text)
                    seq.append(txt_nt)
                    seq.append(" ")
                else:
                    txt_t = process_terminal(sym.text)
                    seq.append(txt_t)
                    seq.append(" ")
            if rules_on_separate_lines:
                indentation = " " * (len(txt_lhs_nt) + len(def_symbol))
                seq.append(_NEWLINE)
                seq.append(indentation)
            seq.append("{} ".format(rule_separator_symbol))
        if rules_on_separate_lines:
            seq.pop()  # remove last lhs indent
            seq.pop()  # remove last newline
        seq.pop()  # remove last rule_separator_symbol
        seq.pop()  # remove last whitespace
        seq.append(_NEWLINE)
    return "".join(seq)


def write_ebnf(grammar, rules_on_separate_lines, *symbols):
    """Write grammar as text in EBNF notation."""
    # Currently reduced to BNF, i.e. no simplifications yet that are possible with EBNF constructs
    return write_bnf(grammar, rules_on_separate_lines, *symbols)


# Helper functions


def _check_symbols_validity(
    defining_symbol,
    rule_separator_symbol,
    start_nonterminal_symbol,
    end_nonterminal_symbol,
    start_terminal_symbol,
    end_terminal_symbol,
    start_terminal_symbol2,
    end_terminal_symbol2,
):
    # Type checking
    def_symbol = _ap.str_arg("defining_symbol", defining_symbol)
    rule_separator_symbol = _ap.str_arg("rule_separator_symbol", rule_separator_symbol)
    start_nt = _ap.str_arg("start_nonterminal_symbol", start_nonterminal_symbol)
    end_nt = _ap.str_arg("end_nonterminal_symbol", end_nonterminal_symbol)
    start_t1 = _ap.str_arg("start_terminal_symbol", start_terminal_symbol)
    end_t1 = _ap.str_arg("end_terminal_symbol", end_terminal_symbol)
    start_t2 = _ap.str_arg("start_terminal_symbol2", start_terminal_symbol2)
    end_t2 = _ap.str_arg("end_terminal_symbol2", end_terminal_symbol2)

    # Consistent: If one symbol is provided, the partner symbol needs to be provided too
    if _ap.logical_xor(start_nt, end_nt):
        _exceptions.raise_delimiter_symbol_error_1()
    if _ap.logical_xor(start_t1, end_t1):
        _exceptions.raise_surrounding_symbol_error_2()
    if _ap.logical_xor(start_t2, end_t2):
        _exceptions.raise_surrounding_symbol_error_3()

    # Ordered: If terminal_symbols1 are not provided, but terminal_symbols2 are, assign 2 to 1
    if start_t1 == "" and start_t2 != "":
        start_t1 = start_t2
        start_t2 = ""
    if end_t1 == "" and end_t2 != "":
        end_t1 = end_t2
        end_t2 = ""

    # Sufficient: Either nonterminals, or terminals, or both need to be enclosed by delimiters
    if not start_nt and not start_t1:
        _exceptions.raise_surrounding_symbol_error_4()

    # Unique: Nonterminal and terminal indicator symbols need to be different
    ind_nt = set([start_nt, end_nt])
    ind_t = set([start_t1, end_t1, start_t2, end_t2])
    overlap = ind_nt.intersection(ind_t)
    if "" in overlap:
        overlap.remove("")
    if overlap:
        listing = ", ".join(repr(x) for x in overlap)
        _exceptions.raise_surrounding_symbol_error_5(listing)
    return (
        def_symbol,
        rule_separator_symbol,
        start_nt,
        end_nt,
        start_t1,
        end_t1,
        start_t2,
        end_t2,
    )


def _check_grammar_validity(grammar):
    """Check criteria that a context-free grammar needs to fulfill.

    Criteria that raise a GrammarError if not fulfilled:
    - The sets of nonterminals, terminals and productions need to be non-empty.
    - Every nonterminal needs a production rule where it appears on the left-hand side,
      otherwise it can not be rewritten.
    - Every nonterminal needs a production rule that is non-recursive
      (also considering co-recursion), otherwise when the symbol appears once
      in a derivation there is no way to reach an end.

    Criteria that only emit warnings if not fulfilled (because algorithms can work anyways):
    - The sets of nonterminals and terminals should be disjoint.
    - Every production rule should be be unique.

    References
    ----------
    - https://en.wikipedia.org/wiki/Context-free_grammar#Proper_CFGs

    """
    # Raise an error if the grammar contains an empty set
    if not grammar.production_rules:
        _exceptions.raise_empty_productions_error()
    if not grammar.nonterminal_symbols:
        _exceptions.raise_empty_nonterminals_error()
    if not grammar.terminal_symbols:
        _exceptions.raise_empty_terminals_error()

    # Emit a warning if nonterminals and terminals are not disjoint sets of symbols
    s1 = set(sym.text for sym in grammar.nonterminal_symbols)
    s2 = set(sym.text for sym in grammar.terminal_symbols)
    intersection = s1.intersection(s2)
    if intersection:
        _warnings._warn_symbol_set_overlap(intersection)

    # Raise an error if a nonterminal (on rhs) has no corresponding production rule (on lhs)
    lhs_nonterminals = grammar.production_rules.keys()
    rhs_nonterminals = set()
    for rhs_multiple in grammar.production_rules.values():
        for rhs in rhs_multiple:
            for sym in rhs:
                if isinstance(sym, _data_structures.NonterminalSymbol):
                    rhs_nonterminals.add(sym)
    if len(lhs_nonterminals) != (len(rhs_nonterminals) - 1):
        missing_nonterminals = [
            "<{}>".format(nt) for nt in rhs_nonterminals if nt not in lhs_nonterminals
        ]
        if missing_nonterminals:
            _exceptions.raise_missing_nonterminals_error(missing_nonterminals)

    # Emit a warning if some production rules are not unique
    repeated_productions = dict()
    for lhs, rhs_vals in grammar.production_rules.items():
        seen = set()
        for rhs in rhs_vals:
            rhs_text = " ".join(
                "<{}>".format(sym.text)
                if isinstance(sym, _data_structures.NonterminalSymbol)
                else sym.text
                for sym in rhs
            )
            if rhs_text in seen:
                text = "  <{}> ::= {}".format(str(lhs), rhs_text)
                repeated_productions[text] = None
            seen.add(rhs_text)
    if repeated_productions:
        _warnings._warn_repeated_productions(repeated_productions)


def _delete_duplicate_rules(grammar, only_helper_rules=False):
    """Remove duplicate rules, i.e. when different lhs share exactly the same rhs."""
    # Delete lhs which have exactly the same rhs as another lhs
    # Example:
    #  S -> A B
    #  A -> 1 | 4
    #  B -> 1 | 4 ... delete it and replace occurrences of B in the rhs of every other lhs
    known_rhs = dict()
    delete_list = []
    replacement_map = dict()

    # Make sure the start symbol is considered first
    lhs_list = list(grammar.production_rules.keys())
    lhs_list.remove(grammar.start_symbol)
    lhs_list = [grammar.start_symbol] + lhs_list
    rule_list = [(lhs, grammar.production_rules[lhs]) for lhs in lhs_list]

    for lhs, rhs in rule_list:
        # Optionally skip rules where the lhs does not start with an underscore
        if only_helper_rules and _SHARED_SUFFIX not in str(lhs):
            continue
        # All lhs (except start symbol) may be deleted if their rhs is the same as in another lhs
        rhs_hashable = str(rhs)
        if rhs_hashable not in known_rhs:
            known_rhs[rhs_hashable] = lhs
        else:
            known_lhs = known_rhs[rhs_hashable]
            replacement_map[lhs] = known_lhs
            delete_list.append(lhs)

    # Replace lhs symbols that will be deleted
    for lhs, rhs_multiple in grammar.production_rules.items():
        for rhs_idx, rhs in enumerate(rhs_multiple):
            for symb_idx, symb in enumerate(rhs):
                if symb in replacement_map:
                    new_symb = replacement_map[symb]
                    grammar.production_rules[lhs][rhs_idx][symb_idx] = new_symb

    # Delete lhs symbols and their productions
    for lhs in delete_list:
        _remove_nonterminal_from_grammar(grammar, lhs)
        _remove_rule_from_grammar(grammar, lhs)


# Helper functions, which have to change when other data structures are used for symbols or rules


def _add_terminal_to_grammar(grammar, symbol):
    grammar.terminal_symbols.add(symbol)


def _add_nonterminal_to_grammar(grammar, symbol):
    grammar.nonterminal_symbols.add(symbol)


def _remove_nonterminal_from_grammar(grammar, symbol):
    grammar.nonterminal_symbols.remove(symbol)


def _add_rule_to_grammar(grammar, sym_lhs_nt, sym_list_rhs, verbose):
    # Special case: Empty rhs is interpreted as empty terminal symbol
    if len(sym_list_rhs) == 0:
        sym_t_empty = _data_structures.TerminalSymbol("")
        _add_terminal_to_grammar(grammar, sym_t_empty)
        sym_list_rhs = [sym_t_empty]
    # Print recognized rule
    if verbose:
        print("    {} -> {}".format(repr(sym_lhs_nt), repr(sym_list_rhs)))
    # Add rule
    if sym_lhs_nt in grammar.production_rules:
        grammar.production_rules[sym_lhs_nt].append(sym_list_rhs)
    else:
        grammar.production_rules[sym_lhs_nt] = [sym_list_rhs]


def _remove_rule_from_grammar(grammar, sym_lhs_nt):
    del grammar.production_rules[sym_lhs_nt]
