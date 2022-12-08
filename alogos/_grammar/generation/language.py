from copy import deepcopy as _deepcopy
from itertools import chain as _chain
from itertools import product as _cartesian_product

from ... import warnings as _warnings
from ..._utilities import argument_processing as _ap
from .. import data_structures as _data_structures


def generate_language(
    grammar, max_steps=None, sort_order=None, verbose=None, return_details=None
):
    """Generate the formal language defined by the grammar."""
    # Argument processing
    warn_if_stopped = max_steps is None
    max_steps = _ap.num_arg("max_steps", max_steps, default=20_000)
    sort_order = _ap.str_arg(
        "sort_order", sort_order, vals=["discovered", "lex", "shortlex"]
    )

    # Preparation of data structures
    # 1) Rules with a) only terminal symbols or b) at least one nonterminal on their rhs
    terminal_rules = []
    nonterminal_rules = []
    for lhs, multiple_rhs in grammar.production_rules.items():
        for rhs in multiple_rhs:
            nonterminals = []
            nonterminal_positions = []
            for idx, symbol in enumerate(rhs):
                if isinstance(symbol, _data_structures.NonterminalSymbol):
                    nonterminals.append(symbol.text)
                    nonterminal_positions.append(idx)
            if nonterminals:
                rule = (lhs.text, _concat(rhs), nonterminals, nonterminal_positions)
                nonterminal_rules.append(rule)
            else:
                rule = (lhs.text, _concat(rhs))
                terminal_rules.append(rule)

    strings_dict_template = {
        nonterminal.text: dict() for nonterminal in grammar.nonterminal_symbols
    }
    strings_all = _deepcopy(strings_dict_template)
    strings_recent = _deepcopy(strings_dict_template)

    # Basis part: Use terminal rules to generate the first strings of some corresponding lhs
    for lhs, rhs in terminal_rules:
        string = _terminal_symbol_seq_to_str(rhs)
        strings_recent[lhs][string] = None  # add the key, value is irrelevant

    if verbose:

        def symbol_seq_repr(symbol_seq):
            seq_repr = []
            for sym in symbol_seq:
                if isinstance(sym, _data_structures.NonterminalSymbol):
                    seq_repr.append("<{}>".format(sym))
                else:
                    seq_repr.append(str(sym))
            return "".join(seq_repr)

        text = "I. Basis"
        print(text)
        print("=" * len(text))
        print()
        print("Production rules with only terminals in the right-hand side:")
        for lhs, rhs in terminal_rules:
            rhs_repr = symbol_seq_repr(rhs)
            if rhs_repr == "":
                rhs_repr = "EPSILON"
            print("  <{}> -> {}".format(lhs, rhs_repr))
        _report_step(1, strings_recent)
        if max_steps > 1:
            text = "II. Recursion"
            print()
            print(text)
            print("=" * len(text))
            print()
            print(
                "Production rules with at least one nonterminal in the right-hand side:"
            )
            for lhs, rhs, _, nonterminal_pos in nonterminal_rules:
                rhs_repr = "".join(
                    "<{}>".format(sym) if i in nonterminal_pos else sym
                    for i, sym in enumerate(rhs)
                )
                print("  <{}> -> {}".format(lhs, rhs_repr))

    # Recursive part: Use nonterminal rules to generate further strings of all lhs
    for num_expansion in range(2, max_steps + 1):
        found_new_strings = False
        new_strings = _deepcopy(strings_dict_template)
        for lhs, rhs, nonterminals, nonterminal_positions in nonterminal_rules:
            # Prepare all combinations of fresh and old strings, except purely old ones
            one_or_another = [(0, 1) for _ in nonterminals]
            all_possible_selections = list(_cartesian_product(*one_or_another))
            wanted_selections = all_possible_selections[:-1]
            for nonterminal_choices_list in wanted_selections:
                string_lists_per_nt = []
                for choice, nonterminal in zip(nonterminal_choices_list, nonterminals):
                    if choice == 0:
                        string_lists_per_nt.append(strings_recent[nonterminal])
                    else:
                        string_lists_per_nt.append(strings_all[nonterminal])
                # Generate all wanted combinations of strings and put them at the places of the
                # corresponding nonterminals to form a new string
                for comb_of_known_strings in _cartesian_product(*string_lists_per_nt):
                    new_string = rhs[:]
                    for pos, known_string in zip(
                        nonterminal_positions, comb_of_known_strings
                    ):
                        new_string[pos] = known_string
                    new_string = _terminal_symbol_seq_to_str(new_string)
                    new_strings[lhs][
                        new_string
                    ] = None  # add the key, value is irrelevant

        # Add fresh strings to old list, and new strings to fresh list
        for nonterminal in grammar.nonterminal_symbols:
            nt_str = nonterminal.text
            strings_all[nt_str].update(strings_recent[nt_str])
            strings_recent[nt_str] = {
                string: None
                for string in new_strings[nt_str]
                if string not in strings_all[nt_str]
            }
            if strings_recent[nt_str]:
                found_new_strings = True
        if not found_new_strings:
            if verbose:
                _report_step(num_expansion, None)
                print()
                print("Finished.")
            break
        if verbose:
            _report_step(num_expansion, strings_recent)
    else:
        if warn_if_stopped:
            _warnings._warn_language_gen_stopped(max_steps)
        for nonterminal in grammar.nonterminal_symbols:
            nt_str = nonterminal.text
            strings_all[nt_str].update(strings_recent[nt_str])

    # Optional sorting
    if sort_order == "discovered":

        def sort_func(given_set):
            unsorted_list = list(given_set)
            return unsorted_list

    elif sort_order == "lex":

        def sort_func(given_set):
            sorted_list = list(given_set)
            sorted_list.sort()
            return sorted_list

    elif sort_order == "shortlex":

        def sort_func(given_set):
            sorted_list = list(given_set)
            sorted_list.sort(key=lambda string: (len(string), string))
            return sorted_list

    # Conditional return
    if return_details:
        # Detailed: A dict with nonterminals as keys and their languages as values
        return {
            nonterminal: sort_func(strings)
            for nonterminal, strings in strings_all.items()
        }
    else:
        # Only the language for the start symbol
        return sort_func(strings_all[grammar.start_symbol.text])


# Helper functions


def _concat(symbol_sequence):
    """Concatenate the text of multiple symbols."""
    return [sym.text for sym in symbol_sequence]


def _terminal_symbol_seq_to_str(sequence):
    """Convert a sequence of terminal symbols to a string."""
    return "".join(_chain.from_iterable(sequence))


def _report_step(num_step, strings_recent):
    """Print an intermediate result after a language generation step."""
    print()
    text = "Step {}".format(num_step)
    print(text)
    print("-" * len(text))
    print()
    if strings_recent:
        print("Newly found strings per nonterminal:")
        for nonterminal, strings in sorted(strings_recent.items()):
            if strings:
                print("  <{}>".format(nonterminal))
                for string in strings:
                    print('    "{}"'.format(string))
    else:
        print("No newly found strings.")
