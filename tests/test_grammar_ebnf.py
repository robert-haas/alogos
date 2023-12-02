import os

import pytest
import shared

import alogos as al


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, "in")


# Read EBNF from file


def test_ebnf_file_simple():
    filepath = os.path.join(IN_DIR, "ebnf_examples", "simple_test_grammar.ebnf")
    grammar = al.Grammar(ebnf_file=filepath, verbose=True)
    shared.check_grammar(
        grammar,
        ["7xR", "8yS", "9xS"],
        ["", "X", "21"],
        language=[
            "7xR",
            "7xS",
            "7yR",
            "7yS",
            "8xR",
            "8xS",
            "8yR",
            "8yS",
            "9xR",
            "9xS",
            "9yR",
            "9yS",
        ],
    )


def test_ebnf_file_from_bnf_with_unusual_symbols_symbolic_regression():
    filepath = os.path.join(IN_DIR, "ebnf_examples", "symbolic_regression.ebnf")
    grammar = al.Grammar(
        ebnf_file=filepath,
        defining_symbol="::=",
        start_nonterminal_symbol="<",
        end_nonterminal_symbol=">",
        start_terminal_symbol="#",
        end_terminal_symbol="#",
    )
    shared.check_grammar(
        grammar,
        ["x", "y", "x+x", "y*x", "x/y-x*y"],
        ["", "+", "+x", "X"],
    )


# Read EBNF from string


def test_ebnf_empty_string_minimal():
    ebnf_variants = [
        # empty string expressed as "" or ''
        # left
        'S = "" | "1"',
        "S = '' | '1'",
        'S=""|"1"',
        "S=''|'1'",
        "            S= ''|           '1'",
        '            S= ""|           "1"',
        """            S= ''|           "1" """,
        """            S= ""|           '1' """,
        # right
        'S = "1" | ""',
        "S = '1' | ''",
        'S="1"|""',
        "S='1'|''",
        '      S=                "1" |""         ',
        "      S=                '1' |''         ",
        """      S=                '1' |""         """,
        """      S=                "1" |''         """,
        # empty string expressed as missing symbol
        # left
        'S =  | "1"',
        "S =  | '1'",
        'S=|"1"',
        "S=|'1'",
        "            S= |           '1'",
        '            S= |           "1"',
        # right
        'S = "1" | ',
        "S = '1' | ",
        'S="1"|',
        "S='1'|",
        '      S=                "1" |         ',
        "      S=                '1' |         ",
    ]
    for ebnf in ebnf_variants:
        grammar = al.Grammar(ebnf_text=ebnf)
        shared.check_grammar(
            grammar,
            ["", "1"],
            [" ", "11", "1 1"],
        )


def test_ebnf_empty_string_recursive():
    ebnf = 'S = S S | "1" | ""'
    grammar = al.Grammar(ebnf_text=ebnf)
    shared.check_grammar(
        grammar,
        ["1", "11", "111"],
        [" ", "1 1"],
    )


def test_ebnf_quantifier_1():
    # + means 1 or more times the preceding symbol or group
    ebnf_text = """
    mid = "0" | "1"+ | Left1 | Right_1
    Left1 = "2"+ "3"
    Right_1 = "4" '5 5'+
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar,
        ["0", "1", "11", "111", "23", "223", "2223", "45 5", "45 55 5"],
        ["", "3", "4", "45"],
    )


def test_ebnf_quantifier_2():
    # * means 0 or more times the preceding symbol or group
    ebnf_text = """
    mid = "0" | "1"* | Left1 | Right_1
    Left1 = "2"* "3"
    Right_1 = "4" '5 5'*
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar,
        ["0", "", "1", "11", "3", "23", "223", "4", "45 5", "45 55 5"],
        ["233", "45"],
    )


def test_ebnf_quantifier_3():
    # ? means 0 or 1 times the preceding symbol or group
    ebnf_text = """
    mid = "0" | "1"? | Left1 | Right_1
    Left1 = "2"? "3"
    Right_1 = "4" '5 5'?
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar,
        ["0", "", "1", "3", "23", "4", "45 5"],
        ["11", "223", "233", "45", "45 55 5"],
    )


def test_ebnf_bracket_1():
    # {} means 0 or more times the included symbol or group
    ebnf_text = """
    mid = {"0" | "1"} | Left1 | Right_1
    Left1 = {"2"} "3"
    Right_1 = "4" {'5 5'}
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar,
        [
            "",
            "0",
            "1",
            "00",
            "01",
            "10",
            "11",
            "1110",
            "3",
            "23",
            "223",
            "4",
            "45 5",
            "45 55 5",
        ],
        ["45"],
    )


def test_ebnf_bracket_2():
    # [] means 0 or 1 times the included symbol or group
    ebnf_text = """
    mid = ["0" | "1"] "1" | Left1 | Right_1
    Left1 = ["2"] "3"
    Right_1 = "4" ['5 5']
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar,
        ["1", "01", "11", "3", "23", "4", "45 5"],
        ["", "00", "10", "1100", "223", "45", "45 55 5"],
        language=["1", "01", "11", "3", "23", "4", "45 5"],
    )


def test_ebnf_bracket_3():
    # () means grouping
    ebnf_text = """
    mid = ("0" | "1")* "1" | Left1 | Right_1
    Left1 = ("2") "3"
    Right_1 = "4" ('5 5')
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar,
        ["1", "01", "11", "001", "011", "101", "111", "0101", "23", "45 5"],
        ["", "00", "10", "110", "223", "4", "3", "45", "45 55 5"],
    )


def test_ebnf_quantifier_bracket_combination():
    ebnf_text = """
    mid = ("a" | "b")? "1" | Left1 | Right_1+
    Left1 = {"2"* "3"}
    Right_1 = "4" ['5 5'+] "8"
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar,
        [
            "1",
            "a1",
            "b1",
            "",
            "3",
            "23",
            "223",
            "2223",
            "2323",
            "223232223",
            "233",
            "48",
            "45 58",
            "45 55 58",
            "484848",
            "4845 55 5845 5848",
        ],
        ["aa1", "45 5", "485 5"],
    )


def test_ebnf_warning_1():
    # A grammar where the sets of nonterminals and terminals overlap
    ebnf_text = """
    S = A B
    A = "B"
    B = "X"
    """
    # Expected warning
    expected_message = '''Problematic grammar specification: The sets of nonterminal and terminal symbols are not disjoint, as required by the mathematical definition of a grammar. This package can deal with it, but in general it is not recommended. Following symbols appear in both sets:
  B as nonterminal <B> and terminal "B"'''
    shared.emits_warning(
        lambda: al.Grammar(ebnf_text=ebnf_text),
        al.warnings.GrammarWarning,
        expected_message,
    )


def test_ebnf_warning_2():
    # A grammar where the same rule is contained more than once
    ebnf_text = """
    S = A | B
    A = "1" | "2"
    B = "a" | "b"
    A = "3" | "4"
    B = "c" | "a"
    """
    # Expected warning
    expected_message = """Problematic grammar specification: Some production rules are redundant. This package can deal with it, but in general it is not recommended. In particular, it introduces bias in random string generation. Following rules are contained more than one time:
  <B> ::= a"""
    shared.emits_warning(
        lambda: al.Grammar(ebnf_text=ebnf_text),
        al.warnings.GrammarWarning,
        expected_message,
    )


def test_ebnf_verbose(capfd):
    ebnf_text = """
A = "1" | {'2' '2' B} | B
B = C ["3" D]
C = "4"+ | "5 6"*
D = ("78")?
"""
    grammar = al.Grammar(ebnf_text=ebnf_text, verbose=True)
    captured_output, captured_err = capfd.readouterr()
    expected_output = """Used symbols
------------

defining_symbol: =
rule_separator_symbol: |
start_nonterminal_symbol: Empty string
end_nonterminal_symbol: Empty string
start_terminal_symbol: "
end_terminal_symbol: "
start_terminal_symbol2: '
end_terminal_symbol2: '

start_group_symbol: (
end_group_symbol: )
start_option_symbol: [
end_option_symbol: ]
start_repeat_symbol: {
end_repeat_symbol: }
quantifier_0_to_1_symbol: ?
quantifier_1_to_n_symbol: +
quantifier_0_to_n_symbol: *

Production list 1
-----------------

Left-hand side
  Text: A
  Interpretation: NT('A')

Right-hand sides:
  Text:  "1" | {'2' '2' B} | B
  Interpretation:
    NT('A') -> [T('1')]
    NT('A') -> [NT('A_§1')]
    NT('A') -> [NT('B')]
    NT('A_§1') -> [NT('A_§1'), NT('A_§0')]
    NT('A_§1') -> [T('')]
    NT('A_§0') -> [T('2'), T('2'), NT('B')]

Production list 2
-----------------

Left-hand side
  Text: B
  Interpretation: NT('B')

Right-hand sides:
  Text:  C ["3" D]
  Interpretation:
    NT('B') -> [NT('C'), NT('B_§0')]
    NT('B_§0') -> [T('3'), NT('D')]
    NT('B_§0') -> [T('')]

Production list 3
-----------------

Left-hand side
  Text: C
  Interpretation: NT('C')

Right-hand sides:
  Text:  "4"+ | "5 6"*
  Interpretation:
    NT('C') -> [NT('C_§0')]
    NT('C') -> [NT('C_§1')]
    NT('C_§0') -> [T('4')]
    NT('C_§0') -> [NT('C_§0'), T('4')]
    NT('C_§1') -> [NT('C_§1'), T('5 6')]
    NT('C_§1') -> [T('')]

Production list 4
-----------------

Left-hand side
  Text: D
  Interpretation: NT('D')

Right-hand sides:
  Text:  ("78")?

  Interpretation:
    NT('D') -> [NT('D_§1')]
    NT('D_§1') -> [NT('D_§0')]
    NT('D_§1') -> [T('')]
    NT('D_§0') -> [T('78')]
"""
    assert captured_output == expected_output

    shared.check_grammar(
        grammar,
        ["", "1", "224", "22224444", "222244443", "22224444378"],
        ["24", "22244"],
    )


def test_ebnf_unusual_symbols():
    # A grammar with weird symbols to mark parts (where do terminals / nonterminals begin and end)
    ebnf_text = r"""
        ZZS% 1314 ZZt_42% ZZabc% ZZQ% ZZla le lu% ZZEND_%
    ZZt_42%   	1314 ""1/ `~ ""2/
    ZZabc% 1314 /11$$ 	`~ ""22/
    ZZQ% 1314 ""111/ `~ ""222/
    ZZla le lu%		1314 ""1111/ `~ /2222$$
    ZZEND_% 1314
    ""/
    """
    grammar = al.Grammar(
        ebnf_text=ebnf_text,
        defining_symbol="1314",
        rule_separator_symbol="`~",
        start_nonterminal_symbol="ZZ",
        end_nonterminal_symbol="%",
        start_terminal_symbol='""',
        end_terminal_symbol="/",
        start_terminal_symbol2="/",
        end_terminal_symbol2="$$",
    )
    shared.check_grammar(
        grammar,
        ["1221112222", "1111111111", "2112221111"],
        ["1", "111111111"],
    )


def test_ebnf_isomorphisms():
    equivalent_formulations = [
        (
            """
            a = "" | 'x'
            """,
            """
            <a> ::= '' | "x"
            """,
        ),
        (
            """
            a = "x" | ''|"y"
            """,
            """
            <a> ::='x'|"" | "y"
            """,
        ),
        (
            """
            a = b |
            b = 'c' | "d" | x
            x = '1' a "2" | "1" "4" b
            """,
            """
            <a> ::= <b> | ''
            <b> ::= 'c' | "d" | <x>
            <x> ::= '1' <a> "2" | '1' "4" <b>
            """,
        ),
        (
            """
            a = b
                  |
            b = "c"
                  | "d"
                  |x
            x = "1" a '2'| '1' "4" b
            """,
            """
            <a> ::= <b>
                  | ''
            <b> ::= 'c'
                  | "d"
                  | <x>
            <x> ::= '1' <a> "2"
                  | '1' "4" <b>
            """,
        ),
    ]

    for ebnf1, ebnf2 in equivalent_formulations:
        grammar1 = al.Grammar(ebnf_text=ebnf1)
        grammar2 = al.Grammar(
            ebnf_text=ebnf2,
            defining_symbol="::=",
            start_nonterminal_symbol="<",
            end_nonterminal_symbol=">",
        )
        assert grammar1.production_rules == grammar2.production_rules


def test_ebnf_fail_due_to_invalid_arguments():
    with pytest.raises(TypeError):
        al.Grammar(ebnf_text=3)
    with pytest.raises(TypeError):
        al.Grammar(ebnf_file=3)
    with pytest.raises(FileNotFoundError):
        al.Grammar(ebnf_file="this is a nonexisting file")


def test_ebnf_fail_due_to_invalid_bnf():
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(ebnf_text="")


def test_ebnf_fail_due_to_missing_lhs():
    ebnf_texts = [
        """
        a = b
        """,
        """
        a = "x" | "y" | b
        b = c | d
        d =
        """,
        """
        a = "x" | "y" | b
        b = c | d
        c = b
        d = a
        e = f | "x"
        """,
    ]
    for text in ebnf_texts:
        with pytest.raises(al.exceptions.GrammarError):
            al.Grammar(ebnf_text=text)
