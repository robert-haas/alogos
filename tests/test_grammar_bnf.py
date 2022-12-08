import os

import pytest
import shared

import alogos as al


TESTFILE_DIR = os.path.dirname(shared.get_path_of_this_file())
IN_DIR = os.path.join(TESTFILE_DIR, "in")


# Read BNF from file


def test_bnf_file_simple():
    filepath = os.path.join(IN_DIR, "bnf_examples", "simple_test_grammar.bnf")
    grammar = al.Grammar(bnf_file=filepath)
    shared.check_grammar(grammar, ["7xR", "8yS", "9xS"], ["", "X", "21"])


def test_bnf_file_symbolic_regression():
    filepath = os.path.join(IN_DIR, "bnf_examples", "symbolic_regression.bnf")
    grammar = al.Grammar(bnf_file=filepath)
    shared.check_grammar(
        grammar, ["x", "y", "x+x", "y*x", "x/y-x*y"], ["", "+", "+x", "X"]
    )


def test_bnf_file_syntax_for_bnf():
    # https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form#Further_examples
    # Required modifications: <EOL> :: "\n"
    filepath = os.path.join(IN_DIR, "bnf_examples", "bnf_syntax_quoted.bnf")
    grammar = al.Grammar(
        bnf_file=filepath,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )

    string1 = """<A> ::= "1" | <B>
<B> ::= "3" | "4"
"""
    string2 = """<PostalNumber> ::= "123" | <OrIsit> "456"
<OrIsit> ::= <OrIsit> "7" | "8"
"""

    shared.check_grammar(
        grammar, [string1, string2], ["", '<A ::= "1" | "2"', "<A> ::"]
    )


# Read BNF from string


def test_bnf_simple_1():
    # Nonterminals enclosed, terminals standalone
    bnf_text = """
<start> ::= <greeting_1> | <greeting 2> | hello world
<greeting_1> ::= hi universe | <greeting 2>
<greeting 2> ::= hey everything
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["helloworld", "hiuniverse", "heyeverything"],
        ["hello", "world", "hi", "universe", "hey", "everything"],
        language=["helloworld", "hiuniverse", "heyeverything"],
    )


def test_bnf_simple_2():
    # Nonterminals standalone, terminals enclosed
    bnf_text = """
start := greeting_1 | greeting-2 | "hello world"
greeting_1 := "hi universe" | greeting-2
greeting-2 := 'hey everything'
"""
    grammar = al.Grammar(
        bnf_text=bnf_text,
        defining_symbol=":=",
        start_nonterminal_symbol="",
        end_nonterminal_symbol="",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        ["hello world", "hi universe", "hey everything"],
        ["hello", "world", "hi", "universe", "hey", "everything"],
        language=["hello world", "hi universe", "hey everything"],
    )


def test_bnf_simple_3():
    # Nonterminals enclosed, terminals enclosed
    bnf_text = """
{{start] := {{greeting_1] | {{greeting 2] | "hello world"
{{greeting_1] := "hi universe" | {{greeting 2]
{{greeting 2] := 'hey everything'
"""
    grammar = al.Grammar(
        bnf_text=bnf_text,
        defining_symbol=":=",
        start_nonterminal_symbol="{{",
        end_nonterminal_symbol="]",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        ["hello world", "hi universe", "hey everything"],
        ["hello", "world", "hi", "universe", "hey", "everything"],
        language=["hello world", "hi universe", "hey everything"],
    )


def test_bnf_single_special_symbol_1():
    bnf_text = r"""
    <EOL> ::= "
"
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        [
            "\n",
            r"""
""",
        ],
        [r"\n"],
    )


def test_bnf_single_special_symbol_2():
    bnf_text = r"""
    <TAB> ::= "	"
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        ["\t", r"	"],
        [r"\t"],
    )


def test_bnf_single_special_symbol_3():
    bnf_text = r"""
    <S> ::= "\"
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        ["\\"],  # single backslash is recognized
        [r"\\"],  # double backslash is not
    )


@pytest.mark.xfail(strict=True, reason="Probably an error in Earley parser of Lark.")
def test_bnf_single_special_symbol_4():
    bnf_text = r"""
    <S> ::= "\\"
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        [r"\\"],  # double backslash should be recognized
        ["\\"],  # single backslash should not
    )


def test_bnf_single_special_symbol_5():
    bnf_text = r"""
    <S> ::= "\\\"
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        [r"\\"],
        [r"\\\\"],
    )


@pytest.mark.xfail(strict=True, reason="Probably an error in Earley parser of Lark.")
def test_bnf_special_sequence_1():
    bnf_text = r"""
    <S> ::= "a\\b"
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        [r"a\\b"],
        [r"a\b"],
    )


def test_bnf_special_sequence_2():
    bnf_text = r"""
    <S> ::= "a\b"
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        [r"a\b"],
        [r"a\\b"],
    )


def test_bnf_verbose(capfd):
    bnf_text = """
<A> ::= 1 | 2 | <B>
<B> ::= <C> 3 <D>
<C> ::= 4 | 5 6
<D> ::= 78
"""
    grammar = al.Grammar(bnf_text=bnf_text, verbose=True)
    captured_output, captured_err = capfd.readouterr()
    expected_output = """Used symbols
------------

defining_symbol: ::=
rule_separator_symbol: |
start_nonterminal_symbol: <
end_nonterminal_symbol: >
start_terminal_symbol: Empty string
end_terminal_symbol: Empty string
start_terminal_symbol2: Empty string
end_terminal_symbol2: Empty string

Production list 1
-----------------

Left-hand side
  Text: <A>
  Interpretation: NT('A')

Right-hand sides:
  Text:  1 | 2 | <B>
  Interpretation:
    NT('A') -> [T('1')]
    NT('A') -> [T('2')]
    NT('A') -> [NT('B')]

Production list 2
-----------------

Left-hand side
  Text: <B>
  Interpretation: NT('B')

Right-hand sides:
  Text:  <C> 3 <D>
  Interpretation:
    NT('B') -> [NT('C'), T('3'), NT('D')]

Production list 3
-----------------

Left-hand side
  Text: <C>
  Interpretation: NT('C')

Right-hand sides:
  Text:  4 | 5 6
  Interpretation:
    NT('C') -> [T('4')]
    NT('C') -> [T('5'), T('6')]

Production list 4
-----------------

Left-hand side
  Text: <D>
  Interpretation: NT('D')

Right-hand sides:
  Text:  78

  Interpretation:
    NT('D') -> [T('78')]
"""
    assert captured_output == expected_output

    shared.check_grammar(
        grammar,
        ["1", "2", "4378", "56378"],
        ["", "13"],
        language=["1", "2", "4378", "56378"],
    )


def test_bnf_problematic():
    # A grammar with redundant rules and symbols used both as nonterminal and terminal
    bnf_text = """
    <S> ::= <A> | 1 2 | <B> 3
    <A> ::= <B> 4 | 5 | B
    <B> ::= 6 7 | <A> <B> | S
    <A> ::= <B> 4
    """
    grammar = al.Grammar(bnf_text=bnf_text, verbose=True)
    shared.check_grammar(
        grammar,
        ["5", "12", "673", "674", "5674", "555673"],
        ["", "X", "21"],
    )


def test_bnf_unusual_symbols_1():
    # A grammar with weird symbols to mark parts
    bnf_text = r"""
        ZZS% 1314 ZZt_42% ZZabc% ZZQ% ZZla le lu% ZZEND_%
    ZZt_42%   	1314 ""1/ `~ ""2/
    ZZabc% 1314 /11$$ 	`~ ""22/
    ZZQ% 1314 ""111/ `~ ""222/
    ZZla le lu%		1314 ""1111/ `~ /2222$$
    ZZEND_% 1314
    ""/
    """
    grammar = al.Grammar(
        bnf_text=bnf_text,
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


def test_bnf_unusual_symbols_2():
    bnf_text = """
(s] /       "1" + {2} + "a" "b" + (x]
\t(x]/\r

{3}
"""
    # Grammar creation
    grammar = al.Grammar(
        bnf_text=bnf_text,
        defining_symbol="/",
        rule_separator_symbol="+",
        start_nonterminal_symbol="(",
        end_nonterminal_symbol="]",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="{",
        end_terminal_symbol2="}",
    )

    shared.check_grammar(grammar, ["1", "2", "ab", "3"], ["", "12", "12", "2a", "2a"])


def test_bnf_unusual_symbols_3():
    bnf_text = """
<A> ::= <B><C><D>
<B> ::= "7" | '8' | "9"
<C> ::= 'x' | 'y'
<D> ::= "R" | 'S'
"""
    # Grammar creation
    grammar = al.Grammar(
        bnf_text=bnf_text,
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
    )
    shared.check_grammar(
        grammar,
        ["7xR", "7yS", "8yS", "9xS"],
        ["", "12", "12", "2a", "2a"],
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


def test_bnf_unusual_symbols_4():
    # http://blog.hvidtfeldts.net/index.php/2008/12/grammars-for-generative-art-part-i/
    bnf_text = """
[English Sentence] = [Simple Sentence]
[Simple Sentence] = [Declarative Sentence]
[Declarative Sentence] = [subject] [predicate]
[subject] = [simple subject]
[simple subject] = [nominative personal pronoun]
[nominative personal pronoun] = "I" | "you" | "he" | "she" | "it" | "we" | "they"
[predicate] = [verb]
[verb] = [linking verb]
[linking verb] = "am" |"are" |"is" | "was"| "were"
"""
    # Grammar creation
    grammar = al.Grammar(
        bnf_text=bnf_text,
        defining_symbol="=",
        start_nonterminal_symbol="[",
        end_nonterminal_symbol="]",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
    )
    shared.check_grammar(
        grammar,
        ["theyare", "wewere", "sheis"],
        ["", "CSCI3125", "MA2743", "PHY120", "EPI65811"],
    )


def test_bnf_palindromes_even_with_interspersed_whitespaces():
    # https://en.wikipedia.org/wiki/Context-free_grammar#Examples
    bnf_text = """
    \n\r\t <S>\r\t\n
    ::= a<S>a
    \f
    <S> ::=\n\r\tb<S>b
    \v
    <S> ::= \n\r\t
    \t
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        [
            "",
            "aa",
            "bb",
            "aaaa",
            "abba",
            "baab",
            "bbbb",
            "aabbaa",
            "aaabababbabaababbababaaa",
        ],
        ["aaba", "aabbaab"],
        language=["", "aa", "bb", "aaaa", "abba", "baab", "bbbb"],
        max_steps=3,
    )


def test_bnf_parentheses_with_interspersed_whitespaces():
    # https://en.wikipedia.org/wiki/Context-free_grammar#Examples
    bnf_text = """<S> ::= <S><S>
        \t <S> ::= (<S>) \r \n
    <S>\t::= ()"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["()", "((()))()(())(((())))"],
        ["", "(", ")", "(((()))()(())(((())))"],
        language=["()", "(())", "()()"],
        max_steps=2,
    )


def test_bnf_isomorphisms():
    equivalent_formulations = [
        (
            """
            <a> ::= | x
            """,
            """
            <a> ::= '' | 'x'
            """,
        ),
        (
            """
            <a> ::= x | | y
            """,
            """
            <a> ::= 'x' | '' | 'y'
            """,
        ),
        (
            """
            <a> ::= <b> |
            <b> ::= c | d | <x>
            <x> ::= 1 <a> 2 | 1 4 <b>
            """,
            """
            <a> ::= <b> | ''
            <b> ::= 'c' | "d" | <x>
            <x> ::= '1' <a> "2" | '1' "4" <b>
            """,
        ),
        (
            """
            <a> ::= <b>
                  |
            <b> ::= c
                  | d
                  | <x>
            <x> ::= 1 <a> 2
                  | 1 4 <b>
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

    for bnf_nonquoted, bnf_quoted in equivalent_formulations:
        grammar1 = al.Grammar(bnf_text=bnf_nonquoted)
        grammar2 = al.Grammar(
            bnf_text=bnf_quoted,
            start_terminal_symbol='"',
            end_terminal_symbol='"',
            start_terminal_symbol2="'",
            end_terminal_symbol2="'",
        )
        assert grammar1.production_rules == grammar2.production_rules


def test_bnf_fail_due_to_invalid_arguments():
    with pytest.raises(TypeError):
        al.Grammar(bnf_text=3)
    with pytest.raises(TypeError):
        al.Grammar(bnf_file=3)
    with pytest.raises(FileNotFoundError):
        al.Grammar(bnf_file="this is a nonexisting file")


def test_bnf_fail_due_to_invalid_bnf():
    with pytest.raises(Exception):
        al.Grammar(bnf_text="")
    with pytest.raises(Exception):
        al.Grammar(bnf_text="A := x | y")


def test_bnf_fail_due_to_missing_lhs():
    bnf_texts = [
        """
        <a> ::= <b>
        """,
        """
        <a> ::= x | y | <b>
        <b> ::= <c> | <d>
        <d> ::=
        """,
        """
        <a> ::= x | y | <b>
        <b> ::= <c> | <d>
        <c> ::= <b>
        <d> ::= <a>
        <e> ::= <f> | x
        """,
    ]
    for text in bnf_texts:
        with pytest.raises(Exception):
            al.Grammar(bnf_text=text)


def test_bnf_and_ebnf_with_multiple_delimiters():
    ebnf_text = """
    <S> = XaY | b
    """
    grammar = al.Grammar(
        ebnf_text=ebnf_text,
        start_nonterminal_symbol="<",
        end_nonterminal_symbol=">",
        start_terminal_symbol="",
        end_terminal_symbol="",
        start_terminal_symbol2="X",
        end_terminal_symbol2="Y",
    )
    assert grammar.recognize_string("a")
    # TODO assert grammar.recognize_string('b')


def test_bnf_and_ebnf_fail_due_to_delimiters():
    # BNF
    bnf_text = """
    <S> ::= 1 | 2
    """
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(
            bnf_text=bnf_text,
            start_nonterminal_symbol="",
            end_nonterminal_symbol="",
            start_terminal_symbol="",
            end_terminal_symbol="",
            start_terminal_symbol2="",
            end_terminal_symbol2="",
        )

    bnf_text = """
    <S> ::= 1 | 2
    """
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(
            bnf_text=bnf_text,
            start_nonterminal_symbol="1",
            end_nonterminal_symbol="",
            start_terminal_symbol="",
            end_terminal_symbol="",
            start_terminal_symbol2="",
            end_terminal_symbol2="",
        )

    bnf_text = """
    <S> ::= 1 | 2
    """
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(
            bnf_text=bnf_text,
            start_nonterminal_symbol="",
            end_nonterminal_symbol="",
            start_terminal_symbol="1",
            end_terminal_symbol="",
            start_terminal_symbol2="",
            end_terminal_symbol2="",
        )

    bnf_text = """
    <S> ::= 1 | 2
    """
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(
            bnf_text=bnf_text,
            start_nonterminal_symbol="",
            end_nonterminal_symbol="",
            start_terminal_symbol="",
            end_terminal_symbol="",
            start_terminal_symbol2="",
            end_terminal_symbol2="1",
        )

    # EBNF
    ebnf_text = """
    S = "1" | "2"
    """
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(
            ebnf_text=ebnf_text,
            start_nonterminal_symbol="",
            end_nonterminal_symbol="",
            start_terminal_symbol="",
            end_terminal_symbol="",
            start_terminal_symbol2="",
            end_terminal_symbol2="",
        )

    ebnf_text = """
    S = "1" | "2"
    """
    with pytest.raises(al.exceptions.GrammarError):
        al.Grammar(
            ebnf_text=ebnf_text,
            start_nonterminal_symbol="<",
            end_nonterminal_symbol=">",
            start_terminal_symbol="",
            end_terminal_symbol="",
            start_terminal_symbol2="",
            end_terminal_symbol2="",
        )


def test_bnf_fails_during_writing():
    bnf_text = """
    <S> ::= 0 | <A>
    <A> ::= 1
    """
    grammar = al.Grammar(bnf_text=bnf_text)

    # Working
    grammar.to_bnf_text(start_nonterminal_symbol="X", end_nonterminal_symbol="Y")
    grammar.to_bnf_text(
        start_nonterminal_symbol="",
        end_nonterminal_symbol="",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
    )

    # Failing
    with pytest.raises(al.exceptions.GrammarError):
        grammar.to_bnf_text(start_nonterminal_symbol="X", end_nonterminal_symbol="A")
    with pytest.raises(al.exceptions.GrammarError):
        grammar.to_bnf_text(start_terminal_symbol="X", end_terminal_symbol="0")
    with pytest.raises(al.exceptions.GrammarError):
        grammar.to_bnf_text(start_terminal_symbol2="X", end_terminal_symbol2="0")
    with pytest.raises(al.exceptions.GrammarError):
        grammar.to_bnf_text(
            start_terminal_symbol="X",
            end_terminal_symbol="0",
            start_terminal_symbol2="",
            end_terminal_symbol2="",
        )
    with pytest.raises(al.exceptions.GrammarError):
        grammar.to_bnf_text(
            start_terminal_symbol="<",
            end_terminal_symbol="0",
            start_terminal_symbol2="",
            end_terminal_symbol2="",
        )
