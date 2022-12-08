import shared

import alogos as al


def test_bnf_algol60_1():
    # https://www.masswerk.at/algol60/report.htm
    bnf_text = """
<ab> ::= ( | [ | <ab> ( | <ab> <d>
<d> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["[(((1(37(", "(12345(", "(((", "[86"],
        ["", "[a"],
        language=["(", "["],
        max_steps=1,
    )


def test_bnf_algol60_2():
    # https://www.masswerk.at/algol60/report.htm
    bnf_text = """
<identifier> ::= <letter> | <identifier> <letter> | <identifier> <digit>
<letter> ::= a | b | c | d | e | f | g | h | i | j | k | l |
        m | n | o | p | q | r | s | t | u | v | w | x | y | z | A |
        B | C | D | E | F | G | H | I | J | K | L | M | N | O | P |
        Q | R | S | T | U | V | W | X | Y | Z
<digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["q", "Soup", "V17a", "a34kTMNs", "MARILYN"],
        ["", "a b"],
    )


def test_bnf_regular():
    # https://en.wikipedia.org/wiki/Context-free_grammar#Examples
    bnf_text = """
<S> ::= a | a<S> | b<S>
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["a", "ba", "abababa"],
        ["", "c"],
        language=["a", "aa", "ba", "aaa", "aba", "baa", "bba"],
        max_steps=3,
    )


def test_bnf_simple_ambiguous():
    # https://en.wikipedia.org/wiki/Context-free_grammar#Derivations_and_syntax_trees
    bnf_text = """
<S> ::= <S> + <S>
<S> ::= 1
<S> ::= a
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["a", "1", "1+a", "a+a", "a+1", "1+1+1", "a+1+a"],
        ["", "a++1", "aa"],
        language=["1", "a", "1+1", "1+a", "a+1", "a+a"],
        max_steps=2,
    )


def test_bnf_number():
    # http://condor.depaul.edu/ichu/csc447/notes/wk3/BNF.pdf on page 7
    bnf_text = """
<number> ::= <digit> | <number> <digit>
<digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(grammar, ["1", "007", "9876543210"], ["", "x", "a4", "123 "])


def test_bnf_natural_language_1():
    # http://athena.ecs.csus.edu/~gordonvs/135/resources/04bnfParseTrees.pdf
    bnf_text = """
<string> ::= <noun-phrase> <predicate>
<noun-phrase> ::= <article> <noun>
<article> ::= the | a | an
<noun> ::= cat | flower
<predicate> ::= jumps | blooms
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(grammar, ["theflowerblooms"], ["", "the flower blooms"])


def test_bnf_natural_language_2():
    # http://web.stanford.edu/class/archive/cs/cs106x/cs106x.1174/assnFiles/recursion/grammarsolver-spec.html
    bnf_text = """
<s>::=<np> <vp>
<np>::=<dp> <adjp> <n>|<pn>
<dp>::=the|a
<adjp>::=<adj>|<adj> <adjp>
<adj>::=big|fat|green|wonderful|faulty|subliminal|pretentious
<n>::=dog|cat|man|university|father|mother|child|television
<pn>::=John|Jane|Sally|Spot|Fred|Elmo
<vp>::=<tv> <np>|<iv>
<tv>::=hit|honored|kissed|helped
<iv>::=died|collapsed|laughed|wept
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["thefatuniversitylaughed", "Elmokissedagreenpretentioustelevision"],
        ["", "Elmo kissed a green pretentious television"],
    )


def test_bnf_expr_1():
    # http://athena.ecs.csus.edu/~gordonvs/135/resources/04bnfParseTrees.pdf
    bnf_text = """
<exp> ::= <exp> +<exp>| <exp> *<exp> | (<exp> )| <digit>
<digit> ::= 0| 1| 2| 3| 4| 5| 6| 7|8| 9
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(grammar, ["3+(4*4+(2*7))", "6+3*4"], ["", "3++4", "*3"])


def test_bnf_expr_2():
    # http://web.stanford.edu/class/archive/cs/cs106x/cs106x.1174/assnFiles/recursion/grammarsolver-spec.html
    bnf_text = """
    <expression> ::= <term> | <expression> + <term> | <expression> - <term>
    <term> ::= <factor> | <term> * <factor> | <term> / <factor>
    <factor> ::= <element> | - <element>
    <element> ::= <NUMBER> | <IDENTIFIER> | ( <expression> )

    <IDENTIFIER> ::= a | b
    <NUMBER> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(grammar, ["3+(4*4+(2*7))", "6+3*4"], ["", "3++4", "*3"])


def test_bnf_course_codes():
    # http://www.cs.utsa.edu/~wagner/CS3723/grammar/examples2.html at Grammar #2
    bnf_text = """
<coursecode>   ::= <acadunit> <coursenumber>
<acadunit>     ::= <letter> <letter> <letter>
<coursenumber> ::= <year> <semesters> <digit> <digit>
<year>         ::= <ugrad> | <grad>
<ugrad>        ::= 0 | 1 | 2 | 3 | 4
<grad>         ::= 5 | 6 | 7 | 9
<semesters>    ::= <onesemester> | <twosemesters>
<onesemester>  ::= <frenchone> | <englishone> | <bilingual>
<frenchone>    ::= 5 | 7
<englishone>   ::= 1 | 3
<bilingual>    ::= 9
<twosemesters> ::= <frenchtwo> | <englishtwo>
<frenchtwo>    ::= 6 | 8
<englishtwo>   ::= 2 | 4
<digit>        ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
<letter>       ::= a | b | c | d | e | f | g | h | i | j | k | l | m
                 | n | o | p | q | r | s | t | u | v | w | x | y | z
                 | A | B | C | D | E | F | G | H | I | J | K | L | M
                 | N | O | P | Q | R | S | T | U | V | W | X | Y | Z
"""
    grammar = al.Grammar(bnf_text=bnf_text)
    shared.check_grammar(
        grammar,
        ["CSI3125", "MAT2743", "PHY1200", "EPI6581", "CSI9999"],
        ["", "CSCI3125", "MA2743", "PHY120", "EPI65811"],
    )
