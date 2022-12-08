# flake8: noqa: W191, E101
import shared

import alogos as al


def test_ebnf_farrell_1():
    # http://www.cs.man.ac.uk/~pjj/farrell/comp2.html#EBNF
    ebnf_text = """
    S :== 'a' [B]
    B :== {'a'}
    """
    grammar = al.Grammar(ebnf_text=ebnf_text, defining_symbol=":==")
    shared.check_grammar(grammar, ["a", "aa", "aaa", "aaaa"], ["", "a ", "a a", " a"])


def test_ebnf_farrell_2():
    # http://www.cs.man.ac.uk/~pjj/farrell/comp2.html#EBNF
    # Required modifications: + to '+', - to '-', * to '*', / to '/'
    ebnf_text = """
    S :== EXPRESSION
    EXPRESSION :== TERM | TERM { ['+'|'-'] TERM] }
    TERM :== FACTOR | FACTOR { ['*'|'/'] FACTOR] }
    FACTOR :== NUMBER | '(' EXPRESSION ')'
    NUMBER :== '1' | '2' | '3' | '4' | '5' |
               '6' | '7' | '8' | '9' | '0' |
               '1' NUMBER | '2' NUMBER | '3' NUMBER |
               '4' NUMBER | '5' NUMBER | '6' NUMBER |
               '7' NUMBER | '8' NUMBER | '9' NUMBER | '0' NUMBER
    """
    grammar = al.Grammar(ebnf_text=ebnf_text, defining_symbol=":==")
    shared.check_grammar(grammar, ["42", "(2*5)+3/19"], ["", "-1"])


def test_ebnf_mod():
    # https://jakobandersen.github.io/mod/dataDesc/dataDesc.html
    # Required modifications: Provide quoteEscapedString and int, add some whitespaces in terminals
    ebnf_text = r"""
graphGML ::=  'graph [' (node | edge)* ']'
node     ::=  ' node [id ' int ' label ' quoteEscapedString ']'
edge     ::=  ' edge [source ' int ' target ' int ' label ' quoteEscapedString ']'
int ::= digit+
digit ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
quoteEscapedString ::= '"C"' | '"O"'
"""
    grammar = al.Grammar(ebnf_text=ebnf_text, defining_symbol="::=")
    shared.check_grammar(
        grammar,
        ["graph []", 'graph [ node [id 15 label "C"] node [id 22 label "O"]]'],
        ["", "graph [ ]"],
    )


def test_ebnf_tiny_c():
    # https://tomassetti.me/ebnf/#examples
    # Required modifications: Regex patterns [a-z]+ [0-9]+ and [ \r\n\t] -> skip
    ebnf_text = r"""
program
   : statement+
   ;

statement
   : 'if' paren_expr statement
   | 'if' paren_expr statement 'else' statement
   | 'while' paren_expr statement
   | 'do' statement 'while' paren_expr ';'
   | '{' statement* '}'
   | expr ';'
   | ';'
   ;

paren_expr
   : '(' expr ')'
   ;

expr
   : test
   | id '=' expr
   ;

test
   : sum
   | sum '<' sum
   ;

sum
   : term
   | sum '+' term
   | sum '-' term
   ;

term
   : id
   | integer
   | paren_expr
   ;

id
   : STRING
   ;

integer
   : INT
   ;


STRING
   : LETTER+
   ;


INT
   : DIGIT+
   ;

LETTER: "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m"
      | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
      | "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M"
      | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
DIGIT: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
WS: ' ' | '\r' | '\n' | '\t';
    """
    grammar = al.Grammar(ebnf_text=ebnf_text, defining_symbol=":")
    shared.check_grammar(
        grammar, [";", "{}", "if(helloworld);", "while(a+b<10);"], ["", "x", "}{"]
    )


def test_ebnf_algebraic_expression():
    # https://karmin.ch/ebnf/examples
    # Required modification: digit made explicit, i.e. replaced ... by actual numbers
    ebnf_text = """
expression = term  { ("+" | "-") term} .
term       = factor  { ("*"|"/") factor} .
factor     = constant | variable | "("  expression  ")" .
variable   = "x" | "y" | "z" .
constant   = digit {digit} .
digit      = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" .
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(
        grammar, ["0", "2-1", "(x+y)*7", "1+x/y*z+44"], ["", "+", "aaba", "aabbaab"]
    )


def test_ebnf_decimal_numbers():
    # http://condor.depaul.edu/ichu/csc447/notes/wk3/BNF.pdf
    ebnf_text = """
<expr> := '-'? <digit>+ ('.' <digit>+)?
<digit> := '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
    """
    grammar = al.Grammar(
        ebnf_text=ebnf_text,
        defining_symbol=":=",
        start_nonterminal_symbol="<",
        end_nonterminal_symbol=">",
    )
    shared.check_grammar(grammar, ["-7", "3.14", "-0.1"], ["", "+-1", "1+", "1."])


def test_ebnf_integer():
    # https://www.ics.uci.edu/~pattis/ICS-31/lectures/ebnf.pdf
    # Required modifications: quote terminals, swap rows to have start symbol in first row
    ebnf_text = """
integer = ["+"|"-"] digit {digit}
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
"""
    grammar = al.Grammar(ebnf_text=ebnf_text)
    shared.check_grammar(grammar, ["+1", "-15"], ["", "+", "0.2"])


def test_java_float_literals_for_antlr():
    # https://github.com/antlr/grammars-v4/blob/master/java8/Java8.g4
    # Required modifications:
    # - FloatTypeSuffix [fFdD] to quoted terminals
    # - ExponentIndicator [eE] to quoted terminals
    # - NonZeroDigit [1-9] to quoted terminals
    # - Sign [+-] to quoted terminals

    ebnf_text = """
DecimalFloatingPointLiteral
	:	Digits '.' Digits? ExponentPart? FloatTypeSuffix?
	|	'.' Digits ExponentPart? FloatTypeSuffix?
	|	Digits ExponentPart FloatTypeSuffix?
	|	Digits FloatTypeSuffix
;

FloatTypeSuffix
	:	'f' | 'F' | 'd' | 'D'
;

ExponentPart
	:	ExponentIndicator SignedInteger
	;

ExponentIndicator
	:	'e' | 'E'
;

SignedInteger
	:	Sign? Digits
;

Sign
	:	'+' | '-'
;

Digits
	:	Digit (DigitsAndUnderscores? Digit)?
;

Digit
	:	'0'
	|	NonZeroDigit
;

NonZeroDigit
	:	'1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
;

DigitsAndUnderscores
	:	DigitOrUnderscore+
;

DigitOrUnderscore
	:	Digit
	|	'_'
;
"""
    grammar = al.Grammar(ebnf_text=ebnf_text, defining_symbol=":")
    shared.check_grammar(
        grammar, ["7e-09", ".000_3f", "0.14f", "1_000_000d"], [".", "a"]
    )


def test_python_import_statement():
    # https://docs.python.org/3/reference/grammar.html
    # Required modifications: Provide a rule for NAME
    # Problem: Whitespaces not encoded
    ebnf_text = """
import_stmt: import_name | import_from
import_name: 'import' dotted_as_names

import_from: ('from' (('.' | '...')* dotted_name | ('.' | '...')+)
              'import' ('*' | '(' import_as_names ')' | import_as_names))
import_as_name: NAME ['as' NAME]
dotted_as_name: dotted_name ['as' NAME]
import_as_names: import_as_name (',' import_as_name)* [',']
dotted_as_names: dotted_as_name (',' dotted_as_name)*
dotted_name: NAME ('.' NAME)*

NAME: LETTER+
LETTER: "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m"
      | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
      | "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M"
      | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
"""
    grammar = al.Grammar(ebnf_text=ebnf_text, defining_symbol=":")
    shared.check_grammar(
        grammar, ["importSfromE", "from.importLib"], ["import S from E"]
    )


def test_rust_number_literals():
    # https://doc.rust-lang.org/grammar.html -> moved, changed, but currently not as easily usable
    # http://tack.sourceforge.net/olddocs/LLgen.html
    # Required modifications: Nothing except selection of suitable rows
    # (Newer version: https://doc.rust-lang.org/reference/tokens.html#number-literals)
    ebnf_text = """
num_lit : nonzero_dec [ dec_digit | '_' ] * float_suffix ?
        | '0' [       [ dec_digit | '_' ] * float_suffix ?
              | 'b'   [ '1' | '0' | '_' ] +
              | 'o'   [ oct_digit | '_' ] +
              | 'x'   [ hex_digit | '_' ] +  ] ;

float_suffix : [ exponent | '.' dec_lit exponent ? ] ? ;

exponent : ['E' | 'e'] ['-' | '+' ] ? dec_lit ;
dec_lit : [ dec_digit | '_' ] + ;
oct_digit : '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' ;
hex_digit : 'a' | 'b' | 'c' | 'd' | 'e' | 'f'
          | 'A' | 'B' | 'C' | 'D' | 'E' | 'F'
          | dec_digit ;
dec_digit : '0' | nonzero_dec ;
nonzero_dec: '1' | '2' | '3' | '4'
           | '5' | '6' | '7' | '8' | '9' ;
"""
    grammar = al.Grammar(ebnf_text=ebnf_text, defining_symbol=":")
    shared.check_grammar(
        grammar, ["1", "1_000", "0b011", "3.14e10", "0xFF", "0o77"], ["", "-4"]
    )
