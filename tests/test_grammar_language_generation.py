import alogos as al

import shared


def test_finite_minimal_bnf_1():
    bnf_text = """
    <S> ::=
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = ['']
    shared.check_language(grammar, expected_strings)


def test_finite_minimal_bnf_2():
    bnf_text = """
    <S> ::= 1
          |
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = ['', '1']
    shared.check_language(grammar, expected_strings)


def test_finite_minimal_ebnf_1():
    ebnf_text = """
    S = ["1"]
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    expected_strings = ['', '1']
    shared.check_language(grammar, expected_strings)


def test_finite_minimal_ebnf_2():
    ebnf_text = """
    S = "1"?
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    expected_strings = ['', '1']
    shared.check_language(grammar, expected_strings)


def test_finite_minimal_ebnf_3():
    ebnf_text = """
    S = "1" | ""
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    expected_strings = ['', '1']
    shared.check_language(grammar, expected_strings)


def test_finite_minimal_ebnf_4():
    ebnf_text = """
    S = [[[["1"]?]??]]
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    expected_strings = ['', '1']
    shared.check_language(grammar, expected_strings)


def test_finite_simple():
    bnf_text = """
    <A> ::= <B><C><D>
    <B> ::= 1 | 2
    <C> ::= x | y
    <D> ::= R | S
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = [
        '1xR',
        '1xS',
        '1yR',
        '1yS',
        '2xR',
        '2xS',
        '2yR',
        '2yS',
    ]
    shared.check_language(grammar, expected_strings)


def test_finite_advanced():
    bnf_text = """
    <A> ::= <B><C> | x
    <B> ::= <D> | <D> <D> |
    <C> ::= <D> <E> | <F> |
    <D> ::= 1
    <E> ::= 2
    <F> ::= <G> | <H>
    <G> ::= <E> <E>
    <H> ::= <E> <D>
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = [
        # A => x
        'x',

        # A => BC
        # .B => D | DD | eps => 1 | 11 | eps
        # ..C => DE => 12
        '1'+'12',
        '11'+'12',
        ''+'12',

        # ..C => F => G | H => EE | ED => 22 | 21
        '1'+'22',
        '11'+'22',
        ''+'22',
        '1'+'21',
        '11'+'21',
        ''+'21',

        # ..C => epsilon
        '1'+'',
        '11'+'',
        ''+'',  # empty string contained in the language
    ]
    shared.check_language(grammar, expected_strings)


def test_infinite_simple_1():
    bnf_text = """
    <S> ::= 1 | 2 | <S> <S>
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = ['1', '2']
    shared.check_language(grammar, expected_strings, max_steps=1)
    expected_strings = ['1', '2', '11', '12', '21', '22']
    shared.check_language(grammar, expected_strings, max_steps=2)
    expected_strings = ['1', '2', '11', '12', '21', '22',
                        '111', '112', '121', '122',      # 1 and rest
                        '211', '212', '221', '222',      # 2 and rest
                        '1111', '1112', '1121', '1122',  # 11 and rest
                        '1211', '1212', '1221', '1222',  # 12 and rest
                        '2111', '2112', '2121', '2122',  # 21 and rest
                        '2211', '2212', '2221', '2222']  # 22 and rest
    shared.check_language(grammar, expected_strings, max_steps=3)


def test_infinite_simple_2():
    ebnf_text = """
    S = "1"*
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    expected_strings = ['']
    shared.check_language(grammar, expected_strings, max_steps=2)
    expected_strings = ['', '1']
    shared.check_language(grammar, expected_strings, max_steps=3)
    expected_strings = ['', '1', '11']
    shared.check_language(grammar, expected_strings, max_steps=4)


def test_infinite_simple_3():
    ebnf_text = """
    S = {"1"} "2"*
    """
    grammar = al.Grammar(ebnf_text=ebnf_text)
    expected_strings = ['']
    shared.check_language(grammar, expected_strings, max_steps=2)
    expected_strings = ['', '1', '2', '12']
    shared.check_language(grammar, expected_strings, max_steps=3)
    expected_strings = ['', '1', '2', '12', '11', '22', '112', '122', '1122']
    shared.check_language(grammar, expected_strings, max_steps=4)


def test_infinite_advanced():
    bnf_text = """
    <S> ::= 1 | <A> <B>
    <A> ::= 2 | <A> <A>
    <B> ::= <B> <B> | 3 <C> | 4
    <C> ::= 5
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = ['1']
    shared.check_language(grammar, expected_strings, max_steps=1)
    expected_strings = ['1', '24']
    shared.check_language(grammar, expected_strings, max_steps=2)
    expected_strings = ['1', '24',              # A has 2 and 22; B has 4, 35 and 44
                        '235', '244',           # 2 and rest
                        '224', '2235', '2244']  # 22 and rest


def test_parameter_max_steps_where_default_value_emits_a_warning_when_being_reached(caplog):
    grammar = al.Grammar(bnf_text='<S> ::= 0 | <S>1')
    expected_message = (
        'Language generation stopped due to reaching max_steps=20000, '
        'but it did not produce all possible strings yet. '
        'To explore it further, the max_steps parameter can be increased.')
    shared.emits_warning(lambda: grammar.generate_language(), caplog, expected_message)
    shared.emits_no_warning(lambda: grammar.generate_language(max_steps=20_000), caplog)


def test_parameter_sort_order():
    bnf = """
    <S> ::= 0 <A> | 1 <B> | a | aa
    <A> ::= 2 | 22
    <B> ::= 4 | 44
    """
    grammar = al.Grammar(bnf_text=bnf)
    l1 = grammar.generate_language(sort_order='discovered')
    assert l1 == ['a', 'aa', '02', '022', '14', '144']
    l2 = grammar.generate_language(sort_order='lex')
    assert l2 == ['02', '022', '14', '144', 'a', 'aa']
    l3 = grammar.generate_language(sort_order='shortlex')
    assert l3 == ['a', '02', '14', 'aa', '022', '144']


def test_parameter_verbose():
    bnf = """
    <S> ::= 0 <A> | 1 <B> | x |
    <A> ::= 2 | 3
    <B> ::= 4 | 5
    """
    grammar = al.Grammar(bnf_text=bnf)
    lang = grammar.generate_language(verbose=True)
    assert set(lang) == set(['', 'x', '02', '03', '14', '15'])


def test_parameter_return_details():
    bnf_text = """
    <A> ::= <B><C><D>
    <B> ::= 1 | 2
    <C> ::= x | y
    <D> ::= R | S
    """
    grammar = al.Grammar(bnf_text=bnf_text)
    expected_strings = ['1xR', '1xS', '1yR', '1yS', '2xR', '2xS', '2yR', '2yS']
    generated_strings_dict = grammar.generate_language(return_details=True)

    def pre(collection):
        return list(sorted(collection))

    assert pre(generated_strings_dict['A']) == pre(expected_strings)
    assert pre(generated_strings_dict['B']) == pre(['1', '2'])
    assert pre(generated_strings_dict['C']) == pre(['x', 'y'])
    assert pre(generated_strings_dict['D']) == pre(['R', 'S'])
