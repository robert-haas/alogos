"""Data structures for a grammar, derivation tree and their parts."""

import collections as _collections
import copy as _copy
import json as _json
import random as _random
from itertools import chain as _chain

from ordered_set import OrderedSet as _OrderedSet

from .. import exceptions as _exceptions
from .. import warnings as _warnings
from .._utilities import argument_processing as _ap
from .._utilities.operating_system import NEWLINE as _NEWLINE


class Grammar:
    """Context-free grammar (CFG) for defining a formal language.

    The role of this class is explained by first introducing
    basic concepts from formal language theory and then
    describing the main tasks a grammar can be used for.

    1. Technical terms from formal language theory

        - A **grammar** is a mathematical device to define a formal
          language.
        - A **language** is a finite or infinite set of strings.
        - A **string** is a finite sequence of symbols that all belong
          to a single alphabet.
        - An **alphabet** is a finite set of symbols.
        - In mathematical terms, a grammar is usually defined as a
          tuple ``(N, T, P, S)`` where

            - ``N`` is a set of **nonterminal symbols**, also known
              as **variables**.
            - ``T`` is a set of **terminal symbols**, which has no
              overlap with ``N``.
            - ``S`` is the **start symbol**, which is a nontermminal
              symbol and therefore an element of ``N``.
            - ``P`` is a set of **production rules**, also known as
              **rewrite rules**. Beginning from the start symbol,
              these rules can be applied until only terminal symbols
              are left, which then form a string of the grammar's
              language.
              Different kinds of grammars can be distinguished by
              putting different restrictions on the form of these rules.
              This influences how expressive the grammars are and
              hence what kind of languages can be defined by them.

        - A **context-free grammar (CFG)** is a type of formal grammar
          that is frequently used in computer science and programming
          language design. The production rules need to fulfill
          following criteria:

            - The left-hand side of each rule is a single nonterminal
              symbol. This means a nonterminal symbol can be rewritten
              into the right-hand side of the rule, no matter which
              context the nonterminal is surrounded with.
            - The right-hand side is a sequence of symbols that can
              consist of a combination of terminal symbols, nonterminal
              symbols and the empty string ``""`` (often denoted by
              the greek letter ``ɛ`` or less often ``λ``).

    2. Tasks a grammar can be used for

        - Define a grammar with a text in a suitable format such as
          Backus-Naur form (BNF) or Extended Backus-Naur form
          (EBNF). Both of these formats can be recognized by
          this class's initialization method `__init__`.
        - Visualize a grammar in form of a syntax diagram with
          the method `plot`.
        - Recognize whether a given string belongs to the grammar's
          language or not with the method `recognize_string`.
        - Parse a given string to see how it can be derived from the
          start symbol by a specific sequence of rule applications
          with the method `parse_string`. The result is a parse tree
          or derivation tree, an instance of the class
          `~alogos._grammar.data_structures.DerivationTree`.
          A derivation tree represents a single string by reading
          its leaf nodes from left to right, but many possible
          derivations that lead to it, since the sequence of rule
          applications is not fixed in the tree.
        - Generate a random derivation, derivation tree or string with
          the methods `generate_derivation`,
          `generate_derivation_tree`, `generate_string`.
        - Generate the grammar's language or a finite subset of it
          with the method `generate_language`.
        - Convert a grammar into a specific normal form or check if it
          is in one with the methods `to_cnf`, `is_cnf`, `to_gnf`,
          `is_gnf`, `to_bnf`, `is_bnf`. Normal forms put further
          restrictions on the form of the production rules, but
          importantly, converting a grammar into it does not change
          its language.

    Notes
    -----
    The concepts and notation mentioned here are based on
    classical textbooks in formal language theory such as [1]_.
    Similar discussions can be found on the web [2]_.

    References
    ----------
    .. [1] J. E. Hopcroft and J. D. Ullman, Introduction to
       Automata Theory, Languages and Computation.
       Addison-Wesley, 1979.

    .. [2] Wikipedia articles

       - `Formal grammar
         <https://en.wikipedia.org/wiki/Formal_grammar>`__
       - `Context-free grammar
         <https://en.wikipedia.org/wiki/Context-free_grammar>`__
       - `Context-free language
         <https://en.wikipedia.org/wiki/Context-free_language>`__
       - `Terminal and nonterminal symbols
         <https://en.wikipedia.org/wiki/Terminal_and_nonterminal_symbols>`__
       - `Production rule
         <https://en.wikipedia.org/wiki/Production_(computer_science)>`__

    """

    __slots__ = (
        "nonterminal_symbols",
        "terminal_symbols",
        "production_rules",
        "start_symbol",
        "_cache",
    )

    # Initialization and reset
    def __init__(
        self, bnf_text=None, bnf_file=None, ebnf_text=None, ebnf_file=None, **kwargs
    ):
        '''Create a grammar from a string or file in BNF or EBNF notation.

        Backus-Naur form (BNF) and Extended Backus-Naur form (EBNF)
        [3]_ are two well-known and often used formats for defining
        context-free grammars.

        Parameters
        ----------
        bnf_text : `str`, optional
            String that contains a grammar in BNF.
        bnf_file : `str`, optional
            Filepath of a text file that contains a grammar in BNF.
        ebnf_text : `str`, optional
            String that contains a grammar in EBNF.
        ebnf_file : `str`, optional
            Filepath of a text file that contains a grammar in EBNF.
        **kwargs : `dict`, optional
            Further keyword arguments are forwarded to the function
            that reads and creates a grammar from text in BNF or EBNF
            notation. The available arguments and further details can
            be found in following methods:

            - BNF: `from_bnf_text`, `from_bnf_file`
            - EBNF: `from_ebnf_text`, `from_ebnf_file`

        Raises
        ------
        TypeError
            If an argument gets a value of unexpected type.

        FileNotFoundError
            If a filepath does not point to an existing file.

        GrammarError
            If a newly generated grammar does not pass all validity
            checks. For example, each nonterminal that appears on the
            right-hand side must also appear on the left-hand side.

        Warns
        -----
        GrammarWarning
            If a newly generated grammar contains a production rule
            more than once.

        References
        ----------
        .. [3] Wikipedia

           - `Backus-Naur form (BNF)
             <https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form>`__
           - `Extended Backus-Naur form (EBNF)
             <https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form>`__

        Examples
        --------
        Use text in Backus-Naur form (BNF) to define a grammar:

        >>> import alogos as al
        >>> bnf_text = """
        <S> ::= <Greeting> _ <Object> !
        <Greeting> ::= Hello | Hi | Hola
        <Object> ::= World | Universe | Cosmos
        """
        >>> grammar = al.Grammar(bnf_text=bnf_text)
        >>> grammar.generate_string()
        Hello_World!

        Use text in or Extended Backus-Naur form (EBNF) to define a
        grammar:

        >>> import alogos as al
        >>> ebnf_text = """
        S = Greeting " " Object "!"
        Greeting = "Hello" | "Hi" | "Hola"
        Object = "World" | "Universe" | "Cosmos"
        """
        >>> grammar = al.Grammar(ebnf_text=ebnf_text)
        >>> grammar.generate_string()
        Hola Universe!

        Use text in an unusual form to define a grammar by passing
        custom symbol definitions as keyword arguments:

        >>> import alogos as al
        >>> bnf_text = """
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
        >>> grammar = al.Grammar(
            bnf_text=bnf_text,
            defining_symbol="=",
            start_nonterminal_symbol="[",
            end_nonterminal_symbol="]",
            start_terminal_symbol='"',
            end_terminal_symbol='"',
        )
        >>> grammar.generate_string(separator=' ')
        I am

        '''
        # Argument processing
        if (
            sum(inp is not None for inp in (bnf_text, bnf_file, ebnf_text, ebnf_file))
            > 1
        ):
            _warnings._warn_multiple_grammar_specs()

        # Transformation
        if bnf_text is not None:
            self.from_bnf_text(bnf_text, **kwargs)
        elif bnf_file is not None:
            self.from_bnf_file(bnf_file, **kwargs)
        elif ebnf_text is not None:
            self.from_ebnf_text(ebnf_text, **kwargs)
        elif ebnf_file is not None:
            self.from_ebnf_file(ebnf_file, **kwargs)
        else:
            # Create an empty grammar if no specification is provided
            self._set_empty_state()

    def _set_empty_state(self):
        """Assign empty containers for symbols and production rules.

        Notes
        -----
        dict() is used as data structure for rules instead of
        OrderedDict from the itertools module, and instead of
        OrderedSet from external orderedset library for symbols,
        because it guarantees order, is faster and introduces no
        dependencies and portability issues. Here is more background:

        - Since Python 3.6, dict in CPython remembers the insertion
          order of keys.
        - Since Python 3.7 this is considered a language feature.
        - If order is not preserved, no algorithm here fails, only
          output becomes less readable.

        """
        self.terminal_symbols = _OrderedSet()
        self.nonterminal_symbols = _OrderedSet()
        self.production_rules = dict()
        self.start_symbol = None
        # Cache to store results of some calculations instead of repeating them
        self._cache = dict()

    # Copying
    def copy(self):
        """Create a deep copy of the grammar.

        The new object is entirely independent of the original object.
        No parts, such as symbols or rules, are shared between the
        objects.

        """
        return self.__deepcopy__()

    def __copy__(self):
        """Create a shallow copy of the grammar.

        Notes
        -----
        The content of the `_cache` attribute is not copied.
        Instead, a new empty dictionary is assigned to it.

        References
        ----------
        - https://docs.python.org/3/library/copy.html

        """
        new_grammar = self.__class__()
        new_grammar.nonterminal_symbols = _copy.copy(self.nonterminal_symbols)
        new_grammar.terminal_symbols = _copy.copy(self.terminal_symbols)
        new_grammar.production_rules = _copy.copy(self.production_rules)
        new_grammar.start_symbol = _copy.copy(self.start_symbol)
        return new_grammar

    def __deepcopy__(self, memo=None):
        """Create a deep copy of the grammar.

        Notes
        -----
        The content of the `_cache` attribute is not copied.
        Instead, a new empty dictionary is assigned to it.

        References
        ----------
        - https://docs.python.org/3/library/copy.html

        """
        new_grammar = self.__class__()
        new_grammar.nonterminal_symbols = _copy.deepcopy(self.nonterminal_symbols)
        new_grammar.terminal_symbols = _copy.deepcopy(self.terminal_symbols)
        new_grammar.production_rules = _copy.deepcopy(self.production_rules)
        new_grammar.start_symbol = _copy.deepcopy(self.start_symbol)
        return new_grammar

    # Representations
    def __repr__(self):
        """Compute the "official" string representation of the grammar.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__repr__

        """
        return "<{} object at {}>".format(self.__class__.__name__, hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the grammar.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__str__

        """
        sep = "{}  ".format(_NEWLINE)
        msg = []
        msg.append("Nonterminal symbols:{sep}".format(sep=sep))
        if self.nonterminal_symbols:
            nt_text = sep.join(
                "{idx}: {nt}".format(idx=i, nt=repr(sym))
                for i, sym in enumerate(list(self.nonterminal_symbols))
            )
        else:
            nt_text = "Empty set"
        msg.append(nt_text)
        msg.append("{nl}{nl}Terminal symbols:{sep}".format(nl=_NEWLINE, sep=sep))
        if self.terminal_symbols:
            t_text = sep.join(
                "{idx}: {terminal}".format(idx=i, terminal=repr(sym))
                for i, sym in enumerate(list(self.terminal_symbols))
            )
        else:
            t_text = "Empty set"
        msg.append(t_text)
        msg.append(
            "{nl}{nl}Start symbol:{sep}{sym}".format(
                nl=_NEWLINE, sep=sep, sym=repr(self.start_symbol)
            )
        )
        msg.append("{nl}{nl}Production rules:".format(nl=_NEWLINE))
        if self.production_rules:
            i = 0
            for lhs, rhs_list in self.production_rules.items():
                for rhs in rhs_list:
                    msg.append(
                        "{sep}{idx}: {lhs} -> {rhs}".format(
                            sep=sep,
                            idx=i,
                            lhs=repr(lhs),
                            rhs=" ".join(repr(sym) for sym in rhs),
                        )
                    )
                    i += 1
        else:
            msg.append("{sep}Empty set".format(sep=sep))
        text = "".join(msg)
        return text

    def _repr_html_(self):
        """Provide rich display representation for Jupyter notebooks.

        References
        ----------
        - https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display

        """
        fig = self.plot()
        return fig._repr_html_()

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython.

        References
        ----------
        - https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display

        """
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Equality and hashing
    def __eq__(self, other):
        """Compute whether two grammars are equal.

        Caution: This checks if two grammars have identical rules and
        symbols. It not check for equivalence in the sense of formal
        language theory, which would mean that two grammars produce
        the same language. This problem is actually undecidable for
        context-free grammars in the general case.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__eq__

        """
        # Type comparison
        if not isinstance(other, self.__class__):
            return NotImplemented

        # Data comparison
        p1 = self.production_rules
        p2 = other.production_rules
        if len(p1) != len(p2):
            return False
        for lhs1, lhs2 in zip(p1, p2):
            if lhs1 != lhs2:
                return False
            rhsm1 = p1[lhs1]
            rhsm2 = p2[lhs2]
            if len(rhsm1) != len(rhsm2):
                return False
            for rhs1, rhs2 in zip(rhsm1, rhsm2):
                if len(rhs1) != len(rhs2):
                    return False
                for sym1, sym2 in zip(rhs1, rhs2):
                    if sym1 != sym2:
                        return False
        return True

    def __hash__(self):
        """Calculate a hash value for this object.

        It is used for operations on hashed collections such as `set`
        and `dict`.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__hash__

        """
        return hash(str(self))

    # Reading
    def from_bnf_text(
        self,
        bnf_text,
        defining_symbol="::=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="<",
        end_nonterminal_symbol=">",
        start_terminal_symbol="",
        end_terminal_symbol="",
        start_terminal_symbol2="",
        end_terminal_symbol2="",
        verbose=False,
    ):
        """Read a grammar specification in BNF notation from a string.

        This method resets the grammar object and then uses the
        provided information.

        Parameters
        ----------
        bnf_text : `str`
            String with grammar specification in BNF notation.

        Other Parameters
        ----------------
        defining_symbol : `str`, optional
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`, optional
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`, optional
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`, optional
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`, optional
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`, optional
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the end of a terminal.
        verbose : `bool`, optional
            If `True`, messages will be printed during processing the
            input text that show which rules and symbols are found one
            after another. This can be useful to see what went wrong
            when the generated grammar does not look or behave as
            expected.

        """
        from . import parsing

        # Reset this grammar object
        self._set_empty_state()

        # Parse BNF
        parsing.read_bnf(
            self,
            bnf_text,
            verbose,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
        )

    def from_bnf_file(
        self,
        filepath,
        defining_symbol="::=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="<",
        end_nonterminal_symbol=">",
        start_terminal_symbol="",
        end_terminal_symbol="",
        start_terminal_symbol2="",
        end_terminal_symbol2="",
        verbose=False,
    ):
        """Read a grammar specification in BNF notation from a file.

        This method resets the grammar object and then uses the
        provided information.

        Parameters
        ----------
        filepath : `str`
            Text file with grammar specification in BNF notation.

        Other Parameters
        ----------------
        defining_symbol : `str`, optional
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`, optional
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`, optional
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`, optional
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`, optional
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`, optional
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the end of a terminal.
        verbose : `bool`, optional
            If `True`, messages will be printed during processing the
            input text that show which rules and symbols are found one
            after another. This can be useful to see what went wrong
            when the generated grammar does not look or behave as
            expected.

        """
        # Argument processing
        filepath = _ap.str_arg("filepath", filepath)

        # Read text from file
        bnf_text = self._read_file(filepath)

        # Transform BNF text to grammar
        self.from_bnf_text(
            bnf_text,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
            verbose,
        )

    def from_ebnf_text(
        self,
        ebnf_text,
        defining_symbol="=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="",
        end_nonterminal_symbol="",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
        verbose=False,
    ):
        """Read a grammar specification in EBNF notation from a string.

        This method resets the grammar object and then uses the
        provided information.

        Parameters
        ----------
        ebnf_text : `str`
            String with grammar specification in EBNF notation.

        Other Parameters
        ----------------
        defining_symbol : `str`, optional
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`, optional
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`, optional
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`, optional
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`, optional
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`, optional
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the end of a terminal.
        verbose : `bool`, optional
            If `True`, messages will be printed during processing the
            input text that show which rules and symbols are found one
            after another. This can be useful to see what went wrong
            when the generated grammar does not look or behave as
            expected.

        """
        from . import parsing

        # Reset this grammar object
        self._set_empty_state()

        # Parse EBNF
        parsing.read_ebnf(
            self,
            ebnf_text,
            verbose,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
        )

    def from_ebnf_file(
        self,
        filepath,
        defining_symbol="=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="",
        end_nonterminal_symbol="",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2="'",
        end_terminal_symbol2="'",
        verbose=False,
    ):
        """Read a grammar specification in EBNF notation from a file.

        This method resets the grammar object and then uses the
        provided information.

        Parameters
        ----------
        filepath : `str`
            Text file with grammar specification in EBNF notation.

        Other Parameters
        ----------------
        defining_symbol : `str`, optional
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`, optional
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`, optional
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`, optional
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`, optional
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`, optional
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the end of a terminal.
        verbose : `bool`, optional
            If `True`, messages will be printed during processing the
            input text that show which rules and symbols are found one
            after another. This can be useful to see what went wrong
            when the generated grammar does not look or behave as
            expected.

        """
        # Argument processing
        filepath = _ap.str_arg("filepath", filepath)

        # Read text from file
        ebnf_text = self._read_file(filepath)

        # Transform EBNF text to grammar
        self.from_ebnf_text(
            ebnf_text,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
            verbose,
        )

    # Writing
    def to_bnf_text(
        self,
        rules_on_separate_lines=True,
        defining_symbol="::=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="<",
        end_nonterminal_symbol=">",
        start_terminal_symbol="",
        end_terminal_symbol="",
        start_terminal_symbol2="",
        end_terminal_symbol2="",
    ):
        """Write the grammar in BNF notation to a string.

        Parameters
        ----------
        rule_on_separate_lines : `bool`, optional
            If `True`, each rule for a nonterminal is put onto a
            separate line. If `False`, all rules for a nonterminal
            are grouped onto one line.
        defining_symbol : `str`, optional
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`, optional
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`, optional
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`, optional
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`, optional
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`, optional
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the end of a terminal.

        """
        from . import parsing

        # Generate BNF text
        bnf_text = parsing.write_bnf(
            self,
            rules_on_separate_lines,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
        )
        return bnf_text

    def to_bnf_file(
        self,
        filepath,
        rules_on_separate_lines=True,
        defining_symbol="::=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="<",
        end_nonterminal_symbol=">",
        start_terminal_symbol="",
        end_terminal_symbol="",
        start_terminal_symbol2="",
        end_terminal_symbol2="",
    ):
        """Write the grammar in BNF notation to a text file.

        Parameters
        ----------
        filepath : `str`
            Filepath of the text file that shall be generated.

        Other Parameters
        ----------------
        rule_on_separate_lines : `bool`, optional
            If `True`, each rule for a nonterminal is put onto a
            separate line. If `False`, all rules for a nonterminal
            are grouped onto one line.
        defining_symbol : `str`, optional
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`
            Alternative symbol indicating the end of a terminal.

        """
        # Argument processing
        filepath = _ap.str_arg("filepath", filepath)

        # Generate BNF text
        bnf_text = self.to_bnf_text(
            rules_on_separate_lines,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
        )

        # Write text to file
        self._write_file(filepath, bnf_text)

    def to_ebnf_text(
        self,
        rules_on_separate_lines=True,
        defining_symbol="=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="",
        end_nonterminal_symbol="",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2='"',
        end_terminal_symbol2='"',
    ):
        """Write the grammar in EBNF notation to a string.

        Parameters
        ----------
        rule_on_separate_lines : `bool`
            If `True`, each rule for a nonterminal is put onto a
            separate line. If `False`, all rules for a nonterminal
            are grouped onto one line.
        defining_symbol : `str`
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`
            Alternative symbol indicating the end of a terminal.

        """
        from . import parsing

        # Generate EBNF text
        ebnf_text = parsing.write_ebnf(
            self,
            rules_on_separate_lines,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
        )
        return ebnf_text

    def to_ebnf_file(
        self,
        filepath,
        rules_on_separate_lines=True,
        defining_symbol="=",
        rule_separator_symbol="|",
        start_nonterminal_symbol="",
        end_nonterminal_symbol="",
        start_terminal_symbol='"',
        end_terminal_symbol='"',
        start_terminal_symbol2='"',
        end_terminal_symbol2='"',
    ):
        """Write the grammar in EBNF notation to a text file.

        Parameters
        ----------
        filepath : `str`
            Filepath of the text file that shall be generated.

        Other Parameters
        ----------------
        rule_on_separate_lines : `bool`, optional
            If `True`, each rule for a nonterminal is put onto a
            separate line. If `False`, all rules for a nonterminal
            are grouped onto one line.
        defining_symbol : `str`, optional
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : `str`, optional
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : `str`, optional
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : `str`, optional
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : `str`, optional
            Symbol indicating the start of a terminal.
        end_terminal_symbol : `str`, optional
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : `str`, optional
            Alternative symbol indicating the end of a terminal.

        """
        # Argument processing
        filepath = _ap.str_arg("filepath", filepath)

        # Generate EBNF text
        ebnf_text = self.to_ebnf_text(
            rules_on_separate_lines,
            defining_symbol,
            rule_separator_symbol,
            start_nonterminal_symbol,
            end_nonterminal_symbol,
            start_terminal_symbol,
            end_terminal_symbol,
            start_terminal_symbol2,
            end_terminal_symbol2,
        )

        # Write text to file
        self._write_file(filepath, ebnf_text)

    # String generation
    def generate_derivation_tree(self, method="weighted", *args, **kwargs):
        """Generate a derivation tree with a chosen method.

        One *derivation tree* represents one *string* but many
        *derivations* [4]_.

        - The leaf nodes of the tree read from left to right form a
          string of the language defined by the grammar.
        - A depth-first walk over the tree that always chooses the
          leftmost item gives the leftmost derivation, while one that
          always chooses the rightmost item gives the rightmost
          derivation. In most cases, there can be many more ways to
          expand one nonterminal after another, leading to a plethora
          of possible derivations, all represented by the same
          derivation tree.

        Parameters
        ----------
        method : `str`, optional
            Name of the method that is used for creating a
            derivation tree.

            Possible values:

            - Deterministic methods that require the keyword argument
              ``genotype`` as input data:

                - ``"cfggp"``: Uses
                  `~alogos.systems.cfggp.mapping.forward`
                  mapping from CFG-GP.
                - ``"cfggpst"``: Uses
                  `~alogos.systems.cfggpst.mapping.forward`
                  mapping from CFG-GP-ST.
                - ``"dsge"``: Uses `~alogos.systems.dsge.mapping.forward`
                  mapping from DSGE.
                - ``"ge"``: Uses `~alogos.systems.ge.mapping.forward`
                  mapping from GE.
                - ``"pige"``: Uses `~alogos.systems.pige.mapping.forward`
                  mapping from piGE.
                - ``"whge"``: Uses `~alogos.systems.whge.mapping.forward`
                  mapping from WHGE.

            - Probabilistic methods that generate a random tree and
              require no input data:

                - ``"uniform"``: Uses
                  `~alogos.systems._shared.init_tree.uniform`.

                - ``"weighted"``: Uses
                  `~alogos.systems._shared.init_tree.weighted`.

                - ``"ptc2"``: Uses
                  `~alogos.systems._shared.init_tree.ptc2`.

                - ``"grow_one_branch_to_max_depth"``: Uses
                  `~alogos.systems._shared.init_tree.grow_one_branch_to_max_depth`.

                - ``"grow_all_branches_within_max_depth"``: Uses
                  `~alogos.systems._shared.init_tree.grow_all_branches_within_max_depth`

                - ``"grow_all_branches_to_max_depth"``: Uses
                  `~alogos.systems._shared.init_tree.grow_all_branches_to_max_depth`.

        **kwargs : `dict`, optional
            Further keyword arguments are forwarded to the chosen
            method. This is especially relevant for controlling the
            behavior of the probabilistic methods.

        Returns
        -------
        derivation_tree : `~alogos._grammar.data_structures.DerivationTree`

        Raises
        ------
        MappingError
            If the method fails to generate a derivation tree,
            e.g. because a limit like maximum number of expansions
            was reached before all leaves contained terminal symbols

        References
        ----------
        .. [4] Wikipedia

           - `Parse tree
             <https://en.wikipedia.org/wiki/Parse_tree>`__

        """
        # Argument processing
        from .. import systems

        name_method_map = {
            # Deterministic derivations with G3P mapping functions
            "cfggp": systems.cfggp.mapping.forward,
            "cfggpst": systems.cfggpst.mapping.forward,
            "dsge": systems.dsge.mapping.forward,
            "ge": systems.ge.mapping.forward,
            "pige": systems.pige.mapping.forward,
            "whge": systems.whge.mapping.forward,
            # Probabilistic derivations with random rule selection schemes
            "uniform": systems._shared.init_tree.uniform,
            "weighted": systems._shared.init_tree.weighted,
            "grow_one_branch_to_max_depth": systems._shared.init_tree.grow_one_branch_to_max_depth,
            "grow_all_branches_within_max_depth": systems._shared.init_tree.grow_all_branches_within_max_depth,
            "grow_all_branches_to_max_depth": systems._shared.init_tree.grow_all_branches_to_max_depth,
            "ptc2": systems._shared.init_tree.ptc2,
        }
        _ap.str_arg("method", method, vals=name_method_map.keys())
        func = name_method_map[method]

        # Mapping
        if method in ("cfggp", "cfggpst", "dsge", "ge", "pige", "whge"):
            kwargs["return_derivation_tree"] = True
            _, dt = func(self, *args, **kwargs)
        else:
            dt = func(self, *args, **kwargs)
        return dt

    def generate_derivation(
        self,
        method="weighted",
        *args,
        derivation_order="leftmost",
        newline=True,
        **kwargs
    ):
        """Generate a derivation with a chosen method.

        Parameters
        ----------
        method : `str`, optional
            Name of the method that is used for creating a derivation.

            Possible values: See `generate_derivation_tree`.
        derivation_order : `str`, optional
            Order in which nonterminals are expanded during the step-by-step derivation.

            Possible values:

            - ``"leftmost"``: Expand the leftmost nonterminal in each step.
            - ``"rightmost"``: Expand the rightmost nonterminal in each step.
            - ``"random"``: Expand a random nonterminal in each step.

        newline : `bool`, optional
             If `True`, the derivation steps are placed on separate
             lines by adding newline characters between them.
        **kwargs : `dict`, optional
            Further keyword arguments are forwarded to the chosen
            method.

        Returns
        -------
        derivation : `str`
            The derivation generated with the chosen method.
            It is a text that shows each step of the derivation
            as it is commonly represented in textbooks.

        Notes
        -----
        This method uses `generate_derivation_tree` to create a
        derivation tree and then reads the leftmost derivation
        contained in it with
        `~alogos._grammar.data_structures.DerivationTree.derivation`.

        """
        dt = self.generate_derivation_tree(method, *args, **kwargs)
        derivation = dt.derivation(derivation_order=derivation_order, newline=newline)
        return derivation

    def generate_string(self, method="weighted", *args, separator="", **kwargs):
        """Generate a string with a chosen method.

        Each string [5]_ is part of the language L(G) defined by grammar G.

        Parameters
        ----------
        method : `str`, optional
            Name of the method that is used for creating a string.

            Possible values: See `generate_derivation_tree`.
        separator : `str`, optional
            A short string that is inserted between each of the
            terminal symbols that make up the string of the grammar's
            language.
        **kwargs : `dict`, optional
            Further keyword arguments are forwarded to the chosen
            method.

        Returns
        -------
        string : `str`
            The string generated with the chosen method.
            It is a text that consists only of terminal symbols
            and optionally a separator symbol between them.

        Notes
        -----
        This method uses `generate_derivation_tree` to create a
        derivation tree and then reads the string contained in it
        from the leaf nodes with
        `~alogos._grammar.data_structures.DerivationTree.string`.

        References
        ----------
        .. [5] Wikipedia

           - `String <https://en.wikipedia.org/wiki/String_(computer_science)>`__

        """
        dt = self.generate_derivation_tree(method, *args, **kwargs)
        string = dt.string(separator)
        return string

    def generate_language(
        self, max_steps=None, sort_order="discovered", verbose=None, return_details=None
    ):
        """Generate the formal language defined by the grammar.

        This algorithm recursively constructs the formal language [6]_
        defined by the grammar, i.e. the set of strings it can generate
        or recognize.

        The algorithm can be stopped prematurely to get only a subset
        of the entire language. By default, only the language of the
        start symbol is returned, which is the language of the grammar,
        but it is also possible to return the languages of each other
        nonterminal symbol, which can be thought of as sublanguages.
        For example, many programming languages are defined by a CFG
        and have a nonterminal symbol that represents all valid integer
        literals in that language.

        Parameters
        ----------
        max_steps : `int`, optional
            The maximum number of recursive steps during language
            generation. It can be used to stop the algorithm before it
            can construct all strings of the language. Instead a list of
            valid strings found so far will be returned, which is a
            subset of the entire language. This is necessary to get a
            result if the grammar defines a very large or infinite
            language.

            Note that each recursive step uses the strings known so far
            and inserts them into the right-hand sides of production
            rules to see if any new strings can be discovered. Therefore
            simpler strings are found before more complex ones that
            require more expansions. If the number of steps is too
            little to form a single string belonging to the language of
            the start symbol, the result will be an empty list.
        sort_order : `str`, optional
            The language is returned as a list of strings, which can be
            sorted in different ways.

            Possible values:

            - ``"discovered"``: Strings are returned in the order they
              were discovered.
            - ``"lex"``: Lexicographic, i.e. the order used in lexicons,
              which means the alphabetic order extended to non-alphabet
              characters like numbers. Python's built-in ``sort()``
              function delivers it by default.
            - ``"shortlex"``: Strings are sorted primarily by their
              length. Those with the same length are further sorted in
              lexicographic order.
        verbose : `bool`, optional
            If `True`, detailed messages are printed during language
            generation. If `False`, no output is generated.
        return_details : `bool`, optional
            If `True`, the return value is a `dict` with nonterminals
            as keys and their languages as values. The language of the
            start symbol is the language of the grammar, but each
            nonterminal has its own sub-language that can be of interest
            too.

        Returns
        -------
        language : `list` of `str`, or `dict` with `list` of `str` as values
            The formal language L(G) defined by the grammar G.
            If the argument `return_details` is set to `True`,
            the return value is a `dict` where each key is a nonterminal
            of the grammar and each value the language (set of strings)
            of the nonterminal.

        Warns
        -----
        GrammarWarning
            If no value is provided for the argument `max_steps`,
            internally an unrealistically large value of is assigned to
            it. In the unlikely case this is ever reached, a warning
            will be raised if the language generation did not generate
            all strings of the language.

        References
        ----------
        .. [6] Wikipedia

           - `Formal language
             <https://en.wikipedia.org/wiki/Formal_language>`__
           - `Shortlex order
             <https://en.wikipedia.org/wiki/Shortlex_order>`__
           - `Lexicographic order
             <https://en.wikipedia.org/wiki/Lexicographical_order>`__

        """
        from . import generation

        strings = generation.generate_language(
            grammar=self,
            sort_order=sort_order,
            max_steps=max_steps,
            verbose=verbose,
            return_details=return_details,
        )
        return strings

    # String parsing
    def recognize_string(self, string, parser="earley"):
        """Test if a string belongs to the language of the grammar.

        Parameters
        ----------
        string : `str`
            Candidate string which can be recognized to be a member of
            the grammar's language.
        parser : `str`, optional
            Parsing algorithm used to analyze the string.

            Possible values: See `parse_string`

        Returns
        -------
        recognized : `bool`
            `True` if the string belongs to the grammar's language,
            `False` if it does not.

        """
        try:
            self.parse_string(string, parser, get_multiple_trees=False)
        except _exceptions.ParserError:
            return False
        return True

    def parse_string(
        self, string, parser="earley", get_multiple_trees=False, max_num_trees=None
    ):
        """Try to parse a given string with a chosen parsing algorithm.

        Parsing means the analysis of a string into its parts or
        constituents [7]_.

        Parameters
        ----------
        string : `str`
            Candidate string which can only be parsed successfully
            if it is a member of the grammar's language.
        parser : `str`, optional
            Parsing algorithm used to analyze the string. This package
            uses parsers from the package Lark [8]_.

            Possible values:

            - ``"earley"``: Can parse any context-free grammar.

              Performance: The algorithm has a time complexity of
              ``O(n^3)``, but if the grammar is unambiguous it is
              reduced to ``O(n^2)`` and for most LR grammars it
              is ``O(n)``.

            - ``"lalr"``: Can parse only a subset of context-free
              grammars, which have a form that allows very efficient
              parsing with a LALR(1) parser.

              Performance: The algorithm has a time complexity
              of ``O(n)``.
        get_multiple_trees : `bool`, optional
            If `True`, a list of parse trees, also known as
            parse forest, is returned instead of a single
            parse tree object.
        max_num_trees : `int`, optional
            An upper limit on how many parse trees will be returned at
            maximum.

        Returns
        -------
        parse_tree : `~alogos._grammar.data_structures.DerivationTree`, or `list` of `~alogos._grammar.data_structures.DerivationTree`
            If the argument `get_multiple_trees` is set to `True`,
            a list of derivation trees is returned instead of a single
            derivation tree object. The list can contain one or more
            trees, dependening on how many ways there are to parse the
            given string. If a grammar is unambiguous, there is only
            one way. If it is ambuguous, there can be multiple ways
            to derive the same string in a leftmost derivation, which
            is captured by different derivation trees.

        Raises
        ------
        `ParserError`
            If the string does not belong to the language and therefore
            no parse tree exists for it.

        Notes
        -----
        If the grammar is ambiguous, there can be more than one way
        to parse a given string, which means that there are multiple
        parse trees for it. By default, only one of these trees is
        returned, but the argument ``get_multiple_trees`` allows to
        get all of them. Caution: This feature is currently only
        supported with Lark's Earley parser.

        References
        ----------
        .. [7] Wikipedia

           - `Parsing
             <https://en.wikipedia.org/wiki/Parsing>`__
           - `Earley parser
             <https://en.wikipedia.org/wiki/Earley_parser>`__
           - `LALR parser
             <https://en.wikipedia.org/wiki/LALR_parser>`__
           - `Ambiguous grammar
             <https://en.wikipedia.org/wiki/Ambiguous_grammar>`__

        .. [8] `Lark <https://lark-parser.readthedocs.io>`__

           - `Earley <https://lark-parser.readthedocs.io/en/latest/parsers.html#earley>`__
           - `LALR(1) <https://lark-parser.readthedocs.io/en/latest/parsers.html#lalr-1>`__

        """
        from . import parsing

        # Parse with lark-parser
        derivation_tree = parsing.parse_string(
            grammar=self,
            string=string,
            parser=parser,
            get_multiple_trees=get_multiple_trees,
            max_num_trees=max_num_trees,
        )
        return derivation_tree

    # Visualization
    def plot(self):
        """Generate a figure containing a syntax diagram of the grammar.

        Returns
        -------
        fig : `Figure`
            Figure object containing the plot, allowing to display or
            export it.

        Notes
        -----
        Syntax diagrams (a.k.a. railroad diagrams) [9]_ are especially
        useful for representing EBNF specifications of a grammar,
        because they capture nicely the extended notations that are
        introduced by EBNF, e.g. optional or repeated items.

        This package supports reading a grammar specification from
        EBNF text. Internally, however, EBNF is automatically converted
        to a simpler form during the reading process, which is done by
        removing any occurrence of extended notation and expressing it
        with newly introduced symbols and rules instead. Only the final
        version of the grammar can be visualized, which is essentially
        BNF with new helper rules and nonterminals. Therefore the
        expressive power of syntax diagrams is unfortunately not fully
        used here.

        There are many websites with explanations [10]_ or
        examples [11]_ of syntax diagrams.
        Likewise, there are many libraries [12]_ for generating
        syntax diagrams. This package uses the library
        Railroad-Diagram Generator [13]_.

        References
        ----------
        .. [9] Wikipedia

           - `Syntax diagram
             <https://en.wikipedia.org/wiki/Syntax_diagram>`__

        .. [10] Explanatory websites

           - Oxford Reference / A Dictionary of Computing 6th edition:
             `Syntax diagram
             <https://www.oxfordreference.com/view/10.1093/oi/authority.20110803100547820>`__
           - Course website by Roger Hartley:
             `Programming language structure 1: Syntax diagrams
             <https://www.cs.nmsu.edu/~rth/cs/cs471/Syntax%20Module/diagrams.html>`__
           - Book chapter by Richard E. Pattis with Syntax Charts
             on p. 11:
             `v1: Languages and Syntax
             <http://www.cs.cmu.edu/~pattis/misc/ebnf.pdf>`__,
             `v2: EBNF - A Notation to Describe Syntax
             <https://www.ics.uci.edu/~pattis/misc/ebnf2.pdf>`__

        .. [11] Example websites

           - `xkcd webcomic <https://xkcd.com/1930/>`__
           - `JSON <https://www.json.org/json-en.html>`__
           - `SQLite <https://www.sqlite.org/lang.html>`__
           - `Oracle Database Lite SQL <https://docs.oracle.com/cd/B19188_01/doc/B15917/sqsyntax.htm>`__
           - `Boost <https://www.boost.org/doc/libs/1_66_0/libs/spirit/doc/html/spirit/abstracts/syntax_diagram.html>`__

        .. [12] Some other tools for drawing syntax diagrams

           - `Railroad Diagram Generator
             <https://bottlecaps.de/rr/ui>`__
             by Gunther Rademacher
           - `ANTLR Development Tools
             <https://www.antlr.org/tools.html>`__
             by Terence Parr
           - `DokuWiki EBNF Plugin
             <https://www.dokuwiki.org/plugin:ebnf>`__
             by Vincent Tscherter
           - `bubble-generator
             <https://www.sqlite.org/docsrc/finfo?name=art/syntax/bubble-generator.tcl>`__
             by the SQLite team
           - `Ebnf2ps <https://github.com/FranklinChen/Ebnf2ps>`__
             by Peter Thiemann
           - `EBNF Visualizer
             <http://dotnet.jku.at/applications/Visualizer/>`__
             by Markus Dopler and Stefan Schörgenhumer
           - `Clapham Railroad Diagram Generator
             <http://clapham.hydromatic.net>`__
             by Julian Hyde
           - `draw-grammar
             <https://github.com/iangodin/draw-grammar>`__
             by Ian Godin

        .. [13] Library used here for generating syntax diagram SVGs

           - `Railroad-Diagram Generator
             <https://github.com/tabatkins/railroad-diagrams>`__
             by Tab Atkins Jr.

        """
        from . import visualization

        fig = visualization.create_syntax_diagram(self)
        return fig

    # Normal forms
    def _is_cnf(self):
        """Check if this grammar is in Chomsky Normal Form (CNF).

        Returns
        -------
        is_cnf : `bool`
            `True` if it is in CNF, `False` otherwise.

        """
        from . import normalization

        return normalization.is_cnf(self)

    def _to_cnf(self):
        """Convert the grammar to Chomsky Normal Form (CNF).

        This normal form was originally defined by Noam Chomsky [14]_.
        Modern definitions and algorithms can be found in standard
        textbooks of formal language theory [15]_, [16]_ and on the
        web [17]_.

        Returns
        -------
        grammar_in_cnf : `Grammar`
            New grammar object where all rules adhere to CNF.
            It is equivalent to the original grammar, which means
            that it generates the same language.

        Notes
        -----
        Assuming that the grammar does not have the empty string as
        part of its language, then all rules need to be in one of
        the following two forms:

        - ``X → a``, where ``a`` is a terminal
        - ``X → BC``, where ``B`` and ``C`` are nonterminals

        Tasks that are simplified by having a grammar in CNF:

        - Deciding if a given string is part of the grammar's language
        - Parsing with efficient data structures for binary trees

        References
        ----------
        .. [14] N. Chomsky, “On certain formal properties of grammars,”
           Information and Control, vol. 2, no. 2, pp. 137–167,
           Jun. 1959.

        .. [15] J. E. Hopcroft and J. D. Ullman, Introduction to
           Automata Theory, Languages and Computation.
           Addison-Wesley, 1979, pp. 92-94.

        .. [16] E. A. Rich, Automata, Computability and Complexity:
           Theory and Applications. Pearson Education, 2007, p. 169.

        .. [17] Wikipedia

           - `Chomsky normal form
             <https://en.wikipedia.org/wiki/Chomsky_normal_form>`__

        """
        from . import normalization

        return normalization.to_cnf(self)

    def _is_gnf(self):
        """Check if this grammar is in Greibach Normal Form (GNF).

        Returns
        -------
        is_gnf : `bool`
            `True` if it is in GNF, `False` otherwise.

        """
        from . import normalization

        return normalization.is_gnf(self)

    def _to_gnf(self):
        """Convert this grammar to Greibach Normal Form (GNF).

        This normal form was originally defined by
        Sheila A. Greibach [18]_.
        Modern definitions and algorithms can be found in standard
        textbooks of formal language theory [19]_, [20]_ and on the
        web [21]_.

        Returns
        -------
        grammar_in_cnf : `Grammar`
            New grammar object where all rules adhere to GNF.
            It is equivalent to the original grammar, which means
            that it generates the same language.

        Notes
        -----
        All rules need to be in the following form:

        - ``X → B`` where ``B`` is a nonterminal and ``a`` is a terminal

        Tasks that are simplified by having a grammar in GNF:

        - Deciding if a given string is part of the grammar's language
        - Converting the grammar to a pushdown automaton (PDA) without
          ε-transitions, which is useful because it is guaranteed to
          halt.

        References
        ----------
        .. [18] S. A. Greibach, “A New Normal-Form Theorem for
           Context-Free Phrase Structure Grammars,”
           J. ACM, vol. 12, no. 1, pp. 42–52, Jan. 1965.

        .. [19] J. E. Hopcroft and J. D. Ullman, Introduction to
           Automata Theory, Languages and Computation.
           Addison-Wesley, 1979, pp. 94-99.

        .. [20] E. A. Rich, Automata, Computability and Complexity:
           Theory and Applications. Pearson Education, 2007, p. 169.

        .. [21] Wikipedia

           - `Greibach normal form
             <https://en.wikipedia.org/wiki/Greibach_normal_form>`__

        """
        from . import normalization

        return normalization.to_gnf(self)

    def _is_bcf(self):
        """Check if this grammar is in Binary Choice Form (BCF).

        Returns
        -------
        is_bcf : `bool`

        """
        from . import normalization

        return normalization.is_bcf(self)

    def _to_bcf(self):
        """Convert this grammar to Binary Choice Form (BCF).

        This is my own attempt at modifying a grammar's form to see
        if some G3P method can perform better on it.

        Returns
        -------
        grammar_in_bnf : `Grammar`
            New grammar object where all rules adhere to BCF.
            It is equivalent to the original grammar, which means
            that it generates the same language.

        Notes
        -----
        All nonterminals need to have a maximum of two productions:

        - ``X → A`` where ``A`` is an arbitrary sequence of symbols
        - ``X → A | B`` where ``A`` and ``B`` are arbitrary sequences
          of symbols

        Tasks that are simplified by having a grammar in BNF:

        - Grammatical Evolution (GE) can use codons of size 1
          (=1 bit, 2 possible states) to perform a
          genotype-to-phenotype mapping, i.e. there is no redundancy
          in the individual genes and the modulo rule becomes irrelevant.

        References
        ----------
        I am not aware of literature that introduces such a form.

        """
        from . import normalization

        return normalization.to_bcf(self)

    # Centralized I/O
    @staticmethod
    def _read_file(filepath):
        """Read a text file and return its content as string."""
        try:
            with open(filepath, "r") as file_handle:
                text = file_handle.read()
        except FileNotFoundError:
            message = (
                'Could not read a grammar from file "{}". '
                "The file does not exist.".format(filepath)
            )
            raise FileNotFoundError(message) from None
        except Exception:
            message = (
                'Could not read a grammar from file "{}". '
                "The file exists, but reading text from it failed.".format(filepath)
            )
            raise ValueError(message)
        return text

    @staticmethod
    def _write_file(filepath, text):
        """Write a given string to a text file."""
        with open(filepath, "w") as file_handle:
            file_handle.write(text)

    # Caching of repeated calculations
    def _lookup_or_calc(self, system, attribute, calc_func, *calc_args):
        """Look up a result in this object's _cache attribute.

        If the result is not availabe yet, calculate it once and store
        it for later reuse.

        """
        try:
            # Cache lookup
            return self._cache[system][attribute]
        except KeyError:
            # Calculation
            result = calc_func(*calc_args)
            try:
                self._cache[system][attribute] = result
            except KeyError:
                self._cache[system] = {attribute: result}
            return result

    def _calc_sym_idx_map(self):
        """Calculate a mapping from a symbol to an index.

        This is used in this module and in some G3P systems.

        """
        return {
            sym: num
            for num, sym in enumerate(
                _chain(self.nonterminal_symbols, self.terminal_symbols)
            )
        }

    def _calc_idx_sym_map(self):
        """Calculate a mapping from an index to a symbol.

        This is used in this module and in some G3P systems.

        """
        return list(_chain(self.nonterminal_symbols, self.terminal_symbols))


class DerivationTree:
    """Derivation tree for representing the structure of a string.

    A **derivation tree** [22]_ is the result from generating or
    parsing a string with a grammar. For the latter reason,
    it is also known as **parse tree** [23]_.

    One **derivation tree** represents a single **string** of the
    grammar's language, but multiple **derivations** by which this
    string can be generated. These derivations differ only in the
    order in which the nonterminals are expanded, which is not
    captured by a derivation tree.

    Notes
    -----
    Structure:

    - The tree consists of linked `Node` objects, each of which
      contains either a `TerminalSymbol` or `NonterminalSymbol`
      belonging to the grammar.
    - A rule of a context-free grammar can transform a nonterminal
      symbol into a sequence of symbols. In a derivation tree this is
      represented by a node, which contains the nonterminal symbol.
      This node is connected to one or more child nodes, which are
      ordered and contain the symbols resulting from the rule
      application. The parent node is said to be expanded.
    - The root node of the derivation tree contains the
      starting symbol of the grammar, which is a `NonterminalSymbol`.
    - Each internal node of the derivation tree contains a
      `NonterminalSymbol`.
    - In a fully expanded derivation tree, where no nonterminal symbol
      is left that could be further expanded, each leaf node contains
      a `TerminalSymbol`. Putting the terminal symbols into a sequence
      results in the string that the derivation tree represents.

    Behaviour:

    - A node with a nonterminal can be expanded using a suitable
      production rule of the grammar.
    - Depth first traversal allows to get all terminals in correct
      order. This allows to retrieve the string represented by the
      derivation tree.

    References
    ----------
    .. [22] J. E. Hopcroft and J. D. Ullman, Introduction to
       Automata Theory, Languages and Computation.
       Addison-Wesley, 1979, pp. 82-87.

    .. [23] Wikipedia

       - `Parse tree <https://en.wikipedia.org/wiki/Parse_tree>`__

    """

    __slots__ = ("grammar", "root_node", "_cache")

    # Initialization
    def __init__(self, grammar, root_symbol=None):
        """Create an empty derivation tree with reference to a grammar.

        Parameters
        ----------
        grammar : `Grammar`
        root_symbol : `Symbol`, optional

        """
        if root_symbol is None:
            if grammar is None:
                root_symbol = NonterminalSymbol("")
            else:
                root_symbol = grammar.start_symbol
        self.grammar = grammar
        self.root_node = Node(root_symbol)

    # Copying
    def copy(self):
        """Create a deep copy of the derivation tree.

        The new object is entirely independent of the original object.
        No parts, such as nodes or symbols, are shared between the
        objects.

        """
        dt = DerivationTree(self.grammar)
        dt.root_node = (
            self.root_node.copy()
        )  # leads to recursive copying of all child nodes
        return dt

    def __copy__(self):
        """Create a deep copy of the derivation tree.

        References
        ----------
        - https://docs.python.org/3/library/copy.html

        """
        return self.copy()

    def __deepcopy__(self, memo=None):
        """Create a deep copy of the derivation tree.

        References
        ----------
        - https://docs.python.org/3/library/copy.html

        """
        return self.copy()

    # Representations
    def __repr__(self):
        """Compute the "official" string representation of the derivation tree.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__repr__

        """
        return "<{} object at {}>".format(self.__class__.__name__, hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the derivation tree.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__str__

        """
        return self.to_parenthesis_notation()

    def _repr_html_(self):
        """Provide rich display representation for Jupyter notebooks.

        References
        ----------
        - https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display

        """
        fig = self.plot()
        return fig._repr_html_()

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython.

        References
        ----------
        - https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display

        """
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Equality and hashing
    def __eq__(self, other):
        """Compute whether two derivation trees are equal.

        This checks if two derivation trees have the same structure
        and contain the same symbols.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__eq__

        """
        # Type comparison
        if not isinstance(other, self.__class__):
            return NotImplemented

        # Data comparison
        stk = []
        stk.append((self.root_node, other.root_node))
        while stk:
            nd1, nd2 = stk.pop(0)
            if nd1.symbol.text != nd2.symbol.text or len(nd1.children) != len(
                nd2.children
            ):
                return False
            if nd1.children or nd2.children:
                stk = [(c1, c2) for c1, c2 in zip(nd1.children, nd2.children)] + stk
        return True

    def __ne__(self, other):
        """Compute whether two derivation trees are not equal.

        This checks if two derivation trees have a different structure
        or contain at least one different symbol.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__ne__

        """
        # Type comparison
        if not isinstance(other, self.__class__):
            return NotImplemented

        # Data comparison
        stk = []
        stk.append((self.root_node, other.root_node))
        while stk:
            nd1, nd2 = stk.pop(0)
            if nd1.symbol.text != nd2.symbol.text or len(nd1.children) != len(
                nd2.children
            ):
                return True
            if nd1.children or nd2.children:
                stk = [(c1, c2) for c1, c2 in zip(nd1.children, nd2.children)] + stk
        return False

    def __hash__(self):
        """Calculate a hash value for this object.

        It is used for operations on hashed collections such as `set`
        and `dict`.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__hash__

        """
        val = 0
        stk = []
        stk.append((self.root_node, 0))
        while stk:
            nd, depth = stk.pop(0)
            val += hash(nd.symbol.text) + hash(depth)
            if nd.children:
                depth += 1
                stk = [(ch, depth) for ch in nd.children] + stk
        return val

    # Further representations
    def to_parenthesis_notation(self):
        """Represent the tree as string in single-line parenthesis notation.

        This notation can be found in the Natural Language Toolkit
        (NLTK) book under the name "bracketed structures" [24]_.
        There are various similar representations of trees or
        nested structures [25]_.

        Returns
        -------
        text : `str`
            Tree in parenthesis notation.

        References
        ----------
        .. [24] `NLTK book: Chapter 8
                <https://www.nltk.org/book/ch08.html>`__

        .. [25] Wikipedia

           - `S-expression
             <https://en.wikipedia.org/wiki/S-expression>`__

           - `Newick format
             <https://en.wikipedia.org/wiki/Newick_format>`__

        """

        def traverse(node, seq):
            if isinstance(node.symbol, NonterminalSymbol):
                text = "<{}>".format(node.symbol.text)
            else:
                text = "{}".format(node.symbol.text)
            seq.append(text)
            if node.children:
                seq.append("(")
                for child in node.children:
                    traverse(child, seq)
                seq.append(")")

        seq = []
        traverse(self.root_node, seq)
        text = "(" + "".join(seq) + ")"
        return text

    def to_tree_notation(self):
        """Represent the tree as string in multi-line tree notation.

        This notation is a way to represent both code and data in
        a minimalistic format [26]_ based on newlines and different
        indentation levels. As such it is also suitable to
        represent derivation trees.

        Returns
        -------
        text : `str`
            Tree in "Tree Notation".

        References
        ----------
        .. [26] `Tree Notation <https://treenotation.org>`__

        """

        def traverse(node, seq, depth):
            if isinstance(node.symbol, NonterminalSymbol):
                text = "<{}>".format(node.symbol.text)
            else:
                text = "{}".format(node.symbol.text)
            indent = " " * depth
            seq.append(indent + text)
            if node.children:
                depth += 1
                for child in node.children:
                    traverse(child, seq, depth)

        seq = []
        traverse(self.root_node, seq, 0)
        text = _NEWLINE.join(seq)
        return text

    # Serialization
    def to_tuple(self):
        """Serialize the tree to a tuple.

        Notes
        -----
        The data structure is a tuple that contains two other tuples
        of integers. A depth-first traversal of the tree visits all
        nodes. For each node, its symbol and number of children is
        remembered in two separate tuples. Instead of storing the
        symbols directly, a number is assigned to each symbol of the
        grammar and that concise number is stored instead of a
        potentially long symbol text.

        Returns
        -------
        serialized_tree : `tuple` of two `tuple` objects
            The first tuple describes symbols that are contained
            in the nodes.
            The second tuple describes the number of children
            each node has.

        """
        # Caching
        sim = self.grammar._lookup_or_calc(
            "serialization", "sym_idx_map", self.grammar._calc_sym_idx_map
        )

        # Serialization: Traverse nodes in DFS order, remember symbol and number of children
        sym = []
        cnt = []
        stk = []
        stk.append(self.root_node)
        while stk:
            nd = stk.pop(0)
            sym.append(sim[nd.symbol])
            cnt.append(len(nd.children))
            if nd.children:
                stk = nd.children + stk
        return tuple(sym), tuple(cnt)

    def from_tuple(self, serialized_tree):
        """Deserialize the tree from a tuple.

        Parameters
        ----------
        serialized_tree : `tuple` containing two `tuple` objects
            The first tuple describes symbols that are contained
            in the nodes.
            The second tuple describes the number of children
            each node has.

        """
        # Caching
        ism = self.grammar._lookup_or_calc(
            "serialization", "idx_sym_map", self.grammar._calc_idx_sym_map
        )

        # Deserialization: Iterate over sequence, add nodes in DFS order, jump when indicated
        def traverse(par):
            nonlocal i
            i += 1
            sym = ism[symbols[i]]
            cnt = counters[i]
            nd = Node(sym)
            par.children.append(nd)
            for _ in range(cnt):
                traverse(nd)

        symbols, counters = serialized_tree
        i = -1
        top = Node("")
        traverse(top)
        self.root_node = top.children[0]
        del top

    def to_json(self):
        """Serialize the tree to a JSON string.

        Returns
        -------
        json_string : `str`
            A string in JSON format that represents the tree.

        """
        # Serialization: Traverse nodes in DFS order, remember symbol, type and number of children
        data = []
        stk = []
        stk.append(self.root_node)
        while stk:
            nd = stk.pop(0)
            is_nt = 1 if isinstance(nd.symbol, NonterminalSymbol) else 0
            item = [nd.symbol.text, is_nt, len(nd.children)]
            data.append(item)
            if nd.children:
                stk = nd.children + stk
        json_string = _json.dumps(data, separators=(",", ":"))
        return json_string

    def from_json(self, serialized_tree):
        """Deserialize the tree from a JSON string.

        Parameters
        ----------
        serialized_tree : `str`
            A string in JSON format that represents a tree.

        """
        # Deserialization: Iterate over sequence, add nodes in DFS order, jump when indicated
        def traverse(par):
            nonlocal i
            i += 1
            text, is_nt, cnt = data[i]
            if is_nt:
                sym = NonterminalSymbol(text)
            else:
                sym = TerminalSymbol(text)
            nd = Node(sym)
            par.children.append(nd)
            for _ in range(cnt):
                traverse(nd)

        data = _json.loads(serialized_tree)
        i = -1
        top = Node("")
        traverse(top)
        self.root_node = top.children[0]
        del top

    # Visualization
    def plot(
        self,
        show_node_indices=None,
        layout_engine=None,
        fontname=None,
        fontsize=None,
        shape_nt=None,
        shape_unexpanded_nt=None,
        shape_t=None,
        fontcolor_nt=None,
        fontcolor_unexpanded_nt=None,
        fontcolor_t=None,
        fillcolor_nt=None,
        fillcolor_unexpanded_nt=None,
        fillcolor_t=None,
    ):
        """Plot the derivation tree as labeled, directed graph.

        This method uses Graphviz [27]_ and therefore requires it
        to be installed on the system.

        Parameters
        ----------
        show_node_indices : `bool`, optional
            If `True`, nodes will contain numbers that indicate the
            order in which they were created during tree construction.
        layout_engine : `str`, optional
            Layout engine that calculates node positions.

            Possible values:

            - ``"circo"``
            - ``"dot"``
            - ``"fdp"``
            - ``"neato"``
            - ``"osage"``
            - ``"patchwork"``
            - ``"sfdp"``
            - ``"twopi"``
        fontname : `str`, optional
            Fontname of text inside nodes.
        fontsize : `int` or `str`, optional
            Fontsize of text inside nodes.
        shape_nt : `str`, optional
            Shape of nodes that represent expanded nonterminals.

            Possible values: See `Graphviz documentation: Node shapes
            <http://www.graphviz.org/doc/info/shapes.html>`__
        shape_unexpanded_nt : `str`, optional
            Shape of nodes that represent unexpanded nonterminals.

            Possible values: See `Graphviz documentation: Node shapes
            <http://www.graphviz.org/doc/info/shapes.html>`__
        shape_t : `str`, optional
            Shape of nodes that represent terminals.

            Possible values: See `Graphviz documentation: Node shapes
            <http://www.graphviz.org/doc/info/shapes.html>`__
        fontcolor_nt : `str`, optional
            Fontcolor of nodes that represent expanded nonterminals.
        fontcolor_unexpanded_nt : `str`, optional
            Fontcolor of nodes that represent unexpanded nonterminals.
        fontcolor_t : `str`, optional
            Fontcolor of nodes that represent terminals.
        fillcolor_nt : `str`, optional
            Fillcolor of nodes that represent expanded nonterminals.
        fillcolor_unexpanded_nt : `str`, optional
            Fillcolor of nodes that represent unexpanded nonterminals.
        fillcolor_t : `str`, optional
            Fillcolor of nodes that represent terminals.

        Returns
        -------
        fig : `Figure`
            Figure object containing the plot, allowing to display or
            export it.

        References
        ----------
        .. [27]  `Graphviz <https://www.graphviz.org>`__

        """
        from . import visualization

        fig = visualization.create_graphviz_tree(
            self,
            show_node_indices,
            layout_engine,
            fontname,
            fontsize,
            shape_nt,
            shape_unexpanded_nt,
            shape_t,
            fontcolor_nt,
            fontcolor_unexpanded_nt,
            fontcolor_t,
            fillcolor_nt,
            fillcolor_unexpanded_nt,
            fillcolor_t,
        )
        return fig

    # Reading contents of the tree
    def nodes(self, order="dfs"):
        """Get all nodes as a list in order of a chosen tree traversal.

        The nodes in a tree can be visited in different orders with
        different tree traversal methods [28]_.

        Parameters
        ----------
        order : `str`, optional
            Possible values:

            - ``"dfs"``: Traversal in order of a depth-first search
            - ``"bfs"``: Traversal in order of a breadth-first search

        Returns
        -------
        nodes : `list` of `Node` objects

        References
        ----------
        .. [28] Wikipedia

           - `Tree traversal <https://en.wikipedia.org/wiki/Tree_traversal>`__
           - `Depth-first_search <https://en.wikipedia.org/wiki/Depth-first_search>`__
           - `Breadth-first search <https://en.wikipedia.org/wiki/Breadth-first_search>`__

        """
        # Argument processing
        _ap.str_arg("order", order, vals=("dfs", "bfs"))

        # Generate node list by tree traversal
        if order == "dfs":
            nodes = self._depth_first_traversal()
        else:
            nodes = self._breadth_first_traversal()
        return nodes

    def _depth_first_traversal(self):
        """Traverse the tree with a depth-first search.

        Returns
        -------
        nodes : `list` of `Node` objects

        """
        # Note: List-based implementation is faster than deque and extend(reversed(node.children))
        nodes = []
        stk = []
        stk.append(self.root_node)
        while stk:
            nd = stk.pop(0)
            nodes.append(nd)
            if nd.children:
                stk = nd.children + stk
        return nodes

    def _breadth_first_traversal(self):
        """Traverse the tree with a breadth-first search.

        Returns
        -------
        nodes : `list` of `Node` objects

        """
        nodes = []
        queue = _collections.deque()
        queue.append(self.root_node)
        while queue:
            node = queue.popleft()
            nodes.append(node)
            if node.children:
                queue.extend(node.children)
        return nodes

    def leaf_nodes(self):
        """Get all leaf nodes by a tree traversal in depth-first order.

        Returns
        -------
        nodes : `list` of `Node` objects

        """
        nodes = []
        stack = _collections.deque()  # LIFO
        stack.append(self.root_node)
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(
                    reversed(node.children)
                )  # add first child last, so it becomes first
            else:
                nodes.append(node)
        return nodes

    def internal_nodes(self):
        """Get all internal nodes by a tree traversal in depth-first order.

        Returns
        -------
        nodes : `list` of `Node` objects

        """
        nodes = []
        stack = _collections.deque()  # LIFO
        stack.append(self.root_node)
        while stack:
            node = stack.pop()
            if node.children:
                nodes.append(node)
                stack.extend(
                    reversed(node.children)
                )  # add first child last, so it becomes first
        return nodes

    def tokens(self):
        """Get a sequence of tokens in leaf nodes from left to right.

        Returns
        -------
        tokens : `list` of `Symbol` objects

        """
        return [nd.symbol for nd in self.leaf_nodes()]

    def string(self, separator=""):
        """Get the string contained in the leaf nodes of the tree.

        - If the tree is fully expanded, no nonterminal symbol is left
          in the leaf nodes, so the obtained string is composed only of
          terminals and belongs to the language of the grammar.

        - If the tree is not fully expanded, the result is not a string
          but a so called sentential form that still includes
          nonterminal symbols.

        Parameters
        ----------
        separator : `str`, optional
            The separator text used between terminal symbols.

        Returns
        -------
        string : `str`

        """
        return separator.join(
            nd.symbol.text
            if isinstance(nd.symbol, TerminalSymbol)
            else "<{}>".format(nd.symbol.text)
            for nd in self.leaf_nodes()
        )

    def derivation(self, derivation_order="leftmost", newline=True):
        """Get a derivation that fits to the structure of the tree.

        Parameters
        ----------
        derivation_order : `str`, optional
            Order in which nonterminals are expanded during the
            step-by-step derivation.

            Possible values:

            - ``"leftmost"``: Expand the leftmost nonterminal
              in each step.
            - ``"rightmost"``: Expand the rightmost nonterminal
              in each step.
            - ``"random"``: Expand a random nonterminal in each step.
        newline : `bool`, optional
            If `True`, the derivation steps are placed on separate
            lines by adding newline characters between them.

        Returns
        -------
        derivation : `str`

        """
        # Argument processing
        _ap.str_arg(
            "derivation_order",
            derivation_order,
            vals=("leftmost", "rightmost", "random"),
        )
        _ap.bool_arg("newline", newline)

        # Helper functions
        def next_derivation_step(derivation, old_node, new_nodes):
            last_sentential_form = derivation[-1]
            new_sentential_form = last_sentential_form[:]
            insert_idx = last_sentential_form.index(old_node)
            new_sentential_form[insert_idx : insert_idx + 1] = new_nodes
            if new_sentential_form:
                derivation.append(new_sentential_form)
            return derivation

        def symbol_seq_repr(symbol_seq):
            seq_repr = []
            for sym in symbol_seq:
                if isinstance(sym, NonterminalSymbol):
                    seq_repr.append("<{}>".format(sym))
                else:
                    seq_repr.append(str(sym))
            return "".join(seq_repr)

        def node_seq_repr(node_seq):
            symbol_seq = [node.symbol for node in node_seq]
            return symbol_seq_repr(symbol_seq)

        # Traverse the tree and collect nodes according to the method
        if derivation_order == "leftmost":
            derivation = [[self.root_node]]
            stack = [self.root_node]
            while stack:
                idx = 0
                nt_node = stack.pop(idx)
                derivation = next_derivation_step(derivation, nt_node, nt_node.children)
                new_nt_nodes = [node for node in nt_node.children if node.children]
                stack = new_nt_nodes + stack
        elif derivation_order == "rightmost":
            derivation = [[self.root_node]]
            stack = [self.root_node]
            while stack:
                idx = len(stack) - 1
                nt_node = stack.pop(idx)
                derivation = next_derivation_step(derivation, nt_node, nt_node.children)
                new_nt_nodes = [node for node in nt_node.children if node.children]
                stack.extend(new_nt_nodes)
        elif derivation_order == "random":
            derivation = [[self.root_node]]
            stack = [self.root_node]
            while stack:
                idx = _random.randint(0, len(stack) - 1)
                nt_node = stack.pop(idx)
                derivation = next_derivation_step(derivation, nt_node, nt_node.children)
                new_nt_nodes = [node for node in nt_node.children if node.children]
                stack.extend(new_nt_nodes)

        if newline:
            sep = "{}=> ".format(_NEWLINE)
        else:
            sep = " => "
        derivation_string = sep.join(node_seq_repr(item) for item in derivation)
        return derivation_string

    def num_expansions(self):
        """Calculate the number of expansions contained in the tree.

        Returns
        -------
        num_expansions : `int`

        """
        num_expansions = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop(0)
            if node.children:
                num_expansions += 1
                stack = node.children + stack
        return num_expansions

    def num_nodes(self):
        """Calculate the number of nodes contained in the tree.

        Returns
        -------
        num_nodes : `int`

        """
        num_nodes = 1
        stack = [self.root_node]
        while stack:
            node = stack.pop(0)
            if node.children:
                num_nodes += len(node.children)
                stack = node.children + stack
        return num_nodes

    def depth(self):
        """Calculate the depth of the derivation tree.

        The depth [29]_ [30]_ [31]_ of a tree is the number of edges
        in the longest path (in the graph-theoretic sense) from the
        root node to a leaf node.

        Returns
        -------
        depth : `int`

        References
        ----------
        .. [29] J. R. Koza, Genetic Programming: On the Programming of
           Computers by Means of Natural Selection. Cambridge, Mass.:
           MIT PR, 1992, p. 92.

           - p. 92: "The depth of a tree is defined as the length of
             the longest nonbacktracking path from the root to an
             endpoint."

        .. [30] R. Poli, W. B. Langdon, and N. F. McPhee, A Field Guide
           to Genetic Programming,
           Morrisville, NC: Lulu Enterprises, UK Ltd, 2008.

           - p. 12: "The depth of a node is the number of edges that
             need to be traversed to reach the node starting from the
             tree’s root node (which is assumed to be at depth 0).
             The depth of a tree is the depth of its deepest leaf"

        .. [31] Wikipedia

           - `Tree (data structure)
             <https://en.wikipedia.org/wiki/Tree_%28data_structure%29>`__

        """
        md = 0
        stk = []
        stk.append((self.root_node, 0))
        while stk:
            node, depth = stk.pop(0)
            if depth > md:
                md = depth
            if node.children:
                depth += 1
                stk = [(ch, depth) for ch in node.children] + stk
        return md

    def is_completely_expanded(self):
        """Check if the tree contains only expanded nonterminal symbols.

        Returns
        -------
        is_completely_expanded : `bool`
            If `True`, the tree is fully expanded which means that it
            contains only terminals in its leave nodes and that it
            represents complete derivations that lead to a string of
            the grammar's language.

        """
        is_completely_expanded = True
        if any(node.contains_unexpanded_nonterminal() for node in self.leaf_nodes()):
            is_completely_expanded = False
        return is_completely_expanded

    # Convenience methods for G3P systems
    def _expand(self, nd, sy):
        """Expand a node in the tree by adding child nodes to it."""
        # Syntax minified for minor optimization due to large number of calls
        d = []
        a = d.append
        for s in sy:
            n = Node(s)
            a(n)
            nd.children.append(n)
        return d

    def _is_deeper_than(self, value):
        """Detect if the derivation tree is deeper than a given value.

        This method is required by some tree-based G3P systems.

        """
        stk = []
        stk.append((self.root_node, 0))
        while stk:
            nd, depth = stk.pop(0)
            if depth > value:
                return True
            if nd.children:
                depth += 1
                stk = [(c, depth) for c in nd.children] + stk
        return False


class Node:
    """Node inside a derivation tree.

    A node contains a symbol and refers to child nodes.

    """

    __slots__ = ("symbol", "children")

    # Initialization
    def __init__(self, sy, ch=None):
        """Create a node.

        Parameters
        ----------
        sy : Symbol
        ch : `list` of child `Node` objects, optional

        """
        self.symbol = sy
        self.children = [] if ch is None else ch

    # Copying
    def copy(self):
        """Create a deep copy of the node and its children."""
        # Recursive copy
        ch = None if self.children is None else [nd.copy() for nd in self.children]
        sy = self.symbol.copy()
        return self.__class__(sy, ch)

    def __copy__(self):
        """Create a deep copy of the node and its children."""
        return self.copy()

    def __deepcopy__(self, memo):
        """Create a deep copy of the node and its children."""
        return self.copy()

    # Representations
    def __repr__(self):
        """Compute the "official" string representation of the node."""
        return "<{} object at {}>".format(self.__class__.__name__, hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the node."""
        return self.symbol.text

    # Symbol type requests
    def contains_terminal(self):
        """Check if the node contains a terminal symbol.

        Returns
        -------
        contains_t : `bool`

        """
        return isinstance(self.symbol, TerminalSymbol)

    def contains_nonterminal(self):
        """Check if the node contains a nonterminal symbol.

        Returns
        -------
        contains_nt : `bool`

        """
        return isinstance(self.symbol, NonterminalSymbol)

    def contains_unexpanded_nonterminal(self):
        """Check if the node contains a nonterminal symbol and has no child nodes.

        Returns
        -------
        contains_unexpanded_nt : `bool`

        """
        return not self.children and isinstance(self.symbol, NonterminalSymbol)


class Symbol:
    """Symbol inside a grammar or derivation tree.

    A symbol can be either a nonterminal or terminal of the grammar
    and it has a `text` attribute.

    """

    __slots__ = ("text",)

    # Initialization
    def __init__(self, text):
        """Create a symbol in the sense of formal language theory.

        A symbol contains a text that can be empty, a single letter
        or multiple letters.

        Parameters
        ----------
        text : `str`
            Text contained in the symbol object.

        """
        self.text = text

    # Copying
    def copy(self):
        """Create a deep copy of the symbol."""
        return self.__class__(self.text)

    def __copy__(self):
        """Create a deep copy of the symbol."""
        return self.__class__(self.text)

    def __deepcopy__(self, memo):
        """Create a deep copy of the symbol."""
        return self.__class__(self.text)

    # Representations
    def __str__(self):
        """Compute the "informal" string representation of the symbol."""
        return self.text

    # Comparison operators for sorting
    def __eq__(self, other):
        """Compute whether two symbols are equal."""
        return self.text == other.text and isinstance(other, self.__class__)

    def __ne__(self, other):
        """Compute whether two symbols are not equal."""
        return self.text != other.text or not isinstance(other, self.__class__)

    def __lt__(self, other):
        """Compute whether a symbol comes before another when ordered."""
        return self.text < other.text

    def __le__(self, other):
        """Compute __lt__ or __eq__ in one call."""
        return self.text <= other.text

    def __gt__(self, other):
        """Compute whether a symbol comes after another when ordered."""
        return self.text > other.text

    def __ge__(self, other):
        """Compute __gt__ or __eq__ in one call."""
        return self.text >= other.text

    # Hash for usage as dict key
    def __hash__(self):
        """Calculate a hash value for the symbol.

        It is used for operations on hashed collections such as `set`
        and `dict`.

        """
        return hash(self.text)


class NonterminalSymbol(Symbol):
    """Nonterminal symbol inside a grammar or derivation tree."""

    __slots__ = ()

    def __repr__(self):
        """Compute the "official" string representation of the symbol."""
        return "NT({})".format(repr(self.text))


class TerminalSymbol(Symbol):
    """Terminal symbol inside a grammar or derivation tree."""

    __slots__ = ()

    def __repr__(self):
        """Compute the "official" string representation of the symbol."""
        return "T({})".format(repr(self.text))
