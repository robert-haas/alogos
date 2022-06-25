import collections as _collections
import copy as _copy
import json as _json
import random as _random
from itertools import chain as _chain

from ordered_set import OrderedSet as _OrderedSet

from .. import _logging
from .. import exceptions as _exceptions
from .._utilities import argument_processing as _ap
from .._utilities.operating_system import NEWLINE as _NEWLINE


class Grammar:
    """Data structure for a context-free grammar (CFG).

    - **Creation**: The grammar can be created from a text in **BNF** or **EBNF** notation.
      There are different variants of these notations, many of which can be read
      and written by this package.

      The input text can be provided as string or text file and may be passed
      during object creation or later to a suitable reading method.

    - **Representation**: The grammar can be represented in different forms,
      for example by listing all its symbols and productions,
      by BNF and EBNF text, or visually as **syntax diagram**.

    - **Usage**: The grammar can be used for various tasks.

        - **Generate strings**

            - A random string of the language
            - A certain string of the language, which results from a derivation
              where the productions used in each step are selected according
              to a list of integers
            - All strings of the language

        - **Parse strings**

            - Recognize whether a given string is part of the language
            - Parse the string to get a parse tree which adds structure to it

        - **Search for optimal strings**

            - Given an objective function that can assign a numerical value to each string,
              a search algorithm can heuristically try to find an optimal string that
              maximizes or minimizes this value without generating each string of the language

    Notes
    -----
    A formal grammar is a tuple ``(N, T, P, S)`` where

    - ``N`` is a set of nonterminal symbols.
    - ``T`` is a set of terminal symbols, which has no overlap with ``N``.
    - ``P`` is a set of production rules.
    - ``S`` is the start symbol, which is an element of ``N``.

    This specification can be implemented in many different ways.
    In this class, they are available as the attributes ``nonterminal_symbols`` (an
    `OrderedDict <https://docs.python.org/3/library/collections.html#collections.OrderedDict>`__
    where only keys are used),
    ``terminal_symbols`` (an
    `OrderedDict <https://docs.python.org/3/library/collections.html#collections.OrderedDict>`__
    where only keys are used),
    ``production_rules`` (a
    `dict <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`__),
    and ``start_symbol`` (a :ref:`NonterminalSymbol`).

    A context-free grammar (CFG) is a type of grammar where the left-hand side of each
    production rule is a single nonterminal symbol, while the right-hand side
    is a sequence of symbols that can consist of a combination of terminal symbols,
    nonterminal symbols and the empty string ``""``
    (often denoted by the greek letter ``ɛ`` or less often ``λ``).

    References
    ----------
    - Books on formal language theory that discuss formal grammars in detail:

        - `Hopcroft et al.: Introduction to Automata Theory, Languages, and Computation,
          3rd edition, 2006
          <https://www.pearson.com/us/higher-education/program/PGM64331.html>`__
          - The definitions and notation used here correspond closely to those
          presented in this classical textbook.

    - Wikipedia articles

        - `Formal grammar <https://en.wikipedia.org/wiki/Formal_grammar>`__
        - `Context-free grammar <https://en.wikipedia.org/wiki/Context-free_grammar>`__
        - `Context-free language <https://en.wikipedia.org/wiki/Context-free_language>`__
        - `Terminal and nonterminal symbols
          <https://en.wikipedia.org/wiki/Terminal_and_nonterminal_symbols>`__
        - `Production rule <https://en.wikipedia.org/wiki/Production_(computer_science)>`__
        - `Backus-Naur form (BNF) <https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form>`__
        - `Extended Backus-Naur form (EBNF)
          <https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form>`__

    """

    __slots__ = (
        'nonterminal_symbols', 'terminal_symbols', 'production_rules', 'start_symbol', '_cache')

    # Initialization and reset
    def __init__(self, bnf_text=None, bnf_file=None, ebnf_text=None, ebnf_file=None, **kwargs):
        """Construct the grammar from text in BNF or EBNF notation.

        Parameters
        ----------
        bnf_text : str
            String that contains a grammar in BNF notation.
        bnf_file : str
            Filepath of a text file that contains a grammar in BNF notation.
        ebnf_text : str
            String that contains a grammar in EBNF notation.
        ebnf_file : str
            Filepath of a text file that contains a grammar in EBNF notation.
        kwargs
            Further keyword arguments are forwarded to the function that parses the grammar
            from text in BNF or EBNF notation.
            See :meth:`~Grammar.from_bnf_text` and :meth:`~Grammar.from_ebnf_text`
            for details.

        """
        # Argument processing
        if sum(inp is not None for inp in (bnf_text, bnf_file, ebnf_text, ebnf_file)) > 1:
            _logging.warn_multiple_grammar_specs()

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
        """Initialize the grammar by assigning empty containers for symbols and productions.

        Notes
        -----
        dict() is used as data structure for rules instead of OrderedDict
        from the itertools module, and instead of OrderedSet from external
        orderedset library for symbols, because it guarantees order and is
        faster and introduces no dependencies and portability issues. Here
        is more background:

        - Since Python 3.6, dict in CPython remembers the insertion order of keys.
        - Since Python 3.7 this is considered a language feature.
        - If order is not preserved, no algorithm here fails, only output becomes less readable.

        """
        self.terminal_symbols = _OrderedSet()
        self.nonterminal_symbols = _OrderedSet()
        self.production_rules = dict()
        self.start_symbol = None
        # Cache to store results of some calculations instead of repeating them
        self._cache = dict()

    # Copying
    def copy(self):
        """Create a deep copy of the grammar, so the new object is entirely independent."""
        return self.__deepcopy__()

    def __copy__(self):
        """Create a shallow copy of the grammar (without cache).

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
        """Create a deep copy of the grammar (without cache).

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
        return '<{} object at {}>'.format(self.__class__.__name__, hex(id(self)))

    def __str__(self):
        """Compute the "informal" or nicely printable string representation of the grammar.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__str__

        """
        sep = '{}  '.format(_NEWLINE)
        msg = []
        msg.append('Nonterminal symbols:{sep}'.format(sep=sep))
        if self.nonterminal_symbols:
            nt_text = sep.join('{idx}: {nt}'.format(idx=i, nt=repr(sym)) for i, sym
                               in enumerate(list(self.nonterminal_symbols)))
        else:
            nt_text = 'Empty set'
        msg.append(nt_text)
        msg.append('{nl}{nl}Terminal symbols:{sep}'.format(nl=_NEWLINE, sep=sep))
        if self.terminal_symbols:
            t_text = sep.join('{idx}: {terminal}'.format(idx=i, terminal=repr(sym)) for i, sym
                              in enumerate(list(self.terminal_symbols)))
        else:
            t_text = 'Empty set'
        msg.append(t_text)
        msg.append('{nl}{nl}Start symbol:{sep}{sym}'.format(
            nl=_NEWLINE, sep=sep, sym=repr(self.start_symbol)))
        msg.append('{nl}{nl}Production rules:'.format(nl=_NEWLINE, sep=sep))
        if self.production_rules:
            i = 0
            for lhs, rhs_list in self.production_rules.items():
                for rhs in rhs_list:
                    msg.append('{sep}{idx}: {lhs} -> {rhs}'.format(
                        sep=sep, idx=i, lhs=repr(lhs), rhs=' '.join(repr(sym) for sym in rhs)))
                    i += 1
        else:
            msg.append('{sep}Empty set'.format(sep=sep))
        text = ''.join(msg)
        return text

    def _repr_html_(self):
        """Provide rich display representation in HTML format for Jupyter notebooks."""
        fig = self.plot()
        return fig._repr_html_()

    def _repr_pretty_(self, p, cycle):
        """Provide a rich display representation for IPython interpreters."""
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Equality and hashing
    def __eq__(self, other):
        # Type comparison
        if not isinstance(other, self.__class__):
            return False

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
        return hash(str(self))

    # Reading
    def from_bnf_text(self, bnf_text,
                      defining_symbol='::=', rule_separator_symbol='|',
                      start_nonterminal_symbol='<', end_nonterminal_symbol='>',
                      start_terminal_symbol='', end_terminal_symbol='',
                      start_terminal_symbol2='', end_terminal_symbol2='',
                      verbose=False):
        """Read a grammar specification in BNF notation from a string.

        This method resets the grammar object and then uses the provided information.

        Parameters
        ----------
        bnf_text : str
            String with grammar specification in BNF notation.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
            Alternative symbol indicating the end of a terminal.
        verbose : bool
            If ``True``, messages will be printed during processing the input text that show
            which rules and symbols are found one after another. This can be useful to see
            what went wrong when the generated grammar does not look or behave as expected.

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

    def from_bnf_file(self, filepath,
                      defining_symbol='::=', rule_separator_symbol='|',
                      start_nonterminal_symbol='<', end_nonterminal_symbol='>',
                      start_terminal_symbol='', end_terminal_symbol='',
                      start_terminal_symbol2='', end_terminal_symbol2='',
                      verbose=False):
        """Read a grammar specification in BNF notation from a text file.

        This method resets the grammar object and then uses the provided information.

        Parameters
        ----------
        filepath : str
            Text file with grammar specification in BNF notation.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
            Alternative symbol indicating the end of a terminal.
        verbose : bool
            If ``True``, messages will be printed during processing the input text that show
            which rules and symbols are found one after another. This can be useful to see
            what went wrong when the generated grammar does not look or behave as expected.

        """
        # Argument processing
        filepath = _ap.str_arg('filepath', filepath)

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

    def from_ebnf_text(self, ebnf_text,
                       defining_symbol='=', rule_separator_symbol='|',
                       start_nonterminal_symbol='', end_nonterminal_symbol='',
                       start_terminal_symbol='"', end_terminal_symbol='"',
                       start_terminal_symbol2="'", end_terminal_symbol2="'",
                       verbose=False):
        """Read a grammar specification in EBNF notation from a string.

        This method resets the grammar object and then uses the provided information.

        Parameters
        ----------
        ebnf_text : str
            String with grammar specification in EBNF notation.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
            Alternative symbol indicating the end of a terminal.
        verbose : bool
            If ``True``, messages will be printed during processing the input text that show
            which rules and symbols are found one after another. This can be useful to see
            what went wrong when the generated grammar does not look or behave as expected.

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

    def from_ebnf_file(self, filepath,
                       defining_symbol='=', rule_separator_symbol='|',
                       start_nonterminal_symbol='', end_nonterminal_symbol='',
                       start_terminal_symbol='"', end_terminal_symbol='"',
                       start_terminal_symbol2="'", end_terminal_symbol2="'",
                       verbose=False):
        """Read a grammar specification in EBNF notation from a text file.

        This method resets the grammar object and then uses the provided information.

        Parameters
        ----------
        filepath : str
            Text file with grammar specification in EBNF notation.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
            Alternative symbol indicating the end of a terminal.
        verbose : bool
            If ``True``, messages will be printed during processing the input text that show
            which rules and symbols are found one after another. This can be useful to see
            what went wrong when the generated grammar does not look or behave as expected.

        """
        # Argument processing
        filepath = _ap.str_arg('filepath', filepath)

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

    @staticmethod
    def _read_file(filepath):
        """Read a text file and return its content as string."""
        try:
            with open(filepath, 'r') as file_handle:
                text = file_handle.read()
        except FileNotFoundError:
            message = (
                'Could not read a grammar from file "{}". '
                'The file does not exist.'.format(filepath))
            raise FileNotFoundError(message) from None
        except Exception:
            message = (
                'Could not read a grammar from file "{}". '
                'The file exists, but reading text from it failed.'.format(filepath))
            raise ValueError(message)
        return text

    # Writing
    def to_bnf_text(self, rules_on_separate_lines=True,
                    defining_symbol='::=', rule_separator_symbol='|',
                    start_nonterminal_symbol='<', end_nonterminal_symbol='>',
                    start_terminal_symbol='', end_terminal_symbol='',
                    start_terminal_symbol2='', end_terminal_symbol2=''):
        """Write the grammar in BNF notation to a string.

        Parameters
        ----------
        rule_on_separate_lines : bool
            If ``True``, each rule for a nonterminal is put onto a separate line.
            If ``False``, all rules for a nonterminal are grouped onto one line.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
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

    def to_bnf_file(self, filepath, rules_on_separate_lines=True,
                    defining_symbol='::=', rule_separator_symbol='|',
                    start_nonterminal_symbol='<', end_nonterminal_symbol='>',
                    start_terminal_symbol='', end_terminal_symbol='',
                    start_terminal_symbol2='', end_terminal_symbol2=''):
        """Write the grammar in BNF notation to a text file.

        Parameters
        ----------
        filepath : str
            Filepath of the text file that shall be generated.
        rule_on_separate_lines : bool
            If ``True``, each rule for a nonterminal is put onto a separate line.
            If ``False``, all rules for a nonterminal are grouped onto one line.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
            Alternative symbol indicating the end of a terminal.

        """
        # Argument processing
        filepath = _ap.str_arg('filepath', filepath)

        # Generate BNF text
        text = self.to_bnf_text(
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
        with open(filepath, 'w') as file_handle:
            file_handle.write(text)

    def to_ebnf_text(self, rules_on_separate_lines=True,
                     defining_symbol='=', rule_separator_symbol='|',
                     start_nonterminal_symbol='', end_nonterminal_symbol='',
                     start_terminal_symbol='"', end_terminal_symbol='"',
                     start_terminal_symbol2='"', end_terminal_symbol2='"'):
        """Write the grammar in EBNF notation to a string.

        Parameters
        ----------
        rule_on_separate_lines : bool
            If ``True``, each rule for a nonterminal is put onto a separate line.
            If ``False``, all rules for a nonterminal are grouped onto one line.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
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

    def to_ebnf_file(self, filepath, rules_on_separate_lines=True,
                     defining_symbol='=', rule_separator_symbol='|',
                     start_nonterminal_symbol='', end_nonterminal_symbol='',
                     start_terminal_symbol='"', end_terminal_symbol='"',
                     start_terminal_symbol2='"', end_terminal_symbol2='"'):
        """Write the grammar in EBNF notation to a text file.

        Parameters
        ----------
        filepath : str
            Filepath of the text file that shall be generated.
        rule_on_separate_lines : bool
            If ``True``, each rule for a nonterminal is put onto a separate line.
            If ``False``, all rules for a nonterminal are grouped onto one line.
        defining_symbol : str
            Symbol between left-hand side and right-hand side of a rule.
        rule_separator_symbol : str
            Symbol between alternative productions of a rule.
        start_nonterminal_symbol : str
            Symbol indicating the start of a nonterminal.
        end_nonterminal_symbol : str
            Symbol indicating the end of a nonterminal.
        start_terminal_symbol : str
            Symbol indicating the start of a terminal.
        end_terminal_symbol : str
            Symbol indicating the end of a terminal.
        start_terminal_symbol2 : str
            Alternative symbol indicating the start of a terminal.
        end_terminal_symbol2 : str
            Alternative symbol indicating the end of a terminal.

        """
        # Argument processing
        filepath = _ap.str_arg('filepath', filepath)

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
        with open(filepath, 'w') as file_handle:
            file_handle.write(ebnf_text)

    # Normal forms
    def is_cnf(self):
        """Check if this grammar is in Chomsky Normal Form (CNF)."""
        from . import normalization

        return normalization.is_cnf(self)

    def to_cnf(self):
        """Convert this grammar to Chomsky Normal Form (CNF)."""
        from . import normalization

        return normalization.to_cnf(self)

    def is_gnf(self):
        """Check if this grammar is in Greibach Normal Form (GNF)."""
        from . import normalization

        return normalization.is_gnf(self)

    def to_gnf(self):
        """Convert this grammar to Greibach Normal Form (CNF)."""
        from . import normalization

        return normalization.to_gnf(self)

    def is_bcnf(self):
        """Check if this grammar is in Binary Choice Normal Form (BCNF)."""
        from . import normalization

        return normalization.is_bcnf(self)

    def to_bcnf(self):
        """Convert this grammar to Binary Choice Normal Form (BCNF)."""
        from . import normalization

        return normalization.to_bcnf(self)

    # String generation
    def generate_derivation_tree(self, method='random-weighted', *args, **kwargs):
        """Generate a derivation tree.

        Notes
        -----
        A tree represents a single string but multiple derivations.

        - The leaf nodes of the tree read from left to right form a string of the
          language defined by the grammar.
        - A depth-first walk over the tree that always chooses the leftmost item gives
          the leftmost derivation, while one that always chooses the rightmost item gives
          the rightmost derivation. There may be many more ways to expand one nonterminal
          after another, leading to a plethora of derivations represented by a single tree.

        Parameters
        ----------
        method : str
            Determines how the derivation is constructed.

            Possible values:
            - 'cfggp' for mapping process from context-free grammar genetic programming (CFG-GP)
            - 'cfggpst' for mapping process from CFG-GP with serialized tree (CFG-GP-ST)
            - 'ge' for mapping process from grammatical evolution (GE)
            - 'random-simple' for random derivation
            - 'random-weighted' for random derivation with improved convergence
        kwargs
            Further keyword arguments are forwarded to the chosen method.

        Returns
        -------
        derivation_tree : :obj:`.DerivationTree`
            The derivation tree generated with the chosen method.

        References
        ----------
        - See section on method description.

        """
        # Argument processing
        from .. import systems
        from . import generation

        name_method_map = {
            # Derivations with GBGP systems
            'cfggp': systems.cfggp.mapping.forward,
            'cfggpst': systems.cfggpst.mapping.forward,
            'ge': systems.ge.mapping.forward,
            # Random derivations
            'random-simple': generation.derivation.generate_derivation_simple,
            'random-weighted': generation.derivation.generate_derivation_weighted,
        }
        _ap.str_arg('method', method, vals=name_method_map.keys())
        method = name_method_map[method]

        # Mapping
        kwargs['return_derivation_tree'] = True
        _, dt = method(self, *args, **kwargs)
        return dt

    def generate_derivation(self, method='random-weighted', *args, separate_lines=True, **kwargs):
        """Generate a derivation.

        Parameters
        ----------
        TODO

        Returns
        -------
        string : str
            The string generated with the chosen method.
        kwargs : TODO

        """
        dt = self.generate_derivation_tree(method, *args, **kwargs)
        derivation = dt.derivation(separate_lines=separate_lines)
        return derivation

    def generate_string(self, method='random-weighted', *args, separator='', **kwargs):
        """Generate a string of the language L(G) defined by the grammar G.

        Parameters
        ----------
        TODO

        Returns
        -------
        string : str
            The string generated with the chosen method.
        kwargs : TODO

        """
        dt = self.generate_derivation_tree(method, *args, **kwargs)
        string = dt.string(separator)
        return string

    def generate_language(self, max_steps=None, sort_order='discovered',
                          verbose=None, return_details=None):
        """Generate the formal language defined by the grammar.

        This algorithm recursively constructs the language, i.e. set of all strings,
        defined by the grammar. It can be stopped prematurely to get a subset of the
        entire language. It is also possible to return not only the language of the
        start symbol, which is the language of the grammar, but also the
        languages of each other nonterminal symbol.

        Parameters
        ----------
        max_steps : int
            The maximum number of recursive steps during language generation.
            It can be used to stop the algorithm before it can construct all strings
            of the language. Instead a list of valid strings found so far will be returned,
            which is a subset of the entire language. This is necessary to get a result
            if the grammar defines a very large or infinite language.

            Note that each recursive step uses the strings known so far and inserts
            them into the right-hand sides of production rules to see if any new
            strings can be discovered. Therefore simpler strings are found before
            more complex ones that require more expansions. If the number of steps
            is too little to form a single string belonging to the language of the
            start symbol, the result will be an empty list.
        sort_order : str
            The language is returned as a list of strings, which can be sorted in different
            ways.

            Possible values:

            - ``"discovered"``: Strings are returned in the order they were discovered.
            - ``"lex"``: Lexicographic, i.e. the order used in lexicons, which means the
              alphabetic order extended to non-alphabet characters like numbers.
              Python's built-in ``sort()`` function delivers it by default.
            - ``"shortlex"``: Strings are sorted primarily by their length.
              Those with the same length are further sorted in lexicographic order.
        verbose : bool
            If ``True``, detailed messages are printed during language generation.
            If ``False``, no output is generated.
        return_details : bool
            If True, the return value is a dict with nonterminals as keys and their
            languages as values. The language of the start symbol is the language of the
            grammar, but each nonterminal has its own sub-language that can be of interest too.

        Returns
        -------
        language : list of str
            The formal language L(G) defined by the grammar G.
            If ``return_details`` is set to ``True``, the return value is a dict
            where each key is a nonterminal of the grammar and each value the
            language (set of strings) of the nonterminal.

        Warns
        -----
        UserWarning
            If no value is provided for ``max_steps``, internally an unrealistically
            large value of 1_000_000_000 is assigned to it. In the unlikely case this is
            ever reached, a warning will be raised if the language generation
            did not generate all strings of the language.

        References
        ----------
        - Wikipedia

            - `Shortlex order <https://en.wikipedia.org/wiki/Shortlex_order>`__
            - `Lexicographic order <https://en.wikipedia.org/wiki/Lexicographical_order>`__

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
    def recognize_string(self, string, parser='earley'):
        """Test if a string belongs to the language of the grammar.

        This package uses parsers provided by
        `Lark <https://lark-parser.readthedocs.io/en/latest/>`__.
        By default, Lark's
        `Earley parser <https://lark-parser.readthedocs.io/en/latest/parsers/#earley>`__
        is chosen, because it works with any context-free grammar.
        If your grammar has a form that is compatible with Lark's
        `LALR(1) parser <https://lark-parser.readthedocs.io/en/latest/parsers/#lalr1>`_,
        then switching to it may give much better performance, especially
        when parsing long strings.

        Parameters
        ----------
        string : str
            Candidate string which can be recognized to be a member of the grammar's language.
        parser : str
            Parsing algorithm used to analyze the string.

            Possible values:

            - ``earley``: Can parse any context-free grammar.

              Performance: The algorithm has a time complexity of O(n^3),
              but if the grammar is unambiguous it is reduced to O(n^2)
              and for most LR grammars it is O(n).

            - ``lalr``: Can parse only a subset of context-free grammars, which have a form
              that allows very efficient parsing with a LALR(1) parser.

              Performance: The algorithm has a time complexity of O(n).

        Returns
        -------
        is_recognized : bool
            ``True`` if the string belongs to the grammar's language, ``False`` if it does not.

        References
        ----------
        - Wikipedia

            - `Parsing <https://en.wikipedia.org/wiki/Parsing>`__
            - `Earley parser <https://en.wikipedia.org/wiki/Earley_parser>`__
            - `LALR parser <https://en.wikipedia.org/wiki/LALR_parser>`__

        """
        try:
            self.parse_string(string, parser, get_multiple_trees=False)
        except _exceptions.ParserError:
            return False
        return True

    def parse_string(self, string, parser='earley', get_multiple_trees=False, max_num_trees=None):
        """Parse a string which belongs to the language of the grammar.

        This package uses parsers provided by
        `Lark <https://lark-parser.readthedocs.io/en/latest/>`__.
        By default, Lark's
        `Earley parser <https://lark-parser.readthedocs.io/en/latest/parsers/#earley>`__
        is chosen, because it works with any context-free grammar.
        If your grammar has a form that is compatible with Lark's
        `LALR(1) parser <https://lark-parser.readthedocs.io/en/latest/parsers/#lalr1>`_,
        then switching to it may give much better performance, especially
        when parsing long strings.

        If the grammar is
        `ambiguous <https://en.wikipedia.org/wiki/Ambiguous_grammar>`__,
        there can be more than one way to parse a given string,
        which means that there are multiple parse trees for it. By default, only one of these
        trees is returned, but the argument ``get_multiple_trees`` allows to get all of
        them. This feature is currently only supported with Lark's Earley parser.

        Parameters
        ----------
        string : str
            Candidate string which can only be parsed if it is a member of the grammar's language.
        parser : str
            Parsing algorithm used to analyze the string.

            Possible values:

            - ``"earley"``: Can parse any context-free grammar.

              Performance: The algorithm has a time complexity of O(n^3),
              but if the grammar is unambiguous it is reduced to O(n^2)
              and for most LR grammars it is O(n).

            - ``"lalr"``: Can parse only a subset of context-free grammars, which have a form
              that allows very efficient parsing with a LALR(1) parser.

              Performance: The algorithm has a time complexity of O(n).
        get_multiple_trees : bool
            If ``True``, a list of parse trees is returned instead of a single parse tree object.
        max_num_trees : int
            An upper limit on how many parse trees will be returned at maximum.

        Returns
        -------
        parse_tree : :ref:`DerivationTree <derivation-tree>`
            If ``get_multiple_trees`` is set to ``True``, a list of parse trees is returned
            instead of a single parse tree object. The list can contain one or more trees,
            dependening on how many ways there are to parse the given string.

        Raises
        ------
        :exc:`~..exceptions.ParserError`
            If the string does not belong to the language
            and therefore no parse tree can be constructed for it.

        References
        ----------
        - Wikipedia

            - `Parsing <https://en.wikipedia.org/wiki/Parsing>`__
            - `Earley parser <https://en.wikipedia.org/wiki/Earley_parser>`__
            - `LALR parser <https://en.wikipedia.org/wiki/LALR_parser>`__

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
        fig : :ref:`Figure <grammar-figure>`
            Figure object containing the plot, allowing to display or export it.

        Notes
        -----
        Syntax diagrams (a.k.a. railroad diagrams) are especially useful for
        representing EBNF specifications of a grammar, because they capture nicely
        the extended notations that are introduced by EBNF, e.g. optional or repeated items.
        This package supports reading a grammar specification from EBNF text.
        Internally, however, EBNF is automatically converted to a simpler form
        during the reading process, which is done by removing any occurrence of
        extended notation and expressing it with newly introduced symbols and rules
        instead. Only the final version of the grammar can be visualized, which is
        essentially BNF with new helper rules and nonterminals.

        References
        ----------
        - What is a syntax diagram?

            - Wikipedia:
              `Syntax diagram
              <https://en.wikipedia.org/wiki/Syntax_diagram>`__
            - Oxford Reference / A Dictionary of Computing 6th edition:
              `Syntax diagram
              <https://www.oxfordreference.com/view/10.1093/oi/authority.20110803100547820>`__
            - Course website by Roger Hartley:
              `Programming language structure 1: Syntax diagrams
              <https://www.cs.nmsu.edu/~rth/cs/cs471/Syntax%20Module/diagrams.html>`__
            - Book chapter by Richard E. Pattis with Syntax Charts on p. 11:
              `v1: Languages and Syntax
              <http://www.cs.cmu.edu/~pattis/misc/ebnf.pdf>`__,
              `v2: EBNF - A Notation to Describe Syntax
              <https://www.ics.uci.edu/~pattis/misc/ebnf2.pdf>`__

        - Library used here for generating syntax diagram SVGs

            - `Railroad-Diagram Generator <https://github.com/tabatkins/railroad-diagrams>`__
              by Tab Atkins Jr.

        - Some other tools for drawing syntax diagrams

            - `Railroad Diagram Generator <https://bottlecaps.de/rr/ui>`__
              by Gunther Rademacher
            - `ANTLR Development Tools <https://www.antlr.org/tools.html>`__
              by Terence Parr
            - `DokuWiki EBNF Plugin <https://www.dokuwiki.org/plugin:ebnf>`__
              by Vincent Tscherter
            - `bubble-generator
              <https://www.sqlite.org/docsrc/finfo?name=art/syntax/bubble-generator.tcl>`__
              by the SQLite team
            - `Ebnf2ps <https://github.com/FranklinChen/Ebnf2ps>`__
              by Peter Thiemann
            - `EBNF Visualizer <http://dotnet.jku.at/applications/Visualizer/>`__
              by Markus Dopler and Stefan Schörgenhumer
            - `Clapham Railroad Diagram Generator <http://clapham.hydromatic.net>`__
              by Julian Hyde
            - `draw-grammar <https://github.com/iangodin/draw-grammar>`__
              by Ian Godin

        - Examples of syntax diagrams

            - `xkcd webcomic <https://xkcd.com/1930/>`__
            - `JSON <https://www.json.org/json-en.html>`__
            - `SQLite <https://www.sqlite.org/lang.html>`__
            - `Oracle Database Lite SQL <https://docs.oracle.com/cd/B19188_01/doc/B15917/sqsyntax.htm>`__
            - `Boost <https://www.boost.org/doc/libs/1_66_0/libs/spirit/doc/html/spirit/abstracts/syntax_diagram.html>`__

        """
        from . import visualization

        fig = visualization.create_syntax_diagram(self)
        return fig

    # Caching of repeated calculations
    def _lookup_or_calc(self, system, attribute, calc_func, *calc_args):
        """Look up a result in the cache. If it is not availabe yet, calculate and store it."""
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
        # Used here and in some G3P systems
        return {sym: num for num, sym
                in enumerate(_chain(self.nonterminal_symbols, self.terminal_symbols))}

    def _calc_idx_sym_map(self):
        # Used here and in some G3P systems
        return list(_chain(self.nonterminal_symbols, self.terminal_symbols))


class DerivationTree:
    """Derivation tree (parse tree) resulting from generating or parsing a string with a grammar.

    Structure:

    - The tree consists of linked nodes, each containing either a terminal or
      nonterminal symbol belonging to the grammar.
    - A rule of the grammar transforms a nonterminal symbol to a list of nonterminal and/or
      terminal symbols. In the tree this is represented by a node, containing the nonterminal
      symbol, being connected to one or more child nodes, containing the symbols derived
      by the expansion.
    - The root node contains the starting symbol (a nonterminal) of the grammar.
    - Each internal node contains a nonterminal symbol.
    - Each leaf node contains a terminal symbol.

    Behaviour:

    - A node with a nonterminal can be expanded using a suitable production rule of the grammar.
    - Depth first traversal allows to get all terminals in correct order. This allows to
      retrieve the string represented by the derivation tree.

    References
    ----------
    - What is a derivation tree or parse tree?

        - Wikipedia: `Parse tree <https://en.wikipedia.org/wiki/Parse_tree>`__
        - tutorialspoint:
          `Context-Free Grammar Introduction
          <https://www.tutorialspoint.com/automata_theory/context_free_grammar_introduction>`__

    """

    __slots__ = ('grammar', 'root_node', '_cache')

    # Initialization
    def __init__(self, grammar, root_symbol=None):
        if root_symbol is None:
            if grammar is None:
                root_symbol = NonterminalSymbol('')
            else:
                root_symbol = grammar.start_symbol
        self.grammar = grammar
        self.root_node = Node(root_symbol)

    # Copying
    def copy(self):
        dt = DerivationTree(self.grammar)
        dt.root_node = self.root_node.copy()  # leads to recursive copying of all child nodes
        return dt

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    # Representations
    def __repr__(self):
        return '<{} object at {}>'.format(self.__class__.__name__, hex(id(self)))

    def __str__(self):
        return self.to_bracket_notation()

    def _repr_html_(self):
        """Provide rich display representation in HTML format for Jupyter notebooks."""
        fig = self.plot()
        return fig._repr_html_()

    def _repr_pretty_(self, p, cycle):
        """Provide a rich display representation for IPython interpreters."""
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Equality and hashing
    def __eq__(self, other):
        # Type comparison
        if not isinstance(other, self.__class__):
            return False

        # Data comparison
        stk = []
        stk.append((self.root_node, other.root_node))
        while stk:
            nd1, nd2 = stk.pop(0)
            if nd1.symbol.text != nd2.symbol.text or len(nd1.children) != len(nd2.children):
                return False
            if nd1.children or nd2.children:
                stk = [(c1, c2) for c1, c2 in zip(nd1.children, nd2.children)] + stk
        return True

    def __ne__(self, other):
        # Type comparison
        if not isinstance(other, self.__class__):
            return True

        # Data comparison
        stk = []
        stk.append((self.root_node, other.root_node))
        while stk:
            nd1, nd2 = stk.pop(0)
            if nd1.symbol.text != nd2.symbol.text or len(nd1.children) != len(nd2.children):
                return True
            if nd1.children or nd2.children:
                stk = [(c1, c2) for c1, c2 in zip(nd1.children, nd2.children)] + stk
        return False

    def __hash__(self):
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
    def to_bracket_notation(self):
        """Create a string that represents the tree in a single-line bracket notation.

        References
        ----------
        - https://www.nltk.org/book/ch08.html

        """
        def traverse(node, seq):
            if isinstance(node.symbol, NonterminalSymbol):
                text = '<{}>'.format(node.symbol.text)
            else:
                text = '{}'.format(node.symbol.text)
            seq.append(text)
            if node.children:
                seq.append('(')
                for child in node.children:
                    traverse(child, seq)
                seq.append(')')

        seq = []
        traverse(self.root_node, seq)
        text = ''.join(seq)
        return text

    def to_tree_notation(self):
        """Create a string that represents the tree in a multi-line tree notation.

        References
        ----------
        - https://treenotation.org

        """
        def traverse(node, seq, depth):
            indent = ' ' * depth
            seq.append(indent + node.symbol.text)
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

        The data structure is a tuple that contains two other tuples of integers.
        A depth-first traversal of the tree visits all nodes. For each node, its symbol
        and number of children is remembered in two separate tuples. Instead of storing
        the symbols directly, a number is assigned to each symbol of the grammar and
        that concise number is stored instead of a potentially long symbol text.

        References
        ----------
        - https://en.wikipedia.org/wiki/Comparison_of_data-serialization_formats

        """
        # Caching
        sim = self.grammar._lookup_or_calc(
            'serialization', 'sym_idx_map', self.grammar._calc_sym_idx_map)

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

    def from_tuple(self, data):
        """Convert two tuples of numbers to the tree they represent."""
        # Caching
        ism = self.grammar._lookup_or_calc(
            'serialization', 'idx_sym_map', self.grammar._calc_idx_sym_map)

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

        symbols, counters = data
        i = -1
        top = Node('')
        traverse(top)
        self.root_node = top.children[0]
        del top

    def to_json(self):
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
        return _json.dumps(data, separators=(',', ':'))

    def from_json(self, data):
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

        data = _json.loads(data)
        i = -1
        top = Node('')
        traverse(top)
        self.root_node = top.children[0]
        del top

    # Visualization
    def plot(self, show_node_indices=None,
             layout_engine=None, fontname=None, fontsize=None,
             shape_nt=None, shape_unexpanded_nt=None, shape_t=None,
             fontcolor_nt=None, fontcolor_unexpanded_nt=None, fontcolor_t=None,
             fillcolor_nt=None, fillcolor_unexpanded_nt=None, fillcolor_t=None):
        """Create a plot that represents the derivation tree as directed graph with Graphviz.

        Parameters
        ----------
        show_node_indices : bool
            If ``True``, nodes will contain numbers that indicate the order
            in which they were created during tree construction.
        layout_engine : str
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
        fontname : str
            Fontname of text inside nodes.
        fontsize : int or str
            Fontsize of text inside nodes.
        shape_nt : str
            Shape of nodes that represent expanded nonterminals.

            Possible values: See `Graphviz documentation: Node shapes
            <http://www.graphviz.org/doc/info/shapes.html>`__
        shape_unexpanded_nt : str
            Shape of nodes that represent unexpanded nonterminals.

            Possible values: See `Graphviz documentation: Node shapes
            <http://www.graphviz.org/doc/info/shapes.html>`__
        shape_t : str
            Shape of nodes that represent terminals.

            Possible values: See `Graphviz documentation: Node shapes
            <http://www.graphviz.org/doc/info/shapes.html>`__
        fontcolor_nt : str
            Fontcolor of nodes that represent expanded nonterminals.
        fontcolor_unexpanded_nt : str
            Fontcolor of nodes that represent unexpanded nonterminals.
        fontcolor_t : str
            Fontcolor of nodes that represent terminals.
        fillcolor_nt : str
            Fillcolor of nodes that represent expanded nonterminals.
        fillcolor_unexpanded_nt : str
            Fillcolor of nodes that represent unexpanded nonterminals.
        fillcolor_t : str
            Fillcolor of nodes that represent terminals.

        Returns
        -------
        fig : :ref:`Figure <derivation-tree-figure>`
            Figure object containing the plot, allowing to display or export it.

        References
        ----------
        - `Graphviz <https://www.graphviz.org>`__

        """
        from . import visualization

        fig = visualization.create_graphviz_tree(
            self, show_node_indices,
            layout_engine, fontname, fontsize,
            shape_nt, shape_unexpanded_nt, shape_t,
            fontcolor_nt, fontcolor_unexpanded_nt, fontcolor_t,
            fillcolor_nt, fillcolor_unexpanded_nt, fillcolor_t)
        return fig

    # Reading contents of the tree
    def nodes(self, order='dfs'):
        """Get all nodes as a list that results from a tree traversal in a chosen order.

        References
        ----------
        - https://en.wikipedia.org/wiki/Tree_traversal
        - https://en.wikipedia.org/wiki/Depth-first_search
        - https://en.wikipedia.org/wiki/Breadth-first_search

        """
        # Argument processing
        _ap.str_arg('order', order, vals=('dfs', 'bfs'))

        # Generate node list by tree traversal
        if order == 'dfs':
            nodes = self._depth_first_traversal()
        else:
            nodes = self._breadth_first_traversal()
        return nodes

    def _depth_first_traversal(self):
        """Traverse the tree with a depth-first search.

        Returns
        -------
        nodes : list of :ref:`Node <derivation-tree-node>` objects

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
        nodes : list of :ref:`Node <derivation-tree-node>` objects

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
        """Get all leaf nodes by traversing the tree in depth-first order.

        Returns
        -------
        node_sequence : list of :class:`.Node` objects

        """
        nodes = []
        stack = _collections.deque()  # LIFO
        stack.append(self.root_node)
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(reversed(node.children))  # add first child last, so it becomes first
            else:
                nodes.append(node)
        return nodes

    def internal_nodes(self):
        """Get all internal nodes by traversing the tree in depth-first order.

        Returns
        -------
        node_sequence : list of :class:`.Node` objects

        """
        nodes = []
        stack = _collections.deque()  # LIFO
        stack.append(self.root_node)
        while stack:
            node = stack.pop()
            if node.children:
                nodes.append(node)
                stack.extend(reversed(node.children))  # add first child last, so it becomes first
        return nodes

    def tokens(self):
        """Get the sequence of tokens found in the leaf nodes from left to right.

        Returns
        -------
        token_sequence : list of :class:`.Symbol` objects

        """
        return [nd.symbol for nd in self.leaf_nodes()]

    def string(self, separator=''):
        """Get the string composed of symbols found in the leaf nodes of the derivation tree.

        If the tree is fully expanded, no nonterminal symbol is left
        in the leaf nodes, so the obtained string is composed only of
        terminals and belongs to the language of the grammar.

        Returns
        -------
        string : str

        """
        return separator.join(nd.symbol.text if isinstance(nd.symbol, TerminalSymbol)
                              else '<{}>'.format(nd.symbol.text) for nd in self.leaf_nodes())

    def derivation(self, derivation_order='leftmost', separate_lines=True):
        """Get a derivation that suits to the structure of the derivation tree.

        Parameters
        ----------
        derivation_order : str
            Order in which nonterminals are expanded during the step-by-step derivation.

            Possible values:

            - ``"leftmost"``: Expand the leftmost nonterminal in each step.
            - ``"rightmost"``: Expand the rightmost nonterminal in each step.
            - ``"random"``: Expand a random nonterminal in each step.
        separate_lines : bool
            If ``True``, the derivation steps are placed on separate lines
            by adding newline characters between them.

        Returns
        -------
        derivation : str

        """
        # Argument processing
        _ap.str_arg(
            'derivation_order', derivation_order, vals=('leftmost', 'rightmost', 'random'))
        _ap.bool_arg(
            'separate_lines', separate_lines)

        # Helper functions
        def next_derivation_step(derivation, old_node, new_nodes):
            last_sentential_form = derivation[-1]
            new_sentential_form = last_sentential_form[:]
            insert_idx = last_sentential_form.index(old_node)
            new_sentential_form[insert_idx:insert_idx + 1] = new_nodes
            if new_sentential_form:
                derivation.append(new_sentential_form)
            return derivation

        def symbol_seq_repr(symbol_seq):
            seq_repr = []
            for sym in symbol_seq:
                if isinstance(sym, NonterminalSymbol):
                    seq_repr.append('<{}>'.format(sym))
                else:
                    seq_repr.append(str(sym))
            return ''.join(seq_repr)

        def node_seq_repr(node_seq):
            symbol_seq = [node.symbol for node in node_seq]
            return symbol_seq_repr(symbol_seq)

        # Traverse the tree and collect nodes according to the method
        if derivation_order == 'leftmost':
            derivation = [[self.root_node]]
            stack = [self.root_node]
            while stack:
                idx = 0
                nt_node = stack.pop(idx)
                derivation = next_derivation_step(derivation, nt_node, nt_node.children)
                new_nt_nodes = [node for node in nt_node.children if node.children]
                stack = new_nt_nodes + stack
        elif derivation_order == 'rightmost':
            derivation = [[self.root_node]]
            stack = [self.root_node]
            while stack:
                idx = len(stack) - 1
                nt_node = stack.pop(idx)
                derivation = next_derivation_step(derivation, nt_node, nt_node.children)
                new_nt_nodes = [node for node in nt_node.children if node.children]
                stack.extend(new_nt_nodes)
        elif derivation_order == 'random':
            derivation = [[self.root_node]]
            stack = [self.root_node]
            while stack:
                idx = _random.randint(0, len(stack)-1)
                nt_node = stack.pop(idx)
                derivation = next_derivation_step(derivation, nt_node, nt_node.children)
                new_nt_nodes = [node for node in nt_node.children if node.children]
                stack.extend(new_nt_nodes)

        if separate_lines:
            sep = '{}=> '.format(_NEWLINE)
        else:
            sep = ' => '
        derivation_str = sep.join(node_seq_repr(item) for item in derivation)
        return derivation_str

    def num_expansions(self):
        """Calculate the number of expansions contained in the derivation tree."""
        num_expansions = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop(0)
            if node.children:
                num_expansions += 1
                stack = node.children + stack
        return num_expansions

    def num_nodes(self):
        """Calculate the number of nodes contained in the derivation tree."""
        num_nodes = 1
        stack = [self.root_node]
        while stack:
            node = stack.pop(0)
            if node.children:
                num_nodes += len(node.children)
                stack = node.children + stack
        return num_nodes

    def depth(self):
        """Calculate the depth of the derivation tree, i.e. the longest path from root to a leaf.

        References
        ----------
        - https://en.wikipedia.org/wiki/Tree_%28data_structure%29

        - Koza, Genetic programming (1992)

            - p. 92: "The depth of a tree is defined as the length of the longest
              nonbacktracking path from the root to an endpoint."

        - Poli, A field guide to genetic programming (2008)

            - p. 12: "The depth of a node is the number of edges that need to be
              traversed to reach the node starting from the tree’s root node (which
              is assumed to be at depth 0). The depth of a tree is the depth of its
              deepest leaf"

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
        """Check if the derivation tree contains only expanded nonterminal symbols.

        Returns
        -------
        is_completely_expanded : bool
            If True, the tree is fully expanded which means that it contains only terminals
            in its leave nodes and together they form a string of the grammar's language
            when concatenated.

        """
        if any(node.contains_unexpanded_nonterminal() for node in self.leaf_nodes()):
            return False
        return True

    # Convenience methods for G3P systems
    def _expand(self, nd, sy):
        """Expand a node in the tree by adding child nodes to it, each getting a symbol."""
        # Syntax minified for minor optimization due to large number of calls
        l=[];a=l.append
        for s in sy:n=Node(s);a(n);nd.children.append(n)
        return l

    def _is_deeper_than(self, value):
        """Detect if the derivation tree is deeper than a given value.

        This is a helper method for some tree-based G3P systems.

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
    """Node inside a derivation tree. It contains a symbol and refers to child nodes."""

    __slots__ = ('symbol', 'children')

    # Initialization
    def __init__(self, sy, ch=None):
        self.symbol = sy
        self.children = [] if ch is None else ch

    # Copying
    def copy(self):
        """Copy a node and its children recursively."""
        ch = None if self.children is None else [nd.copy() for nd in self.children]
        sy = self.symbol.copy()
        return self.__class__(sy, ch)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    # Representations
    def __repr__(self):
        return '<{} object at {}>'.format(self.__class__.__name__, hex(id(self)))

    def __str__(self):
        return self.symbol.text

    # Symbol type requests
    def contains_terminal(self):
        """Check whether the node contains a terminal symbol."""
        return isinstance(self.symbol, TerminalSymbol)

    def contains_nonterminal(self):
        """Check whether the node contains a nonterminal symbol."""
        return isinstance(self.symbol, NonterminalSymbol)

    def contains_unexpanded_nonterminal(self):
        """Check whether the node contains a nonterminal symbol and has no child nodes."""
        return not self.children and isinstance(self.symbol, NonterminalSymbol)


class Symbol:
    """Data structure for symbols in a grammar.

    A symbol can be either a nonterminal or terminal of the grammar
    and it has a ``text`` attribute.

    """

    __slots__ = ('text',)

    # Initialization
    def __init__(self, text):
        self.text = text

    # Copying
    def copy(self):
        return self.__class__(self.text)

    def __copy__(self):
        return self.__class__(self.text)

    def __deepcopy__(self, memo):
        return self.__class__(self.text)

    # Representations
    def __str__(self):
        return self.text

    # Comparison operators for sorting
    def __eq__(self, other):
        return self.text == other.text and isinstance(other, self.__class__)

    def __ne__(self, other):
        return self.text != other.text or not isinstance(other, self.__class__)

    def __lt__(self, other):
        return self.text < other.text

    def __le__(self, other):
        return self.text <= other.text

    def __gt__(self, other):
        return self.text > other.text

    def __ge__(self, other):
        return self.text >= other.text

    # Hash for usage as dict key
    def __hash__(self):
        return hash(self.text)


class NonterminalSymbol(Symbol):
    """Data structure for nonterminal symbols in a grammar."""

    __slots__ = ()

    def __repr__(self):
        return 'NT({})'.format(repr(self.text))


class TerminalSymbol(Symbol):
    """Data structure for terminal symbols in a grammar."""

    __slots__ = ()

    def __repr__(self):
        return 'T({})'.format(repr(self.text))
