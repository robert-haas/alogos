"""All custom warnings used in the package."""

import warnings as _warnings

from ._utilities.operating_system import NEWLINE as _NEWLINE


__all__ = [
    "turn_on",
    "turn_off",
    "GrammarWarning",
    "OptimizationWarning",
    "DatabaseWarning",
]


def turn_on():
    """Turn on all warnings of this package.

    Notes
    -----
    This package uses Python's
    `warnings <https://docs.python.org/3/library/warnings.html>`__
    module.
    This means that functions like
    `warnings.filterwarnings <https://docs.python.org/3/library/warnings.html#warnings.filterwarnings>`__
    will have effect on the warnings issued by this package.

    """
    for cat in _WARNINGS:
        _warnings.filterwarnings("always", category=cat)


def turn_off():
    """Turn off all warnings of this package.

    Notes
    -----
    This package uses Python's
    `warnings <https://docs.python.org/3/library/warnings.html>`__
    module.
    This means that functions like
    `warnings.filterwarnings <https://docs.python.org/3/library/warnings.html#warnings.filterwarnings>`__
    will have effect on the warnings issued by this package.

    """
    for cat in _WARNINGS:
        _warnings.filterwarnings("ignore", category=cat)


class GrammarWarning(Warning):
    """Issued when a grammar is ill-formed but still usable.

    Examples
    --------
    - The same production rule appears more than once.

    """


def _warn_multiple_grammar_specs():
    message = (
        "More than one grammar specification was provided. "
        "Only the first one is used in following order of precedence: "
        "bnf_text > bnf_file > ebnf_text > ebnf_file."
    )
    _warnings.warn(message, category=GrammarWarning)


def _warn_repeated_productions(redundant_rules):
    reported_text = _NEWLINE.join(redundant_rules)
    message = (
        "Problematic grammar specification: Some production rules are "
        "redundant. This package can deal with it, but in general it "
        "is not recommended. In particular, it introduces bias in "
        "random string generation. Following rules are contained more "
        "than one time:{}{}".format(_NEWLINE, reported_text)
    )
    _warnings.warn(message, category=GrammarWarning)


def _warn_symbol_set_overlap(intersection):
    text = _NEWLINE.join(
        "  {sym} as nonterminal <{sym}> and terminal " '"{sym}"'.format(sym=sym)
        for sym in intersection
    )
    message = (
        "Problematic grammar specification: "
        "The sets of nonterminal and terminal symbols are not "
        "disjoint, as required by the mathematical definition of a "
        "grammar. This package can deal with it, but in general it is "
        "not recommended. Following symbols appear in both sets:"
        "{}{}".format(_NEWLINE, text)
    )
    _warnings.warn(message, category=GrammarWarning)


def _warn_language_gen_stopped(max_steps):
    message = (
        "Language generation stopped due to reaching max_steps={}, "
        "but it did not produce all possible strings yet. To explore "
        "it further, the max_steps parameter can be "
        "increased.".format(max_steps)
    )
    _warnings.warn(message, category=GrammarWarning)


class OptimizationWarning(Warning):
    """Issued when an optimization algorithm might behave unexpected.

    Examples
    --------
    - No optimization step was performed due to the parameter settings.

    """


def _warn_no_step_in_ea_performed(generation):
    message = (
        "Started and stopped the run at generation {}. Nothing new was "
        "calculated because a stop criterion was immediately True and "
        "led to an exit before creating a new generation. "
        "If your intention is to re-run the search you can use the "
        "reset() method, which deletes the search state but preserves "
        "the current parameters.".format(generation)
    )
    _warnings.warn(message, category=OptimizationWarning)


class DatabaseWarning(Warning):
    """Issued when a potential problem with a database occurs.

    Examples
    --------
    - Not all the data could be imported from the database.
    - An import argument did not fit to the data and had to be ignored.

    """


def _warn_database_renamed(source, target):
    message = "Renamed database file from {} to {}.".format(repr(source), repr(target))
    _warnings.warn(message, category=DatabaseWarning)


def _warn_database_import_partial():
    message = (
        "Loaded data of a previous run from the provided filepath. "
        "Note, however, that the contained mapping data can not be "
        "added to a cache and therefore also can not be used to "
        'prevent recalculations because the parameter "database_on" is '
        "currently set to False."
    )
    _warnings.warn(message, category=DatabaseWarning)


def _warn_database_import_first(first_gen_in_db, first_generation):
    message = (
        "The provided value for first generation ({}) is invalid. The "
        "first generation in the database ({}) was used "
        "instead.".format(first_generation, first_gen_in_db)
    )
    _warnings.warn(message, category=DatabaseWarning)


def _warn_database_import_last(last_gen_in_db, last_generation):
    message = (
        "The provided value for last generation ({}) is invalid. The "
        "last generation in the database ({}) was used "
        "instead.".format(last_generation, last_gen_in_db)
    )
    _warnings.warn(message, category=DatabaseWarning)


def _warn_import_database_empty(filepath):
    message = (
        "The database at filepath {} contains the correct table for "
        "phenotype-to-fitness evaluation data, but no entries are "
        "stored in it and therefore no data could be "
        "imported.".format(repr(filepath))
    )
    _warnings.warn(message, category=DatabaseWarning)


_WARNINGS = [
    GrammarWarning,
    OptimizationWarning,
    DatabaseWarning,
]
