import logging as _logging

from ._utilities.operating_system import NEWLINE as _NEWLINE


FORMATTER = _logging.Formatter('%(name)s: %(levelname)s: %(message)s')

STREAMHANDLER = _logging.StreamHandler()
STREAMHANDLER.setFormatter(FORMATTER)

LOGGER = _logging.getLogger('alogos')
LOGGER.setLevel(_logging.INFO)
LOGGER.addHandler(STREAMHANDLER)


def warn_user(message):
    LOGGER.warning(message)


def warn_multiple_grammar_specs():
    message = (
        'More than one grammar specification was provided. '
        'Only the first one is used in following order of precedence: '
        'bnf_text > bnf_file > ebnf_text > ebnf_file.')
    warn_user(message)


def warn_repeated_productions(repeated_productions):
    reported_text = _NEWLINE.join(repeated_productions)
    message = (
        'Problematic grammar specification: Some production rules are '
        'redundant. This package can deal with it, but in general it is not '
        'recommended. In particular, it introduces bias in random string '
        'generation. Following rules are contained more than one time:'
        '{}{}'.format(_NEWLINE, reported_text))
    warn_user(message)


def warn_symbol_set_overlap(intersection):
    text = _NEWLINE.join('  {sym} as nonterminal <{sym}> and terminal '
                         '"{sym}"'.format(sym=sym) for sym in intersection)
    message = (
        'Problematic grammar specification: '
        'The sets of nonterminal and terminal symbols are not disjoint, as '
        'required by the mathematical definition of a grammar. This package '
        'can deal with it, but in general it is not recommended. Following '
        'symbols appear in both sets:{}{}'.format(_NEWLINE, text))
    warn_user(message)


def warn_language_gen_stopped(max_steps):
    message = (
        'Language generation stopped due to reaching max_steps={}, but it did '
        'not produce all possible strings yet. To explore it further, the max_steps '
        'parameter can be increased.'.format(max_steps))
    warn_user(message)


def warn_no_step_in_ea_performed(generation):
    message = (
        'Started and stopped the run at generation {}. Nothing new was '
        'calculated because a stop criterion was immediately True and led '
        'to an exit before creating a new generation. '
        'If your intention is to re-run the search you can use the reset() '
        'method, which deletes the search state but preserves the '
        'current parameters.'.format(generation))
    warn_user(message)


def warn_database_renamed(source, target):
    message = 'Renamed database file from {} to {}.'.format(repr(source), repr(target))
    warn_user(message)


def warn_database_import_partial():
    message = (
        'Loaded data of a previous run from the provided filepath. '
        'Note, however, that the contained mapping data can not be added to '
        'a cache and therefore also can not be used to prevent recalculations '
        'because the parameter "database_on" is currently set to False.')
    warn_user(message)


def warn_database_import_first(first_gen_in_db, first_generation):
    message = (
        'The provided value for first generation ({}) is invalid. The first generation '
        'in the database ({}) was used instead.'.format(first_generation, first_gen_in_db))
    warn_user(message)


def warn_database_import_last(last_gen_in_db, last_generation):
    message = (
        'The provided value for last generation ({}) is invalid. The last generation '
        'in the database ({}) was used instead.'.format(last_generation, last_gen_in_db))
    warn_user(message)


def warn_import_database_empty(filepath):
    message = (
        'The database at filepath {} contains the correct table '
        'for phenotype-to-fitness evaluation data, but no entries are stored in it '
        'and therefore no data could be imported.'.format(repr(filepath)))
    warn_user(message)
