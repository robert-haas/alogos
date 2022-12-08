"""All custom exceptions used in the package."""

from ._utilities.operating_system import NEWLINE as _NEWLINE


__all__ = [
    "GrammarError",
    "ParserError",
    "ParameterError",
    "GenotypeError",
    "InitializationError",
    "OperatorError",
    "MappingError",
    "DatabaseError",
]


class GrammarError(Exception):
    """Raised when a grammar can not be read, written or is ill-formed.

    Examples
    --------
    - The set of nonterminal symbols, terminal symbols or production
      rules is empty.
    - A nonterminal symbol has no production rule where its right-hand
      side is defined.

    """


def raise_write_nonterminal_error(text):
    message = (
        "Nonterminal symbol N({}) contains text that does not allow to "
        "write it with the available surrounding markers.".format(text)
    )
    raise GrammarError(message)


def raise_write_terminal_error(text):
    message = (
        "Terminal symbol T({}) contains text that does not allow to "
        "write it with the available surrounding markers.".format(text)
    )
    raise GrammarError(message)


def raise_delimiting_symbol_error_1():
    message = (
        'Either "start_terminal_symbol" and "end_terminal_symbol", '
        'or "start_terminal_symbol2" and "end_terminal_symbol2", or both need to '
        "be nonempty strings in order to serve as indicators for terminal symbols."
    )
    raise GrammarError(message)


def raise_delimiting_symbol_error_2():
    message = (
        'Either "start_terminal_symbol" and "end_terminal_symbol", '
        'or "start_terminal_symbol2" and "end_terminal_symbol2", or both need to '
        "be nonempty strings in order to serve as indicators for terminal symbols."
    )
    raise GrammarError(message)


def raise_empty_productions_error():
    message = "The set of production rules is empty."
    raise GrammarError(message)


def raise_empty_nonterminals_error():
    message = "The set of nonterminal symbols is empty."
    raise GrammarError(message)


def raise_empty_terminals_error():
    message = "The set of terminal symbols is empty."
    raise GrammarError(message)


def raise_missing_nonterminals_error(missing_nonterminals):
    reported_text = _NEWLINE.join(missing_nonterminals)
    message = (
        "Following nonterminals appeared on a right-hand side of a production rule "
        "but on no left-hand side:{}{}".format(_NEWLINE, reported_text)
    )
    raise GrammarError(message)


def raise_delimiter_symbol_error_1():
    message = (
        'Please provide either both a nonempty "start_nonterminal_symbol" '
        'and "end_nonterminal_symbol" or none of them.'
    )
    raise GrammarError(message)


def raise_surrounding_symbol_error_2():
    message = (
        'Please provide either both a nonempty "start_terminal_symbol" '
        'and "end_terminal_symbol" or none of them.'
    )
    raise GrammarError(message)


def raise_surrounding_symbol_error_3():
    message = (
        'Please provide either both a nonempty "start_terminal_symbol2" '
        'and "end_terminal_symbol2" or none of them.'
    )
    raise GrammarError(message)


def raise_surrounding_symbol_error_4():
    message = (
        'Either "start_nonterminal_symbol", '
        '"start_terminal_symbol" or both need to be a nonempty string.'
    )
    raise GrammarError(message)


def raise_surrounding_symbol_error_5(listing):
    message = (
        '"start_nonterminal_symbol" and "end_nonterminal_symbol" may not have '
        'an overlap with any of "start_terminal_symbol", "end_terminal_symbol", '
        '"start_terminal_symbol2" and "end_terminal_symbol2". Following values '
        "were found to cause a problem: {}".format(listing)
    )
    raise GrammarError(message)


class ParserError(Exception):
    """Raised when parsing a string with a grammar fails.

    Examples
    --------
    - The string contains characters that are not part of any terminal
      symbol.
    - The string does not belong to the language of the given grammar.
    - The chosen parser is not compatible with the given grammar.

    """


def raise_lark_parser_mult_error():
    message = (
        'Getting multiple parse trees is currently only supported with parser="earley".'
    )
    raise ParserError(message)


def _raise_parser_creation_error(excp):
    message = (
        "Parsing failed during creation of the chosen parser. Perhaps it is not "
        "compatible with the form of the provided grammar.{nl}{nl}"
        'Lark raised the exception "{name}" with following message: '
        "{text}".format(nl=_NEWLINE, name=type(excp).__name__, text=excp)
    )
    raise ParserError(message) from None


def _raise_parser_string_error(excp):
    message = (
        "Parsing failed during analyzing the string.{nl}{nl}"
        'Lark raised the exception "{name}" with following message: '
        "{text}".format(nl=_NEWLINE, name=type(excp).__name__, text=excp)
    )
    raise ParserError(message) from None


def _raise_parser_node_error(node_label):
    message = (
        "Discovered an unexpected symbol in the Lark tree that contains "
        'all of the discovered parse trees: "{}"'.format(node_label)
    )
    raise ParserError(message) from None


class ParameterError(Exception):
    """Raised when a parameter or collection of parameters is invalid.

    Examples
    --------
    - An unknown parameter is provided or requested by a user.
    - All variation operators of an evolutionary algorithm are turned
      off.
    - All stop criteria of a search algorithm are inactive.

    """


def raise_unknown_parameter_error(parameter_name, default_parameters):
    message = (
        "An unknown parameter was attempted to be used: {val}"
        "{nl}{nl}{defv}".format(
            val=repr(parameter_name), nl=_NEWLINE, defv=default_parameters
        )
    )
    raise ParameterError(message) from None


def raise_initial_parameter_error(name):
    message = (
        "Got an invalid parameter name, which is reserved as "
        "method name: {}".format(repr(name))
    )
    raise ParameterError(message) from None


def raise_stop_parameter_error():
    message = (
        "No stop criterion was set, therefore a search would continue to run indefinitely. "
        "Please provide a value for one or more of the following parameters: "
        "max_generations, max_fitness_evaluations, max_runtime_in_seconds, max_or_min_fitness"
    )
    raise ParameterError(message) from None


def raise_missing_variation_error():
    message = (
        "Neither crossover nor mutation are activated. This means "
        "there is no variation, no new candidate solutions are generated"
        " and the optimization can not progress towards better "
        "objective function values."
    )
    raise ParameterError(message) from None


def raise_operator_lookup_error(description, name, location):
    try:
        available = ", ".join(
            name for name in dir(location) if not name.startswith("_")
        )
    except Exception:
        available = "None"
    message = (
        "{desc} operator is not known to the chosen algorithm in its current configuration. "
        "{nl}Given operator name: {name}"
        "{nl}Available operator names: {av}"
        "{nl}Lookup location: {loc}".format(
            desc=description,
            nl=_NEWLINE,
            name=str(name),
            av=available,
            loc=repr(location),
        )
    )
    raise ParameterError(message) from None


def raise_no_genotype_error():
    message = (
        "No genotype was provided in the parameters dictionary "
        'under the key "init_genotype".'
    )
    raise ValueError(message)


def raise_no_phenotype_error():
    message = (
        "No phenotype was provided in the parameters dictionary "
        'under the key "init_phenotype".'
    )
    raise ValueError(message)


def raise_no_derivation_tree_error():
    message = (
        "No derivation tree was provided in the parameters dictionary "
        'under the key "init_derivation_tree".'
    )
    raise ValueError(message)


def raise_no_genotypes_error():
    message = (
        "No genotypes were provided in the parameters dictionary "
        'under the key "init_genotypes".'
    )
    raise ValueError(message)


def raise_no_phenotypes_error():
    message = (
        "No phenotypes were provided in the parameters dictionary "
        'under the key "init_phenotypes".'
    )
    raise ValueError(message)


def raise_no_derivation_trees_error():
    message = (
        "No derivation trees were provided in the parameters dictionary "
        'under the key "init_derivation_trees".'
    )
    raise ValueError(message)


class GenotypeError(Exception):
    """Raised when a genotype is not well-formed or modified.

    Examples
    --------
    - The genotype can not be interpreted as a list of integers for
      Grammatical Evolution (GE).
    - The genotype can not be interpreted as a derivation tree for
      Context-Free Grammar Genetic Programming (CFGGP).
    - A user tries to modify the immutable genotype attribute of an
      individual.

    """


def raise_data_write_error():
    message = (
        'The attribute "data" is immutable. It can only be assigned during '
        "object creation. Afterwards it is used to identify a genotype, "
        "e.g. when it is added to a set or used as a key in a dictionary, "
        "and therefore it may not be modified."
    )
    raise GenotypeError(message) from None


def raise_cfggp_genotype_error(data):
    message = (
        "The given data could not be interpreted as a CFG-GP genotype. "
        "It needs to be a derivation tree or a serialized derivation tree. "
        "Given data: {}".format(repr(data))
    )
    raise GenotypeError(message) from None


def raise_cfggpst_genotype_error(data):
    message = (
        "The given data could not be interpreted as a CFG-GP-ST genotype. "
        "It needs to be a serialized derivation tree in form of "
        "two tuples of integers. "
        "Given data: {}".format(repr(data))
    )
    raise GenotypeError(message) from None


def raise_dsge_genotype_error(data):
    message = (
        "The given data could not be interpreted as a DSGE genotype. "
        "It needs to be a non-empty tuple of tuples of integers. "
        "Given data: {}".format(repr(data))
    )
    raise GenotypeError(message) from None


def raise_dsge_genotype_repair_error(max_expansions):
    message = (
        "The provided DSGE genotype could not be repaired, because the number of "
        "expansions reached the max_expansions limit: {}".format(max_expansions)
    )
    raise GenotypeError(message) from None


def raise_ge_genotype_error(data):
    message = (
        "The given data could not be interpreted as a GE genotype. "
        "It needs to be a non-empty tuple of integers. "
        "Given data: {}".format(repr(data))
    )
    raise GenotypeError(message) from None


def raise_pige_genotype_error(data):
    message = (
        "The given data could not be interpreted as a piGE genotype. "
        "It needs to be a non-empty tuple of integers. "
        "Given data: {}".format(repr(data))
    )
    raise GenotypeError(message) from None


def raise_whge_genotype_error(data):
    message = (
        "The given data could not be interpreted as a WHGE genotype. "
        "It needs to be a non-empty bitarray. "
        "Given data: {}".format(repr(data))
    )
    raise GenotypeError(message) from None


class InitializationError(Exception):
    """Raised when intitialization of an individual or population fails.

    Examples
    --------
    - A user-provided genotype is not well-formed.
    - A user-provided phenotype can not be parsed and reverse mapped to
      a genotype.

    """


def raise_init_ind_from_gt_error(genotype):
    message = (
        "Initialization of an individual from following given genotype "
        "failed: {}".format(repr(genotype))
    )
    raise InitializationError(message) from None


def raise_init_ind_from_dt_error(derivation_tree):
    message = (
        "Initialization of an individual from following given derivaton tree "
        "failed: {}".format(repr(derivation_tree))
    )
    raise InitializationError(message) from None


def raise_init_ind_from_phe_error(phenotype):
    message = (
        "Initialization of an individual from following given phenotype "
        "failed: {}".format(repr(phenotype))
    )
    raise InitializationError(message) from None


def raise_init_ind_rand_gt_error():
    message = "Initialization of an individual from a random genotype failed."
    raise InitializationError(message) from None


def raise_init_ind_valid_rand_gt_error(max_tries):
    message = (
        "Initialization of a valid individual from a random genotype failed "
        "after {} tries.".format(max_tries)
    )
    raise InitializationError(message) from None


def raise_init_ind_grow_error():
    message = 'Initialization of an individual with the "grow" method failed.'
    raise InitializationError(message) from None


def raise_init_ind_full_error():
    message = 'Initialization of an individual with the "full" method failed.'
    raise InitializationError(message) from None


def raise_init_ind_pi_grow_error():
    message = 'Initialization of an individual with the "pi grow" method failed.'
    raise InitializationError(message) from None


def raise_init_ind_ptc2_error():
    message = 'Initialization of an individual with the "PTC2" method failed.'
    raise InitializationError(message) from None


def raise_init_pop_from_gt_error():
    message = "Initialization of a population from given genotypes failed."
    raise InitializationError(message)


def raise_init_pop_from_dt_error():
    message = "Initialization of a population from given derivation trees failed."
    raise InitializationError(message)


def raise_init_pop_from_phe_error():
    message = "Initialization of a population from given phenotypes failed."
    raise InitializationError(message)


def raise_init_pop_rand_gt_error():
    message = "Initialization of a population from random genotypes failed."
    raise InitializationError(message)


def raise_init_pop_gp_rhh_error():
    message = 'Initialization of a population with gp_rhh (="ramped half and half" from Genetic Programming) failed.'
    raise InitializationError(message)


def raise_init_pop_pi_rhh_error():
    message = 'Initialization of a population with pi_rhh (="ramped half and half" with PI Grow from piGE) failed.'
    raise InitializationError(message)


def raise_init_pop_ptc2_error():
    message = "Initialization of a population with PTC2 (=probabilistic tree creation 2) failed."
    raise InitializationError(message)


def raise_init_pop_unique_gen_error(
    num_found_genotypes, num_wanted_genotypes, num_tries
):
    message = "Found only {} of {} unique genotypes after {} tries.".format(
        num_found_genotypes, num_wanted_genotypes, num_tries
    )
    raise InitializationError(message)


def raise_init_pop_unique_phe_error(
    num_found_phenotypes, num_wanted_phenotypes, num_tries
):
    message = "Found only {} of {} unique phenotypes after {} tries.".format(
        num_found_phenotypes, num_wanted_phenotypes, num_tries
    )
    raise InitializationError(message)


def raise_pop_assignment_error(value):
    message = (
        "Only an individual can be assigned to a population. Got an object of "
        "type {} instead.".format(type(value))
    )
    # TypeError is required here by Python
    raise TypeError(message)


class OperatorError(Exception):
    """Raised when a search operator fails.

    Examples
    --------
    - A crossover operator that requires two parent genotypes of equal
      length fails because it gets two individuals with different
      genotype lengths as input.

    """


def raise_crossover_lp_error1(l1, l2):
    message = (
        "The crossover operator is length preserving and requires that both "
        "parent genotypes have equal length. Instead they had following two "
        "different lengths: {}, {}".format(l1, l2)
    )
    raise OperatorError(message)


def raise_crossover_lp_error2():
    message = (
        "The crossover operator requires that each parent has a minimal "
        "genotype length of 2. This was not the case."
    )
    raise OperatorError(message)


class MappingError(Exception):
    """Raised when a genotype-to-phenotype mapping or its inverse fails.

    Examples
    --------
    - Forward mapping: No string of terminals could be found within the
      allowed number of expansions or wrappings.
    - Reverse mapping: The provided grammar contains no production rule
      that would match a production found in a given derivation tree.

    """


def raise_max_expansion_error(max_expansions):
    message = (
        "No string of the grammar's language was found before reaching the provided "
        "maximum number of expansions: {}".format(max_expansions)
    )
    raise MappingError(message)


def raise_max_wraps_error(max_wraps):
    message = (
        "No string of the grammar's language was found before reaching the provided "
        "maximum number of wraps: {}".format(max_wraps)
    )
    raise MappingError(message)


def raise_missing_nt_error(nonterminal_node):
    nt_text = "<{}>".format(nonterminal_node.symbol)
    message = (
        "For a nonterminal, no production rule could be found in the "
        "grammar: {}".format(nt_text)
    )
    raise MappingError(message) from None


def raise_missing_rhs_error(nonterminal_node, rhs):
    message = (
        "For a derivation step in the tree, no corresponding production rule "
        "could be found in the grammar: <{}> -> {}".format(nonterminal_node, rhs)
    )
    raise MappingError(message) from None


def raise_invalid_mapping_data1(data):
    message = (
        "Reverse mapping got invalid input data. "
        "The given phenotype is a string that is not a "
        "member of the grammar's language: {}".format(repr(data))
    )
    raise MappingError(message)


def raise_invalid_mapping_data2(data):
    message = (
        "Reverse mapping got invalid input data. It is neither a phenotype "
        "nor derivation tree: {}".format(data)
    )
    raise MappingError(message)


def raise_limited_codon_size_error(chosen_rule_idx, max_int):
    message = (
        "Reverse mapping could not encode a choice within the given codon size limit. "
        "The index of the chosen rule is {}, but the maximum integer that can be "
        "encoded is {}.".format(chosen_rule_idx, max_int)
    )
    raise MappingError(message)


def raise_dsge_mapping_error1(genotype, symbols):
    message = (
        "The provided DSGE genotype is invalid, because the length of the "
        "genotype ({}) does not fit to the number of "
        "nonterminal symbols ({}).".format(len(genotype), len(symbols))
    )
    raise MappingError(message) from None


def raise_dsge_mapping_error2(gene_index, symbol):
    message = (
        "The provided DSGE genotype is invalid, because gene number {} (which corresponds to "
        "nonterminal symbol <{}>) does not contain enough integers to complete the "
        "mapping.".format(gene_index, symbol)
    )
    raise MappingError(message) from None


def raise_dsge_mapping_error3(gene_index, symbol, rules, chosen_rule_idx):
    message = (
        "The provided DSGE genotype is invalid, because gene number {} (which corresponds to "
        "nonterminal symbol <{}>) contains the integer {} that can not be used to select "
        "a production out of {} available "
        "ones.".format(gene_index, symbol, chosen_rule_idx, len(rules))
    )
    raise MappingError(message) from None


class DatabaseError(Exception):
    """Raised when storing or retrieving data from a database fails.

    Examples
    --------
    - The provided filepath does not point to a valid file-based
      database.
    - There is no database entry that matches to a user-provided value.

    """


def raise_generation_range_error():
    message = (
        'The argument "generation_range" could not be interpreted as two integers.'
    )
    raise ValueError(message)


def raise_ind_max_id_error():
    message = (
        "The maximum individual id returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_pop_size_min_error():
    message = (
        "The minimum population size returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_pop_size_max_error():
    message = (
        "The maximum population size returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_generation_first_error():
    message = (
        "The first generation returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_generation_last_error():
    message = (
        "The last generation returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_fitness_min_error():
    message = (
        "The minimum fitness returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_fitness_max_error():
    message = (
        "The maximum fitness returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_fitness_min_n_error():
    message = (
        "The minimum fitness after n evaluations returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_fitness_max_n_error():
    message = (
        "The maximum fitness after n evaluations returned by the database is NaN. "
        "This could mean that there are no entries in the database yet."
    )
    raise DatabaseError(message)


def raise_no_database_export_error():
    message = (
        "There is no data to export. "
        'The parameter "database_on" needs to be set to "True" in order '
        "for a database to be created and filled with data "
        "by the evolutionary search."
    )
    raise DatabaseError(message) from None


def raise_reset_database_error(filepath, n_tries):
    message = (
        "Tried to rename the database file located at {} during search reset. "
        "Could not find a new filepath after {} tries.".format(repr(filepath), n_tries)
    )
    raise DatabaseError(message) from None


def raise_import_database_error(filepath):
    message = (
        "Could not load the database from filepath {} because it is "
        "not a file.".format(repr(filepath))
    )
    raise DatabaseError(message) from None


def raise_load_population_error(generation):
    message = (
        "Attempted to load generation {} from the database but could not find any "
        "entries for it. This population seems not to be stored in the "
        "database.".format(generation)
    )
    raise DatabaseError(message) from None


def raise_individual_clash_error():
    message = (
        "Failed to load an individual from the old database. There is "
        "another individual with the same id in the current database."
    )
    raise DatabaseError(message) from None


def raise_serialization_error(string):
    message = (
        "Could not serialize the given string. "
        "It can not be enclosed in any quotes, since it contains "
        "all available variants: {}".format(string)
    )
    raise DatabaseError(message) from None
