import copy as _copy
import json as _json
import os as _os
import sqlite3 as _sqlite3
from collections.abc import Iterable as _Iterable

from ... import _utilities
from ... import exceptions as _exceptions
from ... import warnings as _warnings
from ..._utilities import argument_processing as _ap
from ..._utilities.database_management import Sqlite3Wrapper as _Sqlite3Wrapper
from ..._utilities.operating_system import NEWLINE as _NEWLINE
from . import database as _database
from . import plots as _plots


class Database:
    """Database wrapper for easily storing algorithm results."""

    __slots__ = ("_location", "_system", "_dbms", "_cache", "_deserializer")

    def __init__(self, location, system=None):
        """Create a database object referring to a file-based or in-memory SQLite3 database."""
        self._location = location

        # Cache: Store results of repeated calculations, but recalculate after database changes
        self._cache = dict()

        # Create or connect to SQLite3 database, try to create tables and views
        self._dbms = _Sqlite3Wrapper(self._location)
        self._try_creating_tables()
        self._try_creating_views()

        # Deserializer: Convert database entries back to Python objects
        if system is None:
            system = "cfggpst"
        self._deserializer = Deserializer(system)

    # Representations
    def __repr__(self):
        """Compute the "official" string representation of the database wrapper."""
        return "<EvolutionaryAlgorithmDatabase object at {}>".format(hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the database wrapper."""
        max_shown_phenotypes = 5
        # Read data
        num_changes = self._dbms.get_num_changes()
        num_generations = self.num_generations()
        num_individuals = self.num_individuals()
        num_genotypes = self.num_genotypes()
        num_phenotypes = self.num_phenotypes()
        num_fitness = self.num_fitnesses()
        try:
            pop_size_min = self.population_size_min()
            pop_size_max = self.population_size_max()
        except _exceptions.DatabaseError:
            pop_size_min = "not available"
            pop_size_max = "not available"
        min_individuals = self.individuals_with_min_fitness()
        min_genotypes = self.genotypes_with_min_fitness()
        min_phenotypes = self.phenotypes_with_min_fitness()
        num_min_ind = len(min_individuals)
        num_min_genotypes = len(min_genotypes)
        num_min_phenotypes = len(min_phenotypes)
        try:
            min_fitness = self.fitness_min()
            max_fitness = self.fitness_max()
        except _exceptions.DatabaseError:
            min_fitness = "not available"
            max_fitness = "not available"
        max_individuals = self.individuals_with_max_fitness()
        max_genotypes = self.genotypes_with_max_fitness()
        max_phenotypes = self.phenotypes_with_max_fitness()
        num_max_ind = len(max_individuals)
        num_max_genotypes = len(max_genotypes)
        num_max_phenotypes = len(max_phenotypes)

        # Write message
        msg = []
        msg.append("╭─ Database of the evolutionary search{}".format(_NEWLINE))
        msg.append(
            "│ Number of changes ............ {}{}".format(num_changes, _NEWLINE)
        )
        msg.append(
            "│ Number of generations ........ {}{}".format(num_generations, _NEWLINE)
        )
        msg.append(
            "│ Number of individuals ........ {}{}".format(num_individuals, _NEWLINE)
        )
        msg.append(
            "│ Number of unique genotypes ... {}{}".format(num_genotypes, _NEWLINE)
        )
        msg.append(
            "│ Number of unique phenotypes .. {}{}".format(num_phenotypes, _NEWLINE)
        )
        msg.append(
            "│ Number of unique fitnesses ... {}{}".format(num_fitness, _NEWLINE)
        )
        msg.append(
            "│ Minimum population size ...... {}{}".format(pop_size_min, _NEWLINE)
        )
        msg.append(
            "│ Maximum population size ...... {}{}".format(pop_size_max, _NEWLINE)
        )
        if isinstance(min_fitness, float):
            msg.append(
                "│ Minimum fitness .............. {:.6f}{}".format(
                    min_fitness, _NEWLINE
                )
            )
            msg.append(
                "│   shared by {} individuals, {} genotypes, {} phenotypes{}".format(
                    num_min_ind, num_min_genotypes, num_min_phenotypes, _NEWLINE
                )
            )
            msg.append(
                "│ Maximum fitness .............. {:.6f}{}".format(
                    max_fitness, _NEWLINE
                )
            )
            msg.append(
                "│   shared by {} individuals, {} genotypes, {} phenotypes{}".format(
                    num_max_ind, num_max_genotypes, num_max_phenotypes, _NEWLINE
                )
            )
        else:
            msg.append(
                "│ Minimum fitness .............. {}{}".format(min_fitness, _NEWLINE)
            )
            msg.append(
                "│ Maximum fitness .............. {}{}".format(max_fitness, _NEWLINE)
            )

        # First individual with min fitness
        if min_individuals:
            msg.append("│{}".format(_NEWLINE))
            first_min_individual = min_individuals[0]
            msg.append(
                "│ First individual with minimum fitness {:.6f}{}".format(
                    min_fitness, _NEWLINE
                )
            )
            for line in str(first_min_individual).splitlines():
                msg.append("│   {}{}".format(line, _NEWLINE))

        # All phenotypes with min fitness
        if num_min_phenotypes > 1:
            msg.append("│{}".format(_NEWLINE))
            if num_min_phenotypes <= max_shown_phenotypes:
                num_shown_phenotypes = num_min_phenotypes
            else:
                num_shown_phenotypes = max_shown_phenotypes
            msg.append(
                "│ First {} of {} phenotypes with minimum fitness {:.6f}{}".format(
                    num_shown_phenotypes, num_min_phenotypes, min_fitness, _NEWLINE
                )
            )
            for phe in min_phenotypes[:max_shown_phenotypes]:
                msg.append("│   Phenotype: {}{}".format(phe, _NEWLINE))
            if num_min_phenotypes > max_shown_phenotypes:
                msg.append("│   ...{}".format(_NEWLINE))

        # First individual with max fitness
        if max_individuals:
            msg.append("│{}".format(_NEWLINE))
            first_max_individual = max_individuals[0]
            msg.append(
                "│ First individual with maximum fitness {:.6f}{}".format(
                    max_fitness, _NEWLINE
                )
            )
            for line in str(first_max_individual).splitlines():
                msg.append("│   {}{}".format(line, _NEWLINE))

        # All phenotypes with max fitness
        if num_max_phenotypes > 1:
            msg.append("│{}".format(_NEWLINE))
            if num_max_phenotypes <= max_shown_phenotypes:
                num_shown_phenotypes = num_max_phenotypes
            else:
                num_shown_phenotypes = max_shown_phenotypes
            msg.append(
                "│ First {} of {} phenotypes with maximum fitness {:.6f}{}".format(
                    num_shown_phenotypes, num_max_phenotypes, max_fitness, _NEWLINE
                )
            )
            for phe in max_phenotypes[:max_shown_phenotypes]:
                msg.append("│   Phenotype: {}{}".format(phe, _NEWLINE))
            if num_max_phenotypes > max_shown_phenotypes:
                msg.append("│   ...{}".format(_NEWLINE))
        msg.append("╰─")
        text = "".join(msg)
        return text

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Creation of tables and views
    def _try_creating_tables(self):
        """Try to create tables in the database."""
        try:
            # Table 1: search - contains the main data produced by a search
            query = (
                "CREATE TABLE search ( "
                "  individual_id INTEGER PRIMARY KEY, "
                "  parent_ids TEXT, "
                "  generation INTEGER, "
                "  label TEXT, "
                "  genotype TEXT, "
                "  FOREIGN KEY(genotype) REFERENCES genotype_phenotype_mapping(genotype)"
                ");"
            )
            self._dbms.execute_query(query)

            # Table 2: genotype_phenotype_mapping - contains repetitive mapping data
            query = (
                "CREATE TABLE genotype_phenotype_mapping ( "
                "  genotype TEXT PRIMARY KEY, "
                "  phenotype TEXT, "
                "  FOREIGN KEY(phenotype) REFERENCES phenotype_fitness_mapping(phenotype)"
                ");"
            )
            self._dbms.execute_query(query)

            # Table 3: phenotype_fitness_mapping - contains repetitive mapping data
            query = (
                "CREATE TABLE phenotype_fitness_mapping ( "
                "  phenotype TEXT PRIMARY KEY, "
                "  fitness REAL,"
                "  details TEXT "
                ");"
            )
            self._dbms.execute_query(query)
        except _sqlite3.OperationalError:
            # If tables already exist, ignore the error on trying to create them again
            pass

    def _try_creating_views(self):
        """Try to create views in the database.

        These views join some tables for simpler access to their data.

        """
        try:
            # View 1: full_search - combines search data with repetitive mapping data
            query = (
                "CREATE VIEW full_search AS "
                "SELECT "
                "  search.individual_id, "
                "  search.parent_ids, "
                "  search.generation, "
                "  search.label, "
                "  search.genotype, "
                "  genotype_phenotype_mapping.phenotype, "
                "  phenotype_fitness_mapping.fitness, "
                "  phenotype_fitness_mapping.details "
                "FROM search "
                "LEFT JOIN genotype_phenotype_mapping "
                "  ON genotype_phenotype_mapping.genotype=search.genotype "
                "LEFT JOIN phenotype_fitness_mapping "
                "  ON phenotype_fitness_mapping.phenotype=genotype_phenotype_mapping.phenotype;"
            )
            self._dbms.execute_query(query)
        except _sqlite3.OperationalError:
            # If view already exists, ignore the error on trying to create it again
            pass

    # Storing database entries during a search, loading them during analysis
    def _store_population(self, label, population, generation):
        """Store all individuals of a population in the database.

        Notes
        -----
        - Each individual requires multiple INSERT statements. If they
          were commited as separate transactions, it would take a lot of
          time. Instead, the data of all individuals can be collected
          and stored in a single transaction.

        - The details attribute of each individual can contain an
          evaluation key that refers to user-defined data returned by
          the objective function. It is attempted to be JSON serialized
          and in case it fails the data of this attribute is not stored
          and an empty dictionary instead without warning.

        References
        ----------
        - https://www.sqlite.org/faq.html#q19
        - https://docs.python.org/3/library/sqlite3.html#using-sqlite3-efficiently
        - https://stackoverflow.com/questions/603572/escape-single-quote-character-for-use-in-an-sqlite-query

        """
        # Note: Serialization is performed inline (not in several methods) to increase run speed,
        # while deserialization after a run is performed in methods to increase modularity

        # Argument processing
        generation = str(generation)
        label = str(label)

        # Prepare queries
        query1 = "INSERT OR IGNORE INTO phenotype_fitness_mapping VALUES (?, ?, ?);"
        query2 = "INSERT OR IGNORE INTO genotype_phenotype_mapping VALUES (?, ?);"
        query3 = "INSERT INTO search VALUES (?, ?, ?, ?, ?);"

        # Prepare data
        data1 = []
        data2 = []
        data3 = []
        phe_known = set()
        gt_known = set()
        for ind in population:
            # Serialization: genotype to str in system-specific format
            gt = str(ind.genotype)
            # Serialization: parent_ids to str in list format
            try:
                parent_ids = str(ind.details["parent_ids"])
            except KeyError:
                parent_ids = "[]"
            # Data for table "search"
            data3.append((ind.details["id"], str(parent_ids), generation, label, gt))
            # Data for other tables
            phe = ind.phenotype
            if phe is not None:
                if phe not in phe_known:
                    phe_known.add(phe)
                    # Serialization: details to None, JSON or str
                    details = ind.details["evaluation"]
                    if details is not None:
                        try:
                            details = _json.dumps(details)
                        except Exception:
                            details = str(details)
                    # Data for table "phenotype_fitness_mapping"
                    data1.append((phe, ind.fitness, details))
                if gt not in gt_known:
                    # Data for table "genotype_phenotype_mapping"
                    gt_known.add(gt)
                    data2.append((gt, phe))

        # Insert data
        self._dbms.execute_query_for_many_records(query1, data1)
        self._dbms.execute_query_for_many_records(query2, data2)
        self._dbms.execute_query_for_many_records(query3, data3)

    def _load_population(self, generation, with_parent_ids=True):
        """Load a population identified by its generation from the database.

        The information in from the database is converted to suitable
        Python objects on the level of individual attributes
        (e.g. fitness is float), individual objects (type depends on
        system) and population (type depends on system).

        Raises
        ------
        DatabaseError
            If the loaded population is empty because the database does
            not contain the user-provided generation.

        """
        # Load database entries
        query = 'SELECT * FROM full_search WHERE generation=? AND label="main";'
        rows = self._dbms.execute_query(query, (generation,))

        # Raise error if empty
        if not rows:
            _exceptions.raise_load_population_error(generation)

        # Reconstruct population
        return self._deserializer.population(rows)

    def _store_database_subset(self, data):
        """Store a subset of data to the database."""
        # Split data
        data_search = data["search"]
        data_gen_phe = data["genotype_phenotype_mapping"]
        data_phe_fit = data["phenotype_fitness_mapping"]

        # Table 1: search
        query = "INSERT INTO search VALUES (?, ?, ?, ?, ?);"
        try:
            self._dbms.execute_query_for_many_records(query, data_search)
        except _sqlite3.IntegrityError:
            _exceptions.raise_individual_clash_error()

        # Table 2: genotype_phenotype_mapping
        query = "INSERT OR IGNORE INTO genotype_phenotype_mapping VALUES (?, ?);"
        self._dbms.execute_query_for_many_records(query, data_gen_phe)

        # Table 3: phenotype_fitness_mapping
        query = "INSERT OR IGNORE INTO phenotype_fitness_mapping VALUES (?, ?, ?);"
        self._dbms.execute_query_for_many_records(query, data_phe_fit)

    def _load_database_subset(self, first_gen, last_gen):
        """Load a subset of data defined by first and last generation from the database."""
        # Table 1: search
        query = "SELECT * FROM search WHERE generation>=? AND generation<=?;"
        search = self._dbms.execute_query(query, (first_gen, last_gen))

        # Table 2: genotype_phenotype_mapping
        query = (
            "WITH chosen_genotypes AS ("
            "  SELECT DISTINCT genotype FROM search "
            "  WHERE generation>=? AND generation<=? "
            ") "
            "SELECT * FROM genotype_phenotype_mapping WHERE genotype IN chosen_genotypes"
        )
        genotype_phenotype_mapping = self._dbms.execute_query(
            query, (first_gen, last_gen)
        )

        # Table 3: phenotype_fitness_mapping
        query = (
            "WITH chosen_genotypes AS ("
            "  SELECT DISTINCT genotype FROM search "
            "  WHERE generation>=? AND generation<=? "
            "), chosen_phenotypes AS ("
            "  SELECT DISTINCT phenotype FROM genotype_phenotype_mapping "
            "  WHERE genotype IN chosen_genotypes "
            ") "
            "SELECT * FROM phenotype_fitness_mapping WHERE phenotype IN chosen_phenotypes"
        )
        phenotype_fitness_mapping = self._dbms.execute_query(
            query, (first_gen, last_gen)
        )

        # Combine the results in a dictionary
        data = dict(
            search=search,
            genotype_phenotype_mapping=genotype_phenotype_mapping,
            phenotype_fitness_mapping=phenotype_fitness_mapping,
        )
        return data

    def _load_database_full(self):
        """Load all data from the database."""
        return self._load_database_subset(
            self.generation_first(), self.generation_last()
        )

    # File I/O
    def export_sql(self, filepath, ext="sqlite3"):
        """Export the evolutionary search by storing the current database to an SQL file.

        All tables and views are exported to a single SQL file that
        adheres to
        `SQLite version 3 <https://docs.python.org/3/library/sqlite3.html>`__.
        It can be used for a later import in order to continue and
        analyze a run, or it may be opened with an external tools such
        as `DB Browser for SQLite <https://sqlitebrowser.org/>`__.

        Parameters
        ----------
        filepath : str
            The given filepath may automatically be modified in two
            ways:

            - It is ensured to end with the extension defined by the
              ``ext`` argument.
            - It is ensured to be a filepath that does not exist yet by
              adding a numerical suffix.

              Example: If ``some_file.sqlite3`` exists, it uses
              ``some_file_1.sqlite3`` or if that also exists then
              ``some_file_2.sqlite3`` and so on.
        ext : str
            The extension that the filepath is ensured to end with.

            - If ``None``, no extension is added.
            - If ``db``, the filepath is ensured to end with ``.db``.
            - If ``.sql``, the filepath is ensured to end with ``.sql``.

        Returns
        -------
        filepath_used : str

        """
        # Argument processing
        if ext is None:
            filepath_used = filepath
        else:
            filepath_used = _utilities.operating_system.ensure_file_extension(
                filepath, ext
            )
        filepath_used = _utilities.operating_system.ensure_new_path(filepath_used)

        # Export
        self._dbms.export_sql(filepath_used)
        return filepath_used

    def export_csv(self, filepath, ext="csv"):
        """Export the database of this evolutionary search as CSV file.

        Only the main view, which gathers information from all
        individual tables, is exported to a single CSV file. As such
        it provides all information stored in the database in a
        redundant manner. Currently it can not be used for a later
        import, but it can be opened with external tools that can read
        CSV files, such as
        `LibreOffice Calc <https://www.libreoffice.org/discover/calc/>`__
        or
        `Tad <https://www.tadviewer.com>`__.

        Parameters
        ----------
        filepath : str
            The given filepath may automatically be modified in two ways:

            - It is ensured to end with the extension defined by the
              ``ext`` argument.
            - It is ensured to be a filepath that does not exist yet by
              adding a numerical suffix.

              Example: If ``some_file.sqlite3`` exists, it uses
              ``some_file_1.sqlite3`` or if that also exists then
              ``some_file_2.sqlite3`` and so on.
        ext : str
            The extension that the filepath is ensured to end with.

            - If it is ``None``, no extension is added.
            - If it is ``csv``, the filepath is ensured to end with
              ``.csv``.
            - If it is ``.csv``, the filepath is ensured to end with
              ``.csv``.

        Returns
        -------
        filepath_used : str

        """
        # Argument processing
        if ext is None:
            filepath_used = filepath
        else:
            filepath_used = _utilities.operating_system.ensure_file_extension(
                filepath, ext
            )
        filepath_used = _utilities.operating_system.ensure_new_path(filepath_used)

        # Export
        self._dbms.export_csv(filepath, name="full_search")
        return filepath_used

    def import_sql(self, filepath, generation_range=None):
        """Import an evolutionary search by loading the SQL file of a previous run.

        Either all generations (default) or only a subset defined by the
        interval ``[first_generation, last_generation]`` can be loaded.

        Caution: The method :meth:`reset` is called, so that the current
        state and database are dropped and can be replaced in a clean
        fashion by new data from the SQL file.

        Parameters
        ----------
        filepath : str
            Filepath of an SQLite3 file exported by a previous run.
        generation_range : `tuple` of two `int`, optional
            The first and last generation to include in the import.

        Examples
        --------
        Using ``first_generation=0`` and ``last_generation=2`` loads the
        first three generations of a previous run. The last generation
        is reconstructed as population in memory, so that the search can
        be continued from this point. If :meth:`run_step()` is called,
        the last generation loaded from the database (2) is used to
        construct the next generation (3). The resulting search state
        can again be exported to an SQL file if desired.

        """
        # Argument processing
        first_gen, last_gen = self._process_generation_range(generation_range)
        filepath = _ap.str_arg("filepath", filepath)
        if not _os.path.isfile(filepath):
            _exceptions.raise_import_database_error(filepath)

        # Reset
        self.reset()

        # Databases
        source_db = _database.Database(filepath)
        if self.parameters.database_on:
            target_db = self.database

        # Argument processing (with information from source database)
        first_gen_in_db = source_db.generation_first()
        last_gen_in_db = source_db.generation_last()
        if first_gen < first_gen_in_db or first_gen > last_gen_in_db:
            _warnings._warn_database_import_first(first_gen_in_db, first_gen)
            first_generation = first_gen_in_db
        if last_gen > last_gen_in_db or last_gen < first_gen_in_db:
            _warnings._warn_database_import_last(last_gen_in_db, last_gen)
            last_generation = last_gen_in_db

        # Load subset from old db and store it into new db
        if self.parameters.database_on:
            data = source_db._load_database_subset(first_generation, last_generation)
            if all(len(entries) == 0 for entries in data.values()):
                message = (
                    'Tried to load the chosen data from SQL database "{}" but the '
                    "resulting list was empty.".format(filepath)
                )
                raise ValueError(message)
            target_db._store_database_subset(data)

        # Reconstruct properties to determine current state
        if (last_generation - first_generation) > 1:
            with_parent_ids = True
        else:
            with_parent_ids = False
        last_population = source_db._load_population(
            self.parameters.system, last_generation, with_parent_ids
        )
        if len(last_population) == 0:
            message = (
                'Tried to load the chosen data from SQL database "{}" but the '
                "resulting list was empty.".format(filepath)
            )
            raise ValueError(message)
        min_inds = source_db.individuals_with_min_fit(
            self.parameters.system, first_generation, last_generation
        )
        min_ind = None if not min_inds else min_inds[0]
        max_inds = source_db.individuals_with_max_fit(
            self.parameters.system, first_generation, last_generation
        )
        max_ind = None if not max_inds else max_inds[0]
        max_id = source_db._individual_max_id([first_generation, last_generation])
        if not max_id:
            max_id = 0

        # Set current state, partly based on database content, partly fresh
        self.state.population = last_population
        self.state.generation = last_generation + 1
        self.state.num_generations = 0
        self.state.num_individuals = max_id + 1
        self.state.num_gen_to_phe_evaluations = 0
        self.state.num_phe_to_fit_evaluations = 0
        if True:
            self.state.best_individual = min_ind
        else:
            self.state.best_individual = max_ind
        self.state.min_individual = min_ind
        self.state.max_individual = max_ind

    def import_sql_evaluations(self, filepath, verbose=False):
        """Import only phenotype-to-fitness evaluation data from an SQL file.

        This method allows to load phenotype-to-fitness calculations
        from a previous run. It is relevant when the objective funtion
        is computationally demanding and prevention of some
        recalculations may speed up the search significantly.

        Parameters
        ----------
        filepath : str
            Filepath of an SQLite3 file exported by a previous run.

        """
        # Argument processing
        filepath = _ap.str_arg("filepath", filepath)
        if not _os.path.isfile(filepath):
            _exceptions.raise_import_database_error(filepath)

        # Preparation of source and target database
        source_db = Database(filepath, system="")
        target_db = self

        # Load data from source database, warn if it is empty
        query = "SELECT * FROM phenotype_fitness_mapping;"
        phe_fit_evaluations = source_db._dbms.execute_query(query)
        if len(phe_fit_evaluations) == 0:
            _warnings._warn_import_database_empty(filepath)

        # Store data in target database
        query = "INSERT OR IGNORE INTO phenotype_fitness_mapping VALUES (?, ?, ?);"
        for row in phe_fit_evaluations:
            target_db._dbms.execute_query(query, row)

        # Optional report
        if verbose:
            num_eval = len(phe_fit_evaluations)
            message = (
                "Loaded {} phenotype-to-fitness evaluations from "
                "external database at {}."
            ).format(num_eval, filepath)
            print(message)

    # Getting insights into stored information
    # - Counts
    def num_generations(self):
        """Get the number of generations stored in the database."""
        # Query
        query = "SELECT MAX(generation)-MIN(generation)+1 FROM search;"
        result = self._dbms.execute_query(query)

        # Check
        value = result[0][0]
        if value is None:
            value = 0
        return value

    def num_individuals(self, generation_range=None, only_main=False):
        """Get the number of individuals stored in the database."""
        # Note: individual_id is the primary key, hence counting does not require DISTINCT

        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT COUNT(individual_id) FROM search " "WHERE label={};"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT COUNT(individual_id) FROM search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return result[0][0]

    def num_genotypes(self, generation_range=None, only_main=False):
        """Get the number of unique genotypes stored in the database."""
        # Note: Null is not considered in 'SELECT COUNT(DISTINCT genotype)' but should never occur

        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT COUNT(DISTINCT genotype) FROM full_search " "WHERE label={};"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT COUNT(DISTINCT genotype) FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return result[0][0]

    def num_phenotypes(self, generation_range=None, only_main=False):
        """Get the number of unique phenotypes stored in the database."""
        # Note: Null would not be considered in 'SELECT COUNT(DISTINCT phenotype)' but can occur

        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT COUNT(*) FROM ("
                "  SELECT DISTINCT phenotype FROM full_search"
                "  WHERE label={}"
                ");"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT COUNT(*) FROM ("
                "  SELECT DISTINCT phenotype FROM full_search"
                "  WHERE label={} AND generation BETWEEN ? AND ?"
                ");"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return result[0][0]

    def num_fitnesses(self, generation_range=None, only_main=False):
        """Get the number of unique fitness values stored in the database.

        NaN values are not counted. These values appear in individuals
        that were not evaluated, e.g. those generated by crossover but
        then modified by mutation before being evaluated and selected.

        """
        # Note: Null (NaN) is not considered in 'SELECT COUNT(DISTINCT fitness)' but may occur

        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT COUNT(*) FROM ("
                "  SELECT DISTINCT fitness FROM full_search"
                "  WHERE label={}"
                ");"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT COUNT(*) FROM ("
                "  SELECT DISTINCT fitness FROM full_search"
                "  WHERE label={} AND generation BETWEEN ? AND ?"
                ");"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return result[0][0]

    def num_details(self, generation_range=None, only_main=False):
        """Get the number of unique details stored in the database."""
        # Note: Null would not be considered in 'SELECT COUNT(DISTINCT details)' but can occur

        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT COUNT(*) FROM ("
                "  SELECT DISTINCT details FROM full_search"
                "  WHERE label={}"
                ");"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT COUNT(*) FROM ("
                "  SELECT DISTINCT details FROM full_search"
                "  WHERE label={} AND generation BETWEEN ? AND ?"
                ");"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return result[0][0]

    def num_gen_to_phe_evaluations(self):
        """Get the number of genotype-to-phenotype evaluations."""
        query = "SELECT COUNT(*) FROM genotype_phenotype_mapping;"
        result = self._dbms.execute_query(query)
        return result[0][0]

    def num_phe_to_fit_evaluations(self, only_unique=True):
        """Get the number of phenotype-to-fitness evaluations.

        Note: It assumes that no phenotype was evaluated more than once,
        which depends on the parametrization (cache and/or database
        lookups).

        """
        # Argument processing
        column = "DISTINCT(t1.phenotype)" if only_unique else "t1.phenotype"

        # Query
        query = (
            "SELECT COUNT({}) FROM genotype_phenotype_mapping AS t1 "
            "LEFT JOIN phenotype_fitness_mapping AS t2 "
            "ON t2.phenotype=t1.phenotype;"
        ).format(column)
        result = self._dbms.execute_query(query)
        return result[0][0]

    # - Generation
    def generation_first(self):
        """Get the first generation stored in the database.

        Raises an error if the database does not contain any entries yet.

        """
        # Query
        query = "SELECT MIN(generation) FROM search;"
        result = self._dbms.execute_query(query)

        # Check
        value = result[0][0]
        if value is None:
            _exceptions.raise_generation_first_error()
        return value

    def generation_last(self):
        """Get the last generation stored in the database.

        Raises an error if the database does not contain any entries yet.

        """
        # Query
        query = "SELECT MAX(generation) FROM search;"
        result = self._dbms.execute_query(query)

        # Check
        value = result[0][0]
        if value is None:
            _exceptions.raise_generation_last_error()
        return value

    def _process_generation_range(self, generation_range):
        """Process the user-provided generation range so it can be used safely in a query."""
        _ap.check_arg(
            "generation_range", generation_range, (type(None), int, _Iterable)
        )

        if isinstance(generation_range, int):
            generation_range = (generation_range, generation_range)
        elif isinstance(generation_range, _Iterable):
            try:
                assert not isinstance(generation_range, str)
                first, last = generation_range
                if first is None:
                    try:
                        first = self.generation_first()
                    except _exceptions.DatabaseError:
                        first = 0
                if last is None:
                    try:
                        last = self.generation_last()
                    except _exceptions.DatabaseError:
                        last = 0
                assert not isinstance(first, float)
                assert not isinstance(last, float)
                generation_range = (int(first), int(last))
            except Exception:
                _exceptions.raise_generation_range_error()
        return generation_range

    # - Individual
    def _individual_max_id(self, generation_range=None, only_main=False):
        """Get the largest individual id stored in the database.

        Returns None if the database does not contain any entries yet.

        """
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = ("SELECT MAX(individual_id) FROM search " "WHERE label={};").format(
                label
            )
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT MAX(individual_id) FROM search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)

        # Check
        value = result[0][0]
        if value is None:
            _exceptions.raise_ind_max_id_error()
        return value

    def individuals(self, generation_range=None, only_main=False):
        """Get all individuals stored in the database."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = ("SELECT * FROM full_search " "WHERE label={};").format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT * FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return self._deserializer.individuals(result)

    def individuals_with_given_fitness(
        self, fitness, generation_range=None, only_main=False
    ):
        """Get all individuals that have the same user-provided fitness value."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT * FROM full_search " "WHERE label={} AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (fitness,))
        else:
            query = (
                "SELECT * FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ? AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (*generation_range, fitness))
        return self._deserializer.individuals(result)

    def individuals_with_min_fitness(self, generation_range=None, only_main=False):
        """Get all individuals that have the same minimum fitness value."""
        try:
            value = self.fitness_min(generation_range, only_main)
            individuals = self.individuals_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            individuals = []
        return individuals

    def individuals_with_max_fitness(self, generation_range=None, only_main=False):
        """Get all individuals that have the same maximum fitness value."""
        try:
            value = self.fitness_max(generation_range, only_main)
            individuals = self.individuals_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            individuals = []
        return individuals

    def individuals_with_low_fitness(
        self, n=10, generation_range=None, only_main=False
    ):
        """Get the first n elements from a list of individuals sorted by lowest fitness."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT * FROM full_search "
                "WHERE label={} AND fitness IS NOT NULL "
                "ORDER BY fitness ASC "
                "LIMIT ?;"
            ).format(label)
            result = self._dbms.execute_query(query, (n,))
        else:
            query = (
                "SELECT * FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ? AND fitness IS NOT NULL "
                "ORDER BY fitness ASC "
                "LIMIT ?;"
            ).format(label)
            result = self._dbms.execute_query(query, (*generation_range, n))
        return self._deserializer.individuals(result)

    def individuals_with_high_fitness(
        self, n=10, generation_range=None, only_main=False
    ):
        """Load the first n individuals, when all of them are sorted by hightest fitness."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT * FROM full_search "
                "WHERE label={} AND fitness IS NOT NULL "
                "ORDER BY fitness DESC "
                "LIMIT ?;"
            ).format(label)
            result = self._dbms.execute_query(query, (n,))
        else:
            query = (
                "SELECT * FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ? AND fitness IS NOT NULL "
                "ORDER BY fitness DESC "
                "LIMIT ?;"
            ).format(label)
            result = self._dbms.execute_query(query, (*generation_range, n))
        return self._deserializer.individuals(result)

    # - Population
    def population_size_min(self):
        """Get smallest population size of any generation stored in the database.

        Raises an error if the database does not contain any entries yet.

        """
        # Query
        query = (
            "SELECT MIN(population_size) FROM ("
            "  SELECT COUNT(label) AS population_size FROM search "
            '  WHERE label="main" GROUP BY generation'
            ");"
        )
        result = self._dbms.execute_query(query)

        # Check
        value = result[0][0]
        if value is None:
            _exceptions.raise_pop_size_min_error()
        return value

    def population_size_max(self):
        """Get largest population size of any generation stored in the database.

        Raises an error if the database does not contain any entries yet.

        """
        # Query
        query = (
            "SELECT MAX(population_size) FROM ("
            "  SELECT COUNT(label) AS population_size FROM search "
            '  WHERE label="main" GROUP BY generation'
            ");"
        )
        result = self._dbms.execute_query(query)

        # Check
        value = result[0][0]
        if value is None:
            _exceptions.raise_pop_size_max_error()
        return value

    # - Genotype
    def genotypes(self, generation_range=None, only_main=False):
        """Get unique genotypes stored in the database."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = ("SELECT DISTINCT genotype FROM search " "WHERE label={};").format(
                label
            )
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT DISTINCT genotype FROM search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return self._deserializer.genotypes(result)

    def genotypes_with_given_fitness(
        self, fitness, generation_range=None, only_main=False
    ):
        """Get unique genotypes that have the same user-provided fitness value."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT DISTINCT genotype FROM full_search "
                "WHERE label={} AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (fitness,))
        else:
            query = (
                "SELECT DISTINCT genotype FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ? AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (*generation_range, fitness))
        return self._deserializer.genotypes(result)

    def genotypes_with_min_fitness(self, generation_range=None, only_main=False):
        """Load unique genotypes that have the same minimum fitness value."""
        try:
            value = self.fitness_min(generation_range, only_main)
            genotypes = self.genotypes_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            genotypes = []
        return genotypes

    def genotypes_with_max_fitness(self, generation_range=None, only_main=False):
        """Load unique genotypes that have the same maximum fitness value."""
        try:
            value = self.fitness_max(generation_range, only_main)
            genotypes = self.genotypes_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            genotypes = []
        return genotypes

    # - Phenotype
    def phenotypes(self, generation_range=None, only_main=False):
        """Load unique phenotypes."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT DISTINCT phenotype FROM full_search " "WHERE label={};"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT DISTINCT phenotype FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return self._deserializer.phenotypes(result)

    def phenotypes_with_given_fitness(
        self, fitness, generation_range=None, only_main=False
    ):
        """Load unique phenotypes that have the same user-provided fitness value."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT DISTINCT phenotype FROM full_search "
                "WHERE label={} AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (fitness,))
        else:
            query = (
                "SELECT DISTINCT phenotype FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ? AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (*generation_range, fitness))
        return self._deserializer.phenotypes(result)

    def phenotypes_with_min_fitness(self, generation_range=None, only_main=False):
        """Load unique phenotypes that have the same minimum fitness value."""
        try:
            value = self.fitness_min(generation_range, only_main)
            phenotypes = self.phenotypes_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            phenotypes = []
        return phenotypes

    def phenotypes_with_max_fitness(self, generation_range=None, only_main=False):
        """Load unique phenotypes that have the same minimum fitness value."""
        try:
            value = self.fitness_max(generation_range, only_main)
            phenotypes = self.phenotypes_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            phenotypes = []
        return phenotypes

    # - Details (optionally returned by objective function during phenotype-fitness evaluation)
    def details(self, generation_range=None, only_main=False):
        """Load unique details (returned by the objective function)."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT DISTINCT details FROM full_search " "WHERE label={};"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT DISTINCT details FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return self._deserializer.multiple_details(result)

    def details_with_given_fitness(
        self, fitness, generation_range=None, only_main=False
    ):
        """Load unique details that have the same user-provided fitness value."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT DISTINCT details FROM full_search "
                "WHERE label={} AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (fitness,))
        else:
            query = (
                "SELECT DISTINCT details FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ? AND fitness=?;"
            ).format(label)
            result = self._dbms.execute_query(query, (*generation_range, fitness))
        return self._deserializer.multiple_details(result)

    def details_with_min_fitness(self, generation_range=None, only_main=False):
        """Load unique details that have the same minimum fitness value."""
        try:
            value = self.fitness_min(generation_range, only_main)
            details = self.details_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            details = []
        return details

    def details_with_max_fitness(self, generation_range=None, only_main=False):
        """Load unique details that have the same maximum fitness value."""
        try:
            value = self.fitness_max(generation_range, only_main)
            details = self.details_with_given_fitness(
                value, generation_range, only_main
            )
        except _exceptions.DatabaseError:
            details = []
        return details

    # - Fitness
    def fitnesses(self, generation_range=None, only_main=False):
        """Load unique fitness values."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = (
                "SELECT DISTINCT fitness FROM full_search " "WHERE label={};"
            ).format(label)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT DISTINCT fitness FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)
        return self._deserializer.fitnesses(result)

    def fitness_min(self, generation_range=None, only_main=False):
        """Load the minimum fitness value that was found."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = ("SELECT MIN(fitness) FROM full_search " "WHERE label={};").format(
                label
            )
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT MIN(fitness) FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)

        # Check
        value = self._deserializer.fitness(result[0][0])
        if value != value:  # if NaN
            _exceptions.raise_fitness_min_error()
        return value

    def fitness_max(self, generation_range=None, only_main=False):
        """Load the maximum fitness value that was found."""
        # Argument processing
        generation_range = self._process_generation_range(generation_range)
        label = '"main"' if only_main else "label"

        # Query
        if generation_range is None:
            query = ("SELECT MAX(fitness) FROM full_search " "WHERE label={};").format(
                label
            )
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT MAX(fitness) FROM full_search "
                "WHERE label={} AND generation BETWEEN ? AND ?;"
            ).format(label)
            result = self._dbms.execute_query(query, generation_range)

        # Check
        value = self._deserializer.fitness(result[0][0])
        if value != value:  # if NaN
            _exceptions.raise_fitness_max_error()
        return value

    def fitness_min_after_num_evals(self, num_evaluations):
        """Load the minimum fitness value that was found after a number of fitness evaluations."""
        # Argument processing
        _ap.int_arg("num_evaluations", num_evaluations, min_incl=1)

        # Query
        query = (
            "SELECT MIN(fitness) from ("
            "  SELECT fitness FROM genotype_phenotype_mapping AS t1 "
            "  LEFT JOIN phenotype_fitness_mapping AS t2 "
            "  ON t2.phenotype=t1.phenotype "
            "  LIMIT ?"
            ");"
        )
        result = self._dbms.execute_query(query, (num_evaluations,))

        # Check
        value = self._deserializer.fitness(result[0][0])
        if value != value:  # if NaN
            _exceptions.raise_fitness_min_n_error()
        return value

    def fitness_max_after_num_evals(self, num_evaluations):
        """Load the maximum fitness value that was found after a number of fitness evaluations."""
        # Argument processing
        _ap.int_arg("num_evaluations", num_evaluations, min_incl=1)

        # Query
        query = (
            "SELECT MAX(fitness) from ("
            "  SELECT fitness FROM genotype_phenotype_mapping AS t1 "
            "  LEFT JOIN phenotype_fitness_mapping AS t2 "
            "  ON t2.phenotype=t1.phenotype "
            "  LIMIT ?"
            ");"
        )
        result = self._dbms.execute_query(query, (num_evaluations,))

        # Check
        value = self._deserializer.fitness(result[0][0])
        if value != value:  # if NaN
            _exceptions.raise_fitness_max_n_error()
        return value

    # - Genotype-phenotype evaluations
    def gen_to_phe_evaluations(self, num_evaluations=None):
        """Get genotype-to-phenotype evaluations that were performed during the search.

        Guaranteed:
        - The order of the list is the order of performed evaluations.

        Not guaranteed:
        - The same evaluations may have been performed multiple times during the run,
          depending on cache settings, which is not available as information in the
          database.

        """
        if num_evaluations is None:
            query = "SELECT * FROM genotype_phenotype_mapping;"
            result = self._dbms.execute_query(query)
        else:
            query = "SELECT * FROM genotype_phenotype_mapping LIMIT ?;"
            result = self._dbms.execute_query(query, (num_evaluations,))
        return self._deserializer.gt_phe_map(result)

    # - Phenotype-fitness evaluations
    def phe_to_fit_evaluations(self, num_evaluations=None, with_details=False):
        """Get phenotype-to-fitness evaluations that were performed during the search.

        Guaranteed:
        - The order of the list is the order of performed evaluations.
        - Genotype-phenotype pairs that were loaded from previous runs are not considered.

        Not guaranteed:
        - The same evaluations may have been performed multiple times during the run,
          depending on cache and database lookup settings, which is not available
          as information in the database.

        """
        # Argument processing
        if with_details:
            tables = "DISTINCT(t1.phenotype), fitness, details"
        else:
            tables = "DISTINCT(t1.phenotype), fitness"

        # Query
        # Note: Uses genotype_phenotype_mapping to get the right order of phenotypes and
        # not be influenced by potentially external data present in phenotype_fitness_mapping.
        if num_evaluations is None:
            query = (
                "SELECT {} FROM genotype_phenotype_mapping AS t1 "
                "LEFT JOIN phenotype_fitness_mapping AS t2 "
                "ON t2.phenotype=t1.phenotype;"
            ).format(tables)
            result = self._dbms.execute_query(query)
        else:
            query = (
                "SELECT {} FROM genotype_phenotype_mapping AS t1 "
                "LEFT JOIN phenotype_fitness_mapping AS t2 "
                "ON t2.phenotype=t1.phenotype "
                "LIMIT ?;"
            ).format(tables)
            result = self._dbms.execute_query(query, (num_evaluations,))

        # Conditional return
        if with_details:
            return self._deserializer.phe_fit_det_map(result)
        else:
            return self._deserializer.phe_fit_map(result)

    # Support for memoization of phenotype-fitness mappings
    def _lookup_phenotype_evaluations(self, phenotypes):
        """Get phenotype-to-fitness evaluations for all known phenotypes in a given list.

        References
        ----------
        - https://www.sqlite.org/limits.html
        - https://stackoverflow.com/questions/44012117/what-is-the-most-efficient-way-to-query-multiple-values-from-a-single-column-in

        """
        # Query
        n_max = 999  # SQLITE_MAX_VARIABLE_NUMBER for SQLite versions prior to 3.32.0
        n = len(phenotypes)
        if n > n_max:
            # Split it into multiple queries if the list contains too many phenotypes
            result = []
            for i in range(0, n, n_max):
                partial = self._phenotype_evaluations(phenotypes[i : i + n_max])
                result.extend(partial)
        else:
            # Single query
            values = [str(phe) for phe in phenotypes]
            query = "SELECT * FROM phenotype_fitness_mapping WHERE phenotype IN ({})".format(
                ",".join(["?"] * len(values))
            )
            result = self._dbms.execute_query(query, values)
            ph = self._deserializer.phenotype
            fi = self._deserializer.fitness
            de = self._deserializer.details
            result = [(ph(row[0]), (fi(row[1]), de(row[2]))) for row in result]
        return result

    # Data representations
    def to_list(self, generation_range=None, only_main=False):
        """Convert the database entries to a list of rows and add some derived information.

        It uses lazy loading, i.e. it is only constructed again if the database
        has changed since the last call.

        The first list entry contains the column names.

        """

        def create_list():
            # Load entries
            query = "SELECT * FROM full_search;"
            data = self._dbms.execute_query(query)

            # Deserialize entries and derive new information
            def safe_len(obj):
                try:
                    return len(obj)
                except TypeError:
                    return 0

            for i in range(len(data)):
                row = data[i]
                data[i] = [
                    # contained
                    row[0],  # individual_id [int]
                    self._deserializer.parent_ids(row[1]),  # parent_ids [list, None]
                    row[2],  # generation [int]
                    row[3],  # label [str]
                    self._deserializer.genotype(row[4]),  # genotype [Genotype]
                    self._deserializer.phenotype(row[5]),  # phenotype [str, None]
                    self._deserializer.fitness(row[6]),  # fitness [float, NaN]
                    self._deserializer.details(
                        row[7]
                    ),  # details [None, JSON object, str]
                    # derived
                    safe_len(row[4]),  # genotype_length [int]
                    safe_len(row[5]),  # phenotype_length [int]
                    -1,  # rank [int] (calculated later)
                ]

            # Rank calculation for each population
            def assign_ranks(row_idx, fitnesses):
                # Sort a completed generation by fitness, derive ranks and assign them
                if fitnesses:
                    rank_idx = list(
                        range(len(fitnesses), 0, -1)
                    )  # consider entries -n to -1
                    rank_idx_fit = list(zip(rank_idx, fitnesses))
                    rank_idx_fit.sort(
                        key=lambda x: x[1]
                    )  # sort by fitness to distribute ranks
                    for rank, (rank_idx, _) in enumerate(rank_idx_fit):
                        data[row_idx - rank_idx][
                            10
                        ] = rank  # assign ranks to entries -n to -1

            last_label = "invalid label"
            last_generation = "invalid generation"
            fitnesses = []
            for i in range(len(data)):
                row = data[i]
                generation = row[2]
                label = row[3]
                fitness = row[6]
                if generation != last_generation or label != last_label:
                    assign_ranks(i, fitnesses)
                    fitnesses = []
                last_generation = generation
                last_label = label
                fitnesses.append(fitness)
            if data:
                assign_ranks(i + 1, fitnesses)
            return [tuple(row) for row in data]

        # Calculation or lookup in cache
        data = self._lookup_or_calc("to_list", create_list)

        # Optional filtering
        data = self._filter_list(data, generation_range, only_main)
        return data

    def to_columns(self):
        """Get the columns available in all data."""
        # primary source of truth about columns in data
        columns = (
            "individual_id",
            "parent_ids",
            "generation",
            "label",
            "genotype",
            "phenotype",
            "fitness",
            "details",
            "genotype_length",
            "phenotype_length",
            "rank",
        )
        return columns

    def to_dataframe(self, generation_range=None, only_main=False):
        """Convert the database entries to a Pandas DataFrame.

        Some derived information is added during the conversion.

        Parameters
        ----------
        only_main_populations : bool
            From the complete dataframe, only main populations are kept
            and intermediate ones (selected parents, crossed-over or
            mutated populations) are filtered out. Note that the
            crossed-over population is not evaluated if a mutation
            operator is provided, so that they have no associated
            phenotype and fitness values.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.total_changes

        """
        import pandas

        # Input data
        data = self.to_list(generation_range, only_main)
        columns = self.to_columns()

        # DataFrame
        df = pandas.DataFrame(data, columns=columns)
        return df

    def to_network(self, generation_range=None, only_main=False):
        """Convert the database entries to NetworkX graph and add some derived information."""
        import networkx as nx

        # Input data
        nodes, edges = self._to_graph(generation_range, only_main)

        # Graph
        graph = nx.Graph()

        # Nodes
        columns = self.to_columns() + ("x", "y", "z", "form", "color", "size")
        for row in nodes:
            attributes = {key: val for key, val in zip(columns, row)}
            attributes["hover"] = "\n".join(
                "{}: {}".format(key, val) for key, val in zip(columns, row)
            )
            graph.add_node(row[0], **attributes)

        # Edges
        columns = (
            "parent_id",
            "individual_id",
            "individual_x",
            "parent_x",
            "individual_y",
            "parent_y",
            "individual_z",
            "parent_z",
            "individual_label",
            "style",
            "color",
            "width",
            "individual_generation",
        )
        for row in edges:
            graph.add_edge(
                row[0], row[1], **{key: val for key, val in zip(columns[1:], row[1:])}
            )
        return graph

    def to_jgf(self, generation_range=None, only_main=False):
        """Convert the data into JSON graph format for visualization."""

        def create_jgf(nodes, edges):
            # Graph as dictionary in JSON graph format
            graph = {
                "graph": {
                    "directed": False,
                    "nodes": [],
                    "edges": [],
                }
            }
            gn = graph["graph"]["nodes"]
            ge = graph["graph"]["edges"]

            # Nodes
            columns = self.to_columns() + ("x", "y", "z", "form", "color", "size")
            for row in nodes:
                # Adapt coordinats (which fit only for Matplotlib and Plotly)
                x = row[11] * 100
                y = row[12] * -25
                z = row[13] * 5
                hover = "\n".join(
                    "{}: {}".format(key.title(), val)
                    for key, val in list(zip(columns, row))
                    if key in ["phenotype", "fitness"]
                )
                node = {
                    "id": row[0],
                    "metadata": {
                        "x": x,
                        "y": y,
                        "z": z,
                        "color": row[15],
                        "size": row[16],
                        "hover": hover,
                    },
                }
                gn.append(node)

            # Edges
            columns = (
                "parent_id",
                "individual_id",
                "individual_x",
                "parent_x",
                "individual_y",
                "parent_y",
                "individual_z",
                "parent_z",
                "individual_label",
                "style",
                "color",
                "width",
                "individual_generation",
            )
            for row in edges:
                edge = {
                    "source": row[0],
                    "target": row[1],
                    "metadata": {
                        "size": row[11],
                        "color": row[10],
                    },
                }
                ge.append(edge)
            return graph

        nodes, edges = self._to_graph(generation_range, only_main)
        return create_jgf(nodes, edges)

    def _to_graph(self, generation_range, only_main):
        """Convert the data to a graph object."""
        # Given data
        data = self.to_list()

        # Calculation or lookup in cache: only new information, not contained in given data
        def create_network(data):
            population_size_max = self.population_size_max()
            nodes = self._to_nodes(data, population_size_max)
            edges = self._to_edges(data, nodes)
            return nodes, edges

        nodes, edges = self._lookup_or_calc("_to_graph", create_network, data)

        # Combination: merge given and calculated/cached data
        nodes = [data[i] + nodes[i] for i in range(len(data))]

        # Optional filtering
        nodes, edges = self._filter_graph(
            data, nodes, edges, generation_range, only_main
        )
        return nodes, edges

    def _to_nodes(self, data, population_size_max):
        """Convert the data to nodes for a graph."""
        # Default values
        default_size = 3
        default_form = "o"
        default_color = "black"
        color_for_new_phenotypes = "#00CC00"
        x_offset = 0.0
        y_offset = population_size_max
        z_offset = 0.0
        x_stretch = 0.85

        # Node data generation
        nodes = []
        seen_phenotypes = set()
        for row in data:
            generation = row[2]
            label = row[3]
            phenotype = row[5]
            fitness = row[6]
            rank = row[10]
            # Coordinates
            x = generation
            y = rank
            z = fitness
            # Use default values
            size = default_size
            form = default_form
            color = default_color
            # Adapt them with specific information
            # - Color nodes which represent newly created phenotypes that undergo fitness evaluation
            if label == "main" or label == "mutation":
                if str(phenotype) not in seen_phenotypes:
                    color = color_for_new_phenotypes
            # - Node positions and sizes
            if label == "main":
                size = size * 2.0
                seen_phenotypes.add(str(phenotype))
            else:
                if label == "parent_selection":
                    x -= 2.0 / 4.0 + 1.0 / 4.0 * x_stretch - x_offset
                    y *= 0.5
                    y += y_offset
                    z += z_offset
                elif label == "crossover":
                    x -= 2.0 / 4.0 - x_offset
                    y *= 0.5
                    y += y_offset
                    z += z_offset
                elif label == "mutation":
                    x -= 2.0 / 4.0 - 1.0 / 4.0 * x_stretch - x_offset
                    y *= 0.5
                    y += y_offset
                    z += z_offset
            nodes.append((x, y, z, form, color, size))
        return nodes

    def _to_edges(self, data, nodes):
        """Convert the data to edges for a graph."""
        # Default values
        default_color = "black"
        default_width = 0.2
        default_style = "solid"

        # Edge data generation
        edges = []
        lookback_index = dict()
        for i in range(len(data)):
            # individual_id, parent_ids, generation, label, genotype, phenotype, fitness, details,
            # genotype_length, phenotype_length, rank
            row = data[i] + nodes[i]
            individual_id = row[0]
            individual_parent_ids = row[1]
            individual_generation = row[2]
            individual_label = row[3]
            individual_genotype = row[4]
            # x, y, z, form, color, size
            individual_x = row[11]
            individual_y = row[12]
            individual_z = row[13]
            individual_color = row[15]
            lookback_index[individual_id] = i
            for parent_id in individual_parent_ids:
                try:
                    parent_idx = lookback_index[parent_id]
                except KeyError:
                    continue
                parent_row = data[parent_idx] + nodes[parent_idx]
                parent_label = parent_row[3]
                parent_genotype = parent_row[4]
                parent_x = parent_row[11]
                parent_y = parent_row[12]
                parent_z = parent_row[13]
                # Use default values
                color = default_color
                width = default_width
                style = default_style
                # Adapt them with specific information
                # - Thicken identity relations from main population to main population
                if parent_label == "main" and individual_label == "main":
                    width *= 4
                # - Color inheritance relations that lead to offspring with other gt than parents
                if individual_label == "crossover" or individual_label == "mutation":
                    if individual_genotype != parent_genotype:
                        color = "#EE0000"
                        width *= 2
                # - Color identity relations for nodes with newly discovered phenotype
                if parent_label != "main" and individual_label == "main":
                    color = individual_color
                edge = (
                    parent_id,  # 0
                    individual_id,  # 1
                    individual_x,  # 2
                    parent_x,  # 3
                    individual_y,  # 4
                    parent_y,  # 5
                    individual_z,  # 6
                    parent_z,  # 7
                    individual_label,  # 8
                    style,  # 9
                    color,  # 10
                    width,  # 11
                    individual_generation,  # 12
                )
                edges.append(edge)
        return edges

    def _filter_list(self, data, generation_range, only_main):
        """Filter the data in list form."""
        # Argument processing
        filter_gen = generation_range not in (None, (None, None), [None, None])
        if filter_gen:
            first_gen, last_gen = self._process_generation_range(generation_range)

        # Quick check if any filter has to be applied
        if not data or (not only_main and not filter_gen):
            return data

        # Define the required filter
        if filter_gen and only_main:

            def keep(row):
                return row[3] == "main" and row[2] >= first_gen and row[2] <= last_gen

        elif filter_gen:

            def keep(row):
                return row[2] >= first_gen and row[2] <= last_gen

        else:

            def keep(row):
                return row[3] == "main"

        # Apply filter
        data = [row for row in data if keep(row)]
        return data

    def _filter_graph(self, data, nodes, edges, generation_range=None, only_main=False):
        """Filter the data in graph form."""
        # Argument processing
        filter_gen = generation_range not in (None, (None, None), [None, None])
        if filter_gen:
            first_gen, last_gen = self._process_generation_range(generation_range)

        # Quick check if any filter has to be applied
        if not data or (not only_main and not filter_gen):
            return nodes, edges

        # Define the required filters
        # - node filter
        if filter_gen and only_main:

            def keep_node(row, used_ind_ids):
                gn = row[2]
                if row[3] == "main" and gn >= first_gen and gn <= last_gen:
                    used_ind_ids.add(row[0])
                    return True
                return False

        elif filter_gen:

            def keep_node(row, used_ind_ids):
                gn = row[2]
                if (gn == first_gen and row[3] == "main") or (
                    gn > first_gen and gn <= last_gen
                ):
                    used_ind_ids.add(row[0])
                    return True
                return False

        else:

            def keep_node(row, used_ind_ids):
                if row[3] == "main":
                    used_ind_ids.add(row[0])
                    return True
                return False

        # - edge filter
        def keep_edge(row, used_ind_ids):
            return row[0] in used_ind_ids and row[1] in used_ind_ids

        # Apply filters
        used_ind_ids = set()
        nodes = [n for n in nodes if keep_node(n, used_ind_ids)]
        edges = [e for e in edges if keep_edge(e, used_ind_ids)]
        return nodes, edges

    def plot_genealogy(
        self, backend="vis", generation_range=None, only_main=False, **kwargs
    ):
        """Create a genealogy plot.

        It shows the relationships between all individuals created
        throughout a run.

        """
        # Argument processing
        backend = _ap.str_arg("backend", backend, vals=("d3", "vis", "three"))

        # Data preparation
        if "edge_curvature" not in kwargs:
            kwargs["edge_curvature"] = 0.0
        if "show_node_label" not in kwargs:
            kwargs["show_node_label"] = False
        graph = self.to_jgf(generation_range, only_main)

        # Plot
        fig = _plots.genealogy(graph, backend, **kwargs)
        return fig

    # Caching
    def _lookup_or_calc(self, key, calc_func, *calc_args):
        """Look up a result in the cache or calculate it.

        If it is not availabe yet in the cache, calculate it once and
        store it for later reuse.

        """
        num_changes = self._dbms.get_num_changes()
        try:
            # Cache lookup
            assert self._cache[key]["num_changes"] == num_changes
            result = self._cache[key]["data"]
            result = _copy.copy(
                result
            )  # prevent shallow modification of original, deep too slow
        except (AssertionError, KeyError):
            # Calculation
            result = calc_func(*calc_args)
            self._cache[key] = {"num_changes": num_changes, "data": result}
        return result


class Deserializer:
    """Convert database entries back to Python objects."""

    __slots__ = ("_system",)

    def __init__(self, system):
        """Create a deserializer that knows the chosen G3P system."""
        self._system = system

    def individual_id(self, data):
        """Get the id of an individual."""
        return int(data)

    def parent_ids(self, data):
        """Get the ids of parent individuals."""
        if data is None:
            return []
        return _json.loads(data)

    def genotype(self, data):
        """Reconstruct a genotype.

        This generates a Genotype object. Its exact type depends on the
        grammar-based genetic programming system being used.

        """
        return self._system.representation.Genotype(data)

    def genotypes(self, data):
        """Reconstruct a list of genotypes."""
        return [self.genotype(row[0]) for row in data]

    def phenotype(self, data):
        """Reconstruct a phenotypes."""
        if data is None:
            return ""
        return data

    def phenotypes(self, data):
        """Reconstruct a list of phenotypes."""
        return [self.phenotype(row[0]) for row in data]

    def fitness(self, data):
        """Reconstruct a fitness value."""
        if data is None:
            return float("nan")
        return data

    def fitnesses(self, data):
        """Reconstruct a list of fitness values."""
        return [self.fitness(row[0]) for row in data]

    def details(self, data):
        """Reconstruct a details object."""
        # None
        if data is None:
            return None
        # JSON
        try:
            return _json.loads(data)
        except Exception:
            pass
        # str
        return data

    def multiple_details(self, data):
        """Reconstruct a list of details objects."""
        return [self.details(row[0]) for row in data]

    def gt_phe_map(self, data):
        """Reconstruct genotype-to-phenotype mappings."""
        return [(self.genotype(row[0]), self.phenotype(row[1])) for row in data]

    def phe_fit_map(self, data):
        """Reconstruct phenotype-to-fitness mappings."""
        return [(self.phenotype(row[0]), self.fitness(row[1])) for row in data]

    def phe_fit_det_map(self, data):
        """Reconstruct phenotype-to-fitness-and-details mappings."""
        return [
            (self.phenotype(row[0]), self.fitness(row[1]), self.details(row[2]))
            for row in data
        ]

    def individual(self, data, without_parent_ids=False):
        """Reconstruct an individual.

        This generates an Individual object. Its exact type depends on
        the grammar-based genetic programming system being used.

        """
        # Split row into values
        ind_id, par_ids, gnr, lab, gt, phe, fit, det = data

        # Special case: do not load parent ids
        if without_parent_ids:
            par_ids = None

        # Deserialization: convert each string from the database to a suitable Python object
        ind_id = self.individual_id(ind_id)
        par_ids = self.parent_ids(par_ids)
        gt = self.genotype(gt)
        fit = self.fitness(fit)
        det = self.details(det)

        # Object creation
        ind = self._system.representation.Individual(
            genotype=gt,
            phenotype=phe,
            fitness=fit,
            details=dict(
                id=ind_id,
                parent_ids=par_ids,
                evaluation=det,
            ),
        )
        return ind

    def individuals(self, data):
        """Reconstruct a list of individuals."""
        return [self.individual(row, self._system) for row in data]

    def population(self, data):
        """Reconstruct a population.

        This generates a Population object. Its exact type depends on
        the grammar-based genetic programming system being used.

        """
        individuals = self.individuals(data)
        return self._system.representation.Population(individuals)
