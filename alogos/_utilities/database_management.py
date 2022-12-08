import csv as _csv
import os as _os
import sqlite3 as _sqlite3


class Sqlite3Wrapper:
    """Simple wrapper for easier usage of an SQLite3 database in Python3.

    References
    ----------
    - https://docs.python.org/3/library/sqlite3.html
    - https://www.sqlite.org
    - https://www.sqlite.org/limits.html

    """

    def __init__(self, database_location, try_to_use_wal=True):
        """Open a connection to an existing or new SQLite3 database.

        Parameters
        ----------
        database_location : string
            If the given string is ``":memory:"``, an in-memory database is created.
            Otherwise, a file-based database is created and the given string is
            used as filepath.
        try_to_use_wal : bool
            If True, it is attempted to use a Write-Ahead Log (WAL) instead of
            the default rollback journal, which can bring a significant speedup.
            This is supported since SQLite version 3.7.0 (2010-07-21) if the
            operating system supports it.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.connect

            - You can use ":memory:" to open a database connection to a
              database that resides in RAM instead of on disk.

        - https://www.sqlite.org/wal.html

            - On success, the pragma will return the string "wal"

        - https://www.sqlite.org/pragma.html#pragma_synchronous

            - The synchronous=NORMAL setting is a good choice for most
              applications running in WAL mode.

        """
        self.connection = _sqlite3.connect(database_location)

        #
        if try_to_use_wal:
            try:
                result = self.execute_query("PRAGMA journal_mode=WAL;")
                if result[0][0] == "wal":
                    self.execute_query("PRAGMA synchronous=NORMAL;")
            except Exception:
                pass

    def __del__(self):
        """Close the connection to the database.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.close

        """
        self.connection.close()

    def execute_query(self, query, record=None):
        """Execute an SQL statement, optionally with parameters.

        Parameters
        ----------
        query : str
            String with SQL query
        record : list or tuple
            The record needs to have one entry for each ``?`` placeholder in the query.

        Returns
        -------
        result : list of list of str
            The list of rows and columns returned by the cursor with ``fetchall``.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.execute
        - https://docs.python.org/3/library/sqlite3.html#using-the-connection-as-a-context-manager

        """
        with self.connection as con:
            if record is None:
                result = con.execute(query).fetchall()
            else:
                result = con.execute(query, record).fetchall()
        return result

    def execute_query_for_many_records(self, query, records):
        """Execute an SQL statement for many records.

        This method can only be used for data insertion (INSERT)
        and not for data retrieval (SELECT) because nothing is returned
        by the called ``executemany`` method.

        Parameters
        ----------
        query : str
            String with SQL query
        records : list of tuples
            The list can have an arbitrary number of tuples, each representing a record.
            The tuple needs to have one entry for each ``?`` placeholder in the query.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.executemany
        - https://docs.python.org/3/library/sqlite3.html#using-the-connection-as-a-context-manager

        """
        with self.connection as con:
            con.executemany(query, records)

    def execute_script(self, query_script):
        """Execute a script that can contain multiple SQL statements with a single transaction.

        This method can only be used for data insertion (INSERT)
        and not for data retrieval (SELECT) because nothing is returned
        by the called ``executescript`` method.

        Parameters
        ----------
        query_script : str
            String with multiple SQL queries separated by newlines.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.executescript
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.executescript
        - https://docs.python.org/3/library/sqlite3.html#using-the-connection-as-a-context-manager

        """
        with self.connection as con:
            con.executescript(query_script)

    def get_version(self):
        """Get the version of the running SQLite library.

        Returns
        -------
        version_descriptor : str

        References
        ----------
        - https://sqlite.org/lang_corefunc.html#sqlite_version

        """
        result = self.execute_query("SELECT SQLITE_VERSION();")
        version = result[0][0]
        return "SQLite v{}".format(version)

    def get_table_names(self):
        """Get a list of tables contained in the database.

        Returns
        -------
        table_names : list of str

        References
        ----------
        - https://sqlite.org/fileformat.html#storage_of_the_sql_database_schema

        """
        result = self.execute_query(
            'SELECT name FROM sqlite_master WHERE type="table";'
        )
        table_names = [row[0] for row in result]
        return table_names

    def get_view_names(self):
        """Get a list of views contained in the database.

        Returns
        -------
        view_names : list of str

        References
        ----------
        - https://sqlite.org/fileformat.html#storage_of_the_sql_database_schema

        """
        result = self.execute_query('SELECT name FROM sqlite_master WHERE type="view";')
        view_names = [row[0] for row in result]
        return view_names

    def get_header_names(self, name):
        """Get the headers of a view or table.

        Parameters
        ----------
        name : string
            Name of a table or view.

        Returns
        -------
        header_names : list of str
            Header row of the chosen table or view.

        References
        ----------
        - https://stackoverflow.com/questions/947215/how-to-get-a-list-of-column-names-on-sqlite3-database

        """
        result = self.execute_query("SELECT name FROM PRAGMA_TABLE_INFO(?);", [name])
        header_names = [row[0] for row in result]
        return header_names

    def get_data_per_table(self):
        """Get data of each table in the database in form of a dictionary.

        Returns
        -------
        data : dict
            Each table name is a key in the dict.
            A table's entries form the corresponding value (list of list of str).

        """
        tables = self.get_table_names()
        data = dict()
        for table in tables:
            data[table] = self.execute_query("SELECT * FROM {};".format(table))
        return data

    def get_num_changes(self):
        """Get number of rows which have been modified, inserted, deleted since connection opened.

        Returns
        -------
        num_changes : int

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.total_changes

        """
        return self.connection.total_changes

    def export_sql(self, filepath):
        """Export the entire database to a SQL file at a given filepath.

        Parameters
        ----------
        filepath : str

        Raises
        ------
        FileExistsError
            If there is already a file or directory at the given filepath.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup

        """
        # Argument processing
        self._check_filepath(filepath)

        # Export
        new_con = _sqlite3.connect(filepath)
        with new_con:
            self.connection.backup(new_con)
        new_con.close()

    def export_sql_text(self, filepath):
        """Export the entire database as text file with SQL statements at a given filepath.

        Parameters
        ----------
        filepath : str

        Raises
        ------
        FileExistsError
            If there is already a file or directory at the given filepath.

        References
        ----------
        - https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.iterdump

        """
        # Argument processing
        self._check_filepath(filepath)

        # Export
        with open(filepath, "w", encoding="utf-8") as file_handle:
            for line in self.connection.iterdump():
                file_handle.write("{}\n".format(line))

    def export_csv(self, filepath, name=None, include_header=True):
        """Export a chosen table or view of the database to a CSV file at a given filepath.

        Parameters
        ----------
        filepath : str
        name : str
            Name of a table or view that shall be exported.
            If `None`, each table of the database will be exported to a
            separate file. The filepath is constructed from the original
            filepath followed by the table name.

        Raises
        ------
        FileExistsError
            If there is already a file or directory at the given filepath.

        """
        if name is None:
            # Recursive call for each table name, if no name was provided by the user
            names = self.get_table_names()
            if filepath.endswith(".csv"):
                filepath = filepath[:-4]
            filepaths = ["{}_{}.csv".format(filepath, name) for name in names]
            for fp in filepaths:
                self._check_filepath(fp)
            for fp, name in zip(filepaths, names):
                self.export_csv(fp, name=name)
        else:
            # Argument processing
            self._check_filepath(filepath)
            table_names = self.get_table_names()
            view_names = self.get_view_names()
            if name not in table_names and name not in view_names:
                raise ValueError("The provided name is not a known table or view.")

            # Export
            with open(filepath, "w", encoding="utf-8") as csv_file:
                csv_writer = _csv.writer(csv_file, delimiter=",")
                # Header
                if include_header:
                    header = self.get_header_names(name)
                    csv_writer.writerow(header)
                # Rows
                rows = self.execute_query("select * from {};".format(name))
                csv_writer.writerows(rows)

    @staticmethod
    def _check_filepath(filepath):
        """Check if the given filepath exists.

        Raises
        ------
        FileExistsError
            If the given path is already taken by another file or directory.

        """
        if _os.path.exists(filepath):
            object_at_filepath = "file" if _os.path.isfile(filepath) else "directory"
            message = (
                'Cannot export data to the filepath "{}". '
                "There already is a {} with that path.".format(
                    filepath, object_at_filepath
                )
            )
            raise FileExistsError(message)
