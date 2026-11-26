#----------------------------------------------------------------------------
# This Database class provides an interface to the database
# It's therefore easier to simply inherit the code..
# Created by Brad Nielsen 2019
#-----------------------------------------------------------------------#
import sqlite3
import os, sys
from contextlib import closing

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from loggerinterface import get_logger

class Database:

    def __init__(self, location=""):
        """Initializes the database connection parameters. Errors are logged to logs/database.log."""
        self.location = location
        self.logger = get_logger('Database')
        if not os.path.exists(location):
            self.logger.warning(f"Database file {location} not found - sqlite will create an empty one")

    def connect(self):
        """Returns a configured handle to the Database connection."""
        connection = sqlite3.connect(self.location)
        connection.row_factory = sqlite3.Row 
        
        # Enable WAL mode for simultaneous reads/writes
        connection.execute("PRAGMA journal_mode=WAL;") 
        
        return connection

    def ViewQuery(self, query, params=()):
        """
        Executes a SELECT query and returns the results.
        Returns an empty list [] if no results are found or if an error occurs.
        """
        result = []
        try:
            # closing() guarantees connection.close() is called when the block ends
            with closing(self.connect()) as connection:
                cursor = connection.execute(query, params)
                records = cursor.fetchall()
                if records:
                    result = [dict(row) for row in records]
            self.logger.debug(f"{query} | {params} -> {len(result)} rows")

        except sqlite3.Error as e:
            self.logger.error(f"{type(e).__name__}: {e} | {query} | {params}")
            
        return result

    def ModifyQuery(self, query, params=()):
        """
        Executes an INSERT, UPDATE, or DELETE query.
        Returns True on success, False on failure.
        """
        success = False
        try:
            with closing(self.connect()) as connection:
                # 'with connection:' automatically handles .commit() on success 
                # and .rollback() if an exception is thrown
                with connection:
                    cursor = connection.execute(query, params)
                success = True
            self.logger.info(f"{query} | {params} -> {cursor.rowcount} rows changed")

        except sqlite3.Error as e:
            self.logger.error(f"{type(e).__name__}: {e} | {query} | {params}")
            
        return success

    def log(self, message):
        self.logger.info(message)

    def log_error(self, error):
        self.logger.error(error)