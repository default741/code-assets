# Custom SQL Ops Asset to be Used for any Future Projects - v0.2

# Author: Abdemanaaf Ghadiali
# Copyright: Copyright 2022, DB_Ops, https://HowToBeBoring.com
# Version: 0.0.2
# Email: abdemanaaf.ghadiali.1998@gmail.com
# Status: Development
# Code Style: PEP8 Style Guide
# MyPy Status: NA (Not Tested)

import pandas as pd

import sqlparse
import warnings
import re

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from typing import Union


class InvalidTableName(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidSQLQuery(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidBaseORM(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidTableORM(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class EmptyDataframeDict(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ColumnValueMismatch(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DB_Table_Ops:
    """Database Quering Class with generic functions for MySQL, MSSQL and PostgreSQL.

    Methods:
        __init__: Class Initialization Method. (Creates Engine Connection.)
        is_valid_sql (staticmethod): Checks whether SQL Query is Valid or Not.
        table_exists: Checks whether a table exist or not.
        create_table_using_query: Creates new Table using a CREATE TABLE Query.
        create_table_using_orm: Creates new Table using ORM Classes.

    """

    def __init__(
        self, engine_type: str = 'mssql+pyodbc', driver: str = 'ODBC Driver 17 for SQL Server',
        host: str = 'localhost', port: str = '1433', database: str = 'db', username: str = None, password: str = None
    ) -> None:
        """Class Initialization Method for creating the SQL Connection Engine. Also Tests out the created Engine Connection.

        Args:
            engine_type (str, optional): Engine for Specific SQL Type. Defaults to 'mssql+pyodbc'.
            driver (str, optional): SQL Connection Driver. Defaults to 'ODBC Driver 17 for SQL Server'.
            host (str, optional): Server Name where the SQL Server is Hosted. Defaults to 'localhost'.
            port (str, optional): Port Number to SQL Connection. Defaults to '1433'.
            database (str, optional): Database Name. Defaults to 'db'.
            username (str, optional): Username. Defaults to None.
            password (str, optional): User Password. Defaults to None.
        """

        # Get the type of SQL Connection we are working with.
        self.database_type = engine_type.split('+')[0]
        self.database_name = database
        self.show_table_query = {
            'mysql': 'SHOW TABLES;',
            'mssql': 'SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = \'BASE TABLE\';',
            'postgresql': 'SELECT relname FROM pg_catalog.pg_class WHERE relkind = \'r\';',
            'sqlite': '',
            'oracle': ''
        }

        self.engine = None

        # When the Driver is not Provided.
        if driver is not None:
            self.engine = create_engine(URL.create(engine_type, username=username, password=password,
                                        host=host, port=port, database=database, query=dict(driver=driver)))

        else:
            self.engine = create_engine(URL.create(
                engine_type, username=username, password=password, host=host, port=port, database=database))

        print('Database Engine Connection String Created.')

        # Tests Database Connection.
        try:
            connection = self.engine.connect()
            print('Database Connection Established and Working.')

        except Exception as E:
            print('Error while connecting to database. {E}')

        finally:
            connection.close()

    @staticmethod
    def is_valid_sql(query: str, **kwargs) -> bool:
        """Checks whether the input SQL Query is Valid or Not. Uses SQLParse Library for Parsing SQL Queries.

        Args:
            query (str): SQL Query String

        Raises:
            sqlparse.exceptions.SQLParseError: If SQL Query is not Valid.

        Returns:
            bool: True if Valid SQL Query else False.
        """

        try:
            sqlparse.parse(query, **kwargs)
            return True

        except sqlparse.exceptions.SQLParseError:
            return False

    def table_exists(self, table_name: str = None) -> bool:
        """Check whether a Specified Table Exists in the Database or Not.

        Args:
            table_name (str, optional): Table Name to check its existance in the Database. Defaults to None.

        Raises:
            InvalidTableName: Table Name is either None or not a valid String object.

        Returns:
            bool: True if Table Exists in the Database else False.
        """

        if table_name is None or not isinstance(table_name, str):
            raise InvalidTableName(
                'Table Name is either None or not a valid String object.')

        with self.engine.connect() as conn:
            # Check if Table Exist in Database based on the Type of Database Engine is connect.
            cursor = conn.execute(
                text(self.show_table_query[self.database_type]))

            table_exists = [table[0]
                            for table in cursor if table_name == table[0]]

            return bool(table_exists)

    def show_table_list(self) -> list:
        """Shows a List of Tables in the Database. Also Returns the Same list.

        Returns:
            list: Table List.
        """

        with self.engine.connect() as conn:
            cursor = conn.execute(
                text(self.show_table_query[self.database_type]))
            table_list = [table[0] for table in cursor]

            print(f'Table List in Database ({self.database_name}): ')

            for idx, table in enumerate(table_list):
                print(f'{idx}. {table}')

            return table_list

    def create_table_using_query(self, create_schema_string: str = None) -> None:
        """Creates Table in Database using SQL Query. Need to Provide Schema Structure.

        Args:
            create_schema_string (str, optional): CREATE TABLE SQL Query Schema. Defaults to None.

        Raises:
            InvalidSQLQuery: SQL Schema String is either None or not a Valid String.
            InvalidSQLQuery: SQL Schema String is not a Valid SQL Query.
        """

        if create_schema_string is None or not isinstance(create_schema_string, str):
            raise InvalidSQLQuery(
                'SQL Schema String is either None or not a Valid String.')

        if not DB_Table_Ops.is_valid_sql(create_schema_string):
            raise InvalidSQLQuery(
                'SQL Schema String is not a Valid SQL Query.')

        table_name = create_schema_string.split()[2]

        if not self.table_exists(table_name):
            with self.engine.connect() as conn:
                conn.execute(text(create_schema_string))

        else:
            print(f'Table Already Exist in Database {self.database_name}!')

    def create_table_using_orm(self, base_object: object = None, table_object: Union[str, list] = None) -> None:
        """Creates Table using Object Relationship Manager from SQLAlchemy Library.

        Args:
            base_object (object, optional): Base Object from SQLAlchemy. Defaults to None.
            table_object (Union[str, list], optional): Table Class with columns Defined. Defaults to None.

        Raises:
            InvalidBaseORM: Base ORM Class is either None or Invalid.
            InvalidTableName: Table Name cannot be None.
            InvalidTableName: Table Name cannot be None.
            InvalidTableORM: Table Object is Invalid. Needs to be either str or list.
        """

        if base_object is None:
            raise InvalidBaseORM('Base ORM Class is either None or Invalid.')

        if isinstance(table_object, str):
            table_name = table_object.__tablename__

            if table_name is None:
                raise InvalidTableName('Table Name cannot be None.')

            if not self.table_exists(table_name):
                base_object.metadata.create_all(
                    self.engine, [table_object.__table__])
            else:
                print(f'Table Already Exist in Database {self.database_name}!')

        elif isinstance(table_object, list):
            for table in table_object:
                table_name = table.__tablename__

                if table_name is None:
                    raise InvalidTableName('Table Name cannot be None.')

                if not self.table_exists(table_name):
                    base_object.metadata.create_all(
                        self.engine, [table.__table__])
                else:
                    print(
                        'Table Already Exist in Database {self.database_name}!')

        else:
            raise InvalidTableORM(
                'Table Object is Invalid. Needs to be either str or list.')

    def delete_table(self, table_name: Union[str, list] = None) -> None:
        """Deletes Table from Database.

        Args:
            table_name (Union[str, list], optional): Table Name in Database. Defaults to None.

        Raises:
            InvalidTableName: Table Name (str or list) not Valid.
            InvalidSQLQuery: SQL Schema String is not a Valid SQL Query
            InvalidSQLQuery: SQL Schema String is not a Valid SQL Query.
            InvalidTableName: Table Name Format Mismatch.
        """

        if table_name is None or not isinstance(table_name, (str, list)):
            raise InvalidTableName('Table Name (str or list) not Valid.')

        if isinstance(table_name, str):
            if self.table_exists(table_name):
                sql_query = f'''DROP TABLE {table_name};'''

                if not DB_Table_Ops.is_valid_sql(sql_query):
                    raise InvalidSQLQuery(
                        'SQL Schema String is not a Valid SQL Query.')

                with self.engine.connect() as conn:
                    conn.execute(text(sql_query))

        elif isinstance(table_name, list):
            for table in table_name:
                if self.table_exists(table):
                    sql_query = f'''DROP TABLE {table};'''

                    if not DB_Table_Ops.is_valid_sql(sql_query):
                        raise InvalidSQLQuery(
                            'SQL Schema String is not a Valid SQL Query.')

                    with self.engine.connect() as conn:
                        conn.execute(text(sql_query))

        else:
            raise InvalidTableName('Table Name Format Mismatch.')

    def insert_dataframe_to_table(self, dataframe: Union[pd.DataFrame, dict], table_name: str = None) -> None:
        """Insert Pandas Dataframe to Database Table. Dataframe can be a dictionary, which will be converted to Dataframe.

        Args:
            dataframe (Union[pd.DataFrame, dict]): Data to be saved to Table.
            table_name (str, optional): Table Name in Database. Defaults to None.

        Raises:
            InvalidTableName: Table Name (str or list) not Valid.
            EmptyDataframeDict: Dataframe or Dictionary is Either empty or Invalid.
            EmptyDataframeDict: Dataframe or Dictionary is Either empty or Invalid.

        Warnings:
            1. Table does not exist in database. Creating a new Table.
        """

        if table_name is None or not isinstance(table_name, str):
            raise InvalidTableName('Table Name (str or list) not Valid.')

        if isinstance(dataframe, dict):
            if dataframe == {}:
                raise EmptyDataframeDict(
                    'Dataframe or Dictionary is Either empty or Invalid.')

            dataframe = pd.DataFrame(dataframe)

        if dataframe.empty:
            raise EmptyDataframeDict(
                'Dataframe or Dictionary is Either empty or Invalid.')

        if not self.table_exists(table_name=table_name):
            warnings.warn(
                'Table does not exist in database. Creating a new Table.')

        dataframe.to_sql(table_name, self.engine,
                         if_exists="append", index=False)

    def insert_row_to_table(self, table_name: str, column_list: list, value_list: list) -> None:
        """Insert Single Row to Table in Database.

        Args:
            table_name (str): Table Name in DataBase
            column_list (list): List of Column Names in the Table
            value_list (list): Values for each Column

        Raises:
            ColumnValueMismatch: Columns and Values should be a list.
            ColumnValueMismatch: Length of Column\'s List and Value\'s List should be Equal.
            InvalidSQLQuery: SQL Schema String is not a Valid SQL Query.
        """

        if not isinstance(column_list, list) and not isinstance(value_list, list):
            raise ColumnValueMismatch('Columns and Values should be a list.')

        if len(column_list) != len(value_list):
            raise ColumnValueMismatch(
                'Length of Column\'s List and Value\'s List should be Equal.')

        sql_query = f'''INSERT INTO {table_name} ({", ".join(column_list)})
                        VALUES ({", ".join(value_list)});'''

        if not DB_Table_Ops.is_valid_sql(sql_query):
            raise InvalidSQLQuery(
                'SQL Schema String is not a Valid SQL Query.')

        if self.table_exists(table_name):
            with self.engine.connect() as conn:
                conn.execute(text(sql_query))
        else:
            raise InvalidTableName(
                f'Table Does Not Exist in Database {self.database_name}!')

    def query_dataframe_from_table(self, query: str, index_col: Union[str, list] = None, return_dictionary: bool = False) -> Union[pd.DataFrame, dict]:
        """Reads Data into a Dataframe or Dictionary from Database Table.

        Args:
            query (str): SQL Query.
            index_col (Union[str, list], optional): Column to be specified as index. Defaults to None.
            return_dictionary (bool, optional): Whether to Return the data as dictionary. Defaults to False.

        Raises:
            InvalidSQLQuery: SQL Schema String is either None or not a Valid String.
            InvalidSQLQuery: SQL Schema String is not a Valid SQL Query.
            InvalidTableName: Table Does Not Exist in Database.

        Returns:
            Union[pd.DataFrame, dict]: Data as a Pandas Dataframe or Dictionary.
        """

        if query is None or not isinstance(query, str):
            raise InvalidSQLQuery(
                'SQL Schema String is either None or not Valid.')

        if not DB_Table_Ops.is_valid_sql(query=query):
            raise InvalidSQLQuery(
                'SQL Schema String is not a Valid SQL Query.')

        table_name = re.split(r'(FROM|from|From)', query)[
            2].split(' ')[1].split(';')[0]

        if not self.table_exists(table_name=table_name):
            raise InvalidTableName(
                f'Table Does Not Exist in Database {self.database_name}!')

        try:
            df = pd.read_sql_query(query, con=self.engine, index_col=index_col)
        except Exception as E:
            print(f'Error Reading Table: {E}')

        return df.to_dict() if return_dictionary else df

    def run_general_sql_commands(self, sql_query: str = None) -> list:
        """Execute General SQL Queries for one or many Table in Database.

        Args:
            sql_query (str): SQL Query String.

        Raises:
            InvalidSQLQuery: SQL Schema String is either None or not a Valid String.
            InvalidSQLQuery: SQL Schema String is not a Valid SQL Query.
            TypeError: Cursor is of Ambigous Nature.

        Returns:
            list: Output rows from the Query.
        """

        if sql_query is None or not isinstance(sql_query, str):
            raise InvalidSQLQuery(
                'SQL Schema String is either None or not Valid.')

        if not DB_Table_Ops.is_valid_sql(sql_query):
            raise InvalidSQLQuery(
                'SQL Schema String is not a Valid SQL Query.')

        with self.engine.connect() as conn:
            cursor = conn.execute(text(sql_query))

            try:
                return [list(t) for t in cursor]
            except:
                raise TypeError('Cursor is of Ambigous Nature.')


if __name__ == '__main__':

    dbops = DB_Table_Ops(engine_type='mysql+mysqlconnector', port=3306,
                         username='root', password='root', driver=None, database='mydb')

    dbops.create_table_using_query('''CREATE TABLE users_query_mysql (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL
    );''')

    # dbops2 = DB_Table_Ops(
    #     username='SA', password='Abde@1998', database='dev-test-db')

    # dbops2.create_table_using_query('''CREATE TABLE employees (
    #     employee_id INT PRIMARY KEY,
    #     first_name VARCHAR(50),
    #     last_name VARCHAR(50)
    #     );''')

    # dbops2 = DB_Table_Ops(engine_type='postgresql', port=5432,
    #                       username='postgres', password='root', driver=None, database='test-db')

    # dbops2.create_table_using_query('''CREATE TABLE users (
    # id serial PRIMARY KEY,
    # name VARCHAR(255) NOT NULL,
    # email VARCHAR(255) NOT NULL
    #     );''')
