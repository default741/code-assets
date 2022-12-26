# Custom SQL Ops Asset to be Used for any Future Projects - v0.2

# Author: Abdemanaaf Ghadiali
# Copyright: Copyright 2022, DB_Ops, https://HowToBeBoring.com
# Version: 0.0.2
# Email: abdemanaaf.ghadiali.1998@gmail.com
# Status: Development
# Code Style: PEP8 Style Guide
# MyPy Status: NA (Not Tested)

import pandas as pd
import sqlalchemy

from sqlalchemy.engine import URL


class DB_Table_Ops:

    def __init__(
        self, engine_type: str = 'mssql+pyodbc', driver: str = 'ODBC Driver 17 for SQL Server',
        host: str = 'localhost', port: str = '1433', database: str = 'db', username: str = None, password: str = None
    ) -> None:

        self.database_type = engine_type.split('+')[0]
        connection_url = None

        if driver is not None:
            connection_url = URL.create(
                engine_type, username=username, password=password, host=host, port=port,
                database=database, query=dict(
                    driver=driver)
            )

        else:
            connection_url = URL.create(
                engine_type, username=username, password=password, host=host, port=port,
                database=database
            )

        self.engine = sqlalchemy.create_engine(connection_url)

    def table_exists(self, table_name: str = None) -> bool:
        if table_name is None or not isinstance(table_name, str):
            raise ValueError('Table Name not Valid.')

        query_string = {
            'mysql': '''SHOW TABLES;''',
            'mssql': '''SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE';''',
            'postgresql': '''SELECT relname FROM pg_catalog.pg_class WHERE relkind = 'r';'''
        }

        with self.engine.connect() as conn:
            cursor = conn.execute(sqlalchemy.text(
                query_string[self.database_type]))

            table_exists = [table[0]
                            for table in cursor if table_name == table[0]]

            return bool(table_exists)

    def create_table(self, create_schema_string: str = None) -> None:
        if create_schema_string is None or not isinstance(create_schema_string, str):
            raise ValueError('Schema String Not Valid.')

        table_name = create_schema_string.split()[2]

        if not self.table_exists(table_name):
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text(create_schema_string))

        else:
            print('Table Already Exist!')

    def drop_table(self, table_name: str = None) -> None:
        if table_name is None or not isinstance(table_name, str):
            raise ValueError('Table Name not Valid.')

        if self.table_exists(table_name):
            sql_comm = f'''DROP TABLE {table_name}'''

            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text(sql_comm))

    def insert_df_to_table(self, dataframe: pd.DataFrame, table_name: str = None) -> None:
        if table_name is None or not isinstance(table_name, str):
            raise ValueError('Table Name not Valid.')

        if type(dataframe) is dict:
            if dataframe == {}:
                raise ValueError('Enpty Dictionary.')

            dataframe = pd.DataFrame(data=dataframe)

        if dataframe.empty:
            raise ValueError('DataFrame is Empty.')

        dataframe.to_sql(table_name, self.engine,
                         if_exists="append", index=False)

    def query_to_df(self, query: str, index_col: str = 'ident', return_dictionary: bool = False) -> pd.DataFrame:
        if query is None or not isinstance(query, str):
            raise ValueError('Schema String Not Valid.')

        with self.engine.connect() as conn:
            try:
                df = pd.read_sql_query(query, conn, index_col=index_col)
            except Exception as E:
                print(f'Error Reading Table: {E}')

            return df.to_dict() if return_dictionary else df

    def general_sql_command(self, sql_comm):
        with self.engine.connect() as conn:
            cursor = conn.execute(sqlalchemy.text(sql_comm))
            try:
                return [list(t) for t in cursor]
            except:
                pass


if __name__ == '__main__':

    dbops = DB_Table_Ops(engine_type='mysql+mysqlconnector', port=3306,
                         username='root', password='root', driver=None, database='mydb')

    dbops.create_table('''CREATE TABLE users1 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL
    );''')

    dbops2 = DB_Table_Ops(
        username='SA', password='Abde@1998', database='dev-test-db')

    dbops2.create_table('''CREATE TABLE employees (
        employee_id INT PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(50)
        );''')
