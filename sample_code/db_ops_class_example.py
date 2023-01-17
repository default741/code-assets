from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

from src.db_conn_queries import DB_Table_Ops

import pandas as pd


Base = declarative_base()


class User(Base):
    __tablename__ = 'users_orm_postgresql'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    email = Column(String(255))


class Employee(Base):
    __tablename__ = 'employee_orm_mysql'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    email = Column(String(255))


def test_postgresql_database():
    dbops_postgresql = DB_Table_Ops(engine_type='postgresql', port=5432,
                                    username='postgres', password='root', driver=None, database='test-db')

    try:
        dbops_postgresql.delete_table('customers')
        dbops_postgresql.delete_table(User.__tablename__)
    except:
        print('No Tables Deleted with specified name')

    dbops_postgresql.create_table_using_query('''
        CREATE TABLE customers (
            id serial PRIMARY KEY,
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            email VARCHAR(255),
            phone VARCHAR(255),
            city VARCHAR(255)
        );
    ''')

    dbops_postgresql.create_table_using_orm(Base, User)

    dbops_postgresql.show_table_list()
    dbops_postgresql.insert_row_to_table('customers', ['id', 'first_name', 'last_name', 'email', 'phone', 'city'], [
                                         '1', 'Abdemanaaf', 'Ghadiali', 'abdemanaafzghadiali@gmail.com', '9619268524', 'Mumbai'])

    df = pd.DataFrame(columns=['id', 'first_name', 'last_name', 'email', 'phone', 'city'], data=[[
        '2', 'Abdemanaaf', 'Ghadiali', 'abdemanaafzghadiali@gmail.com', '9619268524', 'Mumbai'], [
        '3', 'Mariya', 'Ghadiali', 'mariyazghadiali@gmail.com', '9619268524', 'Mumbai'], [
        '4', 'Zohair', 'Ghadiali', 'zohairghadiali@gmail.com', '9619268524', 'Mumbai']])

    dbops_postgresql.insert_dataframe_to_table('customers', df)

    print(dbops_postgresql.query_dataframe_from_table(
        'SELECT * from customers;', return_dictionary=True))

    print(dbops_postgresql.run_general_sql_commands('SELECT * from customers;'))

    dbops_postgresql.delete_table(User.__tablename__)
    dbops_postgresql.delete_table('customers')


def test_mysql_database():
    dbops_mysql = DB_Table_Ops(engine_type='mysql+mysqlconnector', port=3306,
                               username='root', password='root', driver=None, database='mydb')

    try:
        dbops_mysql.delete_table('customers')
        dbops_mysql.delete_table(User.__tablename__)
    except:
        print('No Tables Deleted with specified name')

    dbops_mysql.create_table_using_query('''
        CREATE TABLE customers (
            id INT PRIMARY KEY,
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            email VARCHAR(255),
            phone VARCHAR(255),
            city VARCHAR(255)
        );
    ''')

    dbops_mysql.create_table_using_orm(Base, User)

    dbops_mysql.show_table_list()
    dbops_mysql.insert_row_to_table('customers', ['id', 'first_name', 'last_name', 'email', 'phone', 'city'], [
        '1', 'Abdemanaaf', 'Ghadiali', 'abdemanaafzghadiali@gmail.com', '9619268524', 'Mumbai'])

    df = pd.DataFrame(columns=['id', 'first_name', 'last_name', 'email', 'phone', 'city'], data=[[
        '2', 'Abdemanaaf', 'Ghadiali', 'abdemanaafzghadiali@gmail.com', '9619268524', 'Mumbai'], [
        '3', 'Mariya', 'Ghadiali', 'mariyazghadiali@gmail.com', '9619268524', 'Mumbai'], [
        '4', 'Zohair', 'Ghadiali', 'zohairghadiali@gmail.com', '9619268524', 'Mumbai']])

    dbops_mysql.insert_dataframe_to_table('customers', df)

    print(dbops_mysql.query_dataframe_from_table(
        'SELECT * from customers;', return_dictionary=True))

    print(dbops_mysql.run_general_sql_commands('SELECT * from customers;'))

    dbops_mysql.delete_table(User.__tablename__)
    dbops_mysql.delete_table('customers')


if __name__ == '__main__':

    test_postgresql_database()
    print()
    test_mysql_database()

    # dbops.create_table_using_query('''CREATE TABLE users_query_mysql (
    # id INT AUTO_INCREMENT PRIMARY KEY,
    # username VARCHAR(255) NOT NULL,
    # email VARCHAR(255) NOT NULL
    # );''')

    # dbops2 = DB_Table_Ops(
    #     username='SA', password='Abde@1998', database='dev-test-db')

    # dbops2.create_table_using_query('''CREATE TABLE employees (
    #     employee_id INT PRIMARY KEY,
    #     first_name VARCHAR(50),
    #     last_name VARCHAR(50)
    #     );''')

    # dbops = DB_Table_Ops(engine_type='mysql+mysqlconnector', port=3306,
    #                      username='root', password='root', driver=None, database='mydb')

    # dbops.create_table_using_orm(
    #     base_object=Base, table_object=User)
