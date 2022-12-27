from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

from db_conn_queries import DB_Table_Ops


Base = declarative_base()


class User(Base):
    __tablename__ = 'users_orm_mysql'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    email = Column(String(255))


class Employee(Base):
    __tablename__ = 'employee_orm_mysql'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    email = Column(String(255))


if __name__ == '__main__':
    dbops = DB_Table_Ops(engine_type='mysql+mysqlconnector', port=3306,
                         username='root', password='root', driver=None, database='mydb')

    dbops.create_table_using_orm(
        base_object=Base, table_object=User)
