from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import create_engine



DataBase_URL = 'postgresql://postgres:adminadmin@localhost/fastapi_3model'

engine = create_engine(DataBase_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()