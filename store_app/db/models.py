from .database import Base
from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped,mapped_column


class Mnist(Base):
    __tablename__ = 'mnist'

    id: Mapped[int] = mapped_column(Integer, autoincrement=True,primary_key=True)
    image: Mapped[str] = mapped_column(String,nullable=True)
    class_number: Mapped[int] = mapped_column(Integer)


class Fashion(Base):
    __tablename__ = 'fashion'

    id: Mapped[int] = mapped_column(Integer,autoincrement=True,primary_key=True)
    image: Mapped[str] = mapped_column(String)
    class_name: Mapped[str] = mapped_column(String)

class Cifar(Base):
    __tablename__ = 'cifar'

    id: Mapped[int] = mapped_column(Integer,autoincrement=True,primary_key=True)
    image: Mapped[str] = mapped_column(String)
    label_name: Mapped[str] = mapped_column(String)