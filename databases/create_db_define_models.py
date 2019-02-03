import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

#Using the Tutorial
#https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/
 
Base = declarative_base()
 
class ManifestTarFile(Base):
    __tablename__ = 'manifest'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True, nullable=False)
    # ex. src/arXiv_src_1804_001.tar
    filename = Column(String(30), nullable=False)
    # ex. cacbfede21d5dfef26f367ec99384546
    content_md5sum = Column(String(35), nullable=False)
    md5sum = Column(String(35), nullable=False)
    # ex. astro-ph0001001 or 1804.01252
    first_item = Column(String(20), nullable=False)
    last_item = Column(String(20), nullable=False)

    num_items = Column(Integer, nullable=False)
    seq_num = Column(Integer, nullable=False)
    size = Column(BigInteger, nullable=False)

    timestamp = Column(String(25), nullable=False)
    yymm = Column(String(10), nullable=False)

    def __repr__(self):
        return filename.split('/')[1]


 
# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
#engine = create_engine('sqlite:///arxiv1.db')

# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
#Base.metadata.create_all(engine)
