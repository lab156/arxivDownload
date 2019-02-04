import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, BigInteger, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
import time
import datetime as dt

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
        return self.filename.split('/')[1]

class Article(Base):
    __tablename__ = 'articles'
    pk = Column(Integer, primary_key=True)
    tarfile_id = Column(Integer, ForeignKey('manifest.id'), nullable=False)

    # This is from the arxiv stuff
    # This is a naming conflict because the id is the database id
    #  ex.  http://arxiv.org/abs/1601.00104v1
    id = Column(String(50), nullable=False)
    guidislink = Column(Boolean)
    updated_parsed = Column(DateTime)
    published_parsed = Column(DateTime)
    title = Column(String(300), nullable=False)
    summary = Column(String(5000), nullable=False)

    # ex. ['J. Ye', 'R. Gheissari', 'J. Machta', 'C. M. Newman', 'D. L. Stein']
    authors = Column(String(1000), nullable=False)

    # ex. {'name': 'D. L. Stein'}
    author_detail = Column(String(50), nullable=False)
    author = Column(String(50), nullable=False)

    # ex. LaTeX2e, 37 pages
    arxiv_comment = Column(String(100))

    #ex. tags  ::  [{'term': 'math.AG', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'math.KT', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}]
    tags = Column(String(500), nullable=False)

    #ex.  http://arxiv.org/abs/1601.00105v2
    arxiv_url = Column(String(1000), nullable=False)
    doi = Column(String(120))


def new_article_register(D, tarfile_id, session=None):
    '''
    Prepare the data that comes from an arxiv search
    to commit to the articles table
    '''
    # get all the attributes except pk
    attr_list = [a for a in Article.metadata.tables['articles'].columns.keys()\
            if a != 'pk']
    Time = lambda t: dt.datetime.fromtimestamp(time.mktime(t))
    D['tarfile_id'] = tarfile_id
    D['updated_parsed'] = Time(D['updated_parsed'])
    D['published_parsed'] = Time(D['published_parsed'])
    D['authors'] = repr(D['authors'])
    D['author_detail'] = repr(D['author_detail'])
    D['tags'] = repr(D['tags'])
    g = Article(**{k:D[k] for k in attr_list})
    return g




# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
#engine = create_engine('sqlite:///arxiv1.db')

# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
#Base.metadata.create_all(engine)
if __name__ == '__main__':
    db_name = sys.argv[1]
    engine = create_engine(db_name, echo=True)
    Base.metadata.create_all(engine)
