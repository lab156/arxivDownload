import sqlalchemy as sa
import databases.create_db_define_models as cre
import process as pro
import re

# Extract files with metada queried from a local
# database instead of using the arxiv database

def trim_id(id_url):
    '''
    id_url that looks like this:
        http://arxiv.org/abs/1601.00302v1
    output be should be
    '''
    resu = re.match(r'^.*abs/(.*)v\d', id_url)
    return resu.group(1)


def query_files(tar_file, subject, database):
    '''
    Queries database with a join for subject or tar_file

    Ex.
    tar_file: src/arXiv_src_1601_001.tar
    subject: math.AG
    database: arxiv1.db (no protocol sqlite:/// needed)
    '''
    db_name = 'sqlite:///' + database
    engine = sa.create_engine(db_name, echo=False)
    engine.connect()
    SMaker = sa.orm.sessionmaker(bind=engine)
    session = SMaker()
    q = session.query(cre.Article.id)\
            .filter(cre.Article.tags.like("[{'term': 'math.AG'%"))\
            .join(cre.ManifestTarFile)\
            .filter(cre.ManifestTarFile.filename.like(tar_file))
            #.order_by(cre.Article.published_parsed)
    return list(q.all())

if __name__ == '__main__':
    import sys
    tar_file, subject, database = sys.argv[1:]

    res = query_files(tar_file, subject, database)
    for a in res:
        print(trim_id(a[0]))
