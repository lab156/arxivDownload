import sqlalchemy as sa
import databases.create_db_define_models as cre
import process as pro

# Extract files with metada queried from a local
# database instead of using the arxiv database


def query_files(tar_file, subject, database):
    engine = sa.create_engine(database, echo=False)
    engine.connect()
    SMaker = sa.orm.sessionmaker(bind=engine)
    session = SMaker()
    q = sess.query(cre.Article.pk).join(cre.ManifestTarFile)\
            .filter(cre.ManifestTarFile.filename.like(tar_file))
    return list(q.all())

