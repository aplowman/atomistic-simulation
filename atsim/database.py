
import pymysql.cursors


def connect_db(config):
    """Connect to the database."""
    connection = pymysql.connect(**config, charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection
