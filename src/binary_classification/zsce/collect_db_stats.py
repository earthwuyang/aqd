import os
import sys
import psycopg2
from tqdm import tqdm

conn_params = {
    "dbname": "tpc_h",
    "user": "wuy",
    "password": "",
    "host": "localhost"
}
 



def get_result(sql, include_column_names=False, db_created=True, conn_params=conn_params):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.execute(sql)
    records = cur.fetchall()
    # print(f"records: {records}")

    if include_column_names:
        column_names = [desc[0] for desc in cur.description]
        return column_names, records
    
    return records

def transform_dicts(column_names, rows):
    result = []
    for row in rows:
        result.append(dict(zip(column_names, row)))
    return result

def collect_db_statistics(conn_params=conn_params):
    # column stats
    stats_query = """
        SELECT s.tablename, s.attname, s.null_frac, s.avg_width, s.n_distinct, s.correlation, c.data_type 
        FROM pg_stats s
        JOIN information_schema.columns c ON lower(s.tablename)=lower(c.table_name) AND s.attname=c.column_name
        WHERE s.schemaname='public';
    """
    column_stats_names, column_stats_rows = get_result(stats_query, include_column_names=True, conn_params=conn_params)
    column_stats = transform_dicts(column_stats_names, column_stats_rows)
    
    
    # table stats
    # table_stats_query = """
    #     SELECT relname, reltuples, relpages, relallvisible, reltoastrelid, relhasindex, relisshared, relpersistence, relkind, relnatts, relchecks, relhasoids, relhaspkey, relhasrules, relhastriggers, relhassubclass, relrowsecurity, reloptions
    #     FROM pg_class
    #     WHERE relkind IN ('r', 't', 'v', 'f', 'p') AND relnamespace=2200;
    # """
    table_stats_query = """
        SELECT relname, reltuples, relpages from pg_class WHERE relkind = 'r';
    """
    table_stats_names, table_stats_rows = get_result(table_stats_query, include_column_names=True, conn_params=conn_params)
    table_stats = transform_dicts(table_stats_names, table_stats_rows)

    return dict(column_stats=column_stats, table_stats=table_stats)
    
    # # index stats
    # index_stats_query = """         
    #     SELECT c.relname, i.relname, idx_scan, idx_tup_read, idx_tup_fetch, indisunique, indisprimary, indisclustered, indisvalid, indisready, indislive, indisreplident, indisversion, indisautovacuum, indisstats, indisfillfactor, indisconcurrent, indisclusterkey, indissparsity, indisunique_constraint, indisprimary_key, indislike_constraint, indisexclude_constraint, indisgenerated, indisfk, indis_oid, indis_constraint, indis_trigger, indis_rule, indis_tablespace
    #     FROM pg_index i
    #     JOIN pg_class c ON i.indrelid = c.oid
    #     JOIN pg_class t ON i.indrelid = t.oid
    #     JOIN pg_namespace n ON c.relnamespace = n.oid
    #     WHERE n.nspname = 'public' AND c.relkind IN ('r', 't', 'v', 'f', 'p');
    # """
    # index_stats_names, index_stats_rows = get_result(index_stats_query, include_column_names=True)
    # index_stats = transform_dicts(index_stats_names, index_stats_rows)
    


if __name__ == "__main__":
    collect_db_statistics()