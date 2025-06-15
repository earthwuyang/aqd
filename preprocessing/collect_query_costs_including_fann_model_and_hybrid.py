import os
import pymysql
from tqdm import tqdm
import time
import json
import logging
import numpy as np
import csv

import argparse
def execute_row(query, conn, timeout):
    cur = conn.cursor()
    cur.execute(f'set max_execution_time={timeout}')
    cur.execute('set use_imci_engine=off')
    logging.debug(f"Executing query: {query}")
    begin = time.time()
    cur.execute(query)
    cur.fetchall()
    query_time = time.time() - begin
    logging.debug(f"Query time: {query_time} seconds")
    cur.close()
    return query_time

def execute_column(query, conn, timeout):
    cur = conn.cursor()
    cur.execute(f'set max_execution_time={timeout}') 
    cur.execute('set use_imci_engine=forced')
    logging.debug(f"Executing query: {query}")
    begin = time.time()
    cur.execute(query)
    cur.fetchall()
    query_time = time.time() - begin
    logging.debug(f"Query time: {query_time} seconds")
    cur.close()
    return query_time

def write_row_plan(conn, row_plan_dir, query_id, query):
    cur = conn.cursor()
    sql = f"EXPLAIN FORMAT='json' {query}"
    cur.execute(sql)
    result = cur.fetchall()
    plan = json.loads(result[0][0])
    plan_file = os.path.join(row_plan_dir, f"{query_id}.json")
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=4)
    cur.close()

def write_column_plan(conn, column_plan_dir, query_id, query):
    cur = conn.cursor()
    cur.execute(f"set use_imci_engine=forced;")
    cur.execute(f"set imci_explain_print_row_cost   =on;")
    cur.execute(f"set imci_optimizer_switch = 'fast_opt_trivial_query=off';")

    sql = f"EXPLAIN FORMAT='json' {query}"
    cur.execute(sql)
    result = cur.fetchall()
    plan = json.loads(result[0][0])
    plan_file = os.path.join(column_plan_dir, f"{query_id}.json")
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=4)
    cur.close()

def get_hybrid_optimizer_decision(conn, query):
    cur = conn.cursor()
    cur.execute('set use_imci_engine=on;')
    cur.execute("set cost_threshold_for_imci=1;")
    cur.execute("set global imci_optimizer_switch='fast_opt_trivial_query=off';")
    cur.execute("set global imci_auto_update_statistic='SYNC';")
    cur.execute("set global hybrid_opt_compatible_transform_switch=4095;")
    cur.execute("set global hybrid_opt_keep_default_round_trace=on;")
    cur.execute("set global hybrid_opt_fetch_imci_stats_thread_enabled=on;")
    cur.execute("set hybrid_opt_dispatch_enabled=on;")
    cur.execute("set fann_model_routing_enabled=off;")

    explain_query = 'explain format=json ' + query
    try:
        cur.execute(explain_query)
        result = cur.fetchall()[0][0]
        if 'query_block' in result:
            hybrid_use_imci = 0
        else:
            hybrid_use_imci = 1
    except Exception as e:
        raise Exception(f"query {query} hybrid optimzier decision getting failed: {e}")

    cur.execute("set hybrid_opt_dispatch_enabled=off;")
    cur.execute('set use_imci_engine=forced;')
    cur.close()

    return hybrid_use_imci

def get_fann_model_decision(conn, query):
    cur = conn.cursor()
    cur.execute('set use_imci_engine=on;')
    cur.execute("set cost_threshold_for_imci=1;")
    cur.execute("set global imci_optimizer_switch='fast_opt_trivial_query=off';")
    cur.execute("set global imci_auto_update_statistic='SYNC';")
    cur.execute("set global hybrid_opt_compatible_transform_switch=4095;")
    cur.execute("set global hybrid_opt_keep_default_round_trace=on;")
    cur.execute("set global hybrid_opt_fetch_imci_stats_thread_enabled=on;")
    cur.execute("set hybrid_opt_dispatch_enabled=on;")
    cur.execute("set fann_model_routing_enabled=on;")

    explain_query = 'explain format=json ' + query
    try:
        cur.execute(explain_query)
        result = cur.fetchall()[0][0]
        if 'query_block' in result:
            hybrid_use_imci = 0
        else:
            hybrid_use_imci = 1
    except Exception as e:
        raise Exception(f"query {query} hybrid optimzier decision getting failed: {e}")

    cur.execute("set hybrid_opt_dispatch_enabled=off;")
    cur.execute('set use_imci_engine=forced;')
    cur.execute(f"set fann_model_routing_enabled=off;")
    cur.close()

    return hybrid_use_imci

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    argparser.add_argument('--dataset', type=str, default='tpch_sf1', help='dataset name')
    argparser.add_argument('--data_dir', type=str, default='/home/wuy/query_costs/', help='data directory')
    argparser.add_argument('--db', type=str, default='tpch_sf1', help='database name')
    argparser.add_argument('--port', type=int, default=44444, help='database port')
    argparser.add_argument('--timeout', type=int, default=60000, help='timeout in miliseconds, 1min')
    args = argparser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    data_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    csv_file = os.path.join(data_dir, 'query_costs.csv')
    row_plan_dir = os.path.join(data_dir, 'row_plans')
    column_plan_dir = os.path.join(data_dir, 'column_plans')
    if not os.path.exists(row_plan_dir):
        os.makedirs(row_plan_dir)
    if not os.path.exists(column_plan_dir):
        os.makedirs(column_plan_dir)
    timeout = args.timeout

    queries = []
    # queries_path = '/home/wuy/benchtpch/dbgen-src/queries_100k'
    # import os
    # for query in os.listdir(queries_path):
    #     if query.endswith('.sql'):
    #         with open(os.path.join(queries_path, query)) as f:
    #             queries.append(f.read())
    # AP_workload_file_path = os.path.join(args.data_dir, 'workloads', args.dataset, 'workload_100k_s1_group_order_by_more_complex.sql')
    AP_workload_file_path = os.path.join(args.data_dir, 'workloads', args.dataset, 'workload_100k_s1_group_order_by_more_complex.sql')
    with open(AP_workload_file_path) as f:
        queries.extend(f.readlines())
    TP_workload_file_path = os.path.join(args.data_dir, 'workloads', args.dataset, 'TP_queries.sql')
    with open(TP_workload_file_path) as f:
        queries.extend(f.readlines())
    
    # shuffle queries
    np.random.shuffle(queries)
    # queries=['SELECT COUNT(*) FROM REGION;', 'SELECT COUNT(*) FROM NATION;', 'SELECT COUNT(*) FROM PART;', 'SELECT COUNT(*) FROM SUPPLIER;', 'SELECT COUNT(*) FROM PARTSUPP;', 'SELECT COUNT(*) FROM CUSTOMER;', 'SELECT COUNT(*) FROM ORDERS;', 'SELECT COUNT(*) FROM LINEITEM;', 'SELECT COUNT(*) FROM ORDERS WHERE O_ORDERDATE >= DATE \'1993-07-01\';', 'SELECT COUNT(*) FROM ORDERS WHERE O_ORDERDATE >= DATE \'1993-07-01\' AND O_ORDERDATE < DATE \'1993-10-01\';', 'SELECT COUNT(*) FROM ORDERS WHERE O_ORDERDATE >= DATE \'1993-07-01\' AND O_ORDERDATE < DATE \'1993-10-01\' AND O_CLERK = \'Clerk#000000001\';']
    # queries = queries[:2]

    conn = pymysql.connect(host='127.0.0.1', user='root', port=args.port, db=args.db)

    if not os.path.exists(csv_file):
        new_file = True
    else:
        new_file = False
    f = open(csv_file, 'w', newline='')
    writer = csv.writer(f)
    if new_file:
        writer.writerow(['query_id', 'use_imci', 'row_time', 'column_time', 'hybrid_use_imci', 'fann_use_imci', 'query'])

    for i, query in tqdm(enumerate(queries), total=len(queries)):
        if query.startswith('#'):
            continue
        query = query.replace('"', '').strip()
        try:
            row_time = None
            column_time = None
            row_query = f"SELECT /*+ SET_VAR(USE_IMCI_ENGINE=OFF) */ {query.split('select' if 'select' in query else 'SELECT', 1)[1]}"
            try:
                row_time = execute_row(row_query, conn, timeout) # in seconds
            except Exception as e:
                logging.error(f"Error executing row query: {row_query}")
                logging.error(e)
            try:
                column_time = execute_column(query, conn, timeout) # in seconds
            except Exception as e:
                logging.error(f"Error executing column query: {query}")
                logging.error(e)

            write_row_plan(conn, row_plan_dir, i, row_query)
            write_column_plan(conn, column_plan_dir, i, query)
            if row_time is None and column_time is None:
                continue
            elif row_time is None and column_time is not None:
                use_imci = 1
            elif row_time is not None and column_time is not None:
                use_imci = 1 if column_time < row_time else 0;

            hybrid_use_imci = get_hybrid_optimizer_decision(conn, query)
            fann_use_imci = get_fann_model_decision(conn, query)

            writer.writerow([i, use_imci, row_time, column_time, hybrid_use_imci, fann_use_imci, query.replace('\n', ' ')])
            f.flush()
        except Exception as e:
            logging.error(f"Error for query: {query}, query_id: {i}")
            logging.error(e)
            continue
        

    conn.close()
    f.close()

if __name__ == '__main__':
    main()