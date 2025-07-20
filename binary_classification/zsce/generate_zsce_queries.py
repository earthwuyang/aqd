import os
import time
import multiprocessing as mp
import argparse
import sys
sys.path.append('../../zsce')

from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload

workload_defs = {
    'workload_100k_s1_group_order_by': dict(num_queries=100000,
                             max_no_predicates=5,
                             max_no_joins=5,
                             max_no_aggregates=3,
                             max_no_group_by=3,
                             max_cols_per_agg=2,
                             groupby_limit_prob=0.2,
                             groupby_having_prob=0.2,
                             seed=1),
    'workload_100k_s1_group_order_by_complex': dict(num_queries=100000,
                             max_no_predicates=5,
                             max_no_joins=5,
                             max_no_aggregates=3,
                             max_no_group_by=3,
                             max_cols_per_agg=2,
                             groupby_limit_prob=0.2,
                             groupby_having_prob=0.2,
                             complex_predicates = True,
                             seed=1),
    'workload_100k_s1_group_order_by_more_complex': dict(num_queries=100000,
                             max_no_predicates=20,
                             max_no_joins=10,
                             max_no_aggregates=3,
                             max_no_group_by=3,
                             max_cols_per_agg=2,
                             groupby_limit_prob=0.2,
                             groupby_having_prob=0.2,
                             complex_predicates = True,
                             seed=1)
}

def workload_gen(input):
    source_dataset, workload_path, workload_args, overwrite = input
    generate_workload(source_dataset, workload_path, force=overwrite, **workload_args)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload_dir', type=str, default=os.path.join(os.path.dirname(__file__), '/home/wuy/query_costs/workloads'), help='Directory to store generated workloads')
    parser.add_argument('--dataset', nargs='+', default=None, type=str, help='Dataset to generate workloads for')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing workloads')
    args = parser.parse_args()

    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    # from database_list import full_database_list

    if args.dataset is not None:
        full_database_list = args.dataset

    if not os.path.exists(args.workload_dir):
        os.makedirs(args.workload_dir)

    workload_gen_setups = []
    for dataset in full_database_list:
        for workload_name, workload_args in workload_defs.items():
            print(f"Generating workload {workload_name} for {dataset}")
            start_t = time.perf_counter()
            workload_path = os.path.join(args.workload_dir, dataset, f'{workload_name}.sql')
            # not using multiprocessing, use main process to generate workloads
            workload_gen((dataset, workload_path, workload_args, args.overwrite))
            print(f"Generated workload {workload_name} for {dataset} to {workload_path} with {workload_args}, in {time.perf_counter() - start_t:.2f} secs")
            # print(f"Generated workloads in {time.perf_counter() - start_t:.2f} secs")
            # workload_gen_setups.append((dataset, workload_path, 5, workload_args, args.overwrite))
            # print(f"Generating workload {workload_name} for {dataset} to {workload_path} with {workload_args}, max_no_joins=5")

    # start_t = time.perf_counter()
    # proc = mp.cpu_count() - 2
    # proc = 1
    # p = mp.Pool(initargs=('arg',), processes=proc)
    # p.map(workload_gen, workload_gen_setups)
    # print(f"Generated workloads in {time.perf_counter() - start_t:.2f} secs")