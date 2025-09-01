import functools
from json import JSONDecodeError

import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
import os
from cross_db_benchmark.benchmark_tools.utils import load_json
from models.dataset.plan_dataset import PlanDataset
from models.dataset.plan_graph_batching.plan_batchers import plan_collator_dict

def read_workload_runs(logger, data_dir, workload_run_paths, limit_queries=None, limit_queries_affected_wl=None, mode='train', cross_datasets=False):
    # reads several workload runs
    plans = []
    database_statistics = dict()

    # if not isinstance(workload_run_paths, list):
    #     workload_run_paths = [workload_run_paths]
    for i, dataset in enumerate(workload_run_paths):
        try:
            if mode != 'test':
                if cross_datasets:
                    source = os.path.join(data_dir, dataset, 'zsce', 'parsed_plan.json')
                else:
                    source = os.path.join(data_dir, dataset, 'zsce', f'{mode}_plans.json')
            else:
                source = os.path.join(data_dir, dataset, 'zsce', 'test_plans.json')
            print(f"source {source}")
            run = load_json(source) # where SimpleNamespace is imported
        except JSONDecodeError:
            raise ValueError(f"Error reading {source}")
        database_statistics[i] = run.database_stats  # keys: column_stats, table_stats
        database_statistics[i].run_kwars = run.run_kwargs

        limit_per_ds = None
        if limit_queries is not None:
            if i >= len(workload_run_paths) - limit_queries_affected_wl:
                limit_per_ds = limit_queries // limit_queries_affected_wl
                logger.info(f"Capping workload {source} after {limit_per_ds} queries")

        for p_id, plan in enumerate(run.parsed_plans):
            plan.database_id = i
            plans.append(plan)
            if limit_per_ds is not None and p_id > limit_per_ds:
                logger.info("Stopping now")
                break

    # logger.info(f"No of Plans: {len(plans)}")

    return plans, database_statistics


def _inv_log1p(x):  # log1p(x) = log(x+1)
    return np.exp(x) - 1


def create_datasets(logger, data_dir, workload_run_paths, cap_training_samples=None, val_ratio=0.15, limit_queries=None,
                    limit_queries_affected_wl=None, shuffle_before_split=True, loss_class_name=None, mode='train', cross_datasets=False):
    plans, database_statistics = read_workload_runs(logger, data_dir, workload_run_paths, limit_queries=limit_queries,
                                                    limit_queries_affected_wl=limit_queries_affected_wl, mode=mode, cross_datasets=cross_datasets)

    no_plans = len(plans)
    plan_idxs = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_idxs)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_idxs = plan_idxs[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if cap_training_samples is not None:
        prev_train_length = len(train_idxs)
        train_idxs = train_idxs[:cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_idxs), 1) # an integer gte 1
        train_idxs = train_idxs * replicate_factor

    train_dataset = PlanDataset([plans[i] for i in train_idxs], train_idxs)

    val_dataset = None
    if val_ratio > 0:
        val_idxs = plan_idxs[split_train:]
        val_dataset = PlanDataset([plans[i] for i in val_idxs], val_idxs)

    # derive label normalization
    # runtimes = np.array([p.plan_runtime / 1000 for p in plans])
    runtimes = np.array([p.plan_runtime for p in plans])
    peakmems = np.array([p.peakmem for p in plans])
    # label_norm = derive_label_normalizer(loss_class_name, runtimes)
    mem_norm = derive_label_normalizer(loss_class_name, peakmems)
    time_norm = derive_label_normalizer(loss_class_name, runtimes)

    return mem_norm, time_norm, train_dataset, val_dataset, database_statistics


def derive_label_normalizer(loss_class_name, y): # y is `runtimes`
    if loss_class_name == 'MSELoss':
        log_transformer = preprocessing.FunctionTransformer(np.log1p, _inv_log1p, validate=True) # function, and inverse_function, log1p is more accurate for small positive inputs
        scale_transformer = preprocessing.MinMaxScaler()
        pipeline = Pipeline([("log", log_transformer), ("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))
    elif loss_class_name == 'QLoss':
        # scale_transformer = preprocessing.MinMaxScaler(feature_range=(1e-2, 1)) # this feature range is for runtime
        scale_transformer = preprocessing.MinMaxScaler() # this feature range is for peakmem
        pipeline = Pipeline([("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))
    else:
        pipeline = None
    return pipeline


def create_dataloader(logger, data_dir, train_workload_run_paths, val_workload_run_paths, test_workload_run_paths, combined_stats, plan_featurization_name, database,
                      val_ratio=0.15, batch_size=32, shuffle=True, num_workers=1, pin_memory=False,
                      limit_queries=None, limit_queries_affected_wl=None, loss_class_name=None):
    """
    Creates dataloaders that batches physical plans to train the model in a distributed fashion.
    :param workload_run_paths:
    :param val_ratio:
    :param test_ratio:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param pin_memory:
    :return:
    """

    # split plans into train/test/validation
    # label_norm, train_dataset, val_dataset, database_statistics = create_datasets(logger, workload_run_paths,  # wuy comment: training workloads as a whole, while we test on each test_workload separately
    #                                                                               loss_class_name=loss_class_name,
    #                                                                               val_ratio=val_ratio,
    #                                                                               limit_queries=limit_queries,
    #                                                                               limit_queries_affected_wl=limit_queries_affected_wl)
    if isinstance(train_workload_run_paths, str):
        train_workload_run_paths = [train_workload_run_paths]
    if isinstance(val_workload_run_paths, str):
        val_workload_run_paths = [val_workload_run_paths]
    if isinstance(test_workload_run_paths, str):
        test_workload_run_paths = [test_workload_run_paths]

    cross_datasets = len(train_workload_run_paths) > 1

    mem_norm, time_norm, train_dataset, _, database_statistics = create_datasets(logger, data_dir, train_workload_run_paths, loss_class_name= loss_class_name,
                                                                         val_ratio=0.0, shuffle_before_split=False, mode='train', cross_datasets=cross_datasets)
    
    _, _, val_dataset, _, val_database_statistics = create_datasets(logger, data_dir, val_workload_run_paths, loss_class_name= loss_class_name,
                                                                         val_ratio=0.0, shuffle_before_split=False, mode='val', cross_datasets=cross_datasets)

    # postgres_plan_collator does the heavy lifting of creating the graphs and extracting the features and thus requires both
    # database statistics but also feature statistics
    feature_statistics = combined_stats # load_json(statistics_file, namespace=False)

    plan_collator = plan_collator_dict[database]
    train_collate_fn = functools.partial(plan_collator, db_statistics=database_statistics,  # plan_collator here is in postgres_plan_batching.py
                                         feature_statistics=feature_statistics,
                                         plan_featurization_name=plan_featurization_name) # postgres_plan_collator(plans, feature_statistics=None, db_statistics=None, plan_featurization_name=None) in models/dataset/plan_graph_batching/postgres_plan_batching.py
    dataloader_args = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=train_collate_fn,
                           pin_memory=pin_memory)

    train_loader = DataLoader(train_dataset, **dataloader_args)
    val_loader = DataLoader(val_dataset, **dataloader_args)

    # for each test workoad run create a distinct test loader
    if test_workload_run_paths is not None:
        p = test_workload_run_paths
        _, _, test_dataset, _, test_database_statistics = create_datasets(logger, data_dir, p, loss_class_name=loss_class_name,
                                                                        val_ratio=0.0, shuffle_before_split=False, mode='test', cross_datasets=cross_datasets)
        # test dataset
        test_collate_fn = functools.partial(plan_collator, db_statistics=test_database_statistics,
                                            feature_statistics=feature_statistics,
                                            plan_featurization_name=plan_featurization_name)
        # previously shuffle=False but this resulted in bugs
        dataloader_args.update(collate_fn=test_collate_fn)
        test_loader = DataLoader(test_dataset, **dataloader_args)

    return mem_norm, time_norm, feature_statistics, train_loader, val_loader, [test_loader]
