import json
import os
import sys
from enum import Enum
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import numpy as np
import collections
import argparse


def gather_feature_statistics(data_dir, dataset):
    def gather_values_recursively(json_dict, value_dict=None):
        if value_dict is None:
            value_dict = collections.defaultdict(list)

        if isinstance(json_dict, dict):
            for k, v in json_dict.items():
                if not (isinstance(v, list) or isinstance(v, tuple) or isinstance(v, dict)):
                    value_dict[k].append(v)
                elif (isinstance(v, list) or isinstance(v, tuple)) and len(v) > 0 and \
                        (isinstance(v[0], int) or isinstance(v[0], float) or isinstance(v[0], str)):
                    for v_e in v:
                        value_dict[k].append(v_e)
                else:
                    gather_values_recursively(v, value_dict=value_dict)
        elif isinstance(json_dict, tuple) or isinstance(json_dict, list):
            for e in json_dict:
                gather_values_recursively(e, value_dict=value_dict)

        return value_dict

    class FeatureType(Enum):
        numeric = 'numeric'
        categorical = 'categorical'

        def __str__(self):
            return self.value

    run_stats = []
    parsed_plan_file = os.path.join(data_dir, dataset, 'zsce/parsed_plan.json')
    if not os.path.exists(parsed_plan_file):
        print(f"Warning: {parsed_plan_file} does not exist. Skipping dataset '{dataset}'.")
        return {}, {}

    with open(parsed_plan_file, 'r') as f:
        try:
            run_stats.append(json.load(f))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for dataset '{dataset}': {e}")
            return {}, {}

    value_dict = gather_values_recursively(run_stats)

    statistics_dict = dict()
    raw_numeric_values = collections.defaultdict(list)  # To store raw values for numeric features

    for k, values in tqdm(value_dict.items(), desc=f"Processing {dataset}"):
        values = [v for v in values if v is not None]  # filter out None
        if len(values) == 0:
            continue
        # if value is numerical
        if all(isinstance(v, (int, float)) for v in values):
            raw_numeric_values[k].extend(values)  # Collect raw values
            # Temporarily store statistics; actual scale and center will be computed later
            statistics_dict[k] = dict(
                max=float(max(values)),
                type=str(FeatureType.numeric)
            )
        # if value is categorical
        else:
            unique_values = set(values)
            statistics_dict[k] = dict(
                value_dict={v: id for id, v in enumerate(unique_values)},
                no_vals=len(unique_values),
                type=str(FeatureType.categorical)
            )

    return statistics_dict, raw_numeric_values

def merge_categorical_features(existing, new):
    """Merge categorical feature statistics."""
    if existing['type'] != 'categorical' or new['type'] != 'categorical':
        raise ValueError("Feature types do not match for categorical merge.")

    # Combine the value dictionaries
    combined_values = set(existing['value_dict'].keys()).union(new['value_dict'].keys())
    # sorted_values = sorted(combined_values)  # Optional: sort for consistency
    sorted_values = combined_values  # Optional: sort for consistency

    # Assign new unique IDs
    value_dict = {v: idx for idx, v in enumerate(sorted_values)}

    merged = {
        'value_dict': value_dict,
        'no_vals': len(value_dict),
        'type': 'categorical'
    }
    return merged

def combine_statistics(combined_stats, new_stats, new_raw_numeric, combined_raw_numeric):
    """Combine new_stats and new_raw_numeric into combined_stats and combined_raw_numeric."""
    for feature, stat in new_stats.items():
        if stat['type'] == 'numeric':
            if feature not in combined_stats:
                # Initialize with the first occurrence
                combined_stats[feature] = {
                    'max': stat['max'],
                    'type': 'numeric'
                }
            else:
                # Update max
                combined_stats[feature]['max'] = max(combined_stats[feature]['max'], stat['max'])
            # Append raw values
            combined_raw_numeric[feature].extend(new_raw_numeric.get(feature, []))
        elif stat['type'] == 'categorical':
            if feature not in combined_stats:
                combined_stats[feature] = stat
            else:
                # Merge categorical features
                combined_stats[feature] = merge_categorical_features(combined_stats[feature], stat)
        else:
            raise ValueError(f"Unknown feature type: {stat['type']} for feature {feature}")

def compute_numeric_statistics(combined_stats, combined_raw_numeric):
    """Compute scale and center for numeric features using RobustScaler."""
    for feature, stats in tqdm(combined_stats.items(), desc="Computing numeric statistics"):
        if stats['type'] == 'numeric':
            raw_values = combined_raw_numeric.get(feature, [])
            if not raw_values:
                # Handle case with no values
                print(f"Warning: No raw values found for numeric feature '{feature}'. Assigning default scale and center.")
                stats['scale'] = 1.0
                stats['center'] = 0.0
                continue

            scaler = RobustScaler()
            np_values = np.array(raw_values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(np_values)

            stats['scale'] = scaler.scale_.item()
            stats['center'] = scaler.center_.item()
    return combined_stats


# def gather_feature_statistics(data_dir, dataset):
#     def gather_values_recursively(json_dict, value_dict=None):
#         if value_dict is None:
#             value_dict = collections.defaultdict(list)

#         if isinstance(json_dict, dict):
#             for k, v in json_dict.items():

#                 if not (isinstance(v, list) or isinstance(v, tuple) or isinstance(v, dict)):
#                     value_dict[k].append(v)
#                 elif (isinstance(v, list) or isinstance(v, tuple)) and len(v) > 0 and \
#                         (isinstance(v[0], int) or isinstance(v[0], float) or isinstance(v[0], str)):
#                     for v_e in v:
#                         value_dict[k].append(v_e)
#                 else:
#                     gather_values_recursively(v, value_dict=value_dict)
#         elif isinstance(json_dict, tuple) or isinstance(json_dict, list):
#             for e in json_dict:
#                 gather_values_recursively(e, value_dict=value_dict)

#         return value_dict


#     class FeatureType(Enum):
#         numeric = 'numeric'
#         categorical = 'categorical'

#         def __str__(self):
#             return self.value


#     run_stats = []
#     parsed_plan_file = os.path.join(data_dir, dataset, 'zsce', 'total_plans.json')
#     # parsed_plan_file = os.path.join(data_dir, dataset, 'zsce', 'tiny_plans.json')
#     with open(parsed_plan_file, 'r') as f:
#         run_stats.append(json.load(f))
#     value_dict = gather_values_recursively(run_stats)

#     statistics_dict = dict()
#     for k, values in tqdm(value_dict.items()):
#         values = [v for v in values if v is not None] # filter those are None
#         if len(values) == 0:
#             continue
#         # if value is numerical
#         if all([isinstance(v, int) or isinstance(v, float) or v is None for v in values]): # but previously v has been confirmed not be None?
#             scaler = RobustScaler()
#             np_values = np.array(values, dtype=np.float32).reshape(-1, 1)
#             scaler.fit(np_values)

#             statistics_dict[k] = dict(max=float(np_values.max()),
#                                         scale=scaler.scale_.item(),
#                                         center=scaler.center_.item(),
#                                         type=str(FeatureType.numeric))
#         # if value is categorical
#         else:
#             unique_values = set(values)
#             statistics_dict[k] = dict(value_dict={v: id for id, v in enumerate(unique_values)},
#                                         no_vals=len(unique_values),
#                                         type=str(FeatureType.categorical))

#     # save as json
#     target_json_file = os.path.join(data_dir, dataset, 'zsce', "statistics_workload_combined.json")
#     # target = 'tpch_data/statistics_workload_combined.json'
#     os.makedirs(os.path.dirname(target_json_file), exist_ok=True)
#     with open(target_json_file, 'w') as outfile:
#         json.dump(statistics_dict, outfile)


# if __name__ == '__main__':
#     data_dir = '/home/wuy/DB/pg_mem_data'
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--dataset', type=str, default=['tpch_sf1'], nargs='+', help='datasets to process')
#     args = argparser.parse_args()
#     for dataset in args.dataset:
#         gather_feature_statistics(data_dir, dataset)