"""
1. Read all JSON files in subdirectory of a given directory.
2. Aggregate these JSON files in one data structure
3. Save the resulting aggregated data structure in a new JSON file.
"""

import argparse
import fnmatch
import json
import os
import logging

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def read_json_files(directory, file_name_pattern):
    """
    Read all JSON files in subdirectory of a given directory.
    """
    json_files = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, file_name_pattern):
                json_files.append(os.path.join(root, file))
    return json_files


def aggregate_json_files(json_files):
    """
    Aggregate these JSON files in one data structure
    """
    aggregated_data = []
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            aggregated_data = [*aggregated_data, *data]
        except json.decoder.JSONDecodeError as e:
            log.error(
                f"Cannot parse file {json_file}. The error is as follows: {e}"
            )

    return aggregated_data


def read_json_file(json_file):
    """
    Read JSON file
    """
    data = None
    try:
        with open(json_file) as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError as e:
        log.error(
            f"Cannot parse file {json_file}. The error is as follows: {e}"
        )
    return data


def convert_to_tabular(
    aggregated_data: list[dict], starting_experiment_id: int = 0
) -> [pd.DataFrame, int]:
    """
    Convert aggregated JSON to unpivoted tabular format that
    R tidyverse can work with.

    :param aggregated_data: aggregated stats file
    :param starting_experiment_id: starting experiment id (useful when
           parsing multiple files)
    :return: unpivoted dataframe
    """
    # defining column types to preserve memory
    df_list = []
    current_experiment_id = -1
    for experiment_id, experiment in enumerate(aggregated_data):
        current_experiment_id = experiment_id + starting_experiment_id
        batch_method = experiment["batch_method"]
        batch_size = experiment["batch_size"]
        initializer = experiment["initializer"]
        initializer_type = experiment["initializer_type"]
        initializer_mode = experiment.get("initializer_mode", "unknown")
        compute_node = experiment.get("compute_node", "unknown")
        backend = experiment.get("backend", "unknown")
        dataset = experiment.get("dataset", "unknown")
        model = experiment.get("model", "unknown")
        optimizer = experiment.get("optimizer", "unknown")
        reinit = experiment.get("reinit", "unknown")
        learning_rate = experiment.get("learning_rate", -1)
        bias_initializer = experiment.get("bias_initializer", "unknown")
        recurrent_initializer = experiment.get(
            "recurrent_initializer", "unknown"
        )
        max_features = experiment.get("max_features", "unknown")
        units = experiment.get("units", "unknown")
        sequence_scheme = experiment.get("sequence_scheme", "unknown")
        starting_dim_id = experiment.get("starting_dim_id", -1)
        auto_seeds_count = experiment.get("auto_seeds_count", -1)
        auto_min_seed_value = experiment.get("auto_min_seed_value", -1)
        auto_max_seed_value = experiment.get("auto_max_seed_value", -1)
        auto_epoch_count = experiment.get("auto_epoch_count", -1)
        auto_repetition_count = experiment.get("auto_repetition_count", -1)
        embedding_dim = experiment.get("embedding_dim", -1)

        # append metrics
        for metric_name in experiment["summary_metrics"]:
            for epoch_ind, metric_value in enumerate(
                experiment["summary_metrics"][metric_name]
            ):
                df_list.append(
                    {
                        "experiment_id": current_experiment_id,
                        "batch_method": batch_method,
                        "batch_size": batch_size,
                        "initializer": initializer,
                        "initializer_type": initializer_type,
                        "initializer_mode": initializer_mode,
                        "compute_node": compute_node,
                        "backend": backend,
                        "sequence_scheme": sequence_scheme,
                        "starting_dim_id": starting_dim_id,
                        "auto_seeds_count": auto_seeds_count,
                        "auto_min_seed_value": auto_min_seed_value,
                        "auto_max_seed_value": auto_max_seed_value,
                        "auto_epoch_count": auto_epoch_count,
                        "auto_repetition_count": auto_repetition_count,
                        "dataset": dataset,
                        "model": model,
                        "optimizer": optimizer,
                        "reinit": reinit,
                        "learning_rate": learning_rate,
                        "bias_initializer": bias_initializer,
                        "recurrent_initializer": recurrent_initializer,
                        "max_features": max_features,
                        "units": units,
                        "embedding_dim": embedding_dim,
                        "epoch": epoch_ind + 1,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    }
                )
        # append time per epoch
        for epoch_ind, metric_value in enumerate(experiment["time_per_epoch"]):
            # append one row at a time
            df_list.append(
                {
                    "experiment_id": experiment_id,
                    "batch_method": batch_method,
                    "batch_size": batch_size,
                    "initializer": initializer,
                    "initializer_type": initializer_type,
                    "initializer_mode": initializer_mode,
                    "compute_node": compute_node,
                    "backend": backend,
                    "sequence_scheme": sequence_scheme,
                    "starting_dim_id": starting_dim_id,
                    "auto_seeds_count": auto_seeds_count,
                    "auto_min_seed_value": auto_min_seed_value,
                    "auto_max_seed_value": auto_max_seed_value,
                    "auto_epoch_count": auto_epoch_count,
                    "auto_repetition_count": auto_repetition_count,
                    "dataset": dataset,
                    "model": model,
                    "optimizer": optimizer,
                    "reinit": reinit,
                    "learning_rate": learning_rate,
                    "bias_initializer": bias_initializer,
                    "recurrent_initializer": recurrent_initializer,
                    "max_features": max_features,
                    "units": units,
                    "embedding_dim": embedding_dim,
                    "epoch": epoch_ind + 1,
                    "metric_name": "time_per_epoch",
                    "metric_value": metric_value,
                }
            )
    df = pd.DataFrame(data=df_list)
    return df, current_experiment_id + 1


def save_aggregated_data_to_parquet(
    aggregated_data: list[pd.DataFrame], output_file: str
) -> None:
    """
    Save the resulting aggregated data structure in a new csv.zip file.
    """
    append = True if os.path.exists(output_file) else False
    # Note that append is present in fastparquet engine
    # but not in pyarrow engine
    pd.concat(aggregated_data, ignore_index=True).to_parquet(
        output_file, engine="fastparquet", append=append
    )


def parse_args() -> tuple[str, str, str]:
    parser = argparse.ArgumentParser(description="Aggregate stats")
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Input directory containing files",
    )
    parser.add_argument(
        "-m",
        "--input_mask",
        required=True,
        help="Input file mask (e.g., results_*.json)",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Output file name",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    input_mask = args.input_mask
    output_file = args.output_file

    log.info(f"Input Directory: {input_dir}")
    log.info(f"Input File Mask: {input_mask}")
    log.info(f"Output File Name: {output_file}")

    return input_dir, input_mask, output_file


def main() -> None:
    """
    Main function
    """

    root_dir, input_file_mask, output_file_name = parse_args()
    output_file_name = os.path.join(root_dir, output_file_name)

    json_files = read_json_files(root_dir, input_file_mask)
    log.info(
        f"Found {len(json_files)} {input_file_mask} files "
        f"to process in {root_dir}"
    )

    log.info(f"Saving aggregated data to observations to {output_file_name}")

    starting_experiment_id = 0
    dfs_to_save = []  # List to hold dataframes temporarily
    total_rows = 0  # Track total rows to decide when to save
    count_of_rows_to_cache = 100000  # count fo rows to cache before saving

    for json_file in tqdm(json_files):
        data = read_json_file(json_file)
        if data is not None:
            df_to_save, starting_experiment_id = convert_to_tabular(
                data, starting_experiment_id=starting_experiment_id
            )
            # Add the dataframe to the list and update the total row count
            dfs_to_save.append(df_to_save)
            total_rows += len(df_to_save)

            # If the total number of rows exceeds 1000, save and reset
            if total_rows > count_of_rows_to_cache:
                save_aggregated_data_to_parquet(dfs_to_save, output_file_name)
                dfs_to_save = []  # Reset the list
                total_rows = 0  # Reset the row count

    # After looping through all files,
    # check if there are any remaining dataframes to save
    if dfs_to_save:
        save_aggregated_data_to_parquet(dfs_to_save, output_file_name)


if __name__ == "__main__":
    main()
