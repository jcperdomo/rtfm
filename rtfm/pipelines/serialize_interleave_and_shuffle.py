"""
Serialize and interleave a set of datasets. This goes from a
set of individual parquet files, each representing a dataset,
to a set of parquet files where each element is a serialized observation
from one of those datasets. Those observations can be tokenized
and used for training directly, instead of needing to be serialized first.

Usage:
python scripts/serialize_interleave_and_shuffle.py \
    --input-dir "/path/or/wildcard/to/parquet/chunk-001[2-4]/" \
    --output-dir /path/to/output/ \
    --chunk_size 256 \
    --max_tables 100_000
"""
import glob
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Sequence, Optional

import pandas as pd
import ray
import webdataset as wds
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import HfArgumentParser

from rtfm.arguments import DataArguments
from rtfm.configs import SerializerConfig
from rtfm.data import (
    build_formatted_df_from_file,
    example_map_fn,
    NoTargetCandidatesError,
)
from rtfm.serialization.serializers import get_serializer


def write_file_list(files, output: str) -> None:
    print(f"writing list of {len(files)} files to {output}")
    with open(output, "w") as f:
        for i, file in enumerate(files):
            if i + 1 < len(files):
                f.write(os.path.abspath(file) + "\n")
            else:
                f.write(os.path.abspath(file))
    return


def process_file(
    row,
    data_args: DataArguments,
    serializer_config: SerializerConfig,
    model_max_len_tokens=4096,
    appx_chars_per_token=3.5,
):
    filename = row["item"]
    logging.warning(f"loading {filename}")

    try:
        df = build_formatted_df_from_file(
            filename,
            data_args=data_args,
        )
    except NoTargetCandidatesError:
        return {"item": "Failed"}
    except ValueError as ve:
        logging.error(ve)
        return {"item": "Failed"}
    except TypeError as te:
        logging.error(te)
        return {"item": "Failed"}

    records = df.to_dict(orient="records")

    serializer = get_serializer(serializer_config)
    _map_fn = partial(
        example_map_fn,
        data_args=data_args,
        serializer=serializer,
        cfg=None,
    )

    for record in records:
        mapped = _map_fn(record)
        # Do not keep examples that would not fit in the model's context
        if len(mapped["text"]) > int(model_max_len_tokens * appx_chars_per_token):
            logging.warning(
                f"dropping too-long sample with text len {len(mapped['text'])}"
            )
            continue
        # after applying map_fn, each element has fields: 'text', 'class_label_as_text'
        yield {**mapped, "filename": filename}


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


import webdataset as wds
import pandas as pd
import os
import json
import random
from sklearn.model_selection import train_test_split
import logging
from multiprocessing import Pool, Manager
from functools import partial


def process_single_file(
    parquet_file: str,
    shared_writers: dict,
    lock: Manager().Lock(),
    prefix: str,
    eval_split=0.1,
):
    def encode_row(ser: pd.Series):
        return json.dumps(ser.to_dict(), ensure_ascii=False).encode("utf-8")

    df = pd.read_parquet(parquet_file)
    base_filename = os.path.splitext(os.path.basename(parquet_file))[0]

    do_train_eval = prefix == "train" and random.uniform(0.0, 1.0) > 0.9

    if do_train_eval:
        try:
            train_df, eval_df = train_test_split(
                df, test_size=eval_split, random_state=42
            )
        except ValueError:
            logging.warning(f"Skipping train_eval split of df with size {len(df)}")
            train_df = df
            eval_df = None
    else:
        train_df = df
        eval_df = None

    with lock:
        for index, row in train_df.iterrows():
            key = f"{base_filename}__{index}"
            shared_writers["main"].write({"__key__": key, "json": encode_row(row)})

        if eval_df is not None:
            for index, row in eval_df.iterrows():
                key = f"{base_filename}__{index}"
                shared_writers["eval"].write({"__key__": key, "json": encode_row(row)})

    os.remove(parquet_file)


def parquet_to_wds(parquet_files, prefix: str, output_dir: str, target_size_mb=500):
    os.makedirs(output_dir, exist_ok=True)

    main_pattern = os.path.join(output_dir, f"{prefix}-%06d.tar")
    eval_pattern = os.path.join(output_dir, f"{prefix}eval-%06d.tar")

    manager = Manager()
    shared_writers = manager.dict()
    shared_writers["main"] = wds.ShardWriter(
        main_pattern, maxcount=None, maxsize=target_size_mb * 1024 * 1024
    )
    shared_writers["eval"] = wds.ShardWriter(
        eval_pattern, maxcount=None, maxsize=target_size_mb * 1024 * 1024
    )

    lock = manager.Lock()

    # Partial function to fix all arguments except the parquet_file
    process_func = partial(
        process_single_file, shared_writers=shared_writers, lock=lock, prefix=prefix
    )

    # Use all available CPUs
    with Pool() as pool:
        pool.map(process_func, parquet_files)

    # Close the writers
    shared_writers["main"].close()
    shared_writers["eval"].close()

    return shared_writers["main"].shard_names, shared_writers["eval"].shard_names


@dataclass
class PipelineConfig:
    input_dir: str
    output_dir: str
    max_tables: Optional[int] = None
    train_frac: float = 0.975
    split_random_seed = 42
    output_shard_factor: int = 1000
    output_file_prefix: Optional[str] = None
    chunk_size: int = 64
    target_size_mb: int = 500


def main(
    serializer_config: SerializerConfig,
    data_args: DataArguments,
    pipeline_config: PipelineConfig,
):
    data_args.use_config = False
    data_args.feature_name_handling = "none"
    data_args.feature_value_handling = "none"
    data_args.targets_handling = "none"

    if pipeline_config.max_tables:
        logging.warning(f"pipeline_config.max_tables is {pipeline_config.max_tables}")
    print(f"ray version is {ray.__version__}")
    start = time.time()
    files = glob.glob(os.path.join(pipeline_config.input_dir, "*.parquet"))
    print(f"got {len(files)} parquet files")

    if (
        pipeline_config.max_tables is not None
        and len(files) > pipeline_config.max_tables
    ):
        files = files[: pipeline_config.max_tables]
    print(f"files is {files}")
    ray.init(address="auto")

    num_nodes = len(ray.nodes())
    print(f"num nodes = {num_nodes}")
    num_cores = os.cpu_count()
    print(f"num cores = {num_cores}")
    parallelism = num_nodes * num_cores
    print(f"parallelism = {parallelism}")

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True
    ctx.max_errored_blocks = 1000

    # Fully shuffle the input files.
    train_files, split_files = train_test_split(
        files,
        test_size=1 - pipeline_config.train_frac,
        random_state=pipeline_config.split_random_seed,
    )
    train_ds = ray.data.from_items(train_files)
    test_ds = ray.data.from_items(split_files)

    # Repartition to balance the size of shards (smaller shards help avoid OOM),
    # and also to control the size of the output files (this keeps output files small, which helps
    # us shuffle them later).

    fn_kwargs = {
        "data_args": data_args,
        "serializer_config": serializer_config,
    }
    test_ds = test_ds.flat_map(process_file, fn_kwargs=fn_kwargs).repartition(
        parallelism * pipeline_config.output_shard_factor
    )
    train_ds = train_ds.flat_map(process_file, fn_kwargs=fn_kwargs).repartition(
        parallelism * pipeline_config.output_shard_factor
    )

    splits = ("train", "test")

    for ds, split in zip((train_ds, test_ds), splits):
        ds.write_parquet(
            f"local://{os.path.abspath(pipeline_config.output_dir)}/{split}"
        )

    ray.shutdown()

    print(
        f"finished ray pipeline in {time.time() - start} secs. Files are written to {pipeline_config.output_dir}"
    )

    for split in ("test", "train"):
        split_files = glob.glob(
            os.path.join(pipeline_config.output_dir, split, "*.parquet")
        )
        print(f"converting {len(split_files)} files to webdataset for split {split}.")
        prefix = (
            split
            if not pipeline_config.output_file_prefix
            else "-".join((pipeline_config.output_file_prefix, split))
        )
        parquet_to_wds(
            split_files,
            prefix=prefix,
            target_size_mb=pipeline_config.target_size_mb,
            output_dir=os.path.join(pipeline_config.output_dir, split),
        )

    write_file_list(
        glob.glob(os.path.join(pipeline_config.output_dir, "train", "train-*.tar")),
        os.path.join(pipeline_config.output_dir, "train", f"train-files.txt"),
    )
    write_file_list(
        glob.glob(os.path.join(pipeline_config.output_dir, "train", "traineval-*.tar")),
        os.path.join(pipeline_config.output_dir, split, f"traineval-files.txt"),
    )
    write_file_list(
        glob.glob(os.path.join(pipeline_config.output_dir, "test", "test-*.tar")),
        os.path.join(pipeline_config.output_dir, "test", f"test-files.txt"),
    )
    return


if __name__ == "__main__":
    parser = HfArgumentParser((SerializerConfig, DataArguments, PipelineConfig))
    (
        serializer_config,
        data_args,
        pipeline_config,
    ) = parser.parse_args_into_dataclasses()
    main(serializer_config, data_args, pipeline_config)
