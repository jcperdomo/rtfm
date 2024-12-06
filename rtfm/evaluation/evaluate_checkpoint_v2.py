"""
Evaluate a language model for tabular data prediction.
"""
import logging
import os
from typing import Dict, Optional

import pandas as pd
import transformers
from llama_recipes.inference.model_utils import load_model
from transformers import AutoTokenizer

from rtfm.arguments import (
    DataArguments,
)
from rtfm.configs import TrainConfig, TokenizerConfig, SerializerConfig
# from rtfm.evaluation.evaluation_utils import (
#     prepare_eval_kwargs,
#     prepare_eval_datasets,
# )
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import sanity_check_tokenizer, prepare_tokenizer
from rtfm.train_utils import load_model_from_checkpoint
from rtfm.utils import get_task_names_list, initialize_dir, get_latest_checkpoint

#juanky
from rtfm.datasets import get_task_dataset
from rtfm.datasets.data_utils import make_object_json_serializable


from rtfm.datasets.tableshift_utils import (
    fetch_preprocessor_config_from_data_args
)

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=LOG_LEVEL,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from datasets import disable_caching

logging.warning("disabling caching for hf datasets!")
disable_caching()

transformers.logging.set_verbosity_info()


def main(
    data_arguments: DataArguments,
    train_config: TrainConfig,
    tokenizer_config: TokenizerConfig,
    serializer_config: SerializerConfig,
    outfile: str,
    split: str,
    shot_selector: str,
    eval_task_names: Optional[str] = None,
    eval_task_file: Optional[str] = None,
    use_fast_kernels: bool = False,
    overwrite: bool = False,
):
    if os.path.exists(outfile) and not overwrite:
        logging.warning(f"file {outfile} already exists; skipping evaluation.")
        return
    assert not (
        eval_task_names and eval_task_file
    ), "specify either eval_task_names or eval_task_file, not both."

    assert (
        not data_arguments.pack_samples
    ), "are you sure you want to use packing for evals?!"

    assert outfile.endswith(".csv"), "output file must end with .csv"
    initialize_dir(os.path.dirname(outfile))

    eval_task_names = get_task_names_list(
        eval_task_names if eval_task_names else eval_task_file
    )

    assert len(eval_task_names) == 1 or (not data_arguments.tag), (
        f"it is recommended to use this script for only a single task to ensure"
        f"tags and context shift are applied correctly, when tags are used."
        f"Either do not use a tag, or use a single eval task name."
    )

    model = load_model(
        train_config.model_name,
        quantization=False,
        use_fast_kernels=use_fast_kernels,
    )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    serializer = get_serializer(serializer_config)

    tokenizer, model = prepare_tokenizer(
        model,
        tokenizer=tokenizer,
        pretrained_model_name_or_path=train_config.model_name,
        model_max_length=train_config.context_length,
        use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
        serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
        serializer_tokens=serializer.special_tokens
        if tokenizer_config.add_serializer_tokens
        else None,
    )
    # sanity check for tokenizer
    if tokenizer_config.add_serializer_tokens:
        sanity_check_tokenizer(tokenizer, train_config.model_name)
    else:
        logging.warning(
            f"tokenizer_config.add_serializer_tokens is {tokenizer_config.add_serializer_tokens}; "
            f"this is expected if evaluating a base (not fine-tuned) model but unexpected otherwise."
        )

    if train_config.resume and not train_config.use_peft:
        ckpt_dir = get_latest_checkpoint(train_config.resume)
        model, _ = load_model_from_checkpoint(model, ckpt_dir)
    elif train_config.resume and train_config.use_peft:
        from peft import PeftModel

        print("#" * 100)
        ckpt_dir = get_latest_checkpoint(train_config.resume)
        print(f"loading PEFT model from {ckpt_dir}")
        model = PeftModel.from_pretrained(model, ckpt_dir)
        print("#" * 100)

    else:
        print("#" * 50)
        logging.warning(
            f"evaluating the baseline model {train_config.model_name}, NOT a fine-tuned checkpoint"
        )
        print("#" * 50)

    from rtfm.evaluation.evaluators import OpenVocabularyEvaluator
    from rtfm.inference_utils import RandomShotSelector, RICESShotSelector
    from rtfm.inference_utils import InferenceModel

    inference_model = InferenceModel(
        model=model, tokenizer=tokenizer, serializer=serializer
    )


    evaluator = OpenVocabularyEvaluator()

    output_metrics: Dict[str, float] = {}
    for eval_task_name in eval_task_names:
        prefix = f"{split}/{eval_task_name}"

        preprocessor_config = fetch_preprocessor_config_from_data_args(data_arguments, eval_task_name)
        tabular_dataset = get_task_dataset(
            eval_task_name, preprocessor_config=preprocessor_config
        )
        
        # By default fetch the entire dataset.
        df = tabular_dataset._df
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Juanky Edit: fixes ordered datasets
        df[tabular_dataset.target] = make_object_json_serializable(df[tabular_dataset.target])
        # Print out the columns and corresponding datatypes of the dataframe
        print("Columns and datatypes of the dataframe:\n")
        for column, dtype in df.dtypes.items():
            print(f"{column}: {dtype}")

        if shot_selector == "random":
            shot_selector = RandomShotSelector()
        elif shot_selector == "rices":
            shot_selector = RICESShotSelector()
            df = df.iloc[:200] # Juanky Hack
            # Juanky: I'm not convinced this is the right way to do things for target_choices
            # For one, the data is not yet converted to a string so the list below is a list of floats
            # in the current testing script
            target_choices = list(df[tabular_dataset.target].unique())
            shot_selector.post_init(inference_model, 
                                    df, tabular_dataset.target, 
                                    target_choices)
    
        metrics = evaluator.evaluate(
            inference_model=inference_model,
            shot_selector=shot_selector,
            num_shots=data_arguments.num_shots,
            train_config=train_config,
            df=df,
            target_colname=tabular_dataset.target,
        )
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        output_metrics.update(metrics)
        # TODO: log metrics to wandb.
        print(metrics)

    df = (
        pd.DataFrame.from_dict({k: [v] for k, v in output_metrics.items()})
        .T.reset_index()
        .rename(columns={"index": "metric", 0: "value"})
    )
    df["model_arguments.model_name_or_path"] = train_config.model_name
    df["split"] = split
    df.to_csv(outfile, index=False)
    print(df)
    print(f"metrics written to {outfile}")

    return


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (DataArguments, TrainConfig, TokenizerConfig, SerializerConfig)
    )

    parser.add_argument(
        "--eval-task-names",
        required=False,
        type=str,
        default=None,
        help="Comma-delimited list of names of task(s) to use for evaluation.",
    )

    parser.add_argument("--eval-task-file", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--split",
        choices=["train", "test", "validation"],
        default="test",
        help="metrics_prefix of data to use.",
    )

    parser.add_argument(
        "--shot-selector",
        choices=["random", "rices"],
        default="random",
        help="method to select in context examples.",
    )

    parser.add_argument(
        "--outfile",
        required=True,
        help="where to write the metrics files for each task.",
    )

    (
        data_args,
        train_config,
        tokenizer_config,
        serializer_config,
        other_args,
    ) = parser.parse_args_into_dataclasses()

    main(
        data_arguments=data_args,
        train_config=train_config,
        tokenizer_config=tokenizer_config,
        serializer_config=serializer_config,
        **vars(other_args),
    )
