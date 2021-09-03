#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import tensorflow as tf
import transformers 
import seqio

import flax
import jax
import jax.numpy as jnp
import jax.profiler
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )
    shuffle_buffer_size: int = field(
        default=10000, metadata={"help": "The number of examples to pre-load for shuffling."}
    )
    num_train_steps: int = field(default=50000, metadata={"help": "The number of training steps."})
    num_eval_samples: int = field(default=50000, metadata={"help": "The number of samples to be used for evaluation"})


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@flax.struct.dataclass
class FlaxDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # convert list to dict and tensorize input
        batch = BatchEncoding({"input_ids": np.array(examples["input_ids"])})

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.target_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        # to check that tokens are correctly proprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (sentinel_ids + self.tokenizer.vocab_size - 1), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100


def get_default_vocabulary():
      return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    batch_idx = np.split(samples_idx, sections_split)
    return batch_idx

def advance_iter_and_group_samples(train_iterator, num_samples, max_seq_length):
    """
    The training iterator is advanced so that after groupifying the samples,
    `num_samples` of length `max_seq_length` are returned.
    """
    
    num_total_tokens = max_seq_length * num_samples
    samples = defaultdict(list)
    start = time.time()
    i = 0
    import pdb ;pdb.set_trace()
    while i < num_total_tokens:
        tokenized_samples = next(train_iterator)
        i += len(tokenized_samples["input_ids"])

        # concatenate tokenized samples to list
        samples = {k: samples[k] + tokenized_samples[k] for k in tokenized_samples.keys()}
    import pdb; pdb.set_trace()
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
    def group_texts(examples):
        # Concatenate all texts.
        total_length = len(examples['input_ids'])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= expanded_inputs_length:
            total_length = (total_length // (jax.local_device_count() * expanded_inputs_length)) * jax.local_device_count() * expanded_inputs_length
        # Split by chunks of max_len.
        result = {k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)] for k, t in examples.items()}
        return result
    grouped_samples = group_texts(samples)

    return grouped_samples


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        #datasets = load_dataset("/home/martin/datasets/datasets/c4/c4.py", data_args.dataset_name, cache_dir=model_args.cache_dir, streaming=True)
        import tensorflow_datasets as tfds
        #datasets = tfds.load(name="c4/realnewslike", split=["train", "validation"], data_dir="gs://c4-datasets/")
        builder = tfds.builder('c4/realnewslike', data_dir="gs://c4-datasets/")
        train_dataset = builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({"text": True}))
        train_dataset = train_dataset.batch(32).prefetch(10)

        #datasets["train"] = load_dataset(name, data_files=train_files, cache_dir=model_args.cache_dir)
        #datasets["validation"] = load_dataset(name, data_files=validation_files, cache_dir=model_args.cache_dir)

    # Load pretrained model and tokenizer

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    if model_args.config_name:
        config = T5Config.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir, vocab_size=len(tokenizer)
        )
    elif model_args.model_name_or_path:
        config = T5Config.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, vocab_size=len(tokenizer)
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    shuffle_seed = training_args.seed

    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # Since we make sure that all sequences are of the same length, no attention_mask is needed.
    def tokenize_function(examples):
        return tokenizer(str(examples["text"].numpy()), return_attention_mask=False)
    
    # Encoding functions
    def encode(text_tensor):
        encoded_text = tokenizer(str(text_tensor['text'].numpy()), return_attention_mask=False)
        return encoded_text
    DEFAULT_OUTPUT_FEATURES = {
        "inputs": seqio.Feature(
            vocabulary=get_default_vocabulary(), add_eos=True,
            required=False),
        "targets": seqio.Feature(
            vocabulary=get_default_vocabulary(), add_eos=True)
    } 
    from t5.data import preprocessors
    import functools
    seqio.TaskRegistry.add(
        "realnewslike",
        seqio.TfdsDataSource(tfds_name="c4/realnewslike:3.0.1", tfds_data_dir="gs://c4-datasets/"),
        preprocessors=[
            functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
            seqio.preprocessors.tokenize, 
            seqio.CacheDatasetPlaceholder(),
            preprocessors.span_corruption,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[]) 
    import pdb ;pdb.set_trace()
    seqio.preprocessors.tokenize
    train_dataset = datasets[0].batch(32).prefetch(tf.data.AUTOTUNE)
    trian_dataset.map(encode)
    train_data_tokenized = datasets[0].map(encode_map_fn)

    tokenized_datasets = {'train': datasets[0].map(tokenize_function), 'validation': datasets[1].map(tokenize_function)}

    #tokenized_datasets = {'train': datasets['train'].map(tokenize_function, batched=True), 'validation': datasets['validation'].map(tokenize_function, batched=True)}
    tokenized_datasets['train'] = tokenized_datasets['train'].shuffle(buffer_size=data_args.shuffle_buffer_size, seed=shuffle_seed)
    tokenized_datasets['validation'] = tokenized_datasets['validation'].shuffle(buffer_size=data_args.shuffle_buffer_size, seed=shuffle_seed)

    # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
    # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
    # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= expanded_inputs_length:
            total_length = (total_length // expanded_inputs_length) * expanded_inputs_length

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        if len(result['input_ids']) > 232:
            result['input_ids'] = result['input_ids'][:232]
        return result

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    if model_args.model_name_or_path:
        model = FlaxT5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    else:
        model = FlaxT5ForConditionalGeneration(config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype))

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = FlaxDataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Constants
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    num_train_samples = tokenized_datasets["train"]._info.splits['train'].num_examples
    num_validation_samples = tokenized_datasets["validation"]._info.splits['validation'].num_examples
    num_train_steps = num_train_samples // train_batch_size * num_epochs
    num_validation_steps = num_validation_samples // train_batch_size

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # Setup train state
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

            # compute loss
            loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)

        # summarize metrics
        metrics = {"loss": loss.mean(), "accuracy": accuracy.mean()}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    validation_iter = iter(tokenized_datasets['validation'])
    training_iter = iter(tokenized_datasets['train'])
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    train_time = 0
    epoch = 0
    steps = tqdm(range(num_train_steps), desc="Training...", position=0)
    for step in range(num_train_steps):
        # ======================== Training ================================
        try:
            samples = advance_iter_and_group_samples(training_iter, train_batch_size, max_seq_length)
        except StopIteration:
            epoch += 1
            tokenized_datasets['train'].set_epoch(epoch)
            tokenized_datasets['validation'].set_epoch(epoch)
            training_iter = iter(tokenized_datasets['train'])
            samples = advance_iter_and_group_samples(training_iter, train_batch_size, max_seq_length)
        if step < 22402:
            print(step)
            continue
        import pdb; pdb.set_trace()
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        model_inputs = data_collator(samples)

        # Model forward
        model_inputs = shard(model_inputs.data)
        state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)
        train_metrics.append(train_metric)

        if step % training_args.logging_steps == 0 and step > 0:
            # Save metrics
            train_metric = jax_utils.unreplicate(train_metric)
            train_time += time.time() - train_start
            if has_tensorboard and jax.process_index() == 0:
                write_train_metric(summary_writer, train_metrics, train_time, step)

            steps.write(
                f"Step... ({step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
            )

            train_metrics = []

        if step % training_args.eval_steps == 0 and step > 0:
            # ======================== Evaluating ==============================
            eval_metrics = []
            for _ in tqdm(range(num_validation_steps), desc="Evaluating ...", position=2):
                try:
                    samples = advance_iter_and_group_samples(validation_iter, train_batch_size, max_seq_length)
                except StopIteration:
                    validation_iter = iter(tokenized_datasets['validation'])

                model_inputs = data_collator(samples)

                # Model forward
                model_inputs = shard(model_inputs.data)
                metrics = p_eval_step(state.params, model_inputs)
                eval_metrics.append(metrics)

            # get eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            # Update progress bar
            steps.write(f"Step... ({step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})")

            # Save metrics
            if has_tensorboard and jax.process_index() == 0:
                write_eval_metric(summary_writer, eval_metrics, step)

        if step % training_args.save_steps == 0 and step > 0:
            # save checkpoint after each epoch and push checkpoint to the hub
            if jax.process_index() == 0:
                params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                model.save_pretrained(
                    training_args.output_dir,
                    params=params,
                    push_to_hub=training_args.push_to_hub,
                    commit_message=f"Saving weights and logs of step {step}",
                )
        steps.update(1)
        jax.profiler.save_device_memory_profile(f"memory-step-{step}.prof")

