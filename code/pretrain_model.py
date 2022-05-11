from util.localcache.huggingface import cache as HF_CACHE
HF_CACHE.enable()

from util import tokenmasking
from util.argument_parser_from_file import ArgumentParserFromFile

import csv
import math
import itertools
import collections
import contextlib
import dataclasses
import logging
from pathlib import Path

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import IterableDataset

MODULE_LOGGER = logging.getLogger(__name__)

# transformers ignores tokens like this for cross-entropy:
IGNORE_TOKEN_ID = -100

MASKING_ALGORITHM_PRESETS = {}
if hasattr(tokenmasking, "TextInfilling"):
    MASKING_ALGORITHM_PRESETS["text-infilling"] = (lambda mask_token:
                                                   tokenmasking.TextInfilling(mask_token, 0.3, 3))

if hasattr(tokenmasking, "RougeGapSentencesGeneration"):
    MASKING_ALGORITHM_PRESETS["gap-sentences-generation-ind-orig"] = (lambda mask_token:
                                                                      tokenmasking.RougeGapSentencesGeneration(mask_token, 0.3, False))
    MASKING_ALGORITHM_PRESETS["gap-sentences-generation-seq-orig"] = (lambda mask_token:
                                                                      tokenmasking.RougeGapSentencesGeneration(mask_token, 0.3, True))

if hasattr(tokenmasking, "FastRougeGapSentencesGeneration"):
    MASKING_ALGORITHM_PRESETS["gap-sentences-generation-ind-fast"] = (lambda mask_token:
                                                                      tokenmasking.FastRougeGapSentencesGeneration(mask_token, 0.3, False))
    MASKING_ALGORITHM_PRESETS["gap-sentences-generation-seq-fast"] = (lambda mask_token:
                                                                      tokenmasking.FastRougeGapSentencesGeneration(mask_token, 0.3, True))


@dataclasses.dataclass
class PipelineConfiguration:
    model_class: str
    model_name_or_path: str
    masking_algorithm_preset: str
    tokenizer_name_or_path: str = None
    config_overrides: str = None
    use_fast_tokenizer: bool = True
    revision: str = None
    mask_token: str = "mask_token"
    derive_mask_token_from_tokenizer: bool = True
    masking_algorithm_seed: int = None

    def __post_init__(self):
        self.model_class = getattr(transformers, self.model_class)
        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.model_name_or_path

    def _get_mask_token(self):
        if not self.derive_mask_token_from_tokenizer:
            return self._mask_token
        if not hasattr(self, "_tokenizer"):
            self.load_tokenizer()
        return getattr(self._tokenizer, self._mask_token)

    def _set_mask_token(self, mask_token):
        self._mask_token = mask_token

    def load_tokenizer(self, *args, **kwargs):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, use_fast=self.use_fast_tokenizer, revision=self.revision, *args, **kwargs)
        return self._tokenizer

    def load_model(self, *args, **kwargs):
        # Can *not* use AutoModelForMaskedLM, as it does not recognize Pegasus as a model that can perform masked-learning.
        model = self.model_class.from_pretrained(self.model_name_or_path, revision=self.revision, *args, **kwargs)
        if self.config_overrides:
            model.config.update_from_string(self.config_overrides)
        return model

    def load_masker(self):
        masker = MASKING_ALGORITHM_PRESETS[self.masking_algorithm_preset](self.mask_token)
        if self.masking_algorithm_seed is not None:
            masker.set_seed(self.masking_algorithm_seed)
        # NOTE: Some masking algorithms may need to join tokens into sentences,
        # and should therefore be provided with the correct means to do so.
        masker.join_sentence_tokens = self.load_tokenizer().convert_tokens_to_string
        return masker


# Replace public attribute with property.
PipelineConfiguration.mask_token = property(PipelineConfiguration._get_mask_token, PipelineConfiguration._set_mask_token)


@dataclasses.dataclass
class DatasetConfiguration:
    csv_paths: str = dataclasses.field(metadata={"nargs": "+"})
    input_field_name: str
    max_sentences_per_path: int = dataclasses.field(default_factory=list, metadata={"nargs": "*"})
    max_sentences_default: int = None
    sentence_separator: str = None
    sentence_max_length: float = 0.25
    sentences_per_window: int = 10000
    window_stride: float = None
    max_input_length: int = None

    def __post_init__(self):
        specs = len(self.max_sentences_per_path)
        paths = len(self.csv_paths)
        if specs > paths:
            raise ValueError("More specifications for paths ({} via --max_sentences_per_path) have been provided than paths ({}).".format(specs, paths))
        if self.window_stride is not None:
            if self.window_stride < 1:
                self.window_stride = int(self.window_stride * self.sentences_per_window)
            else:
                self.window_stride = int(self.window_stride)


@dataclasses.dataclass
class PaddingCollatorForLabeled(transformers.DataCollatorWithPadding):
    label_max_length: int = None
    label_pad_token_id: int = None

    def __post_init__(self):
        if self.label_max_length is None:
            self.label_max_length = self.max_length

    @contextlib.contextmanager
    def context_max_length(self, max_length):
        original, self.max_length = self.max_length, max_length
        try:
            yield max_length
        finally:
            self.max_length = original

    @contextlib.contextmanager
    def context_pad_token_id(self, pad_token_id):
        # NOTE: This is a bit ugly may break in future transformers-versions.
        # Workaround to pass check if the pad_token is positive:
        hack_active = True
        class PositiveLessThan(int):
            def __lt__(self, other):
                if hack_active:
                    return abs(int(self)).__lt__(other)
                return int(self).__lt__(other)
        pad_token_id = PositiveLessThan(pad_token_id)
        custom_pad_token = "<@CUSTOM PAD_TOKEN@> not-a-token"  # This should not be a token in any tokenizer.
        original_pad_token, self.tokenizer.pad_token = self.tokenizer.pad_token, custom_pad_token
        original_convert = self.tokenizer.convert_tokens_to_ids
        def convert_custom_tokens_to_ids(tokens):
            if tokens == custom_pad_token:
                return pad_token_id
            return original_convert(tokens)
        self.tokenizer.convert_tokens_to_ids = convert_custom_tokens_to_ids
        try:
            yield pad_token_id
        finally:
            hack_active = False
            self.tokenizer.convert_tokens_to_ids = original_convert
            self.tokenizer.pad_token = original_pad_token

    def __call__(self, features):
        batch = super().__call__(features)

        # Only "input_ids" are padded when tokenizer.pad is called,
        # so labels need to be padded separately.
        labels = batch["labels"]
        # Additionally tokenizer.pad does not accept tensorized inputs.
        # See https://github.com/huggingface/transformers/issues/15447
        if not isinstance(labels, list):
            labels = list(labels)
        with contextlib.ExitStack() as context_stack:
            context_stack.enter_context(self.context_max_length(self.label_max_length))
            if self.label_pad_token_id:
                context_stack.enter_context(self.context_pad_token_id(self.label_pad_token_id))
            batch["labels"] = super().__call__({"input_ids": labels})["input_ids"]
        return batch


class CountableIterable:

    def __init__(self, iterable):
        self.iterable = iterable
        self._count = 0

    @property
    def count(self):
        return self._count

    def __iter__(self):
        for element in self.iterable:
            self._count += 1
            yield element

    def __repr__(self):
        return "{}({}, {})".format(type(self).__name__, repr(self.iterable), self.count)


def sliding_window(iterable, size, stride=1):
    """
    Generate windows of given size from the iterable.
    Each window will be shifted n positions from the previous one,
    with n according to the provided stride.

    If stride is set to None, windows are guaranteed to be non-overlapping (except for the last one).
    """
    if stride is None:
        stride = size
    iterator = iter(iterable)
    window = collections.deque(itertools.islice(iterator, size), maxlen=size)
    yield tuple(window)
    has_more = True
    while has_more:
        try:
            for i in range(stride):
                window.append(next(iterator))
        except StopIteration:
            has_more = False
            if i <= 0:
                break
        yield tuple(window)


def replace(x, mask, replacement, return_tensors=None):
    if not return_tensors:
        return [replacement if do_remove else element for element, do_remove in zip(x, mask)]
    if return_tensors == "np":
        import numpy
        where = numpy.where
    elif return_tensors == "pt":
        import torch
        where = torch.where
    elif return_tensors == "tf":
        import tensorflow
        where = tensorflow.where
    return where(mask == 1, replacement, x)


def replace_value(x, value, replacement, return_tensors=None):
    if not return_tensors:
        return [replacement if element == value else element for element in x]
    if return_tensors == "np":
        import numpy
        where = numpy.where
    elif return_tensors == "pt":
        import torch
        where = torch.where
    elif return_tensors == "tf":
        import tensorflow
        where = tensorflow.where

    return where(x == value, replacement, x)


def tokenize_and_mask(tokenizer, masker, sentences, sentence_truncation=True, sentence_max_length=None, sentence_separator=None, return_tensors=None):
    """
    Tokenize every sentence, perform masking and
    return a batch for each one, including label_ids.
    """
    # Tokenize and truncate.
    if sentence_truncation:
        if sentence_max_length is None:
            sentence_max_length = 1.0
        if sentence_max_length <= 1:
            sentence_max_length = int(sentence_max_length * tokenizer.max_len_single_sentence)
        else:
            sentence_max_length = int(sentence_max_length) - tokenizer.num_special_tokens_to_add()

    prepare_opts = {}

    if not sentence_separator:
        sentence_separator = ""
    else:
        prepare_opts["add_special_tokens"] = False
    # NOTE: Python handles out-of-range slices gracefully, no need to worry here.
    tokenized_sentences = tuple(tokenizer.tokenize(sentence)[:sentence_max_length] + [sentence_separator] for sentence in sentences)

    # NOTE: Maskers might add masks that do not replace anything and thus disregard this soft-limit for truncation.
    # On the other hand they may also replace multiple tokens with a single mask, producing a shorter final sequence.
    # In practice this will not matter as long as the max-length per sentence does not approach the max-length of the model.

    prepare_opts["truncation"] = True  # Just in case the masker adds too many masks.
    prepare_opts["return_tensors"] = return_tensors

    masker_state = masker.get_state()
    # Generate ids for the inputs.
    masked_tokenized_sentences = masker.mask(tokenized_sentences)
    ids_masked_sentences = (tokenizer.convert_tokens_to_ids(tokens) for tokens in masked_tokenized_sentences)
    batches = list(tokenizer.prepare_for_model(ids, **prepare_opts) for ids in ids_masked_sentences)

    prepare_opts["verbose"] = False
    prepare_opts["return_special_tokens_mask"] = True

    masker.set_state(masker_state)
    # Generate ids for the labels.
    antimasked_tokenized_sentences = masker.antimask(tokenized_sentences)
    ids_antimasked_sentences = (tokenizer.convert_tokens_to_ids(tokens) for tokens in antimasked_tokenized_sentences)
    anti_batches = (tokenizer.prepare_for_model(ids, **prepare_opts) for ids in ids_antimasked_sentences)

    mask_id = tokenizer.convert_tokens_to_ids(masker.mask_token)
    for batch, anti_batch in zip(batches, anti_batches):
        batch["label_ids"] = replace(anti_batch["input_ids"], anti_batch["special_tokens_mask"], IGNORE_TOKEN_ID, return_tensors=return_tensors)
        # Not all tokenizers identify masks as special tokens, so this needs to be handled.
        if mask_id in batch["label_ids"]:
            batch["label_ids"] = replace_value(batch["label_ids"], mask_id, IGNORE_TOKEN_ID, return_tensors=return_tensors)
    return batches


def group_sentences(encoded_sentences, max_batch_length):
    """
    Groups multiple consecutive whole sentences until the batch reaches a maximum size.
    """
    batch = {}
    # This represents the length of the longest sequence in the batch:
    batch_length = 0
    for sentence_encoding in encoded_sentences:
        sentence_length = max(map(len, sentence_encoding.values()))
        if batch_length + sentence_length <= max_batch_length:
            for k, v in sentence_encoding.items():
                batch.setdefault(k, [])
                batch[k].extend(v)
            batch_length += sentence_length
        else:
            yield batch
            batch = sentence_encoding
            batch_length = sentence_length
    yield batch


def pretrain_model(pipeline_config, dataset_config, training_args):
    """
    Pre-training has roughly the following steps:
    1. load tokenizer and model
    2. load dataset
    3. perform tokenization and masking
    4. group into sequences of max-length and do batching
    5. run Trainer
    """
    MODULE_LOGGER.info("Loading existing tokenizer...")
    tokenizer = pipeline_config.load_tokenizer()
    MODULE_LOGGER.info("Done loading tokenizer.")

    MODULE_LOGGER.info("Loading existing model...")
    model = pipeline_config.load_model()
    details = ""
    if hasattr(model, "device"):
        details += " on '{}'".format(model.device)
    MODULE_LOGGER.info("Done loading model{}.".format(details))

    MODULE_LOGGER.info("Loading masking algorithm...")
    masker = pipeline_config.load_masker()
    MODULE_LOGGER.info("Done loading masking algorithm.")

    if not (training_args.do_train or training_args.do_eval):
        MODULE_LOGGER.warning("Neither 'do_train' nor 'do_eval' is set, so there is no task to perform.")
        return

    def iterate_datasets():
        for length, path in itertools.zip_longest(dataset_config.max_sentences_per_path, dataset_config.csv_paths):
            if length is None:
                length = dataset_config.max_sentences_default
            if length is not None and length <= 0:
                continue
            MODULE_LOGGER.debug("Starting reading from '{}'...".format(path))
            with open(path, "r") as file:
                csvreader = csv.reader(file)
                headers = next(csvreader)
                input_field_index = headers.index(dataset_config.input_field_name)
                sentences = (entry[input_field_index] for entry in csvreader)
                if length is not None:
                    sentences = itertools.islice(sentences, length)
                yield sentences
            MODULE_LOGGER.debug("Done reading from '{}'.".format(path))

    dataset_iterables = iterate_datasets()

    def windowed_processing():
        max_batch_length = dataset_config.max_input_length
        if not max_batch_length:
            max_batch_length = tokenizer.model_max_length
        max_group_length = max_batch_length
        if dataset_config.sentence_separator:
            max_group_length -= tokenizer.num_special_tokens_to_add()
        data_collator = PaddingCollatorForLabeled(tokenizer, "max_length", max_length=max_batch_length, label_pad_token_id=IGNORE_TOKEN_ID)
        for sentences_iterable in dataset_iterables:
            windowed_sentences = sliding_window(sentences_iterable, dataset_config.sentences_per_window, dataset_config.window_stride)
            for sentences in windowed_sentences:
                MODULE_LOGGER.debug("Tokenizing and masking text...")
                batches = tokenize_and_mask(tokenizer, masker, sentences, sentence_max_length=dataset_config.sentence_max_length, sentence_separator=dataset_config.sentence_separator)
                MODULE_LOGGER.debug("Done tokenizing and masking.")

                # NOTE: There may exist an easier way to do this using a default collator,
                # like DataCollatorForSeq2Seq, but this works.
                batched_sentences = group_sentences(batches, max_group_length)
                for batch in batched_sentences:
                    if dataset_config.sentence_separator:
                        if tokenizer.bos_token_id is not None:
                            batch["input_ids"].insert(0, tokenizer.bos_token_id)
                            batch["attention_mask"].append(1)
                            batch["label_ids"].insert(0, tokenizer.bos_token_id)
                        if tokenizer.eos_token_id is not None:
                            batch["input_ids"][-1] = tokenizer.eos_token_id
                            batch["label_ids"][-1] = tokenizer.eos_token_id
                    yield data_collator(batch)

    batches = windowed_processing()
    batches = CountableIterable(batches)

    class IteratorDataset(IterableDataset):

        def __init__(self, iterator):
            self.iterator = iterator

        def __iter__(self):
            yield from self.iterator

    train_dataset = IteratorDataset(batches)
    eval_dataset = IteratorDataset(batches)

    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args,
                                   train_dataset=train_dataset,
                                   eval_dataset=eval_dataset)

    # Suppress warning, that use_cache cannot be used with gradient-checkpointing.
    if training_args.gradient_checkpointing:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if training_args.do_train:
        MODULE_LOGGER.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        MODULE_LOGGER.info("Done training.")

        MODULE_LOGGER.info("Saving model and state...")
        checkpoint_name = "{}-{}".format(PREFIX_CHECKPOINT_DIR, trainer.state.global_step)
        last_checkpoint = Path(training_args.output_dir) / checkpoint_name
        if not last_checkpoint.exists():
            trainer.save_model(last_checkpoint)
            original, training_args.output_dir = training_args.output_dir, last_checkpoint
            trainer.save_state() # save in checkpoint
            training_args.output_dir = original
        MODULE_LOGGER.info("Done saving model and state.")
        metrics = train_result.metrics

        add_metrics(metrics, "train", batches.count, training_args.total_train_batch_size)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if training_args.do_eval:
        previous_count = batches.count

        MODULE_LOGGER.info("Starting evaluation...")
        metrics = trainer.evaluate()
        MODULE_LOGGER.info("Done evaluating.")

        add_metrics(metrics, "eval", batches.count - previous_count, training_args.total_eval_batch_size)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["eval_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def add_metrics(metrics, type_prefix, samples, batch_size):
    steps = samples / batch_size
    metrics[type_prefix + "_samples"] = samples
    metrics[type_prefix + "_batch_size"] = batch_size
    metrics[type_prefix + "_steps"] = steps
    if type_prefix + "_runtime" in metrics:
        seconds = metrics[type_prefix + "_runtime"]
        metrics[type_prefix + "_samples_per_second"] = samples / seconds
        metrics[type_prefix + "_steps_per_second"] = steps / seconds


@dataclasses.dataclass
class CustomTrainingArguments(transformers.TrainingArguments):
    total_train_batch_size: int = None
    total_eval_batch_size: int = None

    def __post_init__(self):
        super().__post_init__()
        if self.total_train_batch_size:
            self.gradient_accumulation_steps = int(self.total_train_batch_size / (self.world_size * self.train_batch_size))
        else:
            self.total_train_batch_size = self.train_batch_size * self.gradient_accumulation_steps * self.world_size
        if self.total_eval_batch_size:
            devices = self.eval_batch_size / self.per_device_eval_batch_size
            self.per_device_eval_batch_size = int(self.total_eval_batch_size / (self.world_size * devices))
        else:
            self.total_eval_batch_size = self.eval_batch_size * self.world_size


class CustomArgumentParser(transformers.HfArgumentParser, ArgumentParserFromFile):
    pass


if __name__ == "__main__":
    parser = CustomArgumentParser((PipelineConfiguration, DatasetConfiguration, CustomTrainingArguments), fromfile_prefix_chars="@",
                                  epilog="Arguments prefixed with '@' will be interpreted as file-locations to parse additional arguments from. Can be either a python or JSON file.")

    pipeline_config, dataset_config, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%x %X")
    MODULE_LOGGER.setLevel(training_args.get_process_log_level())

    if training_args.max_steps <= 0:
        from sys import maxsize
        MODULE_LOGGER.warning("--max_steps not given. Setting it to highest estimate {}.".format(maxsize))
        training_args.max_steps = maxsize

    pretrain_model(pipeline_config, dataset_config, training_args)
