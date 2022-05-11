from util.localcache.huggingface import cache as HF_CACHE
HF_CACHE.enable()
from util.localcache.nltk import cache as NLTK_CACHE
NLTK_CACHE.enable()

from util.argument_parser_from_file import ArgumentParserFromFile
from util import loganalysis

import math
import statistics
import dataclasses
import shlex
import logging
from pathlib import Path

MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

MODULE_LOGGER = logging.getLogger(__name__)

# transformers ignores tokens like this for cross-entropy:
IGNORE_TOKEN_ID = -100


@dataclasses.dataclass
class PipelineConfiguration:
    model_name_or_path: str
    tokenizer_name_or_path: str = None
    config_overrides: str = None
    use_fast_tokenizer: bool = True
    revision: str = None
    output_model_inputs_path: str = None
    output_predictions_path: str = None
    output_references_path: str = None

    def __post_init__(self):
        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.model_name_or_path

    def load_tokenizer(self, *args, **kwargs):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, use_fast=self.use_fast_tokenizer, revision=self.revision, *args, **kwargs)
        return tokenizer

    def load_model(self, *args, **kwargs):
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path, revision=self.revision, *args, **kwargs)
        original_position_embeddings = model.config.max_position_embeddings
        if self.config_overrides:
            model.config.update_from_string(self.config_overrides)
        if original_position_embeddings != model.config.max_position_embeddings:
            model.resize_position_embeddings(model.config.max_position_embeddings)
        return model


@dataclasses.dataclass
class DatasetConfiguration:
    dataset_path: str
    loganalysis_arguments: str
    input_field_name: str
    only_include_common_events: bool = True
    excluded_events_loganalysis_arguments: str = None
    summary_instance_property: str = None
    train_category: str = None
    eval_category: str = None
    keep_empty_references: str = "none"
    sentence_max_length: float = 0.25
    sentence_separator: str = None
    max_input_length: int = None

    def __post_init__(self):
        parser = loganalysis.get_options_parser()
        parser.prog = "loganalysis"
        arguments = shlex.split(self.loganalysis_arguments)
        self.loganalysis_options = vars(parser.parse_args(arguments))
        if self.excluded_events_loganalysis_arguments:
            arguments = shlex.split(self.excluded_events_loganalysis_arguments)
            self.excluded_events_options = vars(parser.parse_args(arguments))
        else:
            self.excluded_events_options = {}

    def _windows_only(self, log_windows_per_instance):
        for windows in log_windows_per_instance.values():
            yield from windows

    def _load_log_windows_with_options(self, load_summaries=False, **options):
        log_windows_per_instance = loganalysis.load_log_windows(self.dataset_path, **options)

        if not load_summaries:
            return log_windows_per_instance
        log_instances = loganalysis.load_log_instances(self.dataset_path)
        if not self.summary_instance_property:
            return log_windows_per_instance, log_instances, None

        summaries_per_instance = {}
        for instance in log_windows_per_instance.keys():
            summaries_per_instance[instance] = log_instances[instance][self.summary_instance_property]
        return log_windows_per_instance, log_instances, summaries_per_instance

    def load_log_windows_and_summaries(self):
        MODULE_LOGGER.info("Loading log windows...")
        x = self._load_log_windows_with_options(load_summaries=True, **self.loganalysis_options)
        self._log_windows_per_instance, self._log_instances, self._summaries_per_instance = x
        MODULE_LOGGER.info("Done loading log windows.")
        return self._log_windows_per_instance, self._log_instances, self._summaries_per_instance

    @property
    def log_instances(self):
        if not hasattr(self, "_log_instances"):
            self.load_log_windows_and_summaries()
        return self._log_instances

    @property
    def log_windows_per_instance(self):
        if not hasattr(self, "_log_windows_per_instance"):
            self.load_log_windows_and_summaries()
        return self._log_windows_per_instance

    @property
    def log_windows(self):
        if not hasattr(self, "_log_windows_per_instance"):
            self.load_log_windows_and_summaries()
        return self._windows_only(self._log_windows_per_instance)

    @property
    def summaries_per_instance(self):
        if not hasattr(self, "_summaries_per_instance"):
            self.load_log_windows_and_summaries()
        return self._summaries_per_instance

    def load_excluded_events(self):
        MODULE_LOGGER.info("Loading excluded events...")
        if self.excluded_events_options:
            log_windows_per_instance = self._load_log_windows_with_options(**self.excluded_events_options)
            log_windows = self._windows_only(log_windows_per_instance)
            self._excluded_events = loganalysis.compute_cumulated_events(log_windows)
        else:
            self._excluded_events = set()
        MODULE_LOGGER.info("Done loading excluded events.")
        return self._excluded_events

    @property
    def excluded_events(self):
        if not hasattr(self, "_excluded_events"):
            self.load_excluded_events()
        return self._excluded_events

    def load_summary_events(self):
        MODULE_LOGGER.info("Loading summary events...")
        if self.only_include_common_events:
            self._summary_events = loganalysis.compute_common_events(self.log_windows)
        else:
            self._summary_events = loganalysis.compute_cumulated_events(self.log_windows)
        self._summary_events.difference_update(dataset_config.excluded_events)
        MODULE_LOGGER.info("Done loading summary events.")
        return self._summary_events

    @property
    def summary_events(self):
        if not hasattr(self, "_summary_events"):
            self.load_summary_events()
        return self._summary_events


def split_by_token(token_ids, split_at, include_at_end=True):
    sequences = [[]]
    for id in token_ids:
        if not include_at_end and id == split_at:
            sequences.append([])
        sequences[-1].append(id)
        if include_at_end and id == split_at:
            sequences.append([])
    return filter(None, sequences)


def split_by_token_pair(token_ids, begin_token, end_token):
    sequences = []
    has_begun = False
    for id in token_ids:
        if id == begin_token:
            has_begun = True
            sequences.append([])
        if has_begun:
            sequences[-1].append(id)
        if id == end_token:
            has_begun = False
    return sequences


def decode_sequences(tokenizer, token_ids, sentence_separator=None, **decoding_options):
    if sentence_separator:
        text = tokenizer.decode(tuple(token_ids), **decoding_options)
        return text.split(sentence_separator)
    if tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None:
        id_sequences = split_by_token_pair(token_ids, tokenizer.bos_token_id, tokenizer.eos_token_id)
    elif tokenizer.bos_token_id is not None:
        id_sequences = split_by_token(token_ids, tokenizer.bos_token_id, False)
    elif tokenizer.eos_token_id is not None:
        id_sequences = split_by_token(token_ids, tokenizer.eos_token_id)
    else:
        MODULE_LOGGER.error("The tokenizer has neither BOS nor EOS tokens. Hence cannot split into multiple sequences.")
        id_sequences = [token_ids]
    return tokenizer.batch_decode(id_sequences, **decoding_options)


def tokenize(tokenizer, sentences, sentence_separator=None, sentence_truncation=True, sentence_max_length=None, return_tensors=None):
    """
    Tokenize every sentence and return a batch for each one.
    """
    options = {}

    if sentence_truncation:
        if sentence_max_length is None:
            sentence_max_length = 1.0
        if sentence_max_length <= 1:
            sentence_max_length = int(sentence_max_length * tokenizer.model_max_length)
        else:
            sentence_max_length = int(sentence_max_length)
        options["truncation"] = True
        options["max_length"] = sentence_max_length

    if not sentence_separator:
        sentence_separator = ""
    else:
        options["add_special_tokens"] = False

    options["return_tensors"] = return_tensors

    for i, sentence in enumerate(sentences):
        if i > 0:
            sentence = sentence_separator + sentence
        yield tokenizer(sentence, **options)


def group_and_label(encoded_sentences, event_ids, label_events, max_batch_length):
    """
    Groups multiple consecutive whole sentences until the batch reaches a maximum size.
    If a sentence's corresponding event id is part of the label-events,
    it will be added as part of the labels.
    """
    batch = {}
    # This represents the length of the longest sequence in the batch:
    batch_length = 0
    for id, sentence_encoding in zip(event_ids, encoded_sentences):
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

        batch.setdefault("labels", [])
        if id in label_events:
            batch["labels"].extend(sentence_encoding["input_ids"])
    yield batch


def write_lines_to(path, line_separated_texts):
    if not path:
        return
    with open(path, "w") as f:
        for text in line_separated_texts:
            for line in text:
                line = line.strip()
                if line:
                    print(line, file=f)
            print("", file=f)


def finetune_model(pipeline_config, dataset_config, training_args):
    """
    Fine-tuning has roughly the following steps:
    1. load tokenizer and model
    2. load dataset
    3. perform tokenization and create summaries from events
    4. define metrics to be evaluated
    5. run Seq2SeqTrainer
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

    tokenizer.model_max_length = model.config.max_position_embeddings

    if not (training_args.do_train or training_args.do_eval):
        MODULE_LOGGER.warning("Neither 'do_train' nor 'do_eval' is set, so there is no task to perform.")
        return

    max_batch_length = dataset_config.max_input_length
    if not max_batch_length:
        max_batch_length = tokenizer.model_max_length
        if dataset_config.sentence_separator:
            max_batch_length -= tokenizer.num_special_tokens_to_add()
    batches_per_instance = {}
    for index, (instance, windows) in enumerate(dataset_config.log_windows_per_instance.items()):
        for entries in windows:
            sentences = map(lambda entry: entry[dataset_config.input_field_name], entries)
            encoded_sentences = tokenize(tokenizer, sentences, sentence_max_length=dataset_config.sentence_max_length,
                                         sentence_separator=dataset_config.sentence_separator)
            summary_event_ids = dataset_config.summary_events
            event_ids = map(loganalysis.get_event_id, entries)
            batched_sentences = group_and_label(encoded_sentences, event_ids, summary_event_ids, max_batch_length)
            if dataset_config.summaries_per_instance:
                # manually supplied summaries
                batch = next(batched_sentences)
                summary = dataset_config.summaries_per_instance[instance]
                if dataset_config.sentence_separator:
                    summary = dataset_config.sentence_separator.join(summary)
                if isinstance(summary, str):
                    batch["labels"] = tokenizer.encode(summary, truncation=True)
                else:
                    encoded_summary = tokenize(tokenizer, summary, sentence_max_length=dataset_config.sentence_max_length)
                    batch["labels"] = [token for sentence in encoded_summary for token in sentence["input_ids"]]
                batched_sentences = [batch]

            batches_per_instance.setdefault(instance, [])
            for batch in batched_sentences:
                if dataset_config.sentence_separator:
                    if tokenizer.bos_token_id is not None:
                        batch["input_ids"].insert(0, tokenizer.bos_token_id)
                        batch["attention_mask"].append(1)
                        if batch["labels"][0] != tokenizer.bos_token_id:
                            batch["labels"].insert(0, tokenizer.bos_token_id)
                    if tokenizer.eos_token_id is not None:
                        batch["input_ids"].append(tokenizer.eos_token_id)
                        batch["attention_mask"].append(1)
                        if batch["labels"][-1] != tokenizer.eos_token_id:
                            batch["labels"].append(tokenizer.eos_token_id)
                batches_per_instance[instance].append(batch)

    compute_metrics = None
    if training_args.predict_with_generate:
        generation_metrics = []

        def compute_rouge(hypotheses_sequences, references_sequences, max_ngram=4, compute_lcs=True, compute_summary_lcs=True, use_stemmer=True, **scorer_options):
            hypotheses = map("\n".join, hypotheses_sequences)
            references = map("\n".join, references_sequences)

            rouge_types = []
            if compute_lcs:
                rouge_types.append("rougeL")
            if compute_summary_lcs:
                rouge_types.append("rougeLsum")
            for i in range(max_ngram):
                rouge_types.append("rouge{}".format(i + 1))
            scorer_options["use_stemmer"] = use_stemmer
            # Note: The default ROUGE-tokenizer removes any non-alphanumerical characters.
            scorer = RougeScorer(rouge_types, **scorer_options)

            scores = {}
            for reference, hypothesis in zip(references, hypotheses):
                for type, score in scorer.score(reference, hypothesis).items():
                    values = scores.setdefault(type + "_p", [])
                    values.append(score.precision)
                    values = scores.setdefault(type + "_r", [])
                    values.append(score.recall)
                    values = scores.setdefault(type + "_f1", [])
                    values.append(score.fmeasure)

            result = {}
            for metric, values in scores.items():
                for k, v in statistics_for_collection(values).items():
                    result["{}_{}".format(metric, k)] = v
            return result
        try:
            from rouge_score.rouge_scorer import RougeScorer
            generation_metrics.append(compute_rouge)
        except Exception as ex:
            MODULE_LOGGER.warning("ROUGE-metric is unavailable.")
            MODULE_LOGGER.warning(repr(ex))

        def compute_bleu(hypotheses_sequences, references_sequences, smoothing=True, **bleu_options):
            if smoothing:
                chencherry = SmoothingFunction()
                bleu_options["smoothing_function"] = chencherry.method2  # ORANGE smoothing

            hypotheses = [[word_tokenize(sequence.lower()) for sequence in hypothesis] for hypothesis in hypotheses_sequences]
            references = [[[word_tokenize(sequence.lower())] for sequence in reference] for reference in references_sequences]

            metric_name = "bleu"
            scores = []
            for reference, hypothesis in zip(references, hypotheses):
                if len(reference) == len(hypothesis):
                    scores.append(corpus_bleu(reference, hypothesis, **bleu_options))
                else:
                    # NOTE: BLEU works only if the amount of sentences is equal;
                    # this is an approximation, but this probably actually makes BLEU unsuitable as a metric in our case.
                    hypothesis = [token for sequence in hypothesis for token in sequence]
                    reference = [token for sequence in reference for token in sequence[0]]
                    scores.append(sentence_bleu([reference], hypothesis, **bleu_options))
                    metric_name = "approx_bleu"

            result = {}
            for k, v in statistics_for_collection(scores).items():
                result["{}_{}".format(metric_name, k)] = v
            return result
        try:
            from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
            import nltk
            nltk.download("punkt")
            word_tokenize = nltk.tokenize.word_tokenize
            generation_metrics.append(compute_bleu)
        except Exception as ex:
            MODULE_LOGGER.warning("BLEU-metric is unavailable.")
            MODULE_LOGGER.warning(repr(ex))

        def compute_meteor(hypotheses_sequences, references_sequences, **meteor_options):
            hypotheses = [word_tokenize("\n".join(hypothesis)) for hypothesis in hypotheses_sequences]
            references = [word_tokenize("\n".join(reference)) for reference in references_sequences]

            scores = []
            for reference, hypothesis in zip(references, hypotheses):
                scores.append(single_meteor_score(reference, hypothesis))

            result = {}
            for k, v in statistics_for_collection(scores).items():
                result["meteor_{}".format(k)] = v
            return result
        try:
            from nltk.translate.meteor_score import single_meteor_score
            import nltk
            nltk.download("punkt")
            nltk.download("wordnet")
            nltk.download("omw-1.4")
            word_tokenize = nltk.tokenize.word_tokenize
            generation_metrics.append(compute_meteor)
        except Exception as ex:
            MODULE_LOGGER.warning("METEOR-score is unavailable.")
            MODULE_LOGGER.warning(repr(ex))

        def compute_metrics(eval_prediction):
            predictions, references = eval_prediction
            references = map(lambda token_ids: filter(lambda id: id != IGNORE_TOKEN_ID, token_ids), references)
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            # TODO decode_sequences is probably not sufficient for generated text, as models/decoders may not insert separators.
            decoded_predictions = map(lambda token_ids: decode_sequences(tokenizer, token_ids,
                                      sentence_separator=dataset_config.sentence_separator, skip_special_tokens=True), predictions)
            decoded_references = map(lambda token_ids: decode_sequences(tokenizer, token_ids,
                                     sentence_separator=dataset_config.sentence_separator, skip_special_tokens=True), references)
            decoded_predictions = tuple(decoded_predictions)
            decoded_references = tuple(decoded_references)

            write_lines_to(pipeline_config.output_predictions_path, decoded_predictions)

            result = {}
            for metric in generation_metrics:
                result.update(metric(decoded_predictions, decoded_references))
            return result

    if dataset_config.keep_empty_references == "none":
        for instance in batches_per_instance.keys():
            batches_per_instance[instance] = list(filter(lambda batch: len(batch["labels"]), batches_per_instance[instance]))

    all_batches = []
    train_dataset = []
    eval_dataset = []
    for instance, batches in batches_per_instance.items():
        category = dataset_config.train_category
        if not category or category in dataset_config.log_instances[instance]["category"]:
            train_dataset.extend(batches)
        category = dataset_config.eval_category
        if not category or category in dataset_config.log_instances[instance]["category"]:
            eval_dataset.extend(batches)
        all_batches.extend(batches)

    if dataset_config.keep_empty_references == "train_only":
        eval_dataset = list(filter(lambda batch: len(batch["labels"]), eval_dataset))

    decoded_inputs = map(lambda batch: decode_sequences(tokenizer, batch["input_ids"],
                                                        sentence_separator=dataset_config.sentence_separator, skip_special_tokens=True), all_batches)
    write_lines_to(pipeline_config.output_model_inputs_path, decoded_inputs)
    decoded_references = map(lambda batch: decode_sequences(tokenizer, batch["labels"],
                                                            sentence_separator=dataset_config.sentence_separator, skip_special_tokens=True), all_batches)
    write_lines_to(pipeline_config.output_references_path, decoded_references)

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length",
                                                        max_length=max_batch_length, label_pad_token_id=IGNORE_TOKEN_ID)

    trainer = transformers.Seq2SeqTrainer(model=model, tokenizer=tokenizer, args=training_args,
                                          data_collator=data_collator,
                                          compute_metrics=compute_metrics,
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
            trainer.save_state()  # save in checkpoint
            training_args.output_dir = original
        MODULE_LOGGER.info("Done saving model and state.")
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if training_args.do_eval:
        MODULE_LOGGER.info("Starting evaluation...")
        metrics = trainer.evaluate()
        MODULE_LOGGER.info("Done evaluating.")

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["eval_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def statistics_for_collection(values):
    if MATPLOTLIB_AVAILABLE:
        metrics = matplotlib.cbook.boxplot_stats(values)[0]
        stat_names = {"mean": "mean", "med": "median",
                      "q1": "lower_quartile", "q3": "higher_quartile",
                      "whislo": "lower_bound", "whishi": "higher_bound"}
        metrics = {stat_names[key]: value for key, value in metrics.items() if key in stat_names}
    else:
        metrics = {}
        metrics["mean"] = statistics.mean(values)
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        metrics["lower_quartile"], metrics["median"], metrics["higher_quartile"] = quartiles
        iqr = quartiles[2] - quartiles[0]
        bound_multiplier = 1.5
        metrics["lower_bound"] = min(filter(lambda x: x >= quartiles[0] - bound_multiplier * iqr, values))
        metrics["higher_bound"] = max(filter(lambda x: x <= quartiles[2] + bound_multiplier * iqr, values))
    metrics["amount_low_outliers"] = sum(map(lambda x: int(x < metrics["lower_bound"]), values))
    metrics["amount_high_outliers"] = sum(map(lambda x: int(x > metrics["higher_bound"]), values))
    metrics["min"] = min(values)
    metrics["max"] = max(values)
    metrics["std"] = statistics.pstdev(values)
    return metrics


@dataclasses.dataclass
class CustomTrainingArguments(transformers.Seq2SeqTrainingArguments):
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

    finetune_model(pipeline_config, dataset_config, training_args)
