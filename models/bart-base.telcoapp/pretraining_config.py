from pathlib import Path

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

options = {
  "model_class": "BartForConditionalGeneration",
  "masking_algorithm_preset": "text-infilling",
  "model_name_or_path": "facebook/bart-base",
  "use_fast_tokenizer": False, # this fast tokenizer cannot recognize masks as special-tokens
  "output_dir": (Path(__file__).parent / "trained").resolve(),
  # dataset
  "csv_paths": list((Path(__file__).parent / ".." / ".." / "data" / "TelcoApp" / "processed" / "with_RCA" / "log_instances").glob("*.csv")),
  "input_field_name": "SimplifiedMessage",
  # training options
  "gradient_checkpointing": True, # otherwise huge amounts of memory are used
  "fp16": True,
  "total_train_batch_size": 8192,
  "per_device_train_batch_size": 8,
  "sentence_separator": ";",
  "sentences_per_window": 250,
  "window_stride": 250,
  "logging_steps": 1,
  "save_steps": 10,
  "save_total_limit": 10,
}

checkpoints = sorted(options["output_dir"].glob(PREFIX_CHECKPOINT_DIR + "*"), key=lambda x: int(str(x).rsplit("-", 1)[1]))
if checkpoints:
    options["model_name_or_path"] = checkpoints[-1]
    options["resume_from_checkpoint"] = checkpoints[-1]
else:
    options["overwrite_output_dir"] = True
