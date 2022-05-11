import os
from collections import OrderedDict
from pathlib import Path

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

base_config = {
    "max_position_embeddings": 1024,
    "max_length": 640,
    "length_penalty": 0.9,
}

models = OrderedDict([
    ("Large", ("google", "pegasus-large", {**base_config, **{"length_penalty": 0.8}})),
    ("CNN/DailyMail", ("google", "pegasus-cnn_dailymail", base_config)),
    ("XSum", ("google", "pegasus-xsum", base_config)),
    ("AESLC", ("google", "pegasus-aeslc", base_config)),
    ("BigPatent", ("google", "pegasus-big_patent", base_config)),
])

name = os.environ.get("FINETUNING_MODEL", None)
if name not in models:
    print("Please set environment variable FINETUNING_MODEL to one of the following choices:")
    print("\n".join(map(lambda x: " * '{}'".format(x), models.keys())))
    exit(1)
model_spec = models[name]

base_data_path = (Path(__file__).parent / ".." / ".." / "data" / "TelcoApp").resolve()

options = {
  "model_name_or_path": "/".join(model_spec[:2]),
  "use_fast_tokenizer": True,
  "output_dir": (Path(__file__).parent / ("finetuned." + model_spec[1])).resolve(),
  # dataset
  "dataset_path": base_data_path / "processed" / "with_RCA",
  "only_include_common_events": True,
  "input_field_name": "SimplifiedMessage",
  # generation
  "config_overrides": ",".join("{}={}".format(k, v) for k, v in model_spec[2].items()),
  "sentence_separator": "<n>",
  "generation_num_beams": 8,
  "predict_with_generate": True,
  # training options
  "num_train_epochs": 20,
  "lr_scheduler_type": "constant",
  "label_smoothing_factor": 0.1,
  "gradient_checkpointing": True, # otherwise huge amounts of memory are used
  "total_train_batch_size": 8,
  "per_device_train_batch_size": 2,
  "per_device_eval_batch_size": 1,
  "evaluation_strategy": "steps",
}

curated_path = base_data_path / "curated"
supervision = os.environ.get("SUPERVISION", "curated").lower()
if supervision == "full":
    pass
elif supervision == "curated" and curated_path.exists():
    options["dataset_path"] = curated_path
    options["train_category"] = "train"
    options["eval_category"] = "eval"
    options["summary_instance_property"] = "summary"
    options["loganalysis_arguments"] = "" # load log-instances as is
    options["only_include_common_events"] = False
else:
    print("Please set environment variable SUPERVISION to one of the following choices:")
    print(" * 'full'")
    print(" * 'curated' (Make sure '{}' points to a dataset with train/eval instances.)".format(curated_path))
    exit(1)

is_zero_shot = os.environ.get("ZERO_SHOT", False)
if is_zero_shot:
    options["output_dir"] = Path(str(options["output_dir"]) + "-zsl")

options["output_predictions_path"] = options["output_dir"] / "model_predictions.txt"

checkpoints = sorted(options["output_dir"].glob(PREFIX_CHECKPOINT_DIR + "*"), key=lambda x: int(str(x).rsplit("-", 1)[1]))
if checkpoints and not is_zero_shot:
    options["model_name_or_path"] = checkpoints[-1]
    options["resume_from_checkpoint"] = checkpoints[-1]
else:
    options["overwrite_output_dir"] = True

if is_zero_shot:
    options["do_eval"] = True
