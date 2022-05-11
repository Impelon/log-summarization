## Installation

1.  clone the repository via
    ```
    git clone https://github.com/Impelon/log-rca-summarization.git
    ```
2.  optionally use `virtualenv` to make a virtual environment for python-packages
3.  open `code`-folder in terminal
4.  make sure system dependencies are installed;
    the required packages can be found in `code/apt_requirements.txt` and `code/apt_optional_requirements.txt`;
    if you use the `apt` package manager you can directly install these via
    ```
    sudo apt-get update
    xargs -o sudo apt-get install < apt_requirements.txt
    xargs -o sudo apt-get install < apt_optional_requirements.txt
    ```
5.  install dependencies via
    ```
    pip3 install -r requirements.txt
    pip3 install -r optional_requirements.txt
    ```
6.  if you want to compare with [LogSummary](https://github.com/WeibinMeng/LogSummary),
    you also need to install it; you can initialize it here as a submodule:
    ```
    git submodule init
    git submodule update
    pip3 install -r LogSummary/requirements.txt
    ```
    The dependencies are not well kept for LogSummary, and `requirements.txt` misses some of them.
    Furthermore some code changes are necessary to include missing functions.
    For more details check [LogSummary's project page](https://github.com/WeibinMeng/LogSummary).
    Additionally a word2vec model is required, which can be trained for log-data with [their other framework Log2Vec](https://github.com/NetManAIOps/Log2Vec).
    *Note: Paths longer than 100 characters cause buffer overflows when training Log2Vec, if the limit is not changed manually.*
    (LogSummary is only used for the purpose of comparison and accessing their dataset; it is not needed otherwise.)

## Prepare datasets

0.  [install repository](#installation)
1.  download a suitable dataset e.g. Hadoop dataset from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3227177.svg)](https://doi.org/10.5281/zenodo.3227177)
2.  open `code`-folder in terminal
3.  run dataset pre-processing via
    ```
    python3 -m "preprocess_dataset" <dataset-type> "<original-dataset-path>" "<destination-path>"
    ```
    e.g.
    ```
    python3 -m "preprocess_dataset" hadoop "../data/Hadoop/raw" "../data/Hadoop/processed"
    ```
    This also works for preprocessing LogSummary's dataset:
    ```
    python3 -m "preprocess_dataset" logsummary "../LogSummary/data/summary/logs" "../data/LogSummary/processed"
    ```

## Analyze logs

0.  [install repository](#installation) and [prepare datasets](#prepare-datasets)
1.  open `code`-folder in terminal
2.  run log-analysis via
    ```
    python3 -m "util.loganalysis"
    ```
    e.g.
    ```
    python3 -m "util.loganalysis" ../data/Hadoop/processed --category "Disk full" --group-by-columns "File" --partition-minimum-size 200 --output-type common-events
    ```
3.  optionally prepare a preset for log-analysis, see the different configurations used in the  `code/log_analysis_configs`-folder;
    the analysis directly with such a configuration, e.g.
    ```
    python3 -m "util.loganalysis" ../data/Hadoop/processed @../data/Hadoop/loganalysis_configs/disk-full.json --output-type common-events
    ```

## Pre-train models

0.  [install repository](#installation) and [prepare datasets](#prepare-datasets)
1.  open `code`-folder in terminal
2.  run pre-training via
    ```
    python3 -m "pretrain_model"
    ```
    e.g.
    ```
    python3 -m "pretrain_model" --model_class "BartForConditionalGeneration" --model_name_or_path "facebook/bart-base" --masking_algorithm_preset "text-infilling" --csv_paths "../data/Hadoop/processed/log-instances"/* --input_field_name "SimplifiedMessage" --output_dir "../models/bart-base.hadoop/trained" --do_train
    ```
3.  optionally prepare a preset for pretraining-configuration, see `pretraining_config.py` used in the  `model`-folder;
    the pre-training can then be run directly with such a configuration, e.g.
    ```
    python3 -m "pretrain_model" @../models/bart-base.hadoop/pretraining_config.py --do_train
    ```
4.  under an environment with multiple GPUs you may want to use [DDP](https://huggingface.co/docs/transformers/performance#dp-vs-ddp) for training. (DDP referes to [PyTorch's DistributedDataParallel](https://pytorch.org/docs/stable/notes/ddp.html));
    in that case the pre-training should be started with `torchrun`
    ```
    torchrun --nproc_per_node <number_of_gpu_you_have> -m "pretrain_model" [arguments_for_pretrain_model]...
    ```
    For example like this:
    ```
    torchrun --nproc_per_node 2 -m "pretrain_model" @../models/bart-base.hadoop/pretraining_config.py --do_train
    ```

## Fine-tune models

0.  [install repository](#installation), [prepare datasets](#prepare-datasets) and optionally [pre-train the model](#pre-train-models)
1.  open `code`-folder in terminal
2.  run pre-training via
    ```
    python3 -m "finetune_model"
    ```
    e.g.
    ```
    python3 -m "finetune_model" --model_name_or_path "facebook/bart-base" --dataset_path "../data/Hadoop/processed" --loganalysis_arguments="@../data/Hadoop/loganalysis_configs/disk-full.json" --excluded_events_loganalysis_arguments="@../data/Hadoop/loganalysis_configs/normal.json" --input_field_name "SimplifiedMessage" --output_dir "../models/bart-base.hadoop/finetuned" --do_train
    ```
3.  optionally prepare a preset for finetuning-configuration, see `finetuning_config.py` used in the  `model`-folder;
    the pre-training can then be run directly with such a configuration, e.g.
    ```
    python3 -m "finetune_model" @../models/bart-base.hadoop/finetuning_config.py --do_train
    ```
4.  under an environment with multiple GPUs you may want to use [DDP](https://huggingface.co/docs/transformers/performance#dp-vs-ddp) for training. (DDP referes to [PyTorch's DistributedDataParallel](https://pytorch.org/docs/stable/notes/ddp.html));
    in that case the fine-tuning should be started with `torchrun`
    ```
    torchrun --nproc_per_node <number_of_gpu_you_have> -m "finetune_model" [arguments_for_finetune_model]...
    ```
    For example like this:
    ```
    torchrun --nproc_per_node 2 -m "finetune_model" @../models/bart-base.hadoop/finetuning_config.py --do_train
    ```

## Monitoring training

Given a suitable platform for monitoring of training is installed
that is [compatible with `🤗 Transformers`](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/callback),
the training-scripts will automatically create logs for those platforms.
The default location is at `runs` in the `output_dir`, but can be controlled with the `--logging_dir` option of training-scripts.

For example with [`tensorboard`](https://www.tensorflow.org/tensorboard) installed,
visualizing previous training-runs could look like this:
```
tensorboard --logdir models/bart-base.hadoop/trained/runs
```

## Use models

Trained models can comfortably be used for making predictions using [pipelines](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines).

To use a pre-trained model for mask-filling (the pipeline is limited to a mask-length of 1):
```python
>>> import transformers
>>> mask_filler = transformers.pipeline("fill-mask", "models/bart-base.hadoop/trained")
>>> sentence = "Scheduled snapshot <mask> at 10 second(s)."
>>> for prediction in mask_filler(sentence):
        print("{:6.2%} {}".format(prediction["score"], prediction["sequence"]))

 9.89% Scheduled snapshot count at 10 second(s).
 5.18% Scheduled snapshot snapshot at 10 second(s).
 4.78% Scheduled snapshot size at 10 second(s).
 2.88% Scheduled snapshot update at 10 second(s).
 2.74% Scheduled snapshot start at 10 second(s).
```

To use a fine-tuned model for summarization:
```python
>>> import transformers
>>> summarizer = transformers.pipeline("summarization", "models/bart-base.hadoop/finetuned")
>>> summary = summarizer("...")
```

## Run trace visualization-tool

0.  [install repository](#installation)
1.  open `code`-folder in terminal
2.  run via
    ```
    python3 -m "tracesviz"
    ```
2.  open any csv-file that contains structured log-data with traces;
    logs can be directly structured into csv via
    ```
    python3 -m "util.logparsing" structure
    ```
