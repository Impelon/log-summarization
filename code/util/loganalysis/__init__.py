from util.argument_parser_from_file import ArgumentParserFromFile

import re
import itertools
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import logging

MODULE_LOGGER = logging.getLogger(__name__)

MESSAGE_KEY = "Message"
TIMESTAMP_KEY = "Timestamp"
COMPONENT_KEY = "Component"
COMPONENT_EVENT_ID_KEY = "ComponentEventID"

TEMPLATE_KEY = "Template"


def get_event_id(entry):
    return (entry[COMPONENT_KEY], entry[COMPONENT_EVENT_ID_KEY])


def as_list(iterable):
    if isinstance(iterable, list):
        return iterable
    return list(iterable)


def as_datetime(object):
    if isinstance(object, datetime):
        return object
    return datetime.fromisoformat(object)


def as_timedelta(object):
    if isinstance(object, timedelta):
        return object
    if object is None or object == "None":
        return None
    return timedelta(milliseconds=int(object))


def in_time_window(entry, start=None, end=None):
    entry[TIMESTAMP_KEY] = as_datetime(entry[TIMESTAMP_KEY])
    if start and entry[TIMESTAMP_KEY] < start:
        return False
    if end and entry[TIMESTAMP_KEY] > end:
        return False
    return True


def as_time_window(base_time, time_range=None):
    if time_range is None:
        return None, None
    try:
        time_start, time_end = time_range
    except TypeError:
        time_start = -time_range
        time_end = time_range

    if time_start is not None:
        time_start = base_time + time_start
    if time_end is not None:
        time_end = base_time + time_end
    return time_start, time_end


def groupby(iterable, key=None):
    return itertools.groupby(sorted(iterable, key=key), key=key)


def partition_by_message_patterns(entries, patterns, partition_must_include_pattern=False):
    if not patterns:
        return [as_list(entries)]

    windows = []
    if not partition_must_include_pattern:
        windows.append([])
    for entry in entries:
        for pattern in patterns:
            if pattern.search(entry[MESSAGE_KEY]):
                windows.append([])
                break
        if windows:
            windows[-1].append(entry)
    return windows


def frame_around_message_patterns(entries, patterns, time_range=None):
    if not patterns:
        return [as_list(entries)]

    windows = []
    for entry in entries:
        for pattern in patterns:
            if pattern.search(entry[MESSAGE_KEY]):
                base_time = as_datetime(entry[TIMESTAMP_KEY])
                time_start, time_end = as_time_window(base_time, time_range)
                window = filter(lambda entry: in_time_window(entry, start=time_start, end=time_end), entries)
                windows.append(as_list(window))
                break
    return windows


def log_instances_with_categories(log_instances, *categories):
    """
    Return only log-instances with matching categories.

    Args:
        log_instances: A dict containing log-instances from a `log_instances.json`-file.
        categories: Any number of categories to look for.
            Each category is a collection of strings that all need to be present
            in the `category`-property of the log-instance, for them to match.
            If no categories are given, this will just return all log_instances.
            Otherwise, in case any given category matches,
            the respective instance will be included in the result.

    Returns:
        An dictionary containing only the matching log-instances.
    """
    if not categories:
        return log_instances
    filtered_instances = {}
    category_sets = []
    for category in categories:
        category_sets.append(set(category))
    for instance, properties in log_instances.items():
        for category_set in category_sets:
            if category_set.issubset(properties["category"]):
                filtered_instances[instance] = properties
    return filtered_instances


def load_log_instances(dataset_path):
    dataset_path = Path(dataset_path)
    with (dataset_path / "log_instances.json").open("r") as file:
        log_instances = json.load(file)
    return log_instances


def load_log_windows(dataset_path, categories, **windowing_options):
    dataset_path = Path(dataset_path)
    log_instances = load_log_instances(dataset_path)

    relevant_instances = log_instances_with_categories(log_instances, *categories)

    logs_path = dataset_path / "log_instances"
    return load_log_windows_from_instances(logs_path, relevant_instances.items(), **windowing_options)


def load_log_windows_from_instances(logs_path, instances_items, time_property=None, time_range=None, group_by_columns=None, partition_patterns=None, partition_must_include_pattern=False, partition_minimum_size=None, frame_patterns=None, frame_time_range=None):
    logs_path = Path(logs_path)
    if time_range is None:
        time_range = timedelta(seconds=1)
    log_windows_per_instance = {}
    for instance, properties in instances_items:
        with (logs_path / instance).with_suffix(".csv").open("r") as file:
            entries = csv.DictReader(file)
            if time_property:
                base_time = as_datetime(properties[time_property])
                time_start, time_end = as_time_window(base_time, time_range)
                entries = filter(lambda entry: in_time_window(entry, start=time_start, end=time_end), entries)
            if group_by_columns:
                groups = groupby(entries, lambda x: tuple(x[c] for c in group_by_columns))
            else:
                groups = [(None, entries)]
            partitions = []
            for key, group_entries in groups:
                partitions.extend(partition_by_message_patterns(group_entries, partition_patterns, partition_must_include_pattern))
            if partition_minimum_size:
                partitions = filter(lambda x: len(x) >= partition_minimum_size, partitions)
        log_windows_per_instance[instance] = []
        for window in partitions:
            framed_windows = frame_around_message_patterns(window, frame_patterns, frame_time_range)
            log_windows_per_instance[instance].extend(filter(None, framed_windows))
    return log_windows_per_instance


def compute_common_events(log_windows):
    common_event_ids = set()
    for log_entries in log_windows:
        if common_event_ids:
            common_event_ids.intersection_update(map(get_event_id, log_entries))
        else:
            common_event_ids.update(map(get_event_id, log_entries))
        MODULE_LOGGER.debug("Common event IDs so far: %d", len(common_event_ids))
    return common_event_ids


def compute_cumulated_events(log_windows):
    cumulated_event_ids = set()
    for log_entries in log_windows:
        cumulated_event_ids.update(map(get_event_id, log_entries))
        MODULE_LOGGER.debug("Cumulated event IDs so far: %d", len(cumulated_event_ids))
    return cumulated_event_ids


def get_options_parser():
    parser = ArgumentParserFromFile(fromfile_prefix_chars="@",
                                    epilog="Arguments prefixed with '@' will be interpreted as file-locations to parse additional arguments from. Can be either a python or JSON file.")

    def add_time_range(name, dest):
        group = parser.add_mutually_exclusive_group()
        name = "--" + name
        group.add_argument(name + "-tolerance-ms", dest=dest, type=as_timedelta)
        group.add_argument(name + "-range-ms", dest=dest, nargs=2, type=as_timedelta)

    parser.add_argument("--category", dest="categories", action="append", nargs="*", default=[])
    parser.add_argument("--time-property")
    add_time_range("time", "time_range")
    parser.add_argument("--group-by-columns", nargs="*")
    parser.add_argument("--partition-patterns", nargs="*", type=re.compile)
    parser.add_argument("--partition-must-include-pattern", action="store_true")
    parser.add_argument("--partition-minimum-size", type=int)
    parser.add_argument("--frame-patterns", nargs="*", type=re.compile)
    add_time_range("frame-time", "frame_time_range")
    return parser
