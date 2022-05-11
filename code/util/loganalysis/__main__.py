from . import *

import sys
import json
import csv
from pathlib import Path
import logging

try:
    from tqdm import tqdm
except:
    def tqdm(iterable, **kwargs):
        return iterable


def write_csv_events(file, events, event_ids):
    events = list(filter(lambda x: get_event_id(x) in event_ids, events))
    return write_csv_entries(file, events)


def write_csv_entries(file, entries):
    headers = next(iter(entries)).keys()
    csvwriter = csv.DictWriter(file, headers)
    csvwriter.writeheader()
    for entry in entries:
        csvwriter.writerow(entry)


def main(dataset_path, categories, output_window_stats=False, output_windowed_entries=False, output_cumulated_events=False, output_common_events=False, output_common_entries=False, **windowing_options):
    dataset_path = Path(dataset_path)
    log_instances = load_log_instances(dataset_path)

    relevant_instances = log_instances_with_categories(log_instances, *categories)
    MODULE_LOGGER.debug("The following instances fit the given categories: %s", relevant_instances.keys())

    MODULE_LOGGER.info("Reading and windowing entries from all relevant log instances...")
    logs_path = dataset_path / "log_instances"
    log_windows_per_instance = load_log_windows_from_instances(logs_path, tqdm(relevant_instances.items()), **windowing_options)
    MODULE_LOGGER.info("Done reading log instances.")

    if output_window_stats:
        stats_per_instance = {}
        for instance, windows in log_windows_per_instance.items():
            stats_per_instance[instance] = []
            for window in windows:
                duration = as_datetime(window[-1][TIMESTAMP_KEY]) - as_datetime(window[0][TIMESTAMP_KEY])
                stats_per_instance[instance].append({"length": len(window), "duration": duration})
        print(json.dumps(stats_per_instance, indent=4, sort_keys=True, default=lambda x: str(x)))

    log_windows = []
    for windows in log_windows_per_instance.values():
        log_windows.extend(windows)
    MODULE_LOGGER.debug("The following windows were collected: %s", list(map(len, log_windows)))

    if output_windowed_entries:
        MODULE_LOGGER.info("Writing windowed entries.")
        for log_entries in tqdm(log_windows):
            write_csv_entries(sys.stdout, log_entries)
            print()  # separate each window
        MODULE_LOGGER.info("Done writing windowed entries.")

    if output_cumulated_events:
        MODULE_LOGGER.info("Computing cumulated events.")
        cumulated_event_ids = compute_cumulated_events(log_windows)
        MODULE_LOGGER.info("Done computing cumulated events.")

        MODULE_LOGGER.info("Writing cumulated events.")
        with (dataset_path / "event_types.csv").open("r") as file:
            csvreader = csv.DictReader(file)
            write_csv_events(sys.stdout, csvreader, cumulated_event_ids)
        MODULE_LOGGER.info("Done writing cumulated events.")

    if not (output_common_events or output_common_entries):
        return

    MODULE_LOGGER.info("Computing common events.")
    common_event_ids = compute_common_events(log_windows)
    MODULE_LOGGER.info("Done computing common events.")

    if output_common_events:
        MODULE_LOGGER.info("Writing common events.")
        with (dataset_path / "event_types.csv").open("r") as file:
            csvreader = csv.DictReader(file)
            write_csv_events(sys.stdout, csvreader, common_event_ids)
        MODULE_LOGGER.info("Done writing common events.")

    if output_common_entries:
        MODULE_LOGGER.info("Writing common log entries.")
        for log_entries in tqdm(log_windows):
            log_entries = list(filter(lambda x: get_event_id(x) in common_event_ids, log_entries))
            write_csv_entries(sys.stdout, log_entries)
            print()  # separate each window
        MODULE_LOGGER.info("Done writing common log entries.")


if __name__ == "__main__":
    parser = get_options_parser()

    output_types = {
        "window-stats": "output_window_stats",
        "windowed-entries": "output_windowed_entries",
        "cumulated-events": "output_cumulated_events",
        "common-events": "output_common_events",
        "common-entries": "output_common_entries",
    }

    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--output-type", choices=output_types.keys(), default="common-entries")
    parser.add_argument("--log-level", nargs="?", type=str.upper, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")

    options = vars(parser.parse_args())
    output_type = options.pop("output_type")
    options[output_types[output_type]] = True

    log_level = options.pop("log_level")
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%x %X")
    MODULE_LOGGER.setLevel(log_level)
    main(**options)
