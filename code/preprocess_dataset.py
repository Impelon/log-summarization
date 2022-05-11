import sys
import re
import json
import csv
import itertools
from collections import OrderedDict, Counter
from pathlib import Path
from datetime import datetime
import contextlib
import logging

from util.heap import Heap
from util.logparsing import structuring
from util.logparsing import abstraction

try:
    from tqdm import tqdm
except:
    def tqdm(iterable, **kwargs):
        return iterable

MODULE_LOGGER = logging.getLogger(__name__)

MESSAGE_COLUMN = "Message"
COMPONENT_COLUMN = "Component"
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"
TEMPLATE_COLUMN = "Template"
TIMESTAMP_KEY = "Timestamp"
LOGNAME_KEY = "File"
EVENT_ID_KEY = "ComponentEventID"
PARAMETERS_KEY = "Parameters"
PARAMETERS_WITH_TYPE_KEY = "PARAMETERS_WITH_TYPE"
SIMPLIFIED_MESSAGE_KEY = "SimplifiedMessage"
CATEGORY_KEY = "category"

PATTERNS = {
    "rtsp_url": r"rtsp://[\w.:@-]*(?:/[\w.-=]+)+",
    "ipv6_address_with_port": r"\[?(?:^|(?<=\W))(?:[\da-fA-F]{1,4}:|:){4,}([\da-fA-F]{1,4}|:)(?:\]|(?=\W)|$)(?::\d{1,5})?",
    "ipv6_address_with_ipv4_with_port": r"\[?(?:^|(?<=\W))(?:[\da-fA-F]{1,4}:|:){2,}(?:\d{1,3}\.){3}\d{1,3}(?:\]|(?=\W)|$)(?::\d{1,5})?",
    "long_numerical_list": r"(?:^|(?<=\W))(?:[-+]?[\dxa-fA-F.]+[ ,;:|/#*-]+){5,}[-+]?[\dxa-fA-F.]+\b",
    "number_with_unit": r"(?<=[<>(\[/.\x27,:=\s]){}(?:[numcdkKMGT]?(?:s|m|g|Ah|A|K|°C|Hz|N|J|W|V|B(?:ytes?)?|b(?:its?)?)(?:[/p]s)?)\b".format(abstraction.PATTERNS["number"].rstrip(r"\b")),
}


def get_mask(key):
    return (key, PATTERNS[key])


DATASET_PARAMETERS = {
    "hadoop": {
        "log_format": structuring.LOG_FORMATS["hadoop"],
        "date_format": "%Y-%m-%d",
        "time_format": "%H:%M:%S,%f",
        "parser_settings": abstraction.PARSER_SETTINGS["hadoop"],
        "patterns_to_replace": {
            MESSAGE_COLUMN: {
                "join_hdfs_path": (re.compile(r"({})\s({})".format(abstraction.PATTERNS["hdfs_url"], abstraction.PATTERNS["linux_path"])), r"\1\2"),
                "thread_dump": (re.compile(r"(thread dump.*\d+ active threads).*$", flags=re.IGNORECASE), r"\1"),
                "msra_sa": (re.compile(r"(MSRA-SA-\d+)(?:\.\w+)+"), r"\1"),
                "java_stacktrace": re.compile(r"[ \t]*at\s+" + abstraction.PATTERNS["java_stackentry"]),
                "java_stacktrace_ellipsis": re.compile(r"[ \t]*...\s+\d+\s+more"),
            },
            COMPONENT_COLUMN: {
                "shorten_package": (re.compile(r"^([\w$]+\.)+"), lambda match: "#{:05x}.{}".format(hash(match.group(0)) % 0x100000, match.group(1))),
            },
        },
        "line_splitting_patterns": [],
        "open_log_kwargs": {},
    },
    "logsummary_bgl": {
        "log_format": "<Type> <Component> <Level> <Message>",
        "parser_settings": abstraction.PARSER_SETTINGS["bgl"],
        "patterns_to_replace": {
            MESSAGE_COLUMN: {
                "untokenize_ido_chip": (re.compile(r"([\da-fA-F]{2} : ){11}([\da-fA-F]{2})"), lambda match: match.group(0).replace(" : ", "")),
                "untokenize_ipv4_with_port": (re.compile(r"((?:\d{1,3}\.){3}\d{1,3}) : (\d{1,5})"), r"\1:\2"),
            },
        },
        "line_splitting_patterns": [],
        "open_log_kwargs": {},
    },
    "logsummary_hdfs": {
        "log_format": "<Level> <Component>: <Message>",
        "parser_settings": abstraction.PARSER_SETTINGS["hdfs"],
        "patterns_to_replace": {
            MESSAGE_COLUMN: {
                "untokenize_ipv4_with_port": (re.compile(r"((?:\d{1,3}\.){3}\d{1,3}) : (\d{1,5})"), r"\1:\2"),
            },
        },
        "line_splitting_patterns": [],
        "open_log_kwargs": {},
    },
    "logsummary_hpc": {
        "log_format": "<Component> <State> <Time> <Flag> <Message>",
        "parser_settings": abstraction.PARSER_SETTINGS["hpc"],
        "patterns_to_replace": {},
        "line_splitting_patterns": [],
        "open_log_kwargs": {},
    },
    "logsummary_proxifier": {
        "log_format": r"\S*\x20?- <Component> : <Port> <Message>",
        "parser_settings": abstraction.PARSER_SETTINGS["proxifier"],
        "patterns_to_replace": {
            MESSAGE_COLUMN: {
                "untokenize_lifetime": (re.compile(r"\b\d{2} : \d{2}( : \d{2})*\b"), lambda match: match.group(0).replace(" : ", ":")),
            },
        },
        "line_splitting_patterns": [],
        "open_log_kwargs": {},
    },
    "logsummary_spark": {
        "log_format": r"(?:(?!Update row : \d+)<Level> <Component>: )?<Message>",
        "parser_settings": abstraction.PARSER_SETTINGS["spark"],
        "patterns_to_replace": {
            MESSAGE_COLUMN: OrderedDict([
                ("untokenize_ipv4_with_port", (re.compile(r"((?:\d{1,3}\.){3}\d{1,3}) : (\d{1,5})"), r"\1:\2")),
                ("untokenize_hdfs_url", (re.compile(r"hdfs : //([\w.:@-]*(?:(?:/[\w.~%+-]+)+/?)?)"), r"hdfs://\1")),
            ]),
        },
        "line_splitting_patterns": [],
        "open_log_kwargs": {},
    },
    "logsummary_zookeeper": {
        "log_format": r"(?:.*? .*? - )?<Level|INFO|WARN|ERROR> (?:\[.*?\] - )?<Message>",
        "parser_settings": abstraction.PARSER_SETTINGS["zookeeper"],
        "patterns_to_replace": {
            MESSAGE_COLUMN: {
                "untokenize_ipv4_with_port": (re.compile(r"((?:\d{1,3}\.){3}\d{1,3}) : (\d{1,5})"), r"\1:\2"),
            },
        },
        "line_splitting_patterns": [],
        "open_log_kwargs": {},
    },
    "telcoapp": {
        "log_format": structuring.LOG_FORMATS["android"],
        "date_format": "%m-%d",
        "time_format": "%H:%M:%S.%f",
        "parser_settings": abstraction.PARSER_SETTINGS["android"],
        "patterns_to_replace": {
            MESSAGE_COLUMN: OrderedDict([
                ("custom_field_unknown_encoding", (re.compile(r"{(.*)(?:,\s+)customField=.*$"), r"{\1}")),
                ("trace", (re.compile(r"(^(?:[\w$@<>.]*: )?)\[(?:[0-9a-fA-F]+, ?){2}[0-9a-fA-F]+\]"), r"\1")),
                ("java_file_marker", re.compile(r"\[ (?:\w+\.java|null): \d+\]")),
                ("java_component_log_marker", re.compile(r"\[(?:[^ ]+ [^ ]+ )?[DIWEF](?:/[^/ ]+)+ \d+:\d+ (?:[\w$<>.]+)?:\d+\]")),
                ("cpp_log_marker", re.compile(r"\[(?:\d+/\d+\.\d+:)?(?:VERBOSE\d+|DEBUG|INFO|WARNING|ERROR|FATAL)[^\]]*:[^\]]+\(\d+\)\]")),
                ("hcdn_log_marker", (structuring.construct_log_format_regex(r"HCDN_LOG: \[<Date> <Time>\]\[T:<TID>\]<Message>")[1], r"\g<Message>")),
                ("teeos_log_marker", (structuring.construct_log_format_regex(r"sn: <LineNumber>: <CPU|cpu\d+>/<PID>:(?: <Date|\d\d/\d\d> <Time>)? \[<Component>\](?: \[<Level>\](?:\[<MessageID|\d+>\])?)? <Message>")[1], r"\g<Message>")),
                ("instance_parameters", re.compile(r"(?<=\w)[{][^{}]*([{][^{}]*([{][^{}]*([{][^{}]*([{][^{}]*([{][^{}]*[}])?[}])?[}])?[}])?[}])?[}]")),
                ("java_stacktrace", re.compile(r"^\s*at\s+(?:[^.\s]+\.)+[^.\s]+.*$")),
                ("java_stacktrace_ellipsis", re.compile(r"^\s*...\s+\d+\s+more")),
            ]),
            COMPONENT_COLUMN: OrderedDict([
                ("data_server", re.compile(r"DataSrv-\d+\.\d+\.\d+\.\d+-.-\d+-\d+:\d+-")),
                ("ipv4_address", (re.compile(r"[\[_-]\d+\.\d+\.\d+\.\d+[_\]-]"), ".")),
                ("ipv4_address_suffix", re.compile(r"-\d+\.\d+\.\d+\.\d+$")),
                ("hex_prefix", re.compile(r"^[0-9a-fA-F]+/")),
                ("minus_prefix", re.compile(r"^-")),
                ("instance_marker", (re.compile(r"[@:]\d{4,}]"), "]")),
                ("gameplugin", (re.compile(r"\.?gameplugin\d?\.P\d+"), "gameplugin")),
                ("ap_with_feedback", (re.compile(r"^APwFeedback\d+$"), "APwFeedback")),
                ("iconnect_machine", (re.compile(r"^iconnect:(.*)machine.*$"), r"iconnect:\1machine")),
                ("network_speed_manager", (re.compile(r"^Net\.\d+.*$"), "NetworkSpeedManager")),
                ("view_root_impl", (re.compile(r"^ViewRootImpl\[.*\]$"), "ViewRootImpl[Component]")),
                ("state_machine", (re.compile(r"^HiSight-\d+-SM$"), "HiSight-StateMachine")),
                ("handoff", (re.compile(r"^(handoff?)(?:(?:[ a-f0-9-]+)|(?:[ -]+[^a-f]))$", flags=re.IGNORECASE), r"\1")),
                ("instance_suffix", re.compile(r"[/:#-][\d_]+$")),
                ("id_suffix", re.compile(r"_[a-fA-F\d]+$")),
                ("number_suffix", re.compile(r"\d{5,}$")),
            ]),
        },
        "line_splitting_patterns": [
            re.compile(r"(One line match string:)"),
        ],
        "open_log_kwargs": {"errors": "replace"},
    },
}

DATASET_PARAMETERS["telcoapp"]["parser_settings"]["masking_patterns"] = [
    get_mask("rtsp_url"),
    abstraction.get_default_mask("http_url"),
    abstraction.get_default_mask("file_url"),
    ("content_or_condition_uri", r"(?:content|condition)://(?:[\w.*~%@-]+:?)?(?:/[\w.*~%?&=-]+)*/?"),
    ("tcp_url", r"tcp://" + abstraction.PATTERNS["ipv4_address_with_port"]),
    abstraction.get_default_mask("linux_path"),
    abstraction.get_default_mask("java_identifier"),
    get_mask("ipv6_address_with_ipv4_with_port"),
    get_mask("ipv6_address_with_port"),
    abstraction.get_default_mask("mac_address"),
    get_mask("long_numerical_list"),
    abstraction.get_default_mask("hex_number"),
    abstraction.get_default_mask("hex_identifier"),
    get_mask("number_with_unit"),
    abstraction.get_default_mask("number"),
    abstraction.get_default_mask("hex_segment"),
]


def parse_hadoop_category_file(lines):
    application = None
    failure_type = None
    instances = {}
    for line in lines:
        line = line.strip()
        if line.startswith("###"):
            application = line[3:].strip()
        elif line.endswith(":"):
            failure_type = line[:-1].strip()
        elif line.startswith("+ ") and application and failure_type:
            instance = line[2:].strip()
            instances[instance] = {CATEGORY_KEY: (failure_type, application)}

    return instances


def parse_logsummary_file(lines, type):
    class MockedPath:
        def __init__(self, content):
            self.content = tuple(content)
            self.name = "{}.txt".format(type)

        @contextlib.contextmanager
        def open(self, *args, **kwargs):
            yield self.content
    instance = None
    next_is_summary = False
    log = []
    instances = {}
    categorized_logs = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line == "#summary:#":
                next_is_summary = True
            else:
                instance = "{}_{}".format(type, len(instances))
        elif not next_is_summary:
            log.append(line)
        else:
            path = MockedPath(log)
            log.clear()
            next_is_summary = False
            instances[instance] = {CATEGORY_KEY: ("logsummary_{}".format(type),), "summary": tuple(map(lambda x: x.strip(), line.strip().strip(";").split(";")))}
            categorized_logs[path] = instance

    return instances, categorized_logs


def parse_telcoapp_diagnosis_file(lines):
    lines = tuple(filter(None, map(lambda x: x.strip(), lines)))
    fault_time = lines[0]
    analysis = None
    for i, line in enumerate(lines):
        if line.lower().startswith("fault time:"):
            fault_time = line.split(":", 1)[1].strip()
        if line.lower().startswith("root cause:"):
            root_cause = line.split(":", 1)[1].strip()
            if not root_cause and i + 1 < len(lines):
                root_cause = lines[i + 1]
            if root_cause:
                analysis = root_cause
    try:
        fault_time = datetime.strptime(fault_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        fault_time = None

    return fault_time, analysis


def gather_categorized_logs(type, dataset_path):
    dataset_path = dataset_path.resolve()
    if type == "hadoop":
        with (dataset_path / "abnormal_label.txt").open("r") as category_file:
            instances = parse_hadoop_category_file(category_file)
        log_files = dataset_path.glob("*/*.log")
        categorized_logs = {}
        for path in log_files:
            instance = path.relative_to(dataset_path).parent.name
            categorized_logs[path] = instance
    elif type.startswith("logsummary"):
        log_files = dataset_path.glob("*.txt")
        instances = {}
        categorized_logs = {}
        for path in log_files:
            with path.open("r") as summary_file:
                file_instances, file_categorized_logs = parse_logsummary_file(summary_file, path.stem.lower())
                instances.update(file_instances)
                categorized_logs.update(file_categorized_logs)
    else:
        log_files = dataset_path.rglob("hiapplogcat*")
        instances = {}
        categorized_logs = {}
        for path in log_files:
            parents = tuple(path.relative_to(dataset_path).parents)
            anomaly_type = parents[-2].name
            root_cause = parents[-3].name
            instance = parents[-4].name
            if not instance.isdigit():
                instance = "1"
            instance = "{}_{}_{}".format(re.search(r"\d+", anomaly_type).group(0), root_cause, instance)
            instances[instance] = {CATEGORY_KEY: (anomaly_type, root_cause)}
            categorized_logs[path] = instance
            with next(path.parents[2].glob("[rR][dD].txt")).open("r") as diagnosis_file:
                fault_time, analysis = parse_telcoapp_diagnosis_file(diagnosis_file)
            if not analysis:
                with next(path.parents[3].glob("[rR][dD].txt")).open("r") as diagnosis_file:
                    _, analysis = parse_telcoapp_diagnosis_file(diagnosis_file)
            instances[instance]["fault_time"] = fault_time
            instances[instance]["analysis"] = analysis
    return instances, categorized_logs


def preprocess_entry(type, entry):
    for column, patterns in DATASET_PARAMETERS[type]["patterns_to_replace"].items():
        for pattern in patterns.values():
            replacement = ""
            try:
                pattern, replacement = pattern
            except:
                pass
            entry[column] = pattern.sub(replacement, entry[column]).strip()

    date = entry.pop(DATE_COLUMN, None)
    time = entry.pop(TIME_COLUMN, None)
    if date and time:
        entry[TIMESTAMP_KEY] = datetime.strptime(date + " " + time,
                                                 DATASET_PARAMETERS[type]["date_format"] + " " + DATASET_PARAMETERS[type]["time_format"])
    return entry

def simplify_parameters(parameters_with_type):
    for i, (value, type) in enumerate(parameters_with_type):
        replacement = ""
        if type == "java_identifier":
            elements = value.split(".")
            word_match = None
            while not word_match:
                word_match = re.match(r"\w+", elements.pop())
            replacement = word_match.group(0)
        elif type.endswith("_path"):
            last_word_match = re.search(r"\w+", value[::-1])
            if last_word_match:
                replacement = last_word_match.group(0)[::-1] + "-path"
            else:
                replacement = "path"
        elif type.endswith(("_uri", "_url")):
            replacement = type[:-4].lower() + "-" + type[-3:].upper()
        elif type.endswith("_id"):
            replacement = "{}#{}".format(type[:-3], hash(value) % 100000)
        elif type.startswith(("ipv4_address", "ipv6_address", "mac_address")):
            replacement = "remote host #{}".format(hash(value) % 100000)
        elif type == "long_numerical_list":
            replacement = ", ".join(x for x, count in Counter(re.findall(r"[-+]?[\dxa-fA-F.]+", value)).most_common(3))
        elif type == "hex_number":
            replacement = int(value[2:], 16)
        elif type == "hex_segment":
            replacement = hex(int(value, 16) % 100000)[2:]
        elif type == "hex_identifier":
            replacement = hex(hash(value) % 100000)[2:]
        elif type.startswith("number"):
            replacement = value
        else:
            try:
                replacement = int(value)
            except ValueError:
                try:
                    replacement = float(value)
                except ValueError:
                    words = re.findall(r"[a-zA-Z]+[:=.?!]?", value)
                    if words:
                        replacement = max(words, key=len)
        parameters_with_type[i] = replacement

def do_additional_line_splitting(type, line_iterator):
    for line in line_iterator:
        additional_lines = (line,)
        for pattern in DATASET_PARAMETERS[type]["line_splitting_patterns"]:
            additional_lines = tuple("".join(l) for x in additional_lines for l in grouper(pattern.split(x), pattern.groups + 1, ""))
        for line in additional_lines:
            yield line + "\n"


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def groupby(iterable, key=None):
    return itertools.groupby(sorted(iterable, key=key), key=key)


def preprocess_dataset(type, dataset_path, destination_path, exist_ok=False):
    current_type = type
    dataset_path = Path(dataset_path)
    destination_path = Path(destination_path)

    log_instances, categorized_logs = gather_categorized_logs(type, dataset_path)

    MODULE_LOGGER.info("Writing meta-information about log instances to destination...")
    destination_path.mkdir(parents=True, exist_ok=exist_ok)
    with (destination_path / "log_instances.json").open("w") as file:
        json.dump(log_instances, file, indent=4, sort_keys=True, default=lambda x: str(x))
    MODULE_LOGGER.info("Done writing about log instances.")

    MODULE_LOGGER.info("Structuring logs into log entries...")
    entries_by_instance = {}
    for path, instance in tqdm(categorized_logs.items()):
        if type == "logsummary":
            current_type = log_instances[instance][CATEGORY_KEY][0]
        with path.open("r", **DATASET_PARAMETERS[current_type]["open_log_kwargs"]) as file:
            entries_by_instance.setdefault(instance, Heap(key=lambda x: x[TIMESTAMP_KEY]))
            entries = entries_by_instance[instance]
            last_inserted = []

            if current_type == "hadoop":
                appending_handler = structuring.create_appending_exception_handler(MESSAGE_COLUMN, " ", previous_log_entries=last_inserted)
                def handler(exception):
                    appending_handler(exception)
                    preprocess_entry(current_type, last_inserted[-1])
            else:
                def handler(exception):
                    if not isinstance(exception, structuring.ImproperlyFormattedLogLineError):
                        raise exception
                    MODULE_LOGGER.warning("Discarding invalid log line:")
                    MODULE_LOGGER.warning(exception)

            line_iterator = do_additional_line_splitting(current_type, file)
            entry_iterator = structuring.lines_to_log_entries(line_iterator, DATASET_PARAMETERS[current_type]["log_format"])
            for entry in structuring.iterate_with_exception_handler(entry_iterator, handler):
                preprocess_entry(current_type, entry)
                if entry[MESSAGE_COLUMN]:
                    if current_type == "telcoapp":
                        entry[TIMESTAMP_KEY] = entry[TIMESTAMP_KEY].replace(year=log_instances[instance]["fault_time"].year)
                    elif current_type.startswith("logsummary"):
                        if not entry.get(COMPONENT_COLUMN):
                            entry[COMPONENT_COLUMN] = ""
                        entry.setdefault(TIMESTAMP_KEY, "")
                    entry[LOGNAME_KEY] = path.name
                    entries.push(entry)
                    last_inserted.clear()
                    last_inserted.append(entry)
    MODULE_LOGGER.info("Done structuring logs.")

    MODULE_LOGGER.info("Parsing log messages into log events...")
    event_types = {}
    if type == "logsummary":
        for instance, entries in entries_by_instance.items():
            for entry in entries:
                entry[COMPONENT_COLUMN] = "{}:{}".format(log_instances[instance][CATEGORY_KEY][0], entry[COMPONENT_COLUMN])
    grouped_by_component = groupby(itertools.chain.from_iterable(entries_by_instance.values()), key=lambda x: x[COMPONENT_COLUMN])
    for component, group in tqdm(grouped_by_component):
        if type == "logsummary":
            current_type = component.split(":", 1)[0]
        configuration = abstraction.apply_settings(**DATASET_PARAMETERS[current_type]["parser_settings"])
        group = tuple(group)
        messages = tuple(x[MESSAGE_COLUMN] for x in group)
        event_instances, component_event_types = abstraction.extract_events_from_log_messages(messages, configuration, include_parameter_masks=True)
        for entry, (id, parameters_with_type) in zip(group, event_instances):
            entry[EVENT_ID_KEY] = id
            entry[PARAMETERS_WITH_TYPE_KEY] = parameters_with_type
            entry[PARAMETERS_KEY] = [parameter for parameter, type in parameters_with_type]
            event_types.setdefault((component, id), component_event_types[id])
    MODULE_LOGGER.info("Done parsing log events.")

    MODULE_LOGGER.info("Simplifing log messages...")
    total_entries = sum(map(len, entries_by_instance.values()))
    for entry in tqdm(itertools.chain.from_iterable(entries_by_instance.values()), total=total_entries):
        if type == "logsummary":
            current_type = entry[COMPONENT_COLUMN].split(":", 1)[0]
        configuration = abstraction.apply_settings(**DATASET_PARAMETERS[current_type]["parser_settings"])
        global_event_id = (entry[COMPONENT_COLUMN], entry[EVENT_ID_KEY])
        template = event_types[global_event_id]
        parameters = entry.pop(PARAMETERS_WITH_TYPE_KEY)
        simplify_parameters(parameters)
        entry[SIMPLIFIED_MESSAGE_KEY] = abstraction.reconstruct_log_message(template, *parameters, configuration=configuration)
    MODULE_LOGGER.info("Done simplifing log messages.")

    MODULE_LOGGER.info("Writing event types to destination...")
    with (destination_path / "event_types.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow([COMPONENT_COLUMN, EVENT_ID_KEY, TEMPLATE_COLUMN])
        writer.writerows(([component, id, template]) for (component, id), template in event_types.items())
    MODULE_LOGGER.info("Done writing event types.")

    MODULE_LOGGER.info("Writing processed log instances to destination...")
    instances_path = destination_path / "log_instances"
    instances_path.mkdir(exist_ok=exist_ok)
    for instance, entries in tqdm(entries_by_instance.items()):
        with (instances_path / (instance + ".csv")).open("w") as file:
            headers = [TIMESTAMP_KEY, LOGNAME_KEY, COMPONENT_COLUMN, EVENT_ID_KEY, SIMPLIFIED_MESSAGE_KEY, MESSAGE_COLUMN, PARAMETERS_KEY]
            all_headers = next(iter(entries)).keys()
            headers[4:4] = filter(lambda x: x not in headers, all_headers)
            writer = csv.DictWriter(file, headers)
            writer.writeheader()
            writer.writerows(entries.consume())
    MODULE_LOGGER.info("Done writing log instances.")

    MODULE_LOGGER.info("Successfully processed dataset.")


if __name__ == "__main__":
    exist_ok = False
    if "-h" in sys.argv or "--help" in sys.argv:
        print("A script for preprocessing a dataset.")
        print()
        print("python3 " + sys.argv[0] + " [option]... <type> <dataset-path> <destination-path>")
        print("Options:")
        print(" -h              Show this help-page.")
        print(" -f              Force preprocessing even if the destination path exists already.")
        print("Type:")
        print(" The type of the dataset to process. Can either be 'telcoapp', 'hadoop' or 'logsummary'.")
        sys.exit(0)
    if "-f" in sys.argv or "--force" in sys.argv:
        exist_ok = True

    arguments = list(filter(lambda x: not x.startswith("-"), sys.argv[1:]))

    if not len(arguments) == 3:
        print("Incorrect amount of arguments.")
        print("Try -h for further information.")
        sys.exit(2)

    arguments[0] = arguments[0].lower()

    if arguments[0] not in ["hadoop", "telcoapp", "logsummary"]:
        print("Unknown dataset type.")
        print("Try -h for further information.")
        sys.exit(2)

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%x %X")
    MODULE_LOGGER.setLevel(logging.INFO)
    preprocess_dataset(*arguments, exist_ok=exist_ok)
