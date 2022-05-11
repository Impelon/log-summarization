import re

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

PATTERNS = {
    "ipv4_address_with_port": r"(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?",
    "mac_address": r"(?:[\dxa-fA-F]{2}[-:]){5}[\dxa-fA-F]{2}",
    "linux_path": r"(?:[\w.*~%@-]+:?)?(?:/[\w.*~%@+,-]+){2,}/?",
    "windows_path": r"(?:[\w.*~%@-]+:?)?(?:\\[\w.*~%@+,-]+){2,}\\?",
    "hdfs_block": r"blk(?:_-?\d+){1,2}",
    "hdfs_url": r"hdfs://[\w.:@-]*(?:(?:/[\w.~%+-]+)+/?)?",
    "http_url": r"https?://[\w.:@-]*(?:(?:/[\w.~%?&=-]+)+/?)?",
    "java_identifier": r"(?:[\w$]+\.){2,}[\w$<>]+",
    "trace": r"^\[(?:[0-9a-fA-F]+, ?){2}[0-9a-fA-F]+\]",
    "hex_number": r"\b0[Xx][a-fA-F\d]+\b",
    "hex_segment": r"\b[a-fA-F\d]{6,}\b",
    "hex_identifier": r"\b(?=[\d-]*[a-fA-F])([a-fA-F\d]{2,}-){2,}[a-fA-F\d]{2,}\b",
    "number": r"[+-]?\b\d+(?:\.\d+(?:-?[eE]\d+)?)?\b",
}
PATTERNS["file_url"] = r"file://(?:(?:{})|(?:{}))".format(PATTERNS["linux_path"], PATTERNS["windows_path"])
PATTERNS["java_stackentry"] = PATTERNS["java_identifier"].replace("{2,}", "+") + r"\((?:\w*\.\w+:\d*|[\w$<>\s]*)\)"

def get_default_mask(key):
    return (key, PATTERNS[key])

PARSER_SETTINGS = {
    "android": {
        "similarity_threshold": 0.2,
        "depth": 6,
        "masking_patterns": [
            get_default_mask("trace"),
            get_default_mask("http_url"),
            get_default_mask("file_url"),
            ("content_or_condition_uri", r"(?:content|condition)://(?:[\w.*~%@-]+:?)?(?:/[\w.*~%?&=-]+)*/?"),
            ("tcp_url", r"tcp://" + PATTERNS["ipv4_address_with_port"]),
            get_default_mask("linux_path"),
            get_default_mask("java_identifier"),
            get_default_mask("mac_address"),
            get_default_mask("hex_number"),
            get_default_mask("hex_identifier"),
            get_default_mask("number"),
            get_default_mask("hex_segment"),
        ],
    },
    "apache": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [PATTERNS["ipv4_address_with_port"]],
    },
    "bgl": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [
            get_default_mask("ipv4_address_with_port"),
            get_default_mask("linux_path"),
            ("core_path", r"core\.\d+"),
            get_default_mask("hex_number"),
            get_default_mask("number"),
            get_default_mask("hex_segment"),
        ],
    },
    "hdfs": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [
            get_default_mask("linux_path"),
            ("block_id", PATTERNS["hdfs_block"]),
            ("ipv4_address_with_port", r"/?" + PATTERNS["ipv4_address_with_port"]),
        ],
    },
    "hpc": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [r"(?<==)\d+"],
    },
    "hadoop": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [
            get_default_mask("hdfs_url"),
            get_default_mask("http_url"),
            get_default_mask("linux_path"),
            get_default_mask("windows_path"),
            ("block_id", r"BP-\d+-" + PATTERNS["ipv4_address_with_port"] + r"-\d+:" + PATTERNS["hdfs_block"]),
            ("attempt_id", r"(?:app)?attempt_\d+_\w+"),
            ("container_id", r"container_\d+_\d+_\d+_\d+"),
            ("task_id", r"task_\d+_\d+_\w_\d+"),
            ("job_id", r"job_\d+_[\d_]+"),
            ("ipv4_address_with_port", r"/?" + PATTERNS["ipv4_address_with_port"]),
            get_default_mask("java_stackentry"),
            get_default_mask("java_identifier"),
            ("number", r"(?<=[:=+])" + PATTERNS["number"].rstrip(r"\b"))
        ],
    },
    "healthapp": {
        "similarity_threshold": 0.2,
        "depth": 4,
        "masking_patterns": [],
    },
    "linux": {
        "similarity_threshold": 0.39,
        "depth": 6,
        "masking_patterns": [PATTERNS["ipv4_address_with_port"], r"\d{2}:\d{2}:\d{2}"],
    },
    "mac": {
        "similarity_threshold": 0.7,
        "depth": 6,
        "masking_patterns": [PATTERNS["java_identifier"]],
    },
    "openssh": {
        "similarity_threshold": 0.6,
        "depth": 5,
        "masking_patterns": [PATTERNS["ipv4_address_with_port"], PATTERNS["java_identifier"]],
    },
    "openstack": {
        "similarity_threshold": 0.5,
        "depth": 5,
        "masking_patterns": [r"(?:{},?)+".format(PATTERNS["ipv4_address_with_port"]), r"/.+?\s", r"\d+"],
    },
    "proxifier": {
        "similarity_threshold": 0.6,
        "depth": 3,
        "masking_patterns": [
            ("host_uri", r"(?:[\w-]+\.)+[\w-]+(?::\d+)?"),
            ("timestamp", r"<\d+\ssec"),
            ("timestamp", r"\b\d{2}:\d{2}(:\d{2})*\b"),
            get_default_mask("number"),
        ],
    },
    "spark": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [
            get_default_mask("hdfs_url"),
            get_default_mask("http_url"),
            get_default_mask("linux_path"),
            get_default_mask("windows_path"),
            ("block_id", r"BP-\d+-" + PATTERNS["ipv4_address_with_port"] + r"-\d+:" + PATTERNS["hdfs_block"]),
            ("attempt_id", r"(?:app)?attempt_\d+_\w+"),
            ("container_id", r"container_\d+_\d+_\d+_\d+"),
            ("task_id", r"task_\d+_\d+_\w_\d+"),
            ("job_id", r"job_\d+_[\d_]+"),
            ("ipv4_address_with_port", r"/?" + PATTERNS["ipv4_address_with_port"]),
            get_default_mask("java_stackentry"),
            get_default_mask("java_identifier"),
            ("number", r"(?<=[:=+])" + PATTERNS["number"].rstrip(r"\b"))
        ],
    },
    "thunderbird": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [PATTERNS["ipv4_address_with_port"]],
    },
    "windows": {
        "similarity_threshold": 0.7,
        "depth": 5,
        "masking_patterns": [r"0x.*?\s"],
    },
    "zookeeper": {
        "similarity_threshold": 0.5,
        "depth": 4,
        "masking_patterns": [
            ("ipv4_address_with_port", r"/?" + PATTERNS["ipv4_address_with_port"]),
            get_default_mask("hex_number"),
        ],
    },
}


def apply_settings(configuration=None, similarity_threshold=None, depth=None, max_children=None, parametrize_numeric_tokens=None, masking_patterns=None):
    """
    Apply the given settings to a configuration object.

    Args:
        configuration: A configuration object used by the Drain parser.
            If this is None, a new configuration object will be created.
        similarity_threshold: A float between 0 and 1 to use as a threshold for similarity to decide when to create a new log event.
            If the ratio of similar tokens is below this threshold, a new log event is created.
            If this is None, the existing value for this configuration will not be changed.
        depth: An integer denoting the maximum depth of the parse tree.
            If this is None, the existing value for this configuration will not be changed.
        max_children: An integer denoting the maximum number of children an internal node of the parse tree can have.
            If this is None, the existing value for this configuration will not be changed.
        parametrize_numeric_tokens: A boolean indicating whether tokens containing numbers have priority to be deemed parameters.
            If this is None, the existing value for this configuration will not be changed.
        masking_patterns: An iterable of strings or string-pairs specifing regular expression patterns that will be used in preprocessing.
            If this is a standalone string, it represents the pattern used and will be assigned a default masking-value.
            If this is a pair of strings, the first string specifies the masking-value and the second string the pattern.
            When parsing messages, text segments matching these patterns will become parameters in the resulting templates,
            and are replaced by a mask with the corresponding masking-value.

    Returns:
        The modified configuration object.
    """
    if not configuration:
        configuration = TemplateMinerConfig()
    if similarity_threshold is not None:
        configuration.drain_sim_th = similarity_threshold
    if depth is not None:
        configuration.drain_depth = depth
    if max_children is not None:
        configuration.drain_max_children = max_children
    if parametrize_numeric_tokens is not None:
        configuration.parametrize_numeric_tokens = parametrize_numeric_tokens
    if masking_patterns:
        for pattern in masking_patterns:
            replacement = "@"
            try:
                replacement, pattern = pattern
            except:
                pass
            configuration.masking_instructions.append(MaskingInstruction(pattern, replacement))
    return configuration


def extract_events_from_log_messages(messages, configuration=None, parseroptions=None, include_parameter_masks=False):
    """
    Extract a sequence of instances of log events from a sequence of log messages.
    A log event is represented by the static parts of a log message, called the template.
    The dynamic parts of a message are the parameters of that instance of the event.
    A template will have wildcards/masks which represent the locations of parameters.

    Args:
        messages: A sequence of log messages.
        configuration: A configuration object used by the Drain parser.
            If this is None, the parser's default configuration will be used.
        parseroptions: Additional keyword-arguments passed to the parser.
            If this is a falsy value (e.g. None) no additional options are passed on.
        include_parameter_masks: Whether to include a pair of (parameter, mask)
            instead of each parameter in any sequence of parameters,
            specifing the mask the parameter was matched with.

    Returns:
        A tuple like (event_instances, event_types).
        event_instances is a sequence of tuples (event_id, parameters) for each processed message,
        pointing to the log event corresponding to that message and a sequence of parameters for that specific message.
        event_types is a dict mapping event ID's to their corresponding templates.
        The original messages can be reconstructed by inserting into the template of each event the corresponding parameters.
    """
    if not parseroptions:
        parseroptions = {}
    parser = TemplateMiner(config=configuration, **parseroptions)

    # Warm up parser to create all templates in their final form.
    for message in messages:
        parser.add_log_message(message)

    event_instances = []
    event_types = {}
    # Gather event-instances and templates.
    for message in messages:
        cluster = parser.match(message, full_search_strategy="fallback")
        id = cluster.cluster_id
        event_types[id] = cluster.get_template()
        extracted_parameters = parser.extract_parameters(event_types[id], message, exact_matching=True)
        if include_parameter_masks:
            event_instances.append((id, extracted_parameters))
        else:
            event_instances.append((id, [parameter.value for parameter in extracted_parameters]))

    return event_instances, event_types


def reconstruct_log_message(template, *parameters, configuration=None):
    """
    Reconstruct a log message from the template of its event and its parameters.

    Args:
        template: A string specifing the template of the message to reconstruct.
        *parameters: Parameter-values that will be filled in for the template.
        configuration: A configuration object used by the Drain parser.
            If this is None, the default configuration will be used.

    Returns:
        A string representing the reconstructed log message.
    """
    if not configuration:
        configuration = apply_settings()
    escaped_prefix = re.escape(configuration.mask_prefix)
    escaped_suffix = re.escape(configuration.mask_suffix)
    possible_masks = set()
    possible_masks.add(re.escape("*"))
    for instruction in configuration.masking_instructions:
        possible_masks.add(re.escape(instruction.mask_with))
    parameter_iterator = iter(parameters)
    result = re.sub(escaped_prefix + r"(?:" + r"|".join(possible_masks) + r")" + escaped_suffix, lambda x: str(next(parameter_iterator)), template)
    return result
