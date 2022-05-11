from .exceptions import *
from . import structuring
from . import abstraction

import unittest
import itertools

line_number_field = "LineNumber"

android_raw = [
    "23-01 21:23:12.715 237 323 I NetworkDiscover: 3 known devices found, 1 unknown device detected.",
    "23-01 21:23:12.843 321 112 W WLANController:  Interface wlan0 down.",
]

android_structured = [
    {line_number_field: 1, "Date": "23-01", "Time": "21:23:12.715", "PID": "237", "TID": "323", "Level": "I",
        "Component": "NetworkDiscover", "Message": "3 known devices found, 1 unknown device detected."},
    {line_number_field: 2, "Date": "23-01", "Time": "21:23:12.843", "PID": "321", "TID": "112", "Level": "W",
        "Component": "WLANController", "Message": "Interface wlan0 down."},
]

hadoop_raw = [
    "2015-10-18 13:22:19,133 INFO [main] AppMaster: Application #4819 started.",
    "2015-10-18 13:22:20,746 WARN [main] AppMaster: Memory low. (276MiB remaining)",
    "2015-10-18 15:13:00,014 INFO [Server on 80] StatusReporter: Status report served.",
]

hadoop_structured = [
    {"Date": "2015-10-18", "Time": "13:22:19,133", "Level": "INFO", "Process": "main",
        "Component": "AppMaster", "Message": "Application #4819 started."},
    {"Date": "2015-10-18", "Time": "13:22:20,746", "Level": "WARN", "Process": "main",
        "Component": "AppMaster", "Message": "Memory low. (276MiB remaining)"},
    {"Date": "2015-10-18", "Time": "15:13:00,014", "Level": "INFO", "Process": "Server on 80",
        "Component": "StatusReporter", "Message": "Status report served."},
]

custom_format = r"<date> <time> <level> \| \(<component>\|<process_id>\) <message>"
custom_fields = ["date", "time", "level", "component", "process_id", "message"]

custom_raw = [
    "24-11     19:02:45 WARN  | (NetworkManager|112) Protocol error (peer 5): socket closed",
    "24-11     19:27:12 WARN  | (NetworkManager|113) Protocol error (peer 3): connection timeout",
    "24-11     19:27:15 FATAL |    (Application|110) Too few peers, needs restart!",
]

custom_structured = [
    {"date": "24-11", "time": "19:02:45", "level": "WARN", "component": "NetworkManager",
        "process_id": "112", "message": "Protocol error (peer 5): socket closed"},
    {"date": "24-11", "time": "19:27:12", "level": "WARN", "component": "NetworkManager",
        "process_id": "113", "message": "Protocol error (peer 3): connection timeout"},
    {"date": "24-11", "time": "19:27:15", "level": "FATAL", "component": "Application",
        "process_id": "110", "message": "Too few peers, needs restart!"},
]

optionals_format  = r"<date> <time> <level> \|( \(<component>\|<process_id>\))? <message>"

optionals_raw = [
    "24-11     19:02:45 WARN  | (NetworkManager|112) Protocol error (peer 5): socket closed",
    "24-11     19:27:12 WARN  | Protocol error (peer 3): connection timeout",
    "24-11     19:27:15 FATAL |    (Application|110) Too few peers, needs restart!",
]

optionals_structured = [
    {"date": "24-11", "time": "19:02:45", "level": "WARN", "component": "NetworkManager",
        "process_id": "112", "message": "Protocol error (peer 5): socket closed"},
    {"date": "24-11", "time": "19:27:12", "level": "WARN", "component": None,
        "process_id": None, "message": "Protocol error (peer 3): connection timeout"},
    {"date": "24-11", "time": "19:27:15", "level": "FATAL", "component": "Application",
        "process_id": "110", "message": "Too few peers, needs restart!"},
]

valid_parsing_examples = {
    "android": (structuring.LOG_FORMATS["android"], android_raw, line_number_field, android_structured),
    "hadoop": (structuring.LOG_FORMATS["hadoop"], hadoop_raw, None, hadoop_structured),
    "custom": (custom_format, custom_raw, None, custom_structured),
    "optionals": (optionals_format, optionals_raw, None, optionals_structured),
}

valid_log_format_examples = {
    r"<name>":
        (False, ["name"],
         r"^(?P<name>.*?)$"),
    r"<comparator|(==)|(\x3C=)|(\x3E=)|(!=)|\x3C|\x3E>":
        (False, ["comparator"],
         r"^(?P<comparator>(==)|(\x3C=)|(\x3E=)|(!=)|\x3C|\x3E)$"),
    r"<id|[a-z0-9]{1,3}>":
        (False, ["id"],
         r"^(?P<id>[a-z0-9]{1,3})$"),
    r"<key>: <value>":
        (True, ["key", "value"],
         r"^\s*(?P<key>.*?):\s+(?P<value>.*?)\s*$"),
    r"<object|[A-Z][a-zA-Z_]*>\((<property>=<value>\|)*\)":
        (False, ["object", "property", "value"],
         r"^(?P<object>[A-Z][a-zA-Z_]*)\(((?P<property>.*?)=(?P<value>.*?)\|)*\)$"),
    r"<optional>? <multiple>+ <integer|(\+|-)?[0-9]+><suffix>":
        (True, ["optional", "multiple", "integer", "suffix"],
         r"^\s*(?P<optional>.*?)?\s+(?P<multiple>.*?)+\s+(?P<integer>(\+|-)?[0-9]+)(?P<suffix>.*?)\s*$"),
}


class TestStructuring(unittest.TestCase):

    def test_construct_log_format_regex(self):
        for log_format, specification in valid_log_format_examples.items():
            ignore_surrounding_whitespace = specification[0]
            with self.subTest(log_format=log_format, ignore_surrounding_whitespace=ignore_surrounding_whitespace):
                fields, regex = structuring.construct_log_format_regex(log_format, ignore_surrounding_whitespace=ignore_surrounding_whitespace)
                self.assertSequenceEqual(specification[1], fields, msg="reference (first) does not match result (second)!")
                self.assertEqual(specification[2], regex.pattern, msg="reference (first) does not match result (second)!")

    def test_construct_log_format_regex_custom(self):
        fields, regex = structuring.construct_log_format_regex(custom_format)
        self.assertSequenceEqual(custom_fields, fields, msg="reference (first) does not match result (second)!")
        for line in custom_raw:
            self.assertIsNotNone(regex.fullmatch(line))

    def test_valid_structuring(self):
        for type, specification in valid_parsing_examples.items():
            with self.subTest(type=type):
                i = 0
                for i, entry in enumerate(structuring.lines_to_log_entries(specification[1], specification[0], line_number_field=specification[2])):
                    self.assertSetEqual(set(specification[3][i].values()), set(entry.values()), msg="reference (first) does not match result (second)!")
                self.assertEqual(len(specification[3]), i + 1, msg="reference (first) does not have the same length as result (second)!")

    def test_valid_structuring_exact(self):
        for type, specification in valid_parsing_examples.items():
            with self.subTest(type=type):
                i = 0
                for i, entry in enumerate(structuring.lines_to_log_entries(specification[1], specification[0], line_number_field=specification[2])):
                    self.assertDictEqual(specification[3][i], entry, msg="reference (first) does not match result (second)!")
                self.assertEqual(len(specification[3]), i + 1, msg="reference (first) does not have the same length as result (second)!")

    def test_invalid_structuring(self):
        entries_iterator = structuring.lines_to_log_entries(custom_raw[:2] + android_raw[:2] + hadoop_raw[1:2] + custom_raw[2:], custom_format)
        self.assertDictEqual(custom_structured[0], next(entries_iterator), msg="reference (first) does not match result (second)!")
        self.assertDictEqual(custom_structured[1], next(entries_iterator), msg="reference (first) does not match result (second)!")
        with self.assertRaises(ImproperlyFormattedLogLineError) as manager:
            next(entries_iterator)
        self.assertEqual(manager.exception.line_number, 3)
        self.assertEqual(manager.exception.line, android_raw[0])
        with self.assertRaises(ImproperlyFormattedLogLineError) as manager:
            next(entries_iterator)
        self.assertEqual(manager.exception.line_number, 4)
        self.assertEqual(manager.exception.line, android_raw[1])
        with self.assertRaises(ImproperlyFormattedLogLineError) as manager:
            next(entries_iterator)
        self.assertEqual(manager.exception.line_number, 5)
        self.assertEqual(manager.exception.line, hadoop_raw[1])
        self.assertDictEqual(custom_structured[2], next(entries_iterator), msg="reference (first) does not match result (second)!")
        self.assertRaises(StopIteration, next, entries_iterator)

    def test_invalid_structuring_with_exception_handler(self):
        def expect_format_error(exception):
            self.assertIsInstance(exception, ImproperlyFormattedLogLineError)
        entries_iterator = structuring.iterate_with_exception_handler(structuring.lines_to_log_entries(android_raw[0:1] + custom_raw[:1] + android_raw[1:2] + hadoop_raw[:2] + custom_raw[1:], custom_format), expect_format_error)
        for i, entry in enumerate(entries_iterator):
            self.assertDictEqual(custom_structured[i], entry, msg="reference (first) does not match result (second)!")
        self.assertEqual(len(custom_structured), i + 1, msg="reference (first) does not have the same length as result (second)!")

    def test_invalid_structuring_with_appending_exception_handler(self):
        with self.subTest("collect_with_exception_handler"):
            handler = structuring.create_appending_exception_handler("message", "\n")
            entries = structuring.collect_with_exception_handler(structuring.lines_to_log_entries(custom_raw + hadoop_raw[1:2] + android_raw[:2], custom_format), handler)
            for i, entry in enumerate(entries[:-1]):
                self.assertDictEqual(custom_structured[i], entry, msg="reference (first) does not match result (second)!")
            reference = custom_structured[-1].copy()
            reference["message"] = "\n".join((reference["message"], hadoop_raw[1], *android_raw[:2]))
            self.assertDictEqual(reference, entries[-1], msg="reference (first) does not match result (second)!")
            with self.subTest("invalid_first (collect_with_exception_handler)"):
                with self.assertRaises(ImproperlyFormattedLogLineError) as manager:
                    structuring.collect_with_exception_handler(structuring.lines_to_log_entries(android_raw[0:1] + custom_raw, custom_format), handler)
        with self.subTest("iterate_with_exception_handler"):
            last_entry = []
            handler = structuring.create_appending_exception_handler("message", "\n", previous_log_entries=last_entry)
            entries_iterator = structuring.iterate_with_exception_handler(structuring.lines_to_log_entries(custom_raw[:2] + hadoop_raw[1:2] + android_raw[:2], custom_format), handler)
            last_entry.append(next(entries_iterator))
            self.assertDictEqual(custom_structured[0], last_entry[0], msg="reference (first) does not match result (second)!")
            last_entry[0] = next(entries_iterator)
            self.assertDictEqual(custom_structured[1], last_entry[0], msg="reference (first) does not match result (second)!")
            self.assertRaises(StopIteration, next, entries_iterator)
            reference = custom_structured[1].copy()
            reference["message"] = "\n".join((reference["message"], hadoop_raw[1], *android_raw[:2]))
            self.assertDictEqual(reference, last_entry[0], msg="reference (first) does not match result (second)!")
            with self.subTest("invalid_first (iterate_with_exception_handler)"):
                entries_iterator = structuring.iterate_with_exception_handler(structuring.lines_to_log_entries(android_raw + custom_raw[0:1] + hadoop_raw + custom_raw[0:1], custom_format), handler)
                self.assertIsNotNone(next(entries_iterator))
                last_entry.clear()
                self.assertRaises(ImproperlyFormattedLogLineError, next, entries_iterator)
                self.assertRaises(StopIteration, next, entries_iterator)

    def test_ignore_surrounding_whitespace(self):
        with self.subTest(ignore_surrounding_whitespace=False):
            _, regex = structuring.construct_log_format_regex(custom_format, ignore_surrounding_whitespace=False)
            match_regular = regex.fullmatch(custom_raw[0])
            self.assertIsNotNone(match_regular)
            match_whitespace = regex.fullmatch(custom_raw[0].center(1000))
            if match_whitespace:
                self.assertNotEqual(match_regular.groups(), match_whitespace.groups())
                self.assertNotEqual(match_regular.groupdict(), match_whitespace.groupdict())
        with self.subTest(ignore_surrounding_whitespace=True):
            _, regex = structuring.construct_log_format_regex(custom_format, ignore_surrounding_whitespace=True)
            match_regular = regex.fullmatch(custom_raw[0])
            self.assertIsNotNone(match_regular)
            match_whitespace = regex.fullmatch(custom_raw[0].center(1000))
            self.assertIsNotNone(match_whitespace)
            self.assertEqual(match_regular.groups(), match_whitespace.groups())
            self.assertEqual(match_regular.groupdict(), match_whitespace.groupdict())

example_messages = [
    "Application #0 (root) started.",
    "Application #100 (NetworkManager) started.",
    "Interface eth0 up",
    "Interface wlan0 up",
    "Application #368 started.",
    "Application #1029 started.",
    "This is peer 10. Starting now.",
    "3 known devices found, 1 unknown device detected.",
    "Operation (join_overlay_network): STARTED",
    "Attepting connection with 127.0.0.1...",
    "Successfully connected to peer 8.",
    "Attepting connection with www.exampe.com...",
    "Successfully connected to peer 3.",
    "Attepting connection with peerserver.com...",
    "Successfully connected to peer 5.",
    "Operation (join_overlay_network): SUCCESS",
    "upload request received.",
    "Operation (create_object): STARTED",
    "upload request received.",
    "upload request received.",
    "Operation (create_object#2): STARTED",
    "Operation (create_object): SUCCESS",
    "Memory low. (276MiB remaining)",
    "Operation (create_object#3): STARTED",
    "Memory low. (132MiB remaining)",
    "Memory CRITICALLY low. (13MiB remaining)",
    "Operation (create_object#3): FAILED (reason: memory error)",
    "Attepting to inform client of failed upload (reason: memory error).",
    "Protocol error (peer 5): socket closed",
    "upload request received.",
    "Operation (create_object): STARTED",
    "Interface wlan0 down",
    "Protocol error (peer 3): connection timeout",
    "Operation (create_object): FAILED (reason: Protocol error)",
    "Attepting to inform client of failed upload (reason: Protocol error).",
    "Protocol error (peer 3): socket closed",
    "Too few peers, needs restart!",
]

example_clusters = {
    "Application <*> started.": [0, 1],
    "Interface <*> <*>": [2, 3, 31],
    "Application <*> <*> started.": [4, 5],
    "This is peer <*>. Starting now.": [6],
    "<*> known devices found, <*> unknown device detected.": [7],
    "Operation <*>: <*>": [8, 15, 17, 20, 21, 23, 30],
    "Attepting connection with <*>...": [9, 11, 13],
    "Successfully connected to peer <*>.": [10, 12, 14],
    "upload request received.": [16, 18, 19, 29],
    "Memory low. (<*> remaining)": [22, 24],
    "Memory CRITICALLY low. (<*> remaining)": [25],
    "Operation <*>: FAILED (reason: <*>)": [26, 33],
    "Attepting to inform client of failed upload (reason: <*>).": [27, 34],
    "Protocol error (peer <*>): <*>": [28, 32, 35],
    "Too few peers, needs restart!": [36],
}

def groupby(sequence, key=None):
    return itertools.groupby(sorted(sequence, key=key), key=key)

class TestAbstraction(unittest.TestCase):

    def setUp(self):
        self.configuration = abstraction.apply_settings(similarity_threshold=0.1, depth=3)

    def test_parser_consistency(self):
        event_instances, events = abstraction.extract_events_from_log_messages(example_messages, self.configuration)
        for i, message in enumerate(example_messages):
            reconstructed = abstraction.reconstruct_log_message(events[event_instances[i][0]], *event_instances[i][1])
            self.assertEqual(message, reconstructed, msg="reference (first) does not match result (second)!")

    def test_correct_event_clusters(self):
        event_instances, _ = abstraction.extract_events_from_log_messages(example_messages, self.configuration)
        clusters = [[x[0] for x in sorted(group)] for _, group in groupby(enumerate(event_instances), lambda x: x[1][0])]
        clusters.sort()
        for i, (template, cluster) in enumerate(example_clusters.items()):
            self.assertSequenceEqual(cluster, clusters[i], msg="reference (first) does not match result (second)!")

if __name__ == "__main__":
    unittest.main()
