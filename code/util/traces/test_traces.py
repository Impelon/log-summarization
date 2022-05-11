from . import traces

import unittest
from unittest.mock import patch
import io
import itertools

message = traces.COLUMN_THAT_CONTAINS_TRACES
trace = traces.COLUMN_THAT_CONTAINS_TRACE_IDS
span = traces.COLUMN_THAT_CONTAINS_SPAN_IDS
pspan = traces.COLUMN_THAT_CONTAINS_PARENT_SPAN_IDS
values = "values"


def extract_explicit_trace_information(entry):
    splitted = entry[message][1:-1].split(", ")
    return {trace: splitted[0], span: splitted[1], pspan: splitted[2]}


simple_entries_a = [{message: "[a, 1, 0]"}, {message: "[a, 2, 1]"}, {message: "[a, 3, 0]"},
                    {message: "[a, 4, 1]"}, {message: "[a, 5, 3]"}, {message: "[a, 0, 6]"}]
simple_entries_b = [{message: "[b, 1, 0]"}, {message: "[b, 2, 0]"}, {message: "[b, 3, 1]"},
                    {message: "[b, 5, 4]"}, {message: "[b, 4, 0]"}]
simple_entries = simple_entries_a + simple_entries_b

simple_explicit_entries = list(map(extract_explicit_trace_information, simple_entries))

simple_entries_csv = message + "\n" + "\n".join('"' + entry[message] + '"' for entry in simple_entries)
simple_entries_csv_with_noise = ("noiseA," + message + ",noiseB\n" +
                                 "\n".join(",".join((str(index), '"' + entry[message] + '"', "B" + str(index))) for index, entry in enumerate(simple_entries)))
simple_explicit_entries_csv = ",".join((trace, span, pspan)) + "\n" + "\n".join(",".join(entry.values()) for entry in simple_explicit_entries)

simple_entries_traces = {
    "a": {
        "6": {"0": {"1": {"2": {}, "4": {}, }, "3": {"5": {}, }, }, },
    },
    "b": {
        "0": {"1": {"3": {}, }, "2": {}, "4": {"5": {}, }, },
    },
}

simple_entries_a_traces_with_values = {
    "a": {
        "6": {
            "0": {"1": {values: [{message: "[a, 1, 0]"}], "2": {values: [{message: "[a, 2, 1]"}]}, "4": {values: [{message: "[a, 4, 1]"}]}},
                  "3": {values: [{message: "[a, 3, 0]"}], "5": {values: [{message: "[a, 5, 3]"}]}},
                  values: [{message: "[a, 0, 6]"}]},
        },
    },
}

complex_entries = [{message: "[0, 3, 13]"}, {message: "[0, 13, 101]"}, {message: "[0, 4, 13]"},
                   {message: "[0, 12, 13]"}, {message: "[0, 5, 12]"}, {message: "[0, 11, 12]"},
                   {message: "[0, 10, 11]"}, {message: "nothing here"}, {message: "[0, 7, 10]"},
                   {message: "[0, 6, 7]"}, {message: "[0, 8, 10]"}, {message: "[0, 9, 10]"},
                   {message: "[0, 14, 102]"}, {message: "[0, 15, 102]"}, {message: "[0, 16, 102]"},
                   {message: "[0, 17, 102]"}, {message: "[0, 18, 102]"}, {message: "[1, 19, 103]"},
                   {message: "[1, 20, 24]"}, {message: "[1, 24, 104]"}, {message: "[1, 21, 24]"},
                   {message: "[1, 22, 24]"}, {message: "nothing here"}, {message: "[1, 23, 24]"},
                   {message: "[1, 25, 105]"}, {message: "[1, 26, 106]"}, {message: "[1, 27, 106]"},
                   {message: "[1, 28, 107]"}, {message: "[1, 30, 107]"}, {message: "[1, 29, 30]"},
                   {message: "[1, 31, 101]"}, {message: "[2, 32, 108]"}, {message: "[2, 33, 35]"},
                   {message: "[2, 35, 108]"}, {message: "[2, 34, 35]"}, {message: "[2, 36, 49]"},
                   {message: "[2, 49, 109]"}, {message: "[2, 39, 49]"}, {message: "[2, 37, 39]"},
                   {message: "[2, 38, 39]"}, {message: "[2, 41, 49]"}, {message: "[2, 40, 41]"},
                   {message: "[2, 42, 49]"}, {message: "[2, 45, 49]"}, {message: "[2, 43, 45]"},
                   {message: "[2, 44, 45]"}, {message: "[2, 47, 49]"}, {message: "[2, 46, 47]"},
                   {message: "[2, 48, 49]"}, {message: "[2, 50, 110]"}, {message: "[2, 51, 110]"},
                   {message: "[2, 53, 111]"}, {message: "[2, 52, 53]"}, {message: "[2, 54, 111]"},
                   {message: "[2, 55, 112]"}, {message: "[2, 57, 113]"}, {message: "[2, 56, 57]"},
                   {message: "[2, 58, 114]"}, {message: "nothing here"}, {message: "[2, 59, 115]"},
                   {message: "[2, 62, 116]"}, {message: "[2, 60, 62]"}, {message: "[2, 61, 62]"},
                   {message: "[2, 64, 116]"}, {message: "[2, 63, 64]"}, {message: "[2, 65, 116]"},
                   {message: "[2, 66, 117]"}, {message: "[2, 74, 101]"}, {message: "[2, 73, 74]"},
                   {message: "[2, 69, 73]"}, {message: "[2, 67, 69]"}, {message: "[2, 68, 69]"},
                   {message: "[2, 71, 73]"}, {message: "[2, 70, 71]"}, {message: "[2, 72, 73]"},
                   {message: "[2, 82, 101]"}, {message: "[2, 81, 82]"}, {message: "[2, 77, 81]"},
                   {message: "[2, 75, 77]"}, {message: "[2, 76, 77]"}, {message: "[2, 79, 81]"},
                   {message: "[2, 78, 79]"}, {message: "[2, 80, 81]"}, {message: "[2, 90, 101]"},
                   {message: "[2, 89, 90]"}, {message: "[2, 85, 89]"}, {message: "[2, 83, 85]"},
                   {message: "[2, 84, 85]"}, {message: "[2, 87, 89]"}, {message: "[2, 86, 87]"},
                   {message: "[2, 88, 89]"}, {message: "[2, 91, 118]"}, {message: "[2, 96, 119]"},
                   {message: "[2, 92, 96]"}, {message: "[2, 94, 96]"}, {message: "[2, 93, 94]"},
                   {message: "[2, 95, 96]"}, {message: "[2, 98, 120]"}, {message: "[2, 97, 98]"},
                   {message: "[2, 99, 121]"}, {message: "nothing here"}, {message: "[2, 100, 122]"}]


class TestTracesBase(unittest.TestCase):

    def assert_paths_equal(self, first, second, msg=None):
        if first is None or second is None:
            self.assertEqual(first, second, msg=msg)
            return
        self.assertEqual(len(first), len(second), msg=msg)
        for i in range(len(first)):
            self.assertEqual(first[i], second[i], msg=msg)


class TestTracesFundamentals(TestTracesBase):

    def test_find_id(self):
        self.assert_paths_equal(traces.find_id(simple_entries_traces, "a"), ("a"))
        self.assert_paths_equal(traces.find_id(simple_entries_traces, "unfindable"), None)
        self.assert_paths_equal(traces.find_id(simple_entries_traces["a"], "1"), ("6", "0", "1"))
        self.assert_paths_equal(traces.find_id(simple_entries_traces, "1"), ("a", "6", "0", "1"))
        self.assert_paths_equal(traces.find_id(simple_entries_traces["b"], "5"), ("0", "4", "5"))

    def test_create_path_parent_same_as_span(self):
        result = {}
        traces.create_path(result, "a", "0", "0")
        self.assertEqual(result, {"a": {"0": {"0": {}}}})
        traces.create_path(result, "a", "0", "1")
        self.assertEqual(result, {"a": {"0": {"0": {}, "1": {}}}})
        result = {}
        traces.create_path(result, "b", "1", "1")
        self.assertEqual(result, {"b": {"1": {"1": {}}}})

    def test_create_path_nested(self):
        result = {}
        traces.create_path(result, "a", "0", "1")
        self.assertEqual(result, {"a": {"0": {"1": {}}}})
        traces.create_path(result, "a", "0", "2")
        self.assertEqual(result, {"a": {"0": {"1": {}, "2": {}}}})
        traces.create_path(result, "a", "2", "3")
        self.assertEqual(result, {"a": {"0": {"1": {}, "2": {"3": {}}}}})

    def test_create_path_change_hierarchy(self):
        result = {}
        traces.create_path(result, "a", "1", "2")
        self.assertEqual(result, {"a": {"1": {"2": {}}}})
        traces.create_path(result, "a", "0", "1")
        self.assertEqual(result, {"a": {"0": {"1": {"2": {}}}}})

    def test_create_path_duplicate_multiple_traces(self):
        result = {}
        traces.create_path(result, "a", "0", "1")
        self.assertEqual(result, {"a": {"0": {"1": {}}}})
        traces.create_path(result, "b", "0", "1")
        self.assertEqual(result, {"a": {"0": {"1": {}}}, "b": {"0": {"1": {}}}})


class TestTracesExtended(TestTracesBase):

    def assert_order_insensitive(self, entries, trace_generator, permutations=None, msg=None):
        result = trace_generator(entries)
        result_reversed = trace_generator(reversed(entries))
        self.assertEqual(result, result_reversed, msg=msg)
        if permutations == "some":
            for permutation in itertools.permutations(entries):
                self.assertEqual(result, trace_generator(permutation), msg=msg)
        elif permutations == "all":
            for permutation1 in itertools.permutations(entries):
                for permutation2 in itertools.permutations(entries):
                    self.assertEqual(trace_generator(permutation1), trace_generator(permutation2), msg=msg)

    def test_find_id_all(self):
        paths = traces.find_id_all(simple_entries_traces, "a")
        self.assertEqual(len(paths), 1)
        self.assert_paths_equal(paths[0], ("a"))
        paths = traces.find_id_all(simple_entries_traces, "unfindable")
        self.assertEqual(len(paths), 0)
        paths = traces.find_id_all(simple_entries_traces["a"], "1")
        self.assertEqual(len(paths), 1)
        self.assert_paths_equal(paths[0], ("6", "0", "1"))
        paths = traces.find_id_all(simple_entries_traces, "1")
        self.assertEqual(len(paths), 2)
        self.assert_paths_equal(paths[0], ("a", "6", "0", "1"))
        self.assert_paths_equal(paths[1], ("b", "0", "1"))

    def test_flatten(self):
        def entries_to_tuples(entries):
            items = [[y[1] for y in sorted(x.items())] for x in entries]
            return sorted(items)
        self.assertEqual(entries_to_tuples(simple_explicit_entries), entries_to_tuples(traces.flatten(simple_entries_traces)),
                         msg="reference (first) does not match result (second)!")

    def traces_from_entries(self, entries, **kwargs):
        return traces.traces_from_entries(entries, **kwargs)

    def traces_from_explicit_entries(self, entries, **kwargs):
        return traces.traces_from_explicit_entries(entries, **kwargs)

    def test_traces_from_entries_with_values_exact_simple(self):
        self.assertEqual(simple_entries_a_traces_with_values, self.traces_from_entries(simple_entries_a, values_key=values),
                         msg="reference (first) does not match result (second)!")

    def test_traces_from_entries_includes_all_simple(self):
        result = self.traces_from_entries(simple_entries)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_traces_from_entries_order_insensitive_simple(self):
        self.assert_order_insensitive(simple_entries, self.traces_from_entries)
        self.assert_order_insensitive(simple_entries_a, self.traces_from_entries, permutations="some")
        self.assert_order_insensitive(simple_entries_b, self.traces_from_entries, permutations="all")
        self.assert_order_insensitive(simple_entries[1:9], self.traces_from_entries, permutations="some")
        self.assert_order_insensitive(simple_entries[3:8], self.traces_from_entries, permutations="all")

    def test_traces_from_entries_order_insensitive_complex(self):
        self.assert_order_insensitive(complex_entries, self.traces_from_entries)

    def test_traces_from_entries_exact_simple(self):
        self.assertEqual(simple_entries_traces, self.traces_from_entries(simple_entries),
                         msg="reference (first) does not match result (second)!")

    def test_traces_from_entries_same_as_traces_from_explicit_entries_simple(self):
        self.assertEqual(self.traces_from_entries(simple_entries),
                         self.traces_from_explicit_entries(simple_explicit_entries))

    def test_traces_from_explicit_entries_exact_simple(self):
        self.assertEqual(simple_entries_traces, self.traces_from_explicit_entries(simple_explicit_entries),
                         msg="reference (first) does not match result (second)!")


class TestTracesFiles(TestTracesBase):

    def test_traces_from_csv_file(self):
        with patch("builtins.open") as mocked_open:
            mocked_open.return_value = io.StringIO(simple_entries_csv)
            self.assertEqual(simple_entries_traces, traces.traces_from_csv_file("", entries_have_explicit_trace_information=False),
                             msg="reference (first) does not match result (second)!")
            mocked_open.assert_called_once()

    def test_traces_from_csv_file_explicit(self):
        with patch("builtins.open") as mocked_open:
            mocked_open.return_value = io.StringIO(simple_explicit_entries_csv)
            self.assertEqual(simple_entries_traces, traces.traces_from_csv_file("", entries_have_explicit_trace_information=True),
                             msg="reference (first) does not match result (second)!")
            mocked_open.assert_called_once()

    def test_traces_from_csv_files(self):
        with patch("builtins.open") as mocked_open:
            mocked_open.side_effect = iter((io.StringIO(simple_entries_csv), io.StringIO(simple_explicit_entries_csv)))
            self.assertEqual(simple_entries_traces, traces.traces_from_csv_files(({"path": "a", "entries_have_explicit_trace_information": False},
                                                                                  {"path": "b", "entries_have_explicit_trace_information": True})),
                             msg="reference (first) does not match result (second)!")
            self.assertEqual(mocked_open.call_count, 2)


if __name__ == "__main__":
    unittest.main()
