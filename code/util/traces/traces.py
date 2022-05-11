import csv
import re
import copy
import collections
import itertools

_id_pattern = r"[0-9a-f]+"

GROUP_NAME_TRACE_ID = "trace_id"
GROUP_NAME_SPAN_ID = "span_id"
GROUP_NAME_PARENT_SPAN_ID = "parent_span_id"

TRACE_ID_PATTERN = r"(?P<" + GROUP_NAME_TRACE_ID + r">" + _id_pattern + r")"
SPAN_ID_PATTERN = r"(?P<" + GROUP_NAME_SPAN_ID + r">" + _id_pattern + r")"
PARENT_SPAN_ID_PATTERN = r"(?P<" + GROUP_NAME_PARENT_SPAN_ID + r">" + _id_pattern + r")"

TRACE_MARKER = re.compile(r"\[" + TRACE_ID_PATTERN + r", ?" + SPAN_ID_PATTERN + r", ?" + PARENT_SPAN_ID_PATTERN + r"\]", flags=re.IGNORECASE)

COLUMN_THAT_CONTAINS_TRACES = "Message"
COLUMN_THAT_CONTAINS_TRACE_IDS = GROUP_NAME_TRACE_ID
COLUMN_THAT_CONTAINS_SPAN_IDS = GROUP_NAME_SPAN_ID
COLUMN_THAT_CONTAINS_PARENT_SPAN_IDS = GROUP_NAME_PARENT_SPAN_ID


def traces_from_csv_files(configurations, **kwargs):
    """
    Extract traces from multiple csv-files.
    The entries from the files are merged together before parsing for traces.

    Args:
        configurations: A iterable of dicts with entries like
            {
                "path": "<path_to_file>",
                "entries_have_explicit_trace_information": <True/False>,
                "readeroptions": dict,
                "parseroptions": dict
            }
            These options have the same semantics as the parameters of `traces_from_csv_file`.
            The path is the only required argument.
        **kwargs: Additional keyword-arguments passed to `traces_from_trace_tuples`.

    Returns:
        A dict of traces.
        Each trace-entry has its ID as key and its value is a dict of other spans.
        A span-entry also has its ID as key and its value is a dict.
        Each span may contain other spans.
    """
    def parse_file(configuration):
        with open(configuration["path"], "r") as file:
            readeroptions = configuration.get("readeroptions", None)
            if not readeroptions:
                readeroptions = {}
            parseroptions = configuration.get("parseroptions", None)
            if parseroptions is None:
                parseroptions = {}
            csv_reader = csv.DictReader(file, **readeroptions)
            parser = entries_to_trace_tuples
            if configuration.get("entries_have_explicit_trace_information", False):
                parser = explicit_entries_to_trace_tuples
            for tuple in parser(csv_reader, **parseroptions):
                yield tuple

    trace_tuples = []
    for configuration in configurations:
        trace_tuples.append(parse_file(configuration))
    return traces_from_trace_tuples(itertools.chain(*trace_tuples), **kwargs)


def traces_from_csv_file(path, entries_have_explicit_trace_information=False, readeroptions=None, parseroptions=None, **kwargs):
    """
    Extract traces from the given csv-file.

    Args:
        path: A path-like object that points to the file.
        entries_have_explicit_trace_information: A boolean indicating whether trace-
            information is explicitly available for every entry or not.
            If True this will use `traces_from_explicit_entries` to construct the traces.
            Otherwise `traces_from_entries` will be used.
        readeroptions: Additional keyword-arguments passed to csv.DictReader().
            If this is a falsy value (e.g. None) no additional options are passed on.
        parseroptions: Additional keyword-arguments passed to `entries_to_trace_tuples` or
            `explicit_entries_to_trace_tuples`.
            If this is a falsy value (e.g. None) no additional options are passed on.
        **kwargs: Additional keyword-arguments passed to `traces_from_trace_tuples`.

    Returns:
        A dict of traces.
        Each trace-entry has its ID as key and its value is a dict of other spans.
        A span-entry also has its ID as key and its value is a dict.
        Each span may contain other spans.
    """
    configuration = {"path": path,
                     "entries_have_explicit_trace_information": entries_have_explicit_trace_information,
                     "readeroptions": readeroptions,
                     "parseroptions": parseroptions}
    return traces_from_csv_files((configuration,), **kwargs)


def traces_from_entries(entries, parseroptions=None, **kwargs):
    """
    Extract traces from the given entries.
    Equivalent to `traces_from_trace_tuples(entries_to_trace_tuples(entries, **parseroptions), **kwargs)`

    Args:
        entries: A iterable of dicts.
        parseroptions: Additional keyword-arguments passed to `entries_to_trace_tuples`.
            If this is a falsy value (e.g. None) no additional options are passed on.
        **kwargs: Additional keyword-arguments passed to `traces_from_trace_tuples`.

    Returns:
        A dict of traces.
        Each trace-entry has its ID as key and its value is a dict of other spans.
        A span-entry also has its ID as key and its value is a dict.
        Each span may contain other spans.
    """
    if not parseroptions:
        parseroptions = {}
    return traces_from_trace_tuples(entries_to_trace_tuples(entries, **parseroptions), **kwargs)


def traces_from_explicit_entries(entries, parseroptions=None, **kwargs):
    """
    Extract traces from the given entries with explicit information about traces.
    Equivalent to `traces_from_trace_tuples(explicit_entries_to_trace_tuples(entries, **parseroptions), **kwargs)`

    Args:
        entries: A iterable of dicts.
        parseroptions: Additional keyword-arguments passed to `explicit_entries_to_trace_tuples`.
            If this is a falsy value (e.g. None) no additional options are passed on.
        **kwargs: Additional keyword-arguments passed to `traces_from_trace_tuples`.

    Returns:
        A dict of traces.
        Each trace-entry has its ID as key and its value is a dict of other spans.
        A span-entry also has its ID as key and its value is a dict.
        Each span may contain other spans.
    """
    if not parseroptions:
        parseroptions = {}
    return traces_from_trace_tuples(explicit_entries_to_trace_tuples(entries, **parseroptions), **kwargs)


def entries_to_trace_tuples(entries, trace_key=None, remove_trace_information=False, can_only_be_prefix=True):
    """
    Convert the given entries into a series of tuples suitable for `traces_from_trace_tuples`.

    Args:
        entries: A iterable of dicts.
        trace_key: A key that points to a string-value in every dict-entry.
            That string is assumed to be able to contain trace-information that can be extracted with TRACE_MARKER.
            If this is a falsy value (e.g. None) COLUMN_THAT_CONTAINS_TRACES will be assumed as default.
        remove_trace_information: A boolean indicating whether to remove the trace-marker for every entry in the result.
            Otherwise entries are left unchanged.
        can_only_be_prefix: A boolean indicating whether the trace-marker can only occur at the start of a string.

    Yields:
        A tuple like (trace_id, parent_span_id, span_id, entry).
        For every element in entries, entry is that original element or a modified copy of it.
    """
    if not trace_key:
        trace_key = COLUMN_THAT_CONTAINS_TRACES

    for entry in entries:
        if can_only_be_prefix:
            match = TRACE_MARKER.match(entry[trace_key])
        else:
            match = TRACE_MARKER.search(entry[trace_key])
        if not match:
            continue
        trace_id = match.group(GROUP_NAME_TRACE_ID)
        parent_span_id = match.group(GROUP_NAME_PARENT_SPAN_ID)
        span_id = match.group(GROUP_NAME_SPAN_ID)
        if remove_trace_information:
            entry = copy.copy(entry)
            halves = [match.string[:match.start()], match.string[match.end():].lstrip()]
            if not halves[1]:
                halves[0] = halves[0].rstrip()
            entry[trace_key] = halves[0] + halves[1]
        yield (trace_id, parent_span_id, span_id, entry)


def explicit_entries_to_trace_tuples(entries, trace_id_key=None, parent_span_id_key=None, span_id_key=None, remove_trace_information=False):
    """
    Extract traces from the given entries with explicit information about traces.

    Args:
        entries: A iterable of dicts.
        trace_id_key: A key that points to the ID of the trace in every dict-entry.
            If this is a falsy value (e.g. None) COLUMN_THAT_CONTAINS_TRACE_IDS will be assumed as default.
        parent_span_id_key: A key that points to the ID of the parent of the span in every dict-entry.
            If this is a falsy value (e.g. None) COLUMN_THAT_CONTAINS_PARENT_SPAN_IDS will be assumed as default.
        span_id_key: A key that points to the ID of the span in every dict-entry.
            If this is a falsy value (e.g. None) COLUMN_THAT_CONTAINS_SPAN_IDS will be assumed as default.
        remove_trace_information: A boolean indicating whether to remove the information about traces for every entry in the result.
            If True the trace-ID, parent-span-ID and span-ID will be removed from every entry in the resulting tuples.
            Otherwise entries are left unchanged.

    Yields:
        A tuple like (trace_id, parent_span_id, span_id, entry).
        For every element in entries, entry is that original element or a modified copy of it.
    """
    if not trace_id_key:
        trace_id_key = COLUMN_THAT_CONTAINS_TRACE_IDS
    if not parent_span_id_key:
        parent_span_id_key = COLUMN_THAT_CONTAINS_PARENT_SPAN_IDS
    if not span_id_key:
        span_id_key = COLUMN_THAT_CONTAINS_SPAN_IDS

    for entry in entries:
        trace_id = entry[trace_id_key]
        parent_span_id = entry[parent_span_id_key]
        span_id = entry[span_id_key]
        if remove_trace_information:
            entry = copy.copy(entry)
            del entry[trace_id_key]
            del entry[parent_span_id_key]
            del entry[span_id_key]
        yield (trace_id, parent_span_id, span_id, entry)


def traces_from_trace_tuples(tuples, values_key=None):
    """
    Extract traces from the given entries.

    Args:
        tuples: A iterable of tuples (trace_id, parent_span_id, span_id, value).
        values_key: A key used to save the contents of each span.
            If this is not None every span will have all values corresponding to it saved under this key.
            If this is None no contents will be saved.

    Returns:
        A dict of traces.
        Each trace-entry has its ID as key and its value is a dict of other spans.
        A span-entry also has its ID as key and its value is a dict.
        Each span may contain other spans.
    """
    traces = dict()
    for trace in tuples:
        trace_id, parent_span_id, span_id, value = trace
        _, _, span = create_path(traces, trace_id, parent_span_id, span_id)
        if not values_key is None:
            if not values_key in span:
                span[values_key] = []
            span[values_key].append(value)

    return traces


def create_path(traces, trace_id, parent_span_id, span_id):
    """
    Ensure the path specified by the given IDs of trace, span and parent-span exists.
    Create it if necessary by creating the needed trace and spans.

    Args:
        traces: A dict of traces to ensure the path's existance in.
        trace_id: The ID of the trace that contains the spans.
        parent_span_id: The ID of the span that contains the span with ID span_id.
        span_id: The ID of the span whose parent-span is the span with ID parent_span_id.

    Returns:
        A tuple (trace, parent_span, span) containing the respective existing or newly-created entries.
    """
    if not trace_id in traces:
        traces[trace_id] = dict()
    trace = traces[trace_id]
    path = find_id(trace, parent_span_id)
    if path is None:
        parent_span = dict()
        trace[parent_span_id] = parent_span
    else:
        parent_span = trace
        for id in path:
            parent_span = parent_span[id]
    if not span_id in parent_span:
        path = find_id(trace, span_id)
        if span_id == parent_span_id or path is None:
            parent_span[span_id] = dict()
        else:
            original_parent = trace
            for id in path[:-1]:
                original_parent = original_parent[id]
            parent_span[span_id] = original_parent[span_id]
            del original_parent[span_id]
    span = parent_span[span_id]
    return trace, parent_span, span


def find_id(trace, id):
    """
    Try to find the ID in the given trace.
    This follows a depth-first approach.

    Args:
        trace: A dict to search through. This may be a trace or span.
        id: A key/ID to search for.

    Returns:
        A tuple of ID's that point to the desired ID in the trace, forming a 'path'.
        By following this 'path' through iterated access on the trace and the returned dict,
        one reaches the value the given ID points to.
        If the search is unsucessful, this returns None.
    """
    try:
        if id in trace:
            return (id,)
        for span_id, span in trace.items():
            result = find_id(span, id)
            if not result is None:
                return (span_id,) + result
    except:
        pass
    return None


def find_id_all(trace, id):
    """
    Try to find all occurrences of the ID in the given trace.
    This follows a depth-first approach.

    Args:
        trace: A dict to search through. This may be a trace or span.
        id: A key/ID to search for.

    Returns:
        A list of tuples of ID's that point to the desired ID in the trace, forming a 'path'.
        By following this 'path' through iterated access on the trace and the returned dict,
        one reaches the value the given ID points to.
        If the search is unsucessful, this returns an empty list.
    """
    paths = []
    try:
        if id in trace:
            paths.append((id,))
        for span_id, span in trace.items():
            result = find_id_all(span, id)
            for path in result:
                paths.append((span_id,) + path)
    except:
        pass
    return paths


def flatten(traces, trace_id_key=None, parent_span_id_key=None, span_id_key=None, values_key=None, parent_span_id_default=None, span_id_default=None):
    """
    Flatten the given traces into a list of entries.
    The resulting list could be parsed with `traces_from_explicit_entries` to reconstruct the original traces.

    Args:
        traces: A dict of traces.
        trace_id_key: A key that will point to the ID of the trace in every dict-entry.
            If this is a falsy value (e.g. None) COLUMN_THAT_CONTAINS_TRACE_IDS will be used as default.
        parent_span_id_key: A that key will point to the ID of the parent ot the span in every dict-entry.
            If this is a falsy value (e.g. None) COLUMN_THAT_CONTAINS_PARENT_SPAN_IDS will be used as default.
        span_id_key: A key that will point to the ID of the span in every dict-entry.
            If this is a falsy value (e.g. None) COLUMN_THAT_CONTAINS_SPAN_IDS will be used as default.
        values_key: A key used to save the contents of each span, if there are any.
            If this is not None the value for each entry will be saved under this key.
            If this is None this will try to unpack the value for each entry instead,
            it is then assumed that value is a mapping type;
            if it is not possible to unpack the value an exeption will be raised.
        parent_span_id_default: A value used if there is no parent-span-ID for a given entry.
            If this is not None an entry without a parent-span-ID (entries of top-level spans)
            will have this as the value for its parent-span-ID.
            If this is None entries without a parent-span-ID will be discarded.
        span_id_default: A value used if there is no span-ID for a given entry.
            If this is not None an entry without a span-ID (entries of traces)
            will have this as the value for its span-ID.
            If this is None entries without a span-ID will be discarded.

    Raises:
        TypeError: If values_key is None and a value can not be unpacked.

    Returns:
        A list of collections.OrderedDict-entries.
    """
    if not trace_id_key:
        trace_id_key = COLUMN_THAT_CONTAINS_TRACE_IDS
    if not parent_span_id_key:
        parent_span_id_key = COLUMN_THAT_CONTAINS_PARENT_SPAN_IDS
    if not span_id_key:
        span_id_key = COLUMN_THAT_CONTAINS_SPAN_IDS

    result = []
    for trace_id, trace in traces.items():
        trace_entries = _flatten_span(trace, parent_span_id_key, span_id_key, values_key)
        def validated_entries():
            for entry in trace_entries:
                if span_id_key not in entry:
                    if span_id_default is None:
                        continue
                    entry[span_id_key] = span_id_default
                    entry.move_to_end(span_id_key, last=False)  # move to first position
                if parent_span_id_key not in entry:
                    if parent_span_id_default is None:
                        continue
                    entry[parent_span_id_key] = parent_span_id_default
                    entry.move_to_end(parent_span_id_key, last=False)
                entry[trace_id_key] = trace_id
                entry.move_to_end(trace_id_key, last=False)
                yield entry
        trace_entries[:] = validated_entries()  # filter and modify entries without creating a copy
        result.extend(trace_entries)

    return result


def _flatten_span(span, parent_span_id_key, span_id_key, values_key):
    """
    Flattens the given span into a list of entries.
    See `flatten` for more information.
    """
    if not span:
        return [collections.OrderedDict()]

    result = []
    has_entries = False
    for span_id, span in span.items():
        if isinstance(span, dict):
            trace_entries = _flatten_span(span, parent_span_id_key, span_id_key, values_key)
            for entry in trace_entries:
                if not span_id_key in entry:
                    entry[span_id_key] = span_id
                    entry.move_to_end(span_id_key, last=False)  # move to first position
                elif not parent_span_id_key in entry:
                    entry[parent_span_id_key] = span_id
                    entry.move_to_end(parent_span_id_key, last=False)
            result.extend(trace_entries)
        elif isinstance(span, list):
            for value in span:
                entry = collections.OrderedDict()
                if values_key is None:
                    try:
                        entry.update(value)
                    except Exception as ex:
                        raise TypeError("could not unpack value") from ex
                else:
                    entry[values_key] = value
                result.append(entry)
                has_entries = True
        else:
            entry = collections.OrderedDict()
            value = {span_id: span}
            if values_key is None:
                entry.update(value)
            else:
                entry[values_key] = value
            result.append(entry)
            has_entries = True

    if not has_entries:
        result.append(collections.OrderedDict())

    return result
