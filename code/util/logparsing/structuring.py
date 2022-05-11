import re

from .exceptions import *

LOG_FORMATS = {
    "android":     "<Date> <Time> <PID> <TID> <Level> <Component>: <Message>",
    "apache":      "\[<Time>\] \[<Level>\] <Message>",
    "bgl":         "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Message>",
    "hdfs":        "<Date> <Time> <PID> <Level> <Component>: <Message>",
    "hpc":         "<LogID> <Node> <Component> <State> <Time> <Flag> <Message>",
    "hadoop":      "<Date> <Time> <Level> \[<Process>\] <Component>: <Message>",
    "healthapp":   "<Time>\|<Component>\|<PID>\|<Message>",
    "linux":       "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Message>",
    "mac":         "<Month> <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Message>",
    "openssh":     "<Date> <Day> <Time> <Component> sshd\[<PID>\]: <Message>",
    "openstack":   "<Logrecord> <Date> <Time> <PID> <Level> <Component> \[<Address>\] <Message>",
    "proxifier":   "\[<Time>\] <Program> - <Message>",
    "spark":       "<Date> <Time> <Level> <Component>: <Message>",
    "thunderbird": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Message>",
    "windows":     "<Date> <Time>, <Level> <Component> <Message>",
    "zookeeper":   "<Date> <Time> - <Level> \[<Node>:<Component>@<ID>\] - <Message>",
}


def construct_log_format_regex(log_format, ignore_surrounding_whitespace=True):
    """
    Generate a regular expression to parse log lines.

    Args:
        log_format: A string specifing the format of the log.
            A string identifier between angle brackets "<" and ">" in the format-string defines a field that needs to be extracted.
            The format of the field can optionally be specified as a regular expression (that does not contain angle brackets)
            following a pipe-character "|" after the string identifier.
            If no such format is specified, the field will be matched by ".*?" as a regular expression.
            Any number of spaces in the format-string define a non-empty sequence of whitespace.
            All other character sequences will be part of the final regular expression and will be interpreted as such.
            For example the following is a crude format specifier matching a class definition in python
            with the class name and potential base classes as fields:

            r"class <ClassName>(\(<BaseClasses|[a-zA-Z0-9_, ]*>\))?:"

        ignore_surrounding_whitespace: A boolean indicating whether the final expression should ignore leading and trailing whitespace.

    Returns:
        A tuple like (fields, regex).
        fields is a sequence of fields specified in log_format.
        regex is a regular expression object (re.Pattern) for parsing log lines according to log_format.
    """
    separators = re.split(r"(<[^<>]+>)", log_format)
    fields = []
    regex = ""
    for k in range(len(separators)):
        if k % 2 == 0:
            separator = re.sub(" +", lambda x: r"\s+", separators[k])
            regex += separator
        else:
            field = separators[k][1:-1]
            pattern = ".*?"
            if "|" in field:
                field, pattern = field.split("|", 1)
            regex += "(?P<{}>{})".format(field, pattern)
            fields.append(field)
    if ignore_surrounding_whitespace:
        regex = r"\s*" + regex + r"\s*"
    regex = re.compile("^" + regex + "$")
    return tuple(fields), regex


def log_to_log_entries(file_path, log_format, ignore_surrounding_whitespace=True, line_number_field=None):
    """
    Generate a sequence of log entries from a log file.

    Args:
        file_path: A path-like object that points to the log file.
        log_format: A string specifing the format of the log.
            String identifiers between angle brackets in the format-string define a field that needs to be extracted.
            Any number of spaces in the format-string define a non-empty sequence of whitespace.
            All other character sequences and will be interpreted as regular expressions.
        ignore_surrounding_whitespace: A boolean indicating whether leading and trailing whitespace should be ignored for each line.
        line_number_field: A string used as the field name for the line number of each entry.
            If this is None, the line number will not be present in the resulting entries.

    Returns:
        An iterator of dicts representing log entries.
        Every dict has a key for each field in log_format and a value for every field.

    Raises:
        ImproperlyFormattedLogLineError: Raised by the iterator when a line did not adhere to the syntax specified by log_format.
            The iterator can safely be resumed after handling this error.
    """
    with open(file_path, "r") as file:
        return lines_to_log_entries(file, log_format, ignore_surrounding_whitespace=ignore_surrounding_whitespace, line_number_field=line_number_field)


def lines_to_log_entries(lines, log_format, ignore_surrounding_whitespace=True, line_number_field=None):
    """
    Generate a sequence of log entries from an iterable of log lines.

    Args:
        lines: An iterable of log lines.
        log_format: A string specifing the format of the log.
            A string identifier between angle brackets in the format-string defines a field that needs to be extracted.
            Any number of spaces in the format-string define a non-empty sequence of whitespace.
            All other character sequences and will be interpreted as regular expressions.
        ignore_surrounding_whitespace: A boolean indicating whether leading and trailing whitespace should be ignored for each line.
        line_number_field: A string used as the field name for the line number of each entry.
            If this is None, the line number will not be present in the resulting entries.

    Returns:
        An iterator of dicts representing log entries.
        Every dict has a key for each field in log_format and a value for every field.

    Raises:
        ImproperlyFormattedLogLineError: Raised by the iterator when a line did not adhere to the syntax specified by log_format.
            The iterator can safely be resumed after handling this error.
    """
    class EntryIterator:
        def __init__(self, fields, regex, lines):
            self.numbered_lines = iter(enumerate(lines, 1))

        def __iter__(self):
            return self

        def __next__(self):
            i, line = next(self.numbered_lines)
            match = regex.search(line)
            if not match:
                raise ImproperlyFormattedLogLineError(line, i)
            entry = {field: match.group(field) for field in fields}
            entry = {field: value if value is None else value.strip() for field, value in entry.items()}
            if line_number_field:
                entry[line_number_field] = i
            return entry

    fields, regex = construct_log_format_regex(log_format, ignore_surrounding_whitespace=ignore_surrounding_whitespace)
    return EntryIterator(fields, regex, lines)


def iterate_with_exception_handler(iterator, exception_handler):
    """
    Yields elements from iterator.
    If the iterator ever raises an exception other than StopIteration,
    exception_handler will be called with that exception.
    If exception_handler does not raise another exception,
    this will continue to yield elements from the iterator.

    Args:
        iterator: An iterator to iterate.
        exception_handler: A function accepting an exception as input.
            It should meaningfully handle the exception if possible and raise it otherwise.

    Yields:
        Elements from the iterator.
    """
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception as ex:
            exception_handler(ex)


def collect_with_exception_handler(iterator, exception_handler):
    """
    Collects all elements from iterator.
    If the iterator ever raises an exception other than StopIteration,
    exception_handler will be called with that exception and all elements collected so far.
    If exception_handler does not raise another exception,
    this will continue to yield elements from the iterator.

    Args:
        iterator: An iterator to iterate.
        exception_handler: A function accepting an exception and a sequence of elements as input.
            It should meaningfully handle the exception if possible and raise it otherwise.

    Returns:
        A sequence of all elements from the iterator.
    """
    elements = []
    while True:
        try:
            elements.append(next(iterator))
        except StopIteration:
            return elements
        except Exception as ex:
            exception_handler(ex, elements)


def create_appending_exception_handler(field_to_append_to, separator, previous_log_entries=None):
    """
    Returns an exception_handler conforming to the specification of collect_with_exception_handler.
    If the exception is ImproperlyFormattedLogLineError and there exists a previous entry,
    the handler appends the problematic line to the specified field of the previous log entry.
    Otherwise the handler will raise the exception again.

    Args:
        field_to_append_to: A string specifing the field to which improperly formatted lines are appended to.
        separator: A string to insert between the specified field of the previous log entry and the improperly formatted line.
        previous_log_entries: An optional list of log entries to use in case no log_entries were supplied.
            If this is not None, the resulting exception_handler can be used to handle exceptions with iterate_with_exception_handler.

    Returns:
        A exception_handler conforming to the specification of collect_with_exception_handler.
    """
    def exception_handler(exception, log_entries=None):
        if not log_entries:
            log_entries = previous_log_entries
        if not isinstance(exception, ImproperlyFormattedLogLineError):
            raise exception
        if not log_entries:
            raise exception
        log_entries[-1][field_to_append_to] += separator + exception.line.strip()
    return exception_handler
