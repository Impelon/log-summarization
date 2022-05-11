from .exceptions import *
from .structuring import *
from .abstraction import *

import sys
import csv
import itertools
import logging
logger = logging.getLogger(__name__)

def main_abstract(program_name, *arguments):
    if "-h" in arguments or "--help" in arguments:
        print("A script that reads log entries as CSV from the standard input, parse static (templates) and dynamic (parameters) segments of log messages",
              "and prints the resulting log events as CSV on the standard output.")
        print()
        print("python3 " + program_name + " abstract [<option>]... <message-column> <similarity-threshold> <depth> [<masking-pattern>]...")
        print("or")
        print("python3 " + program_name + " abstract [<option>]... -p=<preset> <message-column>")
        print("Options:")
        print(" -h              Show help.")
        print(" -l              Print available parser settings presets.")
        print(" -p=<preset>     Use the given preset to specify options passed to the parser.")
        print(" -n=<names>      Specify a name for the template, parameters and event id column-headers.")
        print("                 Each name should be separated by a colon like <template>:<parameters>:<id>")
        print("                 If no names have been specified using this option, no header-row will be printed, even if it is assumed that one exists.")
        print(" -s=<mode>       Print events and templates separately according to the given separation mode.")
        print("Message column:")
        print(" Specifies the name or index of the column that contains log messages.")
        print(" If it is an integer, it will be used as the index of the column and it is assumed that there is no header-row.")
        print(" Otherwise the column with the corresponding header will be chosen, if possible.")
        print("Similarity Threshold:")
        print(" Specifies a threshold for similarity between 0 and 1 to decide when to assign to an existing log event or create a new one.")
        print("Depth:")
        print(" Specifies the maximum depth of the parse tree.")
        print("Masking Pattern:")
        print(" Specifies a regular expression pattern that will be used in preprocessing.")
        print(" When parsing messages, text segments matching these patterns will become parameters in the resulting templates.")
        print("Separation Mode:")
        print(" -s=n[one]       All event information will be printed on standard output together with any other CSV-data. (default)")
        print(" -s=s[tderr]     Event IDs and their parameters will be printed on standard output together with any other CSV-data.")
        print("                 Templates identified by their event ID will be printed on standard error.")
        print(" -s=:<separator> Log event IDs and their parameters will be printed on standard output together with any other CSV-data.")
        print("                 The separator will be printed before the templates identified by their event ID will be printed on standard output.")
        sys.exit(0)
    if "-l" in arguments or "--list-presets" in arguments:
        print("These presets for parser settings are available:")
        for name, settings in PARSER_SETTINGS.items():
            print("{:>15}: similarity-threshold: {:.3f} depth: {} masking_patterns: {}".format(name, *settings.values()))
        sys.exit(0)

    parameters = tuple(filter(lambda x: not x.startswith("-"), arguments))
    if len(parameters) < 1:
        print("You need to specify a column that contains the log message in the CSV.")
        print("Try -h for further information.")
        sys.exit(2)

    preset = None
    template_header, parameters_header, event_header = None, None, None
    separate_templates = False
    template_separator = ""
    template_destination = sys.stdout
    for flag in arguments:
        if flag.startswith("-p=") or flag.startswith("--preset="):
            preset = flag.split("=", 1)[1].lower()
            if preset not in PARSER_SETTINGS.keys():
                print("'{}' is not a known log-format and cannot be used as argument for -p.".format(preset))
                print("Try -l to list all presets for parser settings.")
                sys.exit(2)
        if flag.startswith("-n=") or flag.startswith("--named-columns="):
            template_header, parameters_header, event_header = flag.split("=", 1)[1].split(":")
        if flag.startswith("-s=") or flag.startswith("--separation-mode="):
            mode_specifier = flag.split("=", 1)[1]
            if mode_specifier in ("n", "none"):
                separate_templates = False
            elif mode_specifier in ("s", "stderr"):
                separate_templates = True
                template_destination = sys.stderr
            elif mode_specifier[0] == ":":
                separate_templates = True
                template_separator = mode_specifier[1:]
                template_destination = sys.stdout
            else:
                print("'{}' is not a known mode and cannot be used as argument for -s.".format(mode_specifier))
                sys.exit(2)

    if preset:
        similarity_threshold = PARSER_SETTINGS[preset]["similarity_threshold"]
        depth = PARSER_SETTINGS[preset]["depth"]
        masking_patterns = PARSER_SETTINGS[preset]["masking_patterns"]
        if len(parameters) > 1:
            print("Further parser settings will not be processed when using a preset.")
            print("Try -h for further information.")
            sys.exit(2)
    elif len(parameters) > 2:
        similarity_threshold = float(parameters[1])
        depth = int(parameters[2])
        masking_patterns = parameters[3:]
    else:
        print("You need to specify settings to configure the parser.")
        print("Try -h for further information.")
        sys.exit(2)

    event_writer = csv.writer(sys.stdout)
    if separate_templates:
        template_writer = csv.writer(template_destination)
        template_destination.write(template_separator)

    reader = csv.reader(sys.stdin)
    headers = []
    try:
        message_column_index = int(parameters[0])
    except ValueError:
        headers = next(reader)
        try:
            message_column_index = headers.index(parameters[0])
        except ValueError:
            print("'{}' did not match the name of any column in the first row.".format(parameters[0]))
            sys.exit(2)

    configuration = apply_settings(similarity_threshold=similarity_threshold, depth=depth, masking_patterns=masking_patterns)
    csv_data = list(reader)
    event_instances, event_types = extract_events_from_log_messages([row[message_column_index] for row in csv_data], configuration)

    if template_header is not None and parameters_header is not None and event_header is not None:
        if separate_templates:
            event_writer.writerow(headers + [event_header, parameters_header])
            template_writer.writerow([event_header, template_header])
        else:
            event_writer.writerow(headers + [event_header, parameters_header, template_header])

    if separate_templates:
        for row, instance in zip(csv_data, event_instances):
            event_writer.writerow(itertools.chain(row, instance))
        for id, template in event_types.items():
            template_writer.writerow((id, template))
    else:
        for row, (id, parameters) in zip(csv_data, event_instances):
            event_writer.writerow(itertools.chain(row, (id, parameters, event_types[id])))

def main_structure(program_name, *arguments):
    print_regex = False

    if "-h" in arguments or "--help" in arguments:
        print("A script that reads log lines from the standard input, extracts structured log entries",
              "and prints the result as CSV on the standard output.")
        print()
        print("python3 " + program_name + " structure [<option>]... <log-format> [<line-number-field>]")
        print("Options:")
        print(" -h              Show help.")
        print(" -l              Print available log format presets.")
        print(" -x              Print the fields and RegEx corresponding to the format and exit instead.")
        print(" -n=<field>      Discard entries with empty values for any field specified by this option.")
        print(" -e=<handler>    Specify an exception handler for log lines that are not formatted accordingly.")
        print("Log Format:")
        print(" Specifies how log lines are parsed.")
        print(" Can either be the name of a format preset, or a custom format.")
        print("Field:")
        print(" Fields represent distinct types of information (like time, log level, etc.) in log lines.")
        print(" Each field will have a corresponding column in the CSV output.")
        print("Line Number Field:")
        print(" Specifies a field name for the line number of each entry.")
        print(" If no such field name is given, the line number will not be present in the resulting entries.")
        print("Exception Handler:")
        print(" -e=a[bort]              When a log line does not adhere to the log-format,")
        print("                         the program will raise an exception and abort. (default)")
        print(" -e=l[og]                When a log line does not adhere to the log-format,")
        print("                         the program will print an exception and continue.")
        print(" -e=:<field>             When a log line does not adhere to the log-format,")
        print("                         that line will be appended to the given field of the previous entry")
        print("                         using a newline as separator.")
        print(" -e=:<field>:<separator> When a log line does not adhere to the log-format,")
        print("                         that line will be appended to the given field of the previous entry")
        print("                         using the specified separator.")
        sys.exit(0)
    if "-l" in arguments or "--list-presets" in arguments:
        print("These presets for log formats are available:")
        for name, format in LOG_FORMATS.items():
            print("{:>15}: {}".format(name, format))
        sys.exit(0)
    if "-x" in arguments or "--regex" in arguments:
        print_regex = True

    parameters = tuple(filter(lambda x: not x.startswith("-"), arguments))
    if len(parameters) < 1:
        print("You need to specify the name of a preset or specify your own format for the log data.")
        print("Try -h for further information.")
        sys.exit(2)

    log_format = LOG_FORMATS.get(parameters[0].lower(), parameters[0])
    headers, regex = construct_log_format_regex(log_format)

    if print_regex:
        print("Fields:")
        print(",".join(headers))
        print()
        print("RegEx:")
        print(regex.pattern)
        sys.exit(0)

    line_number_field = None
    if len(parameters) > 1:
        line_number_field = parameters[1]
        headers = (line_number_field,) + headers

    invalid_field = False
    non_empty_fields = []
    def iterator_transformation(x): return x
    for flag in arguments:
        if flag.startswith("-n=") or flag.startswith("--non-empty="):
            field = flag.split("=", 1)[1]
            if field not in headers:
                print("'{}' is not a field contained in the log-format and cannot be used as argument for -n.".format(field))
                invalid_field = True
            else:
                non_empty_fields.append(field)
        if flag.startswith("-e=") or flag.startswith("--exception-handler="):
            handler_specifier = flag.split("=", 1)[1]
            if handler_specifier in ("a", "abort"):
                def iterator_transformation(x): return x
            elif handler_specifier in ("l", "log"):
                def iterator_transformation(x):
                    def handler(exception):
                        if not isinstance(exception, ImproperlyFormattedLogLineError):
                            raise exception
                        logger.warning("Discarding invalid log line:")
                        logger.warning(exception)
                    return iterate_with_exception_handler(x, handler)
            elif handler_specifier[0] == ":":
                handler_specifier = handler_specifier[1:]
                separator = "\n"
                if ":" in handler_specifier:
                    handler_specifier, separator = handler_specifier.split(":", 1)
                if handler_specifier in headers:
                    handler = create_appending_exception_handler(handler_specifier, separator)
                    def iterator_transformation(x): return collect_with_exception_handler(x, handler)
                else:
                    print("'{}' is not a field contained in the log-format and cannot be used as argument for -e.".format(handler_specifier))
                    invalid_field = True
            else:
                print("'{}' is not a known exception handler and cannot be used as argument for -e.".format(handler_specifier))
                sys.exit(2)

    if invalid_field:
        print("Try -x to list the fields of any log-format.")
        sys.exit(2)

    entrywriter = csv.DictWriter(sys.stdout, headers)
    entrywriter.writeheader()
    entries = lines_to_log_entries(sys.stdin, log_format, line_number_field=line_number_field)
    for entry in iterator_transformation(entries):
        if all(bool(entry[field]) for field in non_empty_fields):
            entrywriter.writerow(entry)
        else:
            logger.warning("Discarding invalid log entry with empty fields:")
            logger.warning(entry)


if __name__ == "__main__":
    arguments_iterator = enumerate(sys.argv)
    index, program_name = next(arguments_iterator)
    for index, argument in arguments_iterator:
        if not argument.startswith("-"):
            break
    else:
        index += 1
    options = sys.argv[1:index]

    if "-h" in options or "--help" in options:
        print("A script for parsing log data.")
        print()
        print("python3 " + program_name + " [option]... <command> [<command-arguments>]...")
        print("Options:")
        print(" -h              Show this help-page.")
        print("                 To show help for any command, pass -h as a command-argument.")
        print("Commands:")
        print(" structure       Read log lines from the standard input and extract structured log entries as CSV on the standard output.")
        print(" abstract        Read log entries as CSV from the standard input, parse static and dynamic segments in log messages and extract log events as CSV on the standard output.")
        sys.exit(0)

    if index >= len(sys.argv):
        print("You need to specify a command.")
        print("Try -h for further information.")
        sys.exit(2)

    command = sys.argv[index]
    command_arguments = sys.argv[index + 1:]
    if command == "structure":
        main_structure(program_name, *command_arguments)
    elif command == "abstract":
        main_abstract(program_name, *command_arguments)
    else:
        print("Unknown command: {}".format(command))
        print("Try -h for further information.")
        sys.exit(2)
