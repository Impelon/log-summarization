import argparse
import json
import runpy

class ArgumentParserFromFile(argparse.ArgumentParser):

    def __init__(self, python_file_option_dict="options", **kwargs):
        super().__init__(**kwargs)
        self.python_file_option_dict = python_file_option_dict

    def _read_args_from_files(self, arg_strings):
        new_arg_strings = []
        for arg_string in arg_strings:
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)
                continue
            arg_path = arg_string[1:]
            if arg_path.endswith(".json"):
                with open(arg_path, "r") as file:
                    options = json.load(file)
            elif arg_path.endswith(".py"):
                options = runpy.run_path(arg_path)[self.python_file_option_dict]
            else:
                raise ValueError("Unknown file type for reading arguments from.")
            if hasattr(options, "items"):
                option_items = options.items()
            else:
                option_items = options

            for item in option_items:
                name = item[0]
                new_arg_strings.append("--" + str(name))
                if len(item) == 2:
                    value = item[1]
                    if isinstance(value, (list, tuple, set)):
                        for element in value:
                            new_arg_strings.append(str(element))
                    else:
                        new_arg_strings.append(str(value))
        return new_arg_strings
