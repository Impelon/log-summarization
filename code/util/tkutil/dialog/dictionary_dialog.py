from .abstract_parameter_dialog import *

__all__ = ["askdictionary", "DictionaryDialog", "ParameterSpecification"]

def askdictionary(**kwargs):
    """
    Open a dialog that lets the user configure values contained in a dictionary.

    Args:
        **kwargs: Additional keyword-arguments passed to `DictionaryDialog`.

    Returns:
        A dict or None.
    """
    dialog = DictionaryDialog(**kwargs)
    return dialog.result

class DictionaryDialog(AbstractParameterDialog):

    """
    Class to open a dialog for configuring values contained in a dictionary.

    The result-attribute will be either None or a dict.
    """

    def __init__(self, parent=None, title=None, initialvalue=None, allow_empty_selection=False, parameters=None, separator=None):
        """
        Initialisation of a DictionaryDialog.

        Args:
            parent: A parent window or None.
            title: A string that represents the title of the dialog or None.
            initialvalue: A dict to use as an initial value or None.
                If the user cancels the dialog this value will be the result.
                When possible the parameters of this instance will be chosen as defaults for the configurable parameters.
            allow_empty_selection: A boolean indicating whether the user should be able to specify not to return a TextWrapper.
                If True the user is shown an option not to create a TextWrapper, in which case the result will be None.
                If False the user is not shown this option.
            parameters: An iterable (or a dict of iterables) of tuples like (name, prompt_text, type, allow_none); see also ParameterSpecification.
                For each tuple an input-widget will be created that points to the parameter.
                If this is a dict, every entry will create a labeled frame (with the entry-key as label)
                which contains all input-widgets for the tuples in the entry-value.
                If this is a falsy value (e.g. None), no parameters will be added to the dialog.
            separator: A string that separates dictionary-keys in a parameter name.
                If is possible to configure nested dicts with this, for example:

                Let the separator be "." and the parameter-names be "a.1", "a.2", "a.3.1", "b".
                The resulting dict would look like:
                {"a": {"1": <value of a.1>, "2": <value of a.2>, "3": {"1": <value of a.3.1}}, "b": <value of b>}

                If this is a falsy value (e.g. None), nested dictionaries will not be created.
        """
        self.original_dictionary = initialvalue
        if not initialvalue:
            initialvalue = {}
        self.dictionary = initialvalue.copy()
        self.separator = separator

        super(DictionaryDialog, self).__init__(parent=parent, title=title, allow_empty_selection=allow_empty_selection, parameters=parameters)

    @property
    def _empty_selection_prompt_text(self):
        return "Do not configure any parameters"

    def _get_parameter(self, name):
        if not self.separator:
            return self.dictionary.get(name, None)
        path = name.split(self.separator)
        value = self.dictionary
        for key in path:
            value = value.get(key, None)
            if value is None:
                break
        return value

    def _set_parameter(self, name, value):
        if not self.separator:
            self.dictionary[name] = value
            return
        path = name.split(self.separator)
        dict = self.dictionary
        for key in path[:-1]:
            if not key in dict:
                dict[key] = {}
            dict = dict[key]
        dict[path[-1]] = value

    def _create_parameter_variable(self, variable_class, name, callback=None):
        variable = super(DictionaryDialog, self)._create_parameter_variable(variable_class, name, callback)
        # make sure a value is set for the resulting dict,
        # even if the variable is not changed by the user
        value = variable.get()
        if callback:
            value = callback(value)
        self._set_parameter(name, value)
        return variable

    def body(self, master):
        self.result = self.original_dictionary
        if self.allow_empty_selection and not self.original_dictionary:
            self.has_empty_selection.set(True)
        return super(DictionaryDialog, self).body(master)

    def apply(self):
        self.result = self.dictionary
        super(DictionaryDialog, self).apply()
