from collections import namedtuple

from tkinter.simpledialog import Dialog
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *

__all__ = ["AbstractParameterDialog", "ParameterSpecification"]

ParameterSpecification = namedtuple("ParameterSpecification", ["name", "prompt_text", "type", "allow_none"])
ParameterSpecification.__doc__ = """
Named tuple that includes the specification for a parameter that can be configured with this dialog.

Attributes:
    name: A key to refer the parameter by internally.
    prompt_text: A string to display besides any input-widget.
    type: A type allowed for the parameter. Should either be str, float, int, bool or a callable or a list.
        If this is a callable, its returned value will be used for the parameter when invoked by the user.
        If this is a list, the user will only be able to choose between the provided values.
    allow_none: A boolean indicating whether the parameter can also be None.
"""

def _can_convert_to(x, type_constructor):
    try:
        type_constructor(x)
        return True
    except:
        return False


class AbstractParameterDialog(Dialog):

    """
    Abstract base class for dialogs prompting the user for multiple values.
    This base provides a way to create an uniform UI.

    The result-attribute will be None, if empty selection is enabled and the user selects it.
    """

    def __init__(self, parent=None, title=None, allow_empty_selection=False, parameters=None):
        """
        Initialization of a AbstractParameterDialog.

        Args:
            parent: A parent window or None.
            title: A string that represents the title of the dialog or None.
            allow_empty_selection: A boolean indicating whether the user should be able to specify not to return a result.
                If True the user is shown an option for an empty selection, in which case the result will be None.
                If False the user is not shown this option.
            parameters: An iterable (or a dict of iterables) of tuples like (name, prompt_text, type, allow_none); see also ParameterSpecification.
                For each tuple an input-widget will be created that points to the parameter.
                If this is a dict, every entry will create a labeled frame (with the entry-key as label)
                which contains all input-widgets for the tuples in the entry-value.
                If this is a falsy value (e.g. None), no parameters will be added to the dialog.
        """
        self.allow_empty_selection = allow_empty_selection
        self.has_empty_selection = tk.BooleanVar(value=False)
        self.parameters = parameters
        self._readonly_widgets = set()

        super(AbstractParameterDialog, self).__init__(parent, title)

    @property
    def _empty_selection_prompt_text(self):
        """
        Return a text to be displayed for the button that allows the empty selection.
        """
        pass

    def _get_parameter(self, name):
        """
        Return the value of the parameter with the given name or None if the value is unknown.
        """
        pass

    def _set_parameter(self, name, value):
        """
        Set the value of the parameter with the given name.
        """
        pass

    def _create_parameter_variable(self, variable_class, name, callback=None):
        """
        Create a variable for the given parameter.

        Args:
            variable_class: A class of a tkinter-variable that should be used for this parameter.
            name: The name of the parameter this variable is created for.
            callback: A callback that should be applied to the value of the variable to determine
                its indented value when reading it.
                If this is a falsy value (e.g. None), no such callback is applied.

        Returns:
            The variable created.
        """
        variable = variable_class(self, value=self._get_parameter(name))

        def update(*args):
            value = variable.get()
            if callback:
                value = callback(value)
            self._set_parameter(name, value)
        variable.trace_add("write", update)
        return variable

    def _get_properties_for_parameter(self, parameter):
        """
        Returns a tuple of properties that can be derived for the given parameter.

        Args:
            parameter: A tuple like (name, prompt_text, type, allow_none); see also ParameterSpecification.

        Returns:
            A tuple like (variable_class, validatecommand, callback, pady).
        """
        variable_class = tk.StringVar
        validatecommand = None
        callback = None
        pady = 0
        if isinstance(parameter[2], type):
            if parameter[2] == int:
                variable_class = tk.IntVar
                if len(parameter) > 3 and parameter[3]:
                    validatecommand = self._int_or_empty_validation
                    callback = lambda x: None if not x else int(x)
                else:
                    validatecommand = self._int_validation
            elif parameter[2] == float:
                variable_class = tk.DoubleVar
                if len(parameter) > 3 and parameter[3]:
                    validatecommand = self._float_or_empty_validation
                    callback = lambda x: None if not x else float(x)
                else:
                    validatecommand = self._float_validation
            elif parameter[2] == bool:
                variable_class = tk.BooleanVar
        else:
            variable_class = tk.Variable

        if len(parameter) > 3 and parameter[3]:
            variable_class = tk.Variable
            if callback is None:
                callback = lambda x: None if not x else x
        if "\n" in parameter[1]:
            pady = 5
        return variable_class, validatecommand, callback, pady

    def _place_boolean_widget(self, frame, parameter):
        """
        Place an input-widget for a boolean parameter.
        See also self._place_parameter_widget.
        """
        v = self._create_parameter_variable(tk.BooleanVar, parameter[0])
        w = ttk.Checkbutton(frame, text=parameter[1], variable=v)
        w.pack(fill=Y, side=TOP, anchor=W)

    def _place_enumeration_widget(self, frame, parameter):
        """
        Place an input-widget for a parameter that can have multiple predefined values.
        See also self._place_parameter_widget.
        """
        properties = self._get_properties_for_parameter(parameter)
        g = ttk.Frame(frame)
        g.pack(fill=BOTH, side=TOP, anchor=W, pady=properties[3])
        w = ttk.Label(g, text=parameter[1])
        w.pack(fill=Y, side=LEFT, anchor=E, padx=(0, 5))
        v = self._create_parameter_variable(properties[0], parameter[0], callback=properties[2])
        w = ttk.Combobox(g, textvariable=v, values=list(parameter[2]), state="readonly")
        w.pack(side=RIGHT, anchor=W)
        self._readonly_widgets.add(w)

    def _place_callable_widget(self, frame, parameter):
        """
        Place an input-widget for a parameter whose value is returned by a callable.
        See also self._place_parameter_widget.
        """
        properties = self._get_properties_for_parameter(parameter)
        g = ttk.Frame(frame)
        g.pack(fill=BOTH, side=TOP, anchor=W)
        v = self._create_parameter_variable(properties[0], parameter[0], callback=properties[2])
        w = ttk.Button(g, text=parameter[1], command=lambda: v.set(parameter[2]()))
        w.pack(fill=Y, side=LEFT, anchor=E)
        w = ttk.Entry(g, textvariable=v, state="readonly")
        w.pack(fill=BOTH, side=RIGHT, anchor=W, expand=True)
        self._readonly_widgets.add(w)

    def _place_freeform_widget(self, frame, parameter):
        """
        Place an input-widget for a free-form parameter.
        See also self._place_parameter_widget.
        """
        properties = self._get_properties_for_parameter(parameter)
        g = ttk.Frame(frame)
        g.pack(fill=BOTH, side=TOP, anchor=W, pady=properties[3])
        w = ttk.Label(g, text=parameter[1])
        w.pack(fill=Y, side=LEFT, anchor=E, padx=(0, 5))
        v = self._create_parameter_variable(properties[0], parameter[0], callback=properties[2])
        w = ttk.Entry(g, textvariable=v)
        if properties[1]:
            w.config(validate="key", validatecommand=properties[1])
        w.pack(side=RIGHT, anchor=W)

    def _place_parameter_widget(self, frame, parameter):
        """
        Create an input-widget appropriate for the given parameter and positions it on the given frame.
        The widget's variable will be created and linked to the parameter as well.

        Args:
            frame: A tkinter-widget that is able to contain other widgets.
            parameter: A tuple like (name, prompt_text, type, allow_none); see also ParameterSpecification.
        """
        if isinstance(parameter[2], type):
            if parameter[2] == bool:
                self._place_boolean_widget(frame, parameter)
            else:
                self._place_freeform_widget(frame, parameter)
        else:
            if callable(parameter[2]):
                self._place_callable_widget(frame, parameter)
            else:
                self._place_enumeration_widget(frame, parameter)

    def body(self, master):
        self._int_validation = (self.register(lambda x: _can_convert_to(x, int)), "%P")
        self._int_or_empty_validation = (self.register(lambda x: True if len(x) == 0 else _can_convert_to(x, int)), "%P")
        self._float_validation = (self.register(lambda x: _can_convert_to(x, float)), "%P")
        self._float_or_empty_validation = (self.register(lambda x: True if len(x) == 0 else _can_convert_to(x, float)), "%P")
        frames = []

        def add_parameter_widgets(frame, parameters):
            for parameter in parameters:
                self._place_parameter_widget(frame, parameter)

        def toggle_active(*args, parents=frames):
            state = "normal"
            if self.has_empty_selection.get():
                state = "disabled"
            toggle_next = []
            for parent in parents:
                for widget in parent.winfo_children():
                    widget_state = state
                    try:
                        if widget_state == "normal" and widget in self._readonly_widgets:
                            widget_state = "readonly"
                        widget.configure(state=widget_state)
                    except:
                        toggle_next.append(widget)
            if toggle_next:
                toggle_active(parents=toggle_next)

        if self.parameters:
            try:
                for label_text, parameters in self.parameters.items():
                    frame = ttk.Labelframe(master, text=label_text, relief=RIDGE, borderwidth=2)
                    frame.pack(fill=BOTH, side=BOTTOM)
                    frames.append(frame)
                    add_parameter_widgets(frame, parameters)
            except:
                frame = ttk.Frame(master, relief=RIDGE, borderwidth=2)
                frame.pack(fill=BOTH, side=BOTTOM)
                frames.append(frame)
                add_parameter_widgets(frame, self.parameters)

        toggle_active()
        if self.allow_empty_selection:
            w = ttk.Checkbutton(master, text=str(self._empty_selection_prompt_text), variable=self.has_empty_selection, command=toggle_active)
            w.pack(fill=Y, side=TOP, anchor=W)

        return master

    def apply(self):
        if self.allow_empty_selection and self.has_empty_selection.get():
            self.result = None
