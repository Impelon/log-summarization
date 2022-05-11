from .abstract_parameter_dialog import *

import textwrap
import copy

import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *

__all__ = ["asktextwrapper", "TextWrapperDialog", "ParameterSpecification"]

DEFAULT_PREVIEW_TEXT = ("This is a sample text. "
                        "It will be separated into multiple lines, if it surpasses the character-limit. "
                        "Thus the effects of the different settings can be observed here. ")

_PARAMETERS = {
    "Properties of text-wrapper": [
        ParameterSpecification("width", "The character-width to break text at", int, False),
        ParameterSpecification("max_lines", "The limit of times to break a text after which \nthe rest will be replaced by the placeholder \n(leave blank if there should not be such a limit)", int, True),
        ParameterSpecification("placeholder", "The placeholder to replace text with that is too long", str, False),
        ParameterSpecification("tabsize", "The number of spaces that should be equivalent to a tab", int, False),
        ParameterSpecification("expand_tabs", "Convert tabs to spaces", bool, False),
        ParameterSpecification("replace_whitespace", "Replace whitespace-characters by spaces", bool, False),
        ParameterSpecification("drop_whitespace", "Remove whitespace at beginning and ending of lines", bool, False),
        ParameterSpecification("break_on_hyphens", "Break on hyphens", bool, False),
        ParameterSpecification("break_long_words", "Break words longer than character-width", bool, False),
        ParameterSpecification("fix_sentence_endings", "Separate sentences by two spaces", bool, False),
        ParameterSpecification("initial_indent", "The prefix/indent for the first line", str, False),
        ParameterSpecification("subsequent_indent", "The prefix/indent for all other lines", str, False),
    ]
}


def asktextwrapper(**kwargs):
    """
    Open a dialog that lets the user configure a TextWrapper.

    Args:
        **kwargs: Additional keyword-arguments passed to `TextWrapDialog`.

    Returns:
        An instance of textwrap.TextWrapper or None.
    """
    dialog = TextWrapperDialog(**kwargs)
    return dialog.result


class TextWrapperDialog(AbstractParameterDialog):

    """
    Class to open a dialog for configuring a TextWrapper.

    The result-attribute will be either None or a textwrap.TextWrapper-instance.
    """

    def __init__(self, parent=None, title=None, initialvalue=None, allow_empty_selection=False, show_preview=False, preview_text=None):
        """
        Initialisation of a TextWrapperDialog.

        Args:
            parent: A parent window or None.
            title: A string that represents the title of the dialog or None.
            initialvalue: An instance of textwrap.TextWrapper to use as an initial value or None.
                If the user cancels the dialog this value will be the result.
                When possible the parameters of this instance will be chosen as defaults for the configurable parameters.
            allow_empty_selection: A boolean indicating whether the user should be able to specify not to return a TextWrapper.
                If True the user is shown an option not to create a TextWrapper, in which case the result will be None.
                If False the user is not shown this option.
            show_preview: A boolean indicating whether to show a preview with some text which is wrapped using the configured settings.
                If True the preview is shown.
                Otherwise no preview will be shown to the user.
            preview_text: A string that should be displayed in the preview.
                If this is a falsy value (e.g. None) DEFAULT_PREVIEW_TEXT will be used as default.
        """
        self.original_textwrapper = initialvalue
        self.textwrapper = copy.copy(initialvalue)
        if not self.textwrapper:
            self.textwrapper = textwrap.TextWrapper()
        self.show_preview = show_preview
        self.preview_text = preview_text
        if not self.preview_text:
            self.preview_text = DEFAULT_PREVIEW_TEXT
        self.preview = tk.StringVar()

        super(TextWrapperDialog, self).__init__(parent=parent, title=title, allow_empty_selection=allow_empty_selection, parameters=_PARAMETERS)

    def update_preview(self):
        if self.show_preview:
            if self.allow_empty_selection and self.has_empty_selection.get():
                self.preview.set(self.preview_text)
            else:
                self.preview.set("\n".join(self.textwrapper.wrap(self.preview_text)))

    @property
    def _empty_selection_prompt_text(self):
        return "Do not create an instance for wrapping text"

    def _get_parameter(self, name):
        return getattr(self.textwrapper, name, None)

    def _set_parameter(self, name, value):
        setattr(self.textwrapper, name, value)
        self.update_preview()

    def body(self, master):
        self.result = self.original_textwrapper
        if self.allow_empty_selection and not self.original_textwrapper:
            self.has_empty_selection.set(True)
        self.update_preview()

        if self.show_preview:
            g = ttk.Labelframe(master, text="Preview", relief=RIDGE, borderwidth=2)
            g.pack(fill=BOTH, side=BOTTOM)
            w = ttk.Label(g, textvariable=self.preview)
            w.pack(fill=BOTH, expand=True)
        return super(TextWrapperDialog, self).body(master)

    def apply(self):
        self.result = self.textwrapper
        super(TextWrapperDialog, self).apply()
