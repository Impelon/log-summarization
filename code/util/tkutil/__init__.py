"""
Package for various utilities related to tk/tkinter.
"""

from . import treeview
from . import dialog


def access_variables(variables):
    """
    Returns the values of the given tkinter-variables.

    Args:
        variables: A dict containing tkinter-variables as values.

    Returns:
        A dict with the value of each variable under its corresponding key.
    """
    return {key: variable.get() for key, variable in variables.items()}
