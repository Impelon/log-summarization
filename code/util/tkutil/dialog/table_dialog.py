from tkinter.simpledialog import Dialog
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *

from ..treeview import sort_column

__all__ = ["askentries", "TableDialog"]


def askentries(**kwargs):
    """
    Open a dialog that lets the user add and remove entries in a table.

    Args:
        **kwargs: Additional keyword-arguments passed to `TableDialog`.

    Returns:
        A list of dicts or None.
    """
    dialog = TableDialog(**kwargs)
    return dialog.result


class TableDialog(Dialog):

    """
    Class to open a dialog that lets the user add and remove entries in a table.

    The result-attribute will be either a list of dicts or None.
    """

    def __init__(self, parent=None, title=None, initialvalue=None, columns=None, entry_function=None, allow_entry_deletion=True, sorting_keys=None, add_button_text=None, delete_button_text=None):
        """
        Initialisation of a TableDialog.

        Args:
            parent: A parent window or None.
            title: A string that represents the title of the dialog or None.
            initialvalue: A list of entries to use as an initial value or None.
                If the user cancels the dialog this value will be the result.
            columns: A dict with column-identifiers as keys and column-headings as values for each column to be displayed in the table.
                The columns will appear in the order of the keys.
            entry_function: A function that will be called to create or edit existing entries.
                or None if entries cannot be created or edited.
                The function will receive a dict or None (if adding a new entry) as input for the initial value.
                The returned value should be a dict or None, if no entry should be added / an existing one should remain unchanged.
                By returning None the function can therefore prohibit the user from editing certain entries.
            allow_entry_deletion: A boolean indicating whether the user should be able to delete existing entries.
                If True the user is shown a button to delete selected entries.
                If False the user is not shown this button.
            sorting_keys: A dict with the column-identifiers as keys and key functions passed to the sort-function as values.
                The key-functions will be used to sort the entries by column if the user presses one of the headings.
                If the dict does not contain a value for a column, the default key-function will be used.
            add_button_text: A string that is displayed for the button to add entries or None to use the default value.
                The default value is "Add entry".
            delete_button_text: A string that is displayed for the button to delete selected entries or None to use the default value.
                The default value is "Delete selection".
        """
        self.original_entries = initialvalue
        if not columns:
            columns = {}
        self.columns = columns
        self.entry_function = entry_function
        self.allow_entry_deletion = allow_entry_deletion
        if not sorting_keys:
            sorting_keys = {}
        self.sorting_keys = sorting_keys
        if add_button_text is None:
            add_button_text = "Add entry"
        self.add_button_text = add_button_text
        if delete_button_text is None:
            delete_button_text = "Delete selection"
        self.delete_button_text = delete_button_text
        self.entries = {}

        super(TableDialog, self).__init__(parent, title)

    def _delete_user_selection(self, *args):
        if self.allow_entry_deletion:
            self.table.delete(*self.table.selection())

    def _add_user_entry(self, *args):
        if self.entry_function:
            entry = self.entry_function(None)
            if entry:
                self._insert_entry(entry)

    def _edit_user_entry(self, event):
        if self.entry_function:
            try:
                iid = self.table.identify_row(event.y)
            except:
                iid = self.table.focus()
            if iid:
                entry = self.entry_function(self.entries.get(iid))
                if entry:
                    self._set_entry(entry, iid)

    def _insert_entry(self, entry, parent_iid=""):
        iid = self.table.insert(parent_iid, "end")
        self._set_entry(entry, iid)

    def _set_entry(self, entry, iid):
        self.entries[iid] = entry
        for column in self.columns.keys():
            value = entry.get(column)
            if value is not None:
                self.table.set(iid, column, str(value))

    def body(self, master):
        self.result = self.original_entries

        self.table = ttk.Treeview(master, columns=tuple(self.columns.keys()))
        self.table.column("#0", minwidth=0, width=0)
        for column, heading in self.columns.items():
            self.table.heading(column, text=heading, command=lambda c=column: sort_column(self.table, c, key=self.sorting_keys.get(c)))
        self.table.grid(row=0, column=0, sticky=N + E + S + W)
        w = ttk.Scrollbar(master, orient=HORIZONTAL, command=self.table.xview)
        w.grid(row=1, column=0, sticky=N + E + S + W)
        self.table.config(xscrollcommand=w.set)
        w = ttk.Scrollbar(master, orient=VERTICAL, command=self.table.yview)
        w.grid(row=0, column=1, sticky=N + E + S + W)
        self.table.config(yscrollcommand=w.set)
        self.table.bind("<Delete>", self._delete_user_selection)
        self.table.bind("<Double-Button-1>", self._edit_user_entry)

        toolbar = ttk.Frame(master)
        if self.entry_function:
            w = ttk.Button(toolbar, text=self.add_button_text, command=self._add_user_entry)
            w.pack(fill=X, expand=True, side=LEFT)
        if self.allow_entry_deletion:
            w = ttk.Button(toolbar, text=self.delete_button_text, command=self._delete_user_selection)
            w.pack(fill=X, expand=True, side=LEFT)
        toolbar.grid(row=2, column=0, columnspan=2, sticky=N + E + S + W)

        tk.Grid.columnconfigure(master, 0, weight=1)
        tk.Grid.columnconfigure(master, 1, weight=0)
        tk.Grid.rowconfigure(master, 0, weight=1)
        tk.Grid.rowconfigure(master, 1, weight=0)
        tk.Grid.rowconfigure(master, 2, weight=0)

        if self.original_entries:
            for entry in self.original_entries:
                self._insert_entry(entry)

        return master

    def apply(self):
        self.result = list(self.entries.values())
