def expand_children(treeview, open=True, parent_iid="", recursive=True):
    """
    Open or close the entries in the given treeview.

    Args:
        treeview: A ttk.Treeview to open or close the descendants of.
        open: A boolean indicating whether to expand or collapse entries.
            If True entries will be opened, otherwise they will be closed.
        parent_iid: The iid of the parent of the entries to open or close.
        recursive: A boolean indicating whether to recursively open or close all descendants.
    """
    for iid in treeview.get_children(parent_iid):
        treeview.item(iid, open=open)
        if recursive:
            expand_children(treeview, open=open, parent_iid=iid)


def sort_column(treeview, column, parent_iid="", recursive=True, switch_directions=True, key=None, reverse=False):
    """
    Sort the given treeview by the given column.

    This function was taken and modified from the 'Stack Overflow' network.
    The original author is Sridhar Ratnakumar.
    - original source: https://stackoverflow.com/a/1967793
        answered by: Sridhar Ratnakumar (https://stackoverflow.com/users/55246/sridhar-ratnakumar)
    - original question: https://stackoverflow.com/q/1966929
        asked by: Sridhar Ratnakumar (https://stackoverflow.com/users/55246/sridhar-ratnakumar)

    The original is licensed under Creative Commons Attribution-Share Alike.
    This function and its documentation are therefore licensed under the same license (Adapter's License).
    You can find a copy of this license here: https://creativecommons.org/licenses/by-sa/4.0/legalcode
    And a more readable summary here: https://creativecommons.org/licenses/by-sa/4.0/

    Args:
        treeview: A ttk.Treeview to sort.
        column: The identifier for the column to sort by.
        parent_iid: The iid of the parent of the entries to sort.
        recursive: A boolean indicating whether to recursively sort all descendants.
        switch_directions: A boolean indicating whether to configure the column to sort in the reverse direction after this.
        key: A key function passed to the sort-function.
        reverse: A boolean indicating whether the order should be reversed.

    Returns:
        The value for the column of the first descendant after the sort was completed or
        an empty string, if the parent did not have any descendants.
    """
    def element_generator():
        for iid in treeview.get_children(parent_iid):
            if column == "#0":
                value = treeview.item(iid, "text")
            else:
                value = treeview.set(iid, column)
            if recursive:
                first_child_value = sort_column(treeview, column, parent_iid=iid, switch_directions=False, key=key, reverse=reverse)
                if value == "":
                    value = first_child_value
            yield (value, iid)

    elements = sorted(element_generator(), key=key, reverse=reverse)
    for index, (_, iid) in enumerate(elements):
        treeview.move(iid, parent_iid, index)

    if switch_directions:
        treeview.heading(column, command=lambda: sort_column(treeview, column, parent_iid=parent_iid,
                                                             recursive=recursive, key=key, reverse=not reverse))
    if not elements:
        return ""
    return elements[0][0]
