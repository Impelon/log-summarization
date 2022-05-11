from util import traces
from util.traces import graph
from util import tkutil

import collections
import itertools
import traceback
import textwrap
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import colorchooser
from tkinter import messagebox

import networkx
import matplotlib.font_manager
from matplotlib.patches import ArrowStyle
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

NATSORT_AVAILABLE = False
try:
    import natsort
    NATSORT_AVAILABLE = True
except:
    pass

GRAPHVIZ_AVAILABLE = False
try:
    import networkx.drawing.nx_agraph as graphviz
    GRAPHVIZ_AVAILABLE = True
except:
    pass
try:
    import networkx.drawing.nx_pydot as graphviz
    GRAPHVIZ_AVAILABLE = True
except:
    pass


DIALOG_PARAMETER_SEPARATOR = "."

CSV_IMPORT_DIALOG_PARAMETERS = [
    tkutil.dialog.ParameterSpecification("path", "Select File", filedialog.askopenfilename, True),
    tkutil.dialog.ParameterSpecification("entries_have_explicit_trace_information",
                                         "Entries contain information about traces explicitly", bool, False),
]

TRACES_ENTRIES_DIALOG_PARAMETERS = [
    tkutil.dialog.ParameterSpecification(
        "trace_key", "Which column contains information about traces?\nLeave blank to use the default column.", str, True),
    tkutil.dialog.ParameterSpecification("remove_trace_information", "Remove trace-marker from the text", bool, False),
    tkutil.dialog.ParameterSpecification("can_only_be_prefix", "Trace-marker are only recognized at the start of the text", bool, True),
]

TRACES_EXPLICIT_ENTRIES_DIALOG_PARAMETERS = [
    tkutil.dialog.ParameterSpecification(
        "trace_id_key", "Which column contains the ID of the trace?\nLeave blank to use the default column.", str, True),
    tkutil.dialog.ParameterSpecification(
        "parent_span_id_key", "Which column contains the ID of the parent of the span?\nLeave blank to use the default column.", str, True),
    tkutil.dialog.ParameterSpecification(
        "span_id_key", "Which column contains the ID of the span?\nLeave blank to use the default column.", str, True),
    tkutil.dialog.ParameterSpecification("remove_trace_information", "Remove the columns with trace-information", bool, False),
]

NETWORKX_AVAILABLE_LAYOUTS = collections.OrderedDict((
    ("planar", networkx.drawing.layout.planar_layout),
    ("kamada_kawai", networkx.drawing.layout.kamada_kawai_layout),
    ("spiral", networkx.drawing.layout.spiral_layout),
    ("spring", networkx.drawing.layout.spring_layout),
    ("shell", networkx.drawing.layout.shell_layout),
    ("spectral", networkx.drawing.layout.spectral_layout),
    ("circular", networkx.drawing.layout.circular_layout),
    ("random", networkx.drawing.layout.random_layout),
))
if GRAPHVIZ_AVAILABLE:
    for program in ["circo", "twopi", "fdp", "sfdp", "neato", "dot"]:
        key = "graphviz_" + program
        NETWORKX_AVAILABLE_LAYOUTS[key] = lambda *args, prog=program, **kwargs: graphviz.graphviz_layout(*args, prog=prog, **kwargs)
        NETWORKX_AVAILABLE_LAYOUTS.move_to_end(key, last=False)

NETWORK_DISPLAY_OPTIONS_DIALOG_PARAMETERS = {
    "Font": [
        tkutil.dialog.ParameterSpecification("font_size", "Font size", int, True),
        tkutil.dialog.ParameterSpecification("font_weight", "Font weight", list(matplotlib.font_manager.weight_dict.keys()), True),
        tkutil.dialog.ParameterSpecification("font_family", "Font family", list(matplotlib.font_manager.font_family_aliases), True),
        tkutil.dialog.ParameterSpecification("font_color", "Font color", lambda: colorchooser.askcolor()[1], True),
    ],
    "Nodes": [
        tkutil.dialog.ParameterSpecification("node_size", "Node size", int, False),
        tkutil.dialog.ParameterSpecification("layout_algorithm", "Node Position Algorithm", list(NETWORKX_AVAILABLE_LAYOUTS.keys()), False),
        tkutil.dialog.ParameterSpecification("node_color", "Node color", lambda: colorchooser.askcolor()[1], True),
        tkutil.dialog.ParameterSpecification("with_labels", "Draw labels", bool, False),
    ],
    "Edges": [
        tkutil.dialog.ParameterSpecification("width", "Edge width", float, True),
        tkutil.dialog.ParameterSpecification("edge_color", "Edge color", lambda: colorchooser.askcolor()[1], True),
    ],
    "Arrows": [
        tkutil.dialog.ParameterSpecification("arrows", "Draw arrows", bool, False),
        tkutil.dialog.ParameterSpecification("arrowsize", "Size of arrows", int, False),
        tkutil.dialog.ParameterSpecification("arrowstyle", "Style of arrows", list(ArrowStyle.get_styles().keys()), True),
    ],
}


class TracesVisualizer(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.traces = None
        self.traces_forest = None
        self.entries_key = "Entries"
        self.trace_id_key = "TraceId"
        self.span_id_key = "SpanId"
        self.parent_span_id_key = "ParentSpanId"
        self.sorting_key = None
        if NATSORT_AVAILABLE:
            self.sorting_key = natsort.natsort_keygen()
        self.textwrapper = textwrap.TextWrapper(width=25, drop_whitespace=True, break_on_hyphens=False, replace_whitespace=True)
        self.network_display_options = {"font_size": 8, "font_weight": "medium", "font_family": "monospace", "with_labels": True,
                                        "node_size": 350, "arrowsize": 5, "arrowstyle": "->", "arrows": True, "width": 0.75, "layout_algorithm": next(iter(NETWORKX_AVAILABLE_LAYOUTS.keys()))}

        self.title("TracesVisualizer")

        toolbar = ttk.Frame(self, relief=RIDGE, borderwidth=2)
        toolbar.grid(row=0, column=0, sticky=N + E + S + W)
        self.tabs = ttk.Notebook(self)
        self.tabs.grid(row=1, column=0, sticky=N + E + S + W)
        self.tabs.enable_traversal()
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.rowconfigure(self, 0, weight=0)
        tk.Grid.rowconfigure(self, 1, weight=1)

        # toolbar
        w = ttk.Button(toolbar, text="Import CSV...", command=self.csv_import_dialog)
        w.pack(fill=Y, side=LEFT)
        w = ttk.Button(toolbar, text="Clear", command=self.clear_traces)
        w.pack(fill=Y, side=RIGHT)
        self.opened_traces_path = tk.StringVar()
        w = ttk.Entry(toolbar, textvariable=self.opened_traces_path, justify=LEFT, state="readonly")
        w.pack(fill=BOTH, expand=True, side=LEFT)

        # log-tree view tab
        frame = ttk.Frame(self.tabs, relief=RIDGE, borderwidth=2)
        self.log_tree = ttk.Treeview(frame)
        self.log_tree.heading("#0", command=lambda: tkutil.treeview.sort_column(self.log_tree, "#0", key=self.sorting_key))
        self.log_tree.grid(row=1, column=0, sticky=N + E + S + W)
        w = ttk.Scrollbar(frame, orient=HORIZONTAL, command=self.log_tree.xview)
        w.grid(row=2, column=0, sticky=N + E + S + W)
        self.log_tree.config(xscrollcommand=w.set)
        w = ttk.Scrollbar(frame, orient=VERTICAL, command=self.log_tree.yview)
        w.grid(row=1, column=1, sticky=N + E + S + W)
        self.log_tree.config(yscrollcommand=w.set)
        # tab-specific toolbar
        self.options_update_log_tree = {}
        toolbar = ttk.Frame(frame)
        separate = tk.BooleanVar(value=False)
        self.options_update_log_tree["separate_element_for_entries"] = separate
        w = ttk.Checkbutton(toolbar, text="Separate element for entries", variable=separate, command=self.update_log_tree)
        w.pack(fill=Y, side=LEFT)
        flatten = tk.BooleanVar(value=False)
        self.options_update_log_tree["flatten"] = flatten
        w = ttk.Checkbutton(toolbar, text="Flatten", variable=flatten, command=self.update_log_tree)
        w.pack(fill=Y, side=LEFT)
        w = ttk.Button(toolbar, text="Collapse all", command=lambda *args: tkutil.treeview.expand_children(self.log_tree, open=False))
        w.pack(fill=Y, side=RIGHT)
        w = ttk.Button(toolbar, text="Expand all", command=lambda *args: tkutil.treeview.expand_children(self.log_tree, open=True))
        w.pack(fill=Y, side=RIGHT)
        w = ttk.Button(toolbar, text="Reorganize headings", command=lambda *args: self.update_log_tree(appearance_only=True))
        w.pack(fill=Y, side=RIGHT)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=N + E + S + W)
        # tab-layout
        tk.Grid.columnconfigure(frame, 0, weight=1)
        tk.Grid.columnconfigure(frame, 1, weight=0)
        tk.Grid.rowconfigure(frame, 0, weight=0)
        tk.Grid.rowconfigure(frame, 1, weight=1)
        tk.Grid.rowconfigure(frame, 2, weight=0)
        self.tabs.add(frame, text="Log-Tree View", underline=0)

        # graph view tab
        frame = ttk.Frame(self.tabs, relief=RIDGE, borderwidth=2)
        matplotlib_figure = Figure(tight_layout=True)
        self.matplotlib_axes = matplotlib_figure.add_subplot(1, 1, 1)
        canvas = FigureCanvasTkAgg(matplotlib_figure, frame)
        canvas.get_tk_widget().grid(row=1, column=0, sticky=N + E + S + W)
        w = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        w.grid(row=2, column=0, sticky=N + E + S + W)
        # tab-specific toolbar
        self.options_update_graph = {}
        toolbar = ttk.Frame(frame)
        display = tk.BooleanVar(value=True)
        self.options_update_graph["display_trace_node"] = display
        group_lines = tk.BooleanVar(value=True)
        self.options_update_graph["group_lines"] = group_lines
        selected = tk.StringVar()
        self.options_update_graph["trace_id"] = selected
        selected.trace_add("write", self.update_graph)
        column = tk.StringVar()
        self.options_update_graph["column_to_display"] = column
        column.trace_add("write", self.update_graph)
        w = ttk.Checkbutton(toolbar, text="Show root-element", variable=display, command=self.update_graph)
        w.grid(row=0, column=0, sticky=N + E + S + W)
        w = ttk.Checkbutton(toolbar, text="Group identical label-lines", variable=group_lines, command=self.update_graph)
        w.grid(row=1, column=0, sticky=N + E + S + W)
        self.graph_options = ttk.OptionMenu(toolbar, selected)
        self.graph_options.grid(row=0, column=1, sticky=N + E + S + W)
        self.column_options = ttk.OptionMenu(toolbar, column)
        self.column_options.grid(row=1, column=1, sticky=N + E + S + W)
        w = ttk.Button(toolbar, text="Configure text wrapping", command=self.textwrapper_dialog)
        w.grid(row=0, column=2, sticky=N + E + S + W)
        w = ttk.Button(toolbar, text="Configure graph display", command=self.network_display_options_dialog)
        w.grid(row=1, column=2, sticky=N + E + S + W)
        # tab-toolbar-layout
        tk.Grid.columnconfigure(toolbar, 0, weight=0)
        tk.Grid.columnconfigure(toolbar, 1, weight=1)
        tk.Grid.columnconfigure(toolbar, 2, weight=0)
        tk.Grid.rowconfigure(toolbar, 0, weight=0)
        tk.Grid.rowconfigure(toolbar, 1, weight=0)
        toolbar.grid(row=0, column=0, sticky=N + E + S + W)
        # tab-layout
        tk.Grid.columnconfigure(frame, 0, weight=1)
        tk.Grid.rowconfigure(frame, 0, weight=0)
        tk.Grid.rowconfigure(frame, 1, weight=1)
        tk.Grid.rowconfigure(frame, 2, weight=0)
        self.tabs.add(frame, text="Graph View", underline=0)

    def textwrapper_dialog(self):
        self.textwrapper = tkutil.dialog.asktextwrapper(
            parent=self, title="Configure Text-Wrapper", initialvalue=self.textwrapper, allow_empty_selection=True, show_preview=True)
        self.update_graph()

    def network_display_options_dialog(self):
        self.network_display_options = tkutil.dialog.askdictionary(parent=self, title="Configure Display Options", initialvalue=self.network_display_options,
                                                                   parameters=NETWORK_DISPLAY_OPTIONS_DIALOG_PARAMETERS, separator=DIALOG_PARAMETER_SEPARATOR)
        self.update_graph()

    def csv_import_dialog(self):
        def file_import(initialvalue):
            configuration = tkutil.dialog.askdictionary(parent=self, title="Import CSV", initialvalue=initialvalue,
                                                        parameters=CSV_IMPORT_DIALOG_PARAMETERS, separator=DIALOG_PARAMETER_SEPARATOR)
            if configuration:
                advanced_options_parameters = TRACES_ENTRIES_DIALOG_PARAMETERS
                if not initialvalue:
                    initialvalue = {}
                advanced_options_defaults = {"remove_trace_information": initialvalue.get("remove_trace_information", True)}
                if configuration["entries_have_explicit_trace_information"]:
                    advanced_options_parameters = TRACES_EXPLICIT_ENTRIES_DIALOG_PARAMETERS
                    advanced_options_defaults["trace_id_key"] = initialvalue.get("trace_id_key", self.trace_id_key)
                    advanced_options_defaults["span_id_key"] = initialvalue.get("span_id_key", self.span_id_key)
                    advanced_options_defaults["parent_span_id_key"] = initialvalue.get("parent_span_id_key", self.parent_span_id_key)
                configuration["parseroptions"] = tkutil.dialog.askdictionary(parent=self, title="Advanced Parser Options", initialvalue=advanced_options_defaults,
                                                                             parameters=advanced_options_parameters, separator=DIALOG_PARAMETER_SEPARATOR)
                return configuration
            return initialvalue

        columns = collections.OrderedDict((("path", "Path"), ("entries_have_explicit_trace_information", "Explicit trace-information"),
                                           ("parseroptions", "Parser options")))
        configurations = tkutil.dialog.askentries(parent=self, title="Select traces to import", entry_function=file_import, columns=columns,
                                                  sorting_keys={c: self.sorting_key for c in columns.keys()},
                                                  add_button_text="Add file...", delete_button_text="Remove selection")
        if configurations:
            self.import_csv_files(configurations, values_key=self.entries_key)

    def import_csv_files(self, configurations, **kwargs):
        try:
            self.traces = traces.traces_from_csv_files(configurations, **kwargs)
        except Exception as ex:
            messagebox.showerror(title="Could not import from csv!", message="An error occured: {}\n{}".format(repr(ex), traceback.format_exc()))
            return False
        paths = [configuration["path"] for configuration in configurations]
        self.opened_traces_path.set("; ".join(paths))
        self.on_traces_update()
        return True

    def clear_traces(self):
        self.traces = None
        self.on_traces_update()

    def get_traces_columns(self):
        if not self.traces:
            return tuple()
        path = traces.find_id(self.traces, self.entries_key)
        if not path:
            return tuple()
        entries = self.traces
        for id in path:
            entries = entries[id]
        return tuple(entries[0].keys())

    def on_traces_update(self):
        self.update_log_tree()
        if self.traces:
            self.traces_forest = graph.to_forest(self.traces, attribute_extractor=graph.retain_keys([self.entries_key]))
            columns = (" ",) + self.get_traces_columns()
            self.column_options.set_menu(columns[0], *columns)
            self.graph_options.set_menu(next(iter(self.traces_forest.keys())), *self.traces_forest.keys())
        else:
            self.opened_traces_path.set("")
            self.traces_forest = None
            self.column_options.set_menu(default=" ")
            self.graph_options.set_menu(default=" ")

    def update_log_tree(self, appearance_only=False, *args):
        options = tkutil.access_variables(self.options_update_log_tree)
        self.set_headings_log_tree(**options)
        if appearance_only:
            return
        self.log_tree.delete(*self.log_tree.get_children())
        if self.traces:
            self.fill_log_tree(self.traces, **options)

    def set_headings_log_tree(self, flatten=False, separate_element_for_entries=False):
        self.log_tree.update()
        self.log_tree.column("#0", width=self.log_tree.winfo_width() - 2, stretch=True)
        self.log_tree.heading("#0", text="")
        self.log_tree.configure(columns=())
        if not self.traces:
            return
        columns = self.get_traces_columns()
        if not columns:
            return
        if flatten:
            self.log_tree.heading("#0", text=self.trace_id_key)
            if separate_element_for_entries:
                columns = (self.entries_key,)
            columns = (self.parent_span_id_key, self.span_id_key) + columns
        columnwidth, remainder = divmod((self.log_tree.winfo_width() - 150), len(columns))
        self.log_tree.column("#0", width=150, stretch=False)
        self.log_tree.configure(columns=columns)
        for column in columns:
            self.log_tree.column(column, stretch=False, width=columnwidth)
            self.log_tree.heading(column, text=column, command=lambda c=column: tkutil.treeview.sort_column(self.log_tree, c, key=self.sorting_key))
        self.log_tree.column(column, stretch=True, width=columnwidth + remainder - 2)

    def fill_flat_log_tree(self, entries, parent_iid=""):
        for entry in entries:
            entry = tuple(entry.values())
            self.log_tree.insert(parent_iid, "end", text=entry[0], values=entry[1:])

    def fill_log_tree(self, trace, parent_iid="", flatten=False, separate_element_for_entries=False):
        if flatten:
            values_key = None
            if separate_element_for_entries:
                values_key = self.entries_key
            entries = traces.flatten(self.traces, trace_id_key=self.trace_id_key, parent_span_id_key=self.parent_span_id_key,
                                     span_id_key=self.span_id_key, values_key=values_key)
            return self.fill_flat_log_tree(entries, parent_iid=parent_iid)
        for key, value in trace.items():
            if key == self.entries_key:
                iid = parent_iid
                if separate_element_for_entries:
                    iid = self.log_tree.insert(parent_iid, "end", text=key, open=False)
                for entry in value:
                    self.log_tree.insert(iid, "end", values=tuple(entry.values()))
            else:
                iid = self.log_tree.insert(parent_iid, "end", text=key, open=False)
                self.fill_log_tree(value, parent_iid=iid, separate_element_for_entries=separate_element_for_entries)

    def update_graph(self, *args):
        self.draw_graph(**tkutil.access_variables(self.options_update_graph))

    def draw_graph(self, trace_id=None, display_trace_node=False, column_to_display=None, group_lines=False):
        self.matplotlib_axes.clear()
        if self.traces_forest and trace_id in self.traces_forest:
            trace_graph = self.traces_forest[trace_id]
            nodes_to_display = trace_graph.nodes()
            if not display_trace_node:
                nodes_to_display -= {trace_id}
            if column_to_display and column_to_display.strip():
                def get_label(node):
                    if not self.entries_key in trace_graph.nodes[node]:
                        return ""
                    lines = filter(None, (entry.get(column_to_display, "") for entry in trace_graph.nodes[node][self.entries_key]))
                    if group_lines:
                        lines = (key for key, group in itertools.groupby(lines))
                    label = "\n".join(lines)
                    return label
                labels_to_display = {node: get_label(node) for node in nodes_to_display}
            else:
                labels_to_display = {node: node for node in nodes_to_display}
            edges_to_display = trace_graph.edges(nodes_to_display)
            if self.textwrapper:
                labels_to_display = {node: "\n".join("\n".join(self.textwrapper.wrap(line))
                                     for line in label.splitlines()) for node, label in labels_to_display.items()}
            draw_options = self.network_display_options.copy()
            layout = NETWORKX_AVAILABLE_LAYOUTS[draw_options.pop("layout_algorithm")](trace_graph)
            networkx.draw(trace_graph, pos=layout, ax=self.matplotlib_axes, labels=labels_to_display,
                          nodelist=nodes_to_display, edgelist=edges_to_display, **draw_options)
        self.matplotlib_axes.figure.canvas.draw()


def display():
    """
    Starts an instance of TracesVisualizer, displaying a GUI.
    """
    window = TracesVisualizer()
    window.mainloop()


if __name__ == "__main__":
    display()
