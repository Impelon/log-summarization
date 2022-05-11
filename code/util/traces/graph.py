import networkx as nx

def retain_keys(keys):
    """
    Creates an attribute-extractor-function that retains all given trace-entries as node-attributes.

    Args:
        keys: A collection of keys for attributes that all traces and spans should retain.

    Returns:
        A function that accepts a trace and returns a dictionary of entries extracted from the trace given the keys.
    """
    return lambda trace: {key: trace[key] for key in keys if key in trace}

def to_forest(traces, graph_constructor=None, attribute_extractor=None):
    """
    Transforms traces into graphs.

    Args:
        traces: A dict containing traces.
        graph_constructor: A networkx-like constructor for graphs.
            If this is a falsy value (e.g. None) networkx.DiGraph is used as a default.
        attribute_extractor: A function that is called on each trace/span and
            returns a dictionary of attributes to set for the corresponding node.
            If this is a falsy value (e.g. None) no attributes will be retained.

    Returns:
        A dict containing a graph for each trace, accessed through its ID.
        Each graph contains all spans as nodes, as well as the trace itself.
    """
    if not graph_constructor:
        graph_constructor = nx.DiGraph
    forest = dict()
    for trace_id, trace in traces.items():
        forest[trace_id] = _to_graph(trace, graph_constructor(), trace_id, attribute_extractor)
    return forest


def _to_graph(trace, graph, root, attribute_extractor=None):
    attributes = {}
    if attribute_extractor:
        attributes = attribute_extractor(trace)
    graph.add_node(root, **attributes)
    for span_id, span in trace.items():
        if isinstance(span, dict):
            graph.add_edge(root, span_id)
            _to_graph(span, graph, span_id, attribute_extractor)
    return graph
