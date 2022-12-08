def genealogy(graph, backend, **kwargs):
    """Create a graph visualization of the genealogy of individuals."""
    import gravis as gv

    if backend == "d3":
        fig = gv.d3(graph, **kwargs)
    elif backend == "vis":
        fig = gv.vis(graph, **kwargs)
    else:
        fig = gv.three(graph, **kwargs)
    return fig
