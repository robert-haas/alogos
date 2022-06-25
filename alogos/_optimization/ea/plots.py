def genealogy(data, graph, backend, **kwargs):
    import gravis as gv

    if backend == 'd3':
        fig = gv.d3(graph, **kwargs)
    elif backend == 'vis':
        fig = gv.vis(graph, **kwargs)
    else:
        fig = gv.three(graph, **kwargs)
    return fig
