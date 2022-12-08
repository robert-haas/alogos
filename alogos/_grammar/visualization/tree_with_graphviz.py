import os as _os

from ..._utilities import argument_processing as _ap
from ..._utilities import operating_system as _operating_system


def create_graphviz_tree(
    tree,
    show_node_indices=None,
    layout_engine=None,
    fontname=None,
    fontsize=None,
    shape_nt=None,
    shape_unexpanded_nt=None,
    shape_t=None,
    fontcolor_nt=None,
    fontcolor_unexpanded_nt=None,
    fontcolor_t=None,
    fillcolor_nt=None,
    fillcolor_unexpanded_nt=None,
    fillcolor_t=None,
):
    """Represent the derivation tree as Digraph in GraphViz.

    References
    ----------
    - `Graphviz <https://www.graphviz.org>`__
    - `An inofficial Python wrapper for Graphviz <https://pypi.org/project/graphviz>`__ used here

    """
    from graphviz import Digraph

    # Argument processing
    # Node shapes from http://www.graphviz.org/doc/info/shapes.html
    shapes = [
        "Mcircle",
        "Mdiamond",
        "Msquare",
        "assembly",
        "box",
        "box3d",
        "cds",
        "circle",
        "component",
        "cylinder",
        "diamond",
        "doublecircle",
        "doubleoctagon",
        "egg",
        "ellipse",
        "fivepoverhang",
        "folder",
        "hexagon",
        "house",
        "insulator",
        "invhouse",
        "invtrapezium",
        "invtriangle",
        "larrow",
        "lpromoter",
        "none",
        "note",
        "noverhang",
        "octagon",
        "oval",
        "parallelogram",
        "pentagon",
        "plain",
        "plaintext",
        "point",
        "polygon",
        "primersite",
        "promoter",
        "proteasesite",
        "proteinstab",
        "rarrow",
        "rect",
        "rectangle",
        "restrictionsite",
        "ribosite",
        "rnastab",
        "rpromoter",
        "septagon",
        "signature",
        "square",
        "star",
        "tab",
        "terminator",
        "threepoverhang",
        "trapezium",
        "triangle",
        "tripleoctagon",
        "underline",
        "utr",
        None,
    ]

    show_node_indices = _ap.bool_arg(
        "show_node_indices", show_node_indices, default=False
    )
    layout_engine = _ap.str_arg(
        "layout_engine",
        layout_engine,
        default="dot",
        vals=[
            "circo",
            "dot",
            "fdp",
            "neato",
            "osage",
            "patchwork",
            "sfdp",
            "twopi",
            None,
        ],
    )
    fontname = _ap.str_arg("fontname", fontname, default="Mono")
    fontsize = _ap.int_arg("fontsize", fontsize, default=12)
    shape_nt = _ap.str_arg("shape_nt", shape_nt, vals=shapes, default="box")
    shape_unexpanded_nt = _ap.str_arg(
        "shape_unexpanded_nt", shape_unexpanded_nt, vals=shapes, default="box"
    )
    shape_t = _ap.str_arg("shape_t", shape_t, vals=shapes, default="ellipse")
    fontcolor_nt = _ap.str_arg("fontcolor_nt", fontcolor_nt, default="black")
    fontcolor_unexpanded_nt = _ap.str_arg(
        "fontcolor_unexpanded_nt", fontcolor_unexpanded_nt, default="white"
    )
    fontcolor_t = _ap.str_arg("fontcolor_t", fontcolor_t, default="white")
    fillcolor_nt = _ap.str_arg("fillcolor_nt", fillcolor_nt, default="white")
    fillcolor_unexpanded_nt = _ap.str_arg(
        "fillcolor_unexpanded_nt", fillcolor_unexpanded_nt, default="#b81118"
    )
    fillcolor_t = _ap.str_arg("fillcolor_t", fillcolor_t, default="#00864b")

    # Graph construction
    digraph = Digraph(
        engine=layout_engine,
        node_attr=dict(fontname=fontname, fontsize=str(fontsize)),
        encoding="utf-8",
    )
    stack = [tree.root_node]
    cnt = _AutoCounter()
    while stack:
        # Get node
        current_node = stack.pop(0)
        if current_node.children:
            stack = current_node.children + stack
            child_nodes = current_node.children
        else:
            child_nodes = []

        # Determine node style
        node_id = str(cnt[current_node])
        if show_node_indices:
            node_label = "{}: {}".format(node_id, current_node.symbol)
        else:
            node_label = current_node.symbol.text
            if node_label == "":
                node_label = "ɛ"
        if current_node.contains_nonterminal():
            if not current_node.children:
                node_shape = shape_unexpanded_nt
                node_fillcolor = fillcolor_unexpanded_nt
                node_fontcolor = fontcolor_unexpanded_nt
            else:
                node_shape = shape_nt
                node_fillcolor = fillcolor_nt
                node_fontcolor = fontcolor_nt
        else:
            node_shape = shape_t
            node_fillcolor = fillcolor_t
            node_fontcolor = fontcolor_t

        # Create node in visualized graph
        digraph.node(
            node_id,
            node_label,
            shape=node_shape,
            fontcolor=node_fontcolor,
            style="filled",
            fillcolor=node_fillcolor,
        )

        # Create edges in visualized graph
        for child_node in child_nodes:
            child_node_id = str(cnt[child_node])
            digraph.edge(node_id, child_node_id)

    fig = DerivationTreeFigure(digraph)
    return fig


class DerivationTreeFigure:
    """Data structure for wrapping, displaying and exporting a Graphviz graph of a tree."""

    # Initialization
    def __init__(self, given_graph):
        """Initialize a figure with a Graphviz graph object."""
        self.fig = given_graph

    # Representations
    def __repr__(self):
        """Compute the "official" string representation of the figure."""
        return "<{} object at {}>".format(self.__class__.__name__, hex(id(self)))

    def _repr_html_(self):
        """Provide rich display representation in HTML format for Jupyter notebooks."""
        return self.html_text_partial

    # Display in browser or notebook
    def display(self, inline=False):
        """Display the plot in a webbrowser or as IPython rich display representation.

        Parameters
        ----------
        inline : bool
            If True, the plot will be shown inline in a Jupyter notebook.

        """
        if inline:
            from IPython.display import HTML, display

            display(HTML(self.html_text_standalone))
        else:
            _operating_system.open_in_webbrowser(self.html_text_standalone)

    # Further representations
    @property
    def html_text(self):
        """Create a HTML text representation."""
        return self.html_text_standalone

    @property
    def html_text_standalone(self):
        """Create a standalone HTML text representation."""
        html_text = _HTML.format(self.svg_text)
        return html_text

    @property
    def html_text_partial(self):
        """Create a partial HTML text representation without html, head and body tags."""
        return self.svg_text

    @property
    def svg_text(self):
        """Create an SVG text representation of the plot, usable in HTML context or SVG file."""
        try:
            svg_text = self.fig._repr_svg_()
        except AttributeError:
            # API change: https://graphviz.readthedocs.io/en/latest/changelog.html#version-0-19
            mime_type = "image/svg+xml"
            data = self.fig._repr_mimebundle_(include=[mime_type])
            svg_text = data[mime_type]
        lines = svg_text.splitlines()
        for _i, line in enumerate(lines):
            if line.startswith("<svg"):
                break
        svg_text = "".join(lines[_i:])
        return svg_text

    # Export as HTML file
    def export_html(self, filepath):
        """Export the plot as text file in HTML format.

        Parameters
        ----------
        filepath : str
            Filepath of the created HTML file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".html" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated HTML file, guaranteed to end with ".html".

        """
        # Argument processing
        used_filepath = _operating_system.ensure_file_extension(filepath, "html")

        # Create HTML file
        with open(used_filepath, "w", encoding="utf-8") as file_handle:
            file_handle.write(self.html_text)
        return used_filepath

    # Export in various other file formats
    def _export(self, filepath, fileformat):
        """Export the digraph in various formats.

        References
        ----------
        - https://graphviz.readthedocs.io/en/stable/api.html#digraph
        - https://www.graphviz.org/doc/info/output.html

        """
        # Argument processing
        filepath = _ap.str_arg("filepath", filepath)
        filepath = _ap.ensure_no_file_extension(filepath, fileformat)

        # Check if file already exists, only to prevent deletion
        did_not_exist = not _os.path.exists(filepath)

        # Create image file
        used_filepath = self.fig.render(filename=filepath, format=fileformat)

        # Remove side product generated by GraphViz
        if _os.path.isfile(filepath) and did_not_exist:
            _operating_system.delete_file(filepath)
        return used_filepath

    def export_dot(self, filepath):
        """Export the plot as text file in DOT format.

        Parameters
        ----------
        filepath : str
            Filepath of the created DOT file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".dot" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated DOT file, guaranteed to end with ".dot".

        """
        return self._export(filepath, "dot")

    def export_eps(self, filepath):
        """Export the plot as vector graphic in EPS format.

        Parameters
        ----------
        filepath : str
            Filepath of the created EPS file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".eps" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated EPS file, guaranteed to end with ".eps".

        """
        return self._export(filepath, "eps")

    def export_gv(self, filepath):
        """Export the plot as text file in GV format.

        Parameters
        ----------
        filepath : str
            Filepath of the created GV file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".gv" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated GV file, guaranteed to end with ".gv".

        """
        return self._export(filepath, "gv")

    def export_pdf(self, filepath):
        """Export the plot as vector graphic in PDF format.

        Parameters
        ----------
        filepath : str
            Filepath of the created PDF file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".pdf" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated PDF file, guaranteed to end with ".pdf".

        """
        return self._export(filepath, "pdf")

    def export_png(self, filepath):
        """Export the plot as raster graphic in PNG format.

        Parameters
        ----------
        filepath : str
            Filepath of the created PNG file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".png" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated PNG file, guaranteed to end with ".png".

        """
        return self._export(filepath, "png")

    def export_ps(self, filepath):
        """Export the plot as vector graphic in PS format.

        Parameters
        ----------
        filepath : str
            Filepath of the created PS file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".ps" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated PS file, guaranteed to end with ".ps".

        """
        return self._export(filepath, "ps")

    def export_svg(self, filepath):
        """Export the plot as vector graphic in SVG format.

        Parameters
        ----------
        filepath : str
            Filepath of the created SVG file.
            If the file exists it will be overwritten without warning.
            If the path does not end with ".svg" it will be changed to do so.
            If the parent directory does not exist it will be created.

        Returns
        -------
        filepath_used : str
            Filepath of the generated SVG file, guaranteed to end with ".svg".

        """
        return self._export(filepath, "svg")


_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
</head>
<body>

{}

</body>
</html>"""


class _AutoCounter:
    def __init__(self):
        self.map = dict()
        self.cnt = 0

    def __getitem__(self, key):
        if key not in self.map:
            self.map[key] = self.cnt
            self.cnt += 1
        return self.map[key]
