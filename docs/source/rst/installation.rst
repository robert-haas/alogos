Installation Guide
##################

This page explains the installation of the Python 3 package
:doc:`alogos <package_references>`
and its optional dependencies.



Required: alogos
================

The package
`alogos <https://pypi.org/project/alogos>`__
is available on the
`Python Package Index (PyPI) <https://pypi.org>`__
and therefore can be easily installed with Python's
default package manager
`pip <https://pypi.org/project/pip>`__ by using the following
command in a shell:

.. code-block:: console

   $ pip install alogos

Additional remarks:

- alogos is compatible with
  `Python 3.6 upwards <https://www.python.org/downloads>`_.
- Using an environment manager like
  `virtualenv <https://virtualenv.pypa.io>`__ or
  `conda <https://docs.conda.io>`__
  is a good idea in almost any case. These tools allow to create a
  `virtual environment <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`__
  into which pip can install the package in an isolated fashion instead
  of globally on your system. Thereby it does not interfere with the
  installation of other projects, which may require a different version
  of some shared dependency.



Optional: Graphviz
==================

alogos requires the standalone tool
`Graphviz <https://graphviz.org/>`__
and the Python package
`graphviz <https://pypi.org/project/graphviz/>`__
to visualize derivation trees.

Graphviz itself can not be installed via pip, but package managers of
various operating systems support it. Instructions are available in the
`Graphviz installation guide <https://graphviz.org/download/>`__.
In Debian and Ubuntu it can be done with the package manager apt:

.. code-block:: console

   $ sudo apt install graphviz libgraphviz-dev

Afterwards graphviz can be installed with pip. More information can
be found in the
`graphviz installation guide <https://github.com/xflr6/graphviz#installation>`__:

.. code-block:: console

   $ pip install graphviz



Optional: Jupyter notebook
==========================

alogos requires
`Jupyter notebook <https://jupyter.org>`__
for displaying grammars and derivation trees inline in notebooks in form 
of embedded HTML visualizations, accompanied by Python code and
Markdown comments.
A notebook is a file ending with ``.ipynb``, which can be created, modified and
executed with the
`notebook server <https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#starting-the-notebook-server>`__
and a webbrowser opened by it. Jupyter notebook can be installed according
to its `recommended installation <https://jupyter.org/install#jupyter-notebook>`__
with pip:

.. code-block:: console

   $ pip install notebook

Further remarks:

- **Caution**: Some plots may not show up in the notebook with default settings.
  Instead only a blank area is visible. The reason is a parameter called
  ``iopub_data_rate_limit`` in Jupyter's
  `config system <https://jupyter-notebook.readthedocs.io/en/stable/config.html>`__.
  Its value `is chosen rather low by default <https://github.com/jupyter/notebook/issues/2287>`__.
  Plots that contain much data can therefore be blocked.
  This problem can be solved by increasing the value of the parameter,
  which can be done in two ways:
  
  1. Permanently change it with a config file in the directory ``~/.jupyter`` or
  2. Temporarily change it when opening a notebook by adding an optional argument
     to the start-up command:

     .. code-block:: console

        $ jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e12
