alogos
######

Welcome! This is the documentation for
:doc:`alogos <rst/package_references>`,
an open-source Python 3.6+ package for solving optimization problems
with grammar-guided genetic programming.


.. figure:: images/girl_with_a_pearl_earring.gif
   :scale: 100 %
   :align: center
   :alt: Animation of an optimization example

   :doc:`Example <../code/examples/image_approximation>`:
   Evolution of a program that draws 150 semi-transparent triangles
   to approximate an image.



What is this project about?
===========================

The name alogos is an acronym for
"a library of grammatical optimization systems".

- "`Optimization
  <https://en.wikipedia.org/wiki/Mathematical_optimization>`__"
  means to search for the best item in a set of candidates.
  This set is also known as the
  `search space <https://en.wikipedia.org/wiki/Search_space>`__
  and it can be explored by various
  `search methods <https://en.wikipedia.org/wiki/Search_algorithm>`__
  in order to find a good or optimal element in it.
  Some methods can be applied to a wide range of
  `optimization problems
  <https://en.wikipedia.org/wiki/Optimization_problem>`__
  but do not come with a guarantee to find the global optimum.
  These methods are called
  `metaheuristics <https://en.wikipedia.org/wiki/Metaheuristic>`__
  in contrast to problem-specific heuristics or exact algorithms,
  which for many problems are not available or not feasible with
  limited time or computational resources.
  A well-known example of a metaheuristic is an
  `evolutionary algorithm
  <https://en.wikipedia.org/wiki/Evolutionary_algorithm>`__
  and this package provides a specialized implementation of it.

- "Best item" implies that the user needs to provide a way to
  `rank <https://en.wikipedia.org/wiki/Ranking>`__
  the candidates from best to worst. This is usually done by
  defining an
  `objective function
  <https://en.wikipedia.org/wiki/Loss_function>`__,
  which gets a candidate as input and needs
  to return a numerical value as output. This value represents a
  quality or fitness score for each candidate, which makes them
  pairwise comparable and thereby induces a
  `weak ordering <https://en.wikipedia.org/wiki/Weak_ordering>`__
  on them. Additionally, the user must define whether the search
  should find an item with maximum or minimum value.

- "`Grammatical <https://en.wikipedia.org/wiki/Formal_grammar>`__"
  means that a
  `context-free grammar
  <https://en.wikipedia.org/wiki/Context-free_grammar>`__
  is used to define
  the search space. A grammar is a mathematical device frequently used
  in computer science to define a
  `formal language
  <https://en.wikipedia.org/wiki/Formal_language>`__
  such as JSON, Python, Rust but also much more basic ones.
  A language is just a set of strings and therefore each item of the
  search space can be a simple string such as "Hello world!" or a
  complex string such as a multi-line program.

- To summarize the previous points: This package provides several
  systems to search for the
  best string in a grammar's language, where best means the item
  with highest or lowest value assigned by an objective function.

The name alogos also stands for the greek word
`άλογος
<https://en.wiktionary.org/wiki/%CE%AC%CE%BB%CE%BF%CE%B3%CE%BF%CF%82>`__,
which means "irrational" or "without reason".
This property characterizes the stochastic search algorithms provided
in this package well, yet does not hinder them from finding good
solutions to hard optimization problems in limited time.
An interesting open question is whether reasoning capabilities could
be added to make the search process more effective without impacting
its efficiency too much.



Which goals are pursued by this project?
========================================

This project revolves around a particular family of
`optimization algorithms
<https://en.wikipedia.org/wiki/List_of_algorithms#Optimization_algorithms>`__
that are mainly studied in the field of
`evolutionary computation (EC)
<https://en.wikipedia.org/wiki/Evolutionary_computation>`__
and can be referred to with the umbrella term
`Grammar-Guided Genetic Programming (GGGP, G3P)
<https://scholar.google.de/citations?view_op=view_citation&citation_for_view=MVjWALEAAAAJ:4JMBOYKVnBMC>`__.
This package currently provides implementations of five different
G3P methods:

1. `Context-Free Grammar Genetic Programming (CFG-GP)
   <https://scholar.google.de/citations?view_op=view_citation&user=BA0ubm4AAAAJ&citation_for_view=BA0ubm4AAAAJ:u5HHmVD_uO8C>`__

2. `Grammatical Evolution (GE)
   <https://scholar.google.de/citations?view_op=view_citation&user=KlZHzFgAAAAJ&citation_for_view=KlZHzFgAAAAJ:AvfA0Oy_GE0C>`__

3. `Position-independent Grammatical Evolution (piGE)
   <https://scholar.google.de/citations?view_op=view_citation&user=KlZHzFgAAAAJ&citation_for_view=KlZHzFgAAAAJ:hFOr9nPyWt4C>`__

4. `Dynamic Structured Grammatical Evolution (DSGE)
   <https://scholar.google.de/citations?view_op=view_citation&user=IC4uQLcAAAAJ&citation_for_view=IC4uQLcAAAAJ:2P1L_qKh6hAC>`__

5. `Weighted Hierarchical Grammatical Evolution (WHGE)
   <https://scholar.google.de/citations?view_op=view_citation&user=PMy0x0MAAAAJ&citation_for_view=PMy0x0MAAAAJ:3WNXLiBY60kC>`__

This project tries to achieve two main goals:

1. Provide a modular implementation of several G3P methods
   to make them easier to study, compare and extend.

2. Democratize the access to these methods, so they can be used
   more frequently and by a wider audience.



How can you get started?
========================

If you want to use this package, the following steps should
help you to get results in a short time:

1. The :doc:`Installation Guide <rst/installation>`
   describes how you can install the package and its optional
   dependencies.

2. The :doc:`Quickstart Example <rst/quickstart>`
   is a short piece of code you can run immediately
   after the installation to see if it worked and to
   get a first impression of the package.

3. The `Getting Started Tutorial <rst/getting_started.ipynb>`_
   explains the main concepts required to define and solve your
   own optimization problems, such as defining a grammar and objective
   function as well as applying an evolutionary algorithm to it.

4. The :doc:`Code Examples <rst/examples>`
   contain solutions to optimization problems from different domains.
   They illustrate the generality and flexibility of the methods
   provided in this package. You can use the examples as templates or
   recipes that can be adapted to tackle similar problems.

5. The :doc:`API Documentation <autoapi/index>`
   contains a comprehensive description of all user-facing
   functionality available in this package, so you can see the full
   range of available options.



Where can everything be found?
==============================

The :doc:`Package References <rst/package_references>`
page contains a collection of links to all parts of this project,
including the source code, distributed code and documentation.



Who is involved in this project?
================================

- The design, implementation and documentation of this package was
  done by
  `Robert Haas <https://github.com/robert-haas>`_.

- Several research groups in the field of evolutionary computation
  invented the original G3P algorithms that are reproduced here.
  For a list of involved persons, please have a look at the authors of
  the research articles referenced above and in the API Documentation.

- Financial support to realize this project was granted by the
  `Deep Funding <https://deepfunding.ai>`__
  initiative of
  `SingularityNET <https://singularitynet.io>`__.
  If you are interested in using or providing ML/AI algorithms,
  you may want to have a look at SingularityNET's decentralized
  `marketplace of AI services
  <https://beta.singularitynet.io/getstarted>`__.



Table of contents
=================

.. toctree::
   :maxdepth: 1

   rst/package_references
   rst/installation
   rst/quickstart
   rst/getting_started
   rst/examples
