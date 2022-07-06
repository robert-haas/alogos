alogos
======

Welcome! You have found alogos, a library of grammatical optimization systems.


Installation
------------

This repository can be installed as a Python 3 package using ``git`` and ``pip``
with a single command: ``pip install git+https://github.com/robert-haas/alogos.git``


Content
-------

The goal is to implement following methods from the field of 
`Grammar-Guided Genetic Programming (GGGP, G3P) <https://scholar.google.com/citations?view_op=view_citation&citation_for_view=BA0ubm4AAAAJ:UeHWp8X0CEIC>`__
in order to make them easily accessible and directly comparable:

- `Context-Free Grammar Genetic Programming (CFG-GP) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=BA0ubm4AAAAJ&citation_for_view=BA0ubm4AAAAJ:u5HHmVD_uO8C>`_

  - Context-Free Grammar Genetic Programming on Serialized Trees (CFG-GP-ST) is a slight variation of CFG-GP that operates on serialized trees in a preliminary attempt to improve its performance, which might be further improved in future.
- `Grammatical Evolution (GE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=KlZHzFgAAAAJ&citation_for_view=KlZHzFgAAAAJ:AvfA0Oy_GE0C>`_
- `Position-Independent Grammatical Evolution (piGE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=KlZHzFgAAAAJ&citation_for_view=KlZHzFgAAAAJ:hFOr9nPyWt4C>`_
- `Dynamic Structured Grammatical Evolution (DSGE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=IC4uQLcAAAAJ&citation_for_view=IC4uQLcAAAAJ:2P1L_qKh6hAC>`_
- `Weighted Hierarchical Grammatical Evolution (WHGE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=PMy0x0MAAAAJ&citation_for_view=PMy0x0MAAAAJ:3WNXLiBY60kC>`_


Releases
--------

- The current alpha version (v0.1.0) covers CFG-GP, CFG-GP-ST and GE.
- Upcoming beta versions will additionally cover piGE, DSGE and WHGE and
  also become available on PyPI.
- A benchmark suite will be collected according to
  `guidelines from the Genetic Programming community <https://scholar.google.de/citations?view_op=view_citation&citation_for_view=_mzk1w4AAAAJ:ZHo1McVdvXMC>`__
  in order to compare the performance of the methods.
- A documentation website will be created with an API reference and example gallery.
- The best performing method will be provided on the SingularityNET platform,
  where its role is that of a general-purpose optimization service that
  can be called by users or other services to solve hard optimization problems
  from various domains.


Further information
-------------------

- This project is supported by the
  `Deep Funding <https://deepfunding.ai/>`_
  initiative from
  `SingularityNET <https://singularitynet.io/>`_.
- A `demonstration website <https://robert-haas.github.io/g3p/>`_ contains an explanation of
  the proposed projects, an example gallery with code examples and references to the original
  proposals that were submitted to DeepFunding round 1.
- A `short video <https://youtu.be/0wKIBCdLMuQ?t=1501>`__
  (10min) illustrates what can be done with this package.
- A `longer video <https://www.youtube.com/watch?v=8klrXhelAiQ>`__
  (36min) introduces necessary concepts from formal language theory
  (alphabet, string, language, grammar) and demonstrates how this package can be used to
  1) define a formal language with a context-free grammar and
  2) search for an optimal string in this language with an evolutionary algorithm.
