alogos
======

This package implements different methods from the field of
**Grammar-Guided Genetic Programming (G3P)**.

The current alpha version (v0.1.0) covers two methods and one variation:

- `Context-Free Grammar Genetic Programming (CFG-GP) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=BA0ubm4AAAAJ&citation_for_view=BA0ubm4AAAAJ:u5HHmVD_uO8C>`_
- Context-Free Grammar Genetic Programming on Serialized Trees (CFG-GP-ST) is a slight
  variation of CFG-GP that operates on serialized trees in a preliminary attempt
  to improve its performance.
- `Grammatical Evolution (GE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=KlZHzFgAAAAJ&citation_for_view=KlZHzFgAAAAJ:AvfA0Oy_GE0C>`_

Future versions will cover three further methods:

- `Position-Independent Grammatical Evolution (piGE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=KlZHzFgAAAAJ&citation_for_view=KlZHzFgAAAAJ:hFOr9nPyWt4C>`_
- `Dynamic Structured Grammatical Evolution (DSGE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=IC4uQLcAAAAJ&citation_for_view=IC4uQLcAAAAJ:2P1L_qKh6hAC>`_
- `Weighted Hierarchical Grammatical Evolution (WHGE) <https://scholar.google.de/citations?view_op=view_citation&hl=de&user=PMy0x0MAAAAJ&citation_for_view=PMy0x0MAAAAJ:3WNXLiBY60kC>`_

The further plan is to collect a comprehensive benchmark suite
and provide statistical evaluation methods
according to guidelines from the genetic programming community
in order to compare the implemented methods in a meaningful way.
The package will also come with examples from different application areas,
have a documentation website, and be available as service on the
SingularityNET platform where it can be used as general-purpose optimization
service for various use cases.

This project is supported by the 
`Deep Funding <https://deepfunding.ai/>`_
initiative from
`SingularityNET <https://singularitynet.io/>`_.
More background information can be found on the
`proposal website <https://robert-haas.github.io/g3p/>`_.


Preliminary installation
------------------------

This repository can be installed as a Python 3 package using ``git`` and ``pip``
with the following command: ``pip install git+https://github.com/robert-haas/alogos.git``
