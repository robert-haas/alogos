"""Core components of Grammar-Guided Genetic Programming (G3P) systems.

This subpackage implements the core functionality of each
G3P system in a highly modular form:

- ``Representation``: What kind of genotype is used to indirectly
  represent a phenotype?
- ``Mapping``: How is a given genotype deterministically translated
  into a certain phenotype?
- ``Initialization``: Which methods are there to generate genotypes
  for individuals of an initial population?
- ``Mutation``: Which methods are there to mutate a given genotype
  in order to generate a random but related new genotype?
- ``Crossover``: Which methods are there to recombine two genotypes
  in order to generate two random but related new genotypes?
- ``Neighborhood``: Which methods are there to generate all
  or some genotypes that are close to a given genotype?

The consequence of this modularization is that the components
can not only be used within an evolutionary algorithm,
but also within other metaheuristic search methods such as
hill climbing or simulated annealing.

"""

from . import cfggp, cfggpst, dsge, ge, pige, whge
