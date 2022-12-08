"""Initialization functions to generate a population for WHGE."""

from .._shared import init_population as _init_population
from . import default_parameters as _dp
from . import init_individual as _init_individual
from . import representation as _representation


def given_genotypes(grammar, parameters=None):
    """Create a population from given genotypes.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``init_pop_given_genotypes`` (`list` of `~.representation.Genotype` objects or data that can be converted to a genotype) : A
          list of genotypes, which are used to initialize the
          individuals of the population. Note that the length
          of this list determines the size of the generated population.

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    """
    return _init_population.given_genotypes(
        grammar, parameters, _dp, _representation, _init_individual
    )


def random_genotypes(grammar, parameters=None):
    """Create a population from random genotypes.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``population_size`` (`int`)
        - ``init_pop_unique_genotypes`` (`bool`)
        - ``init_pop_unique_phenotypes`` (`bool`)
        - ``init_pop_unique_max_tries`` (`int`)

    Returns
    -------
    population : `~.representation.Population`

    Raises
    ------
    InitializationError
        If creating the population fails.

    Notes
    -----
    Choice of genotype size:

    - In the 2017 paper on HGE and WHGE the genotype size is set to
      256, 512 and 1024 bits in the experiments.
    - In the 2018 paper on WHGE the genotype size is set to 1024 bits.
    - The reference implementation in Java comes with examples where the
      genotype size is set mostly to 256 or 1024 bits.

    Construction of a population of random individuals:

    - The reference implementation in Java seems to construct random
      individuals independently, without checking for duplicates on
      genotype or phenotype level. Is is extremely unlikely to ever
      create two identical individuals on the genotype level
      (2^256 or 2^1024 possibilities), but on the phenotype level it
      can happen quite often (depending on structure of the grammar).

    References
    ----------
    - Papers

        - 2017: Medvet,
          `Hierarchical Grammatical Evolution
          <https://doi.org/10.1145/3067695.3075972>`__

            - p. 250: "For each GE variant, problem, and genotype size, we performed
              30 independent executions of the evolutionary search by varying the
              random seed and with the following evolution parameters:
              population of 500 individuals randomly initialized [...]"

            - p. 250, Table 1: genotype sizes |g| 256, 512, 1024

        - 2018: Bartoli, Castelli, Medvet:
          `Weighted Hierarchical Grammatical Evolution
          <https://doi.org/10.1109/TCYB.2018.2876563>`__

            - p. 7: "Concerning the variant-specific parameters, we set the
              genotype size to 1024 bits for GE, πGE, and WHGE"

            - p. 8: "This finding could be explained by the lower degeneracy of WHGE
              (see Section IV-D) which results in a tendency of WHGE to better sample
              the phenotype space given a random set of genotypes."

    - Reference implementation in Java: `evolved-ge
      <https://github.com/ericmedvet/evolved-ge>`__

        - `Folder with initialization classes
          <https://github.com/ericmedvet/evolved-ge/tree/master/src/main/java/it/units/malelab/ege/core/initializer>`__

        - `Folder with example scripts
          <https://github.com/ericmedvet/evolved-ge/blob/master/src/main/java/it/units/malelab/ege>`__

            - Using 256 bits: ``GOM.java``
            - Using 1024 bits: ``DeepExperimenter.java``, ``DeepDistributedExperimenter.java``
            - Using 256 and 1024 bits: ``MapperGenerationExperimenter.java``,
              ``MapperGenerationDistributedExperimenter.java``
            - Using 64, 128, 256, 512 and 1024 bits: ``MappingPropertiesExperimenter.java``

    """
    return _init_population.random_genotypes(
        grammar, parameters, _dp, _representation, _init_individual
    )
