"""Shared individual initialization functions of different systems."""

import random as _random

from ... import exceptions as _exceptions
from ..._utilities.parametrization import get_given_or_default as _get_given_or_default
from . import init_tree as _init_tree


def given_genotype(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a given genotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``init_ind_given_genotype`` : Data for a Genotype.
    default_parameters : `~alogos._utilities.parametrization.ParameterCollection`
        Default parameters of the system that calls this generic
        function.
    representation : module
        Representation module of the system that calls this generic
        function. This module contains the specific
        `~.representation.Genotype` subclass of the system, which is
        used here to create the genotype for the individual.
    mapping : function
        Mapping module of the system that calls this generic
        function. This module contains the specific `forward` function
        of the system, which is used here to map the genotype of the
        individual to a derivation tree and phenotype.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    """
    # Parameter extraction
    gt = _get_given_or_default(
        "init_ind_given_genotype", parameters, default_parameters
    )

    # Argument processing
    if not isinstance(gt, representation.Genotype):
        try:
            if gt is None:
                _exceptions.raise_no_genotype_error()
            gt = representation.Genotype(gt)
        except Exception:
            _exceptions.raise_init_ind_from_gt_error(gt)

    # Transformation
    try:
        phe, dt = mapping.forward(grammar, gt, parameters, return_derivation_tree=True)
    except _exceptions.MappingError:
        phe = None
        dt = None
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))


def given_derivation_tree(
    grammar, parameters, default_parameters, representation, mapping
):
    """Create an individual from a given derivation tree.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``init_ind_given_derivation_tree`` : Data for a
          `~alogos._grammar.data_structures.DerivationTree`.
    default_parameters : `~alogos._utilities.parametrization.ParameterCollection`
        Default parameters of the system that calls this generic
        function.
    representation : module
        Representation module of the system that calls this generic
        function. This module contains the specific
        `~.representation.Genotype` subclass of the system, which is
        used here to create the genotype for the individual.
    mapping : function
        Mapping module of the system that calls this generic
        function. This module contains the specific `reverse` function
        of the system, which is used here to map the phenotype (read
        from the leaves of the derivation tree) to a genotype.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    """
    # Parameter extraction
    dt = _get_given_or_default(
        "init_ind_given_derivation_tree", parameters, default_parameters
    )

    # Transformation
    try:
        if dt is None:
            _exceptions.raise_no_derivation_tree_error()
        gt = mapping.reverse(grammar, dt, parameters)
    except Exception:
        _exceptions.raise_init_ind_from_dt_error(dt)
    phe = dt.string()
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))


def given_phenotype(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a given phenotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``init_ind_given_phenotype`` (`str`) : Data for a phenotype,
          which needs to be a string of the grammar's language.
    default_parameters : `~alogos._utilities.parametrization.ParameterCollection`
        Default parameters of the system that calls this generic
        function.
    representation : module
        Representation module of the system that calls this generic
        function. This module contains the specific
        `~.representation.Genotype` subclass of the system, which is
        used here to create the genotype for the individual.
    mapping : function
        Mapping module of the system that calls this generic
        function. This module contains the specific `reverse` function
        of the system, which is used here to map the phenotype
        to a genotype and derivation tree.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    """
    # Parameter extraction
    phe = _get_given_or_default(
        "init_ind_given_phenotype", parameters, default_parameters
    )

    # Transformation
    try:
        if phe is None:
            _exceptions.raise_no_phenotype_error()
        gt, dt = mapping.reverse(grammar, phe, parameters, return_derivation_tree=True)
    except Exception:
        _exceptions.raise_init_ind_from_phe_error(phe)
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))


def random_genotype(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a random genotype.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``genotype_length`` (`int`) : Number of integers in the GE
          or piGE `~.representation.Genotype`.
        - ``codon_size`` (`int`) : Number of bits used for a codon,
          which determines the maximum integer value a codon can take.
          For example, a codon size of 8 bits allows integers from
          0 to 255 (from 2**8-1).
    default_parameters : `~alogos._utilities.parametrization.ParameterCollection`
        Default parameters of the system that calls this generic
        function.
    representation : module
        Representation module of the system that calls this generic
        function. This module contains the specific
        `~.representation.Genotype` subclass of the system, which is
        used here to create the genotype for the individual.
    mapping : function
        Mapping module of the system that calls this generic
        function. This module contains the specific `forward` function
        of the system, which is used here to map the genotype
        to a phenotype and derivation tree.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    """
    # Parameter extraction
    gl = _get_given_or_default("genotype_length", parameters, default_parameters)
    cs = _get_given_or_default("codon_size", parameters, default_parameters)

    # Transformation
    try:
        assert cs > 0
        max_int = 2**cs - 1
        random_genotype = [_random.randint(0, max_int) for _ in range(gl)]
    except Exception:
        _exceptions.raise_init_ind_rand_gt_error()

    if parameters is None:
        parameters = dict()
    parameters["init_ind_given_genotype"] = random_genotype
    return given_genotype(
        grammar, parameters, default_parameters, representation, mapping
    )


def random_valid_genotype(
    grammar, parameters, default_parameters, representation, mapping
):
    """Create an individual from a random genotype likely to be valid.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`
        Following keyword-value pairs are considered by this function:

        - ``genotype_length`` (`int`) : Number of integers in the GE
          or piGE `~.representation.Genotype`.
        - ``codon_size`` (`int`) : Number of bits used for a codon,
          which determines the maximum integer value a codon can take.
          For example, a codon size of 8 bits allows integers from
          0 to 255 (from 2**8-1).
        - ``init_ind_random_valid_genotype_max_tries`` (`int`) : Number
          of tries to generate a random valid genotype.
    default_parameters : `~alogos._utilities.parametrization.ParameterCollection`
        Default parameters of the system that calls this generic
        function.
    representation : module
        Representation module of the system that calls this generic
        function. This module contains the specific
        `~.representation.Genotype` subclass of the system, which is
        used here to create the genotype for the individual.
    mapping : function
        Mapping module of the system that calls this generic
        function. This module contains the specific `forward` function
        of the system, which is used here to map the genotype
        to a phenotype and derivation tree.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    Notes
    -----
    This function repeatedly calls `random_genotype` until it returns
    a genotype that can be mapped to a phenotype or a given maximum
    number of tries is reached.

    """
    # Parameter extraction
    mt = _get_given_or_default(
        "init_ind_random_valid_genotype_max_tries", parameters, default_parameters
    )

    # Transformation
    for _ in range(mt):
        ind = random_genotype(
            grammar, parameters, default_parameters, representation, mapping
        )
        if ind.phenotype is not None:
            break
    else:
        _exceptions.raise_init_ind_valid_rand_gt_error(mt)
    return ind


def gp_grow_tree(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a tree created with "Grow" from GP.

    References
    ----------
    - 2016, Fagan et al.: `Exploring position independent
      initialisation in grammatical evolution
      <https://doi.org/10.1109/CEC.2016.7748331>`__

      - "Traditional Grow in GP looks to construct a tree that is at
        most a certain depth. The trees are generally grown in a
        recursive manner, randomly picking production choices until
        only leaf nodes remain. If the tree approaches the depth limit
        then a terminating sequence is selected to make sure the depth
        limit is not breached"

      - "Grow randomly builds a tree up to a maximum speciﬁed depth.
        If a branch of the tree reaches the imposed depth limit the
        branch is ﬁnished by selecting only terminals. There is no
        guarantee of the tree reaching the depth limit."

    """
    # Parameter extraction
    md = _get_given_or_default(
        "init_ind_gp_grow_max_depth", parameters, default_parameters
    )

    try:
        # Create the tree with the "grow" strategy
        dt = _init_tree.grow_all_branches_within_max_depth(grammar, md)

        # Map tree to genotype and phenotype
        ind = _create_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_grow_error()
    return ind


def gp_full_tree(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a tree created with "Full" from GP.

    References
    ----------
    - 2016, Fagan et al.: `Exploring position independent
      initialisation in grammatical evolution
      <https://doi.org/10.1109/CEC.2016.7748331>`__

        - "Full constructs a tree where every branch extends to the set
          maximum depth. This generally results in very bushy or full
          trees."

    """
    # Parameter extraction
    md = _get_given_or_default(
        "init_ind_gp_full_max_depth", parameters, default_parameters
    )

    try:
        # Create the tree with the "full" strategy
        dt = _init_tree.grow_all_branches_to_max_depth(grammar, md)

        # Map tree to genotype and phenotype
        ind = _create_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_full_error()
    return ind


def pi_grow_tree(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a tree created with Fagan's "PI Grow".

    References
    ----------
    - 2016, Fagan et al.: `Exploring position independent
      initialisation in grammatical evolution
      <https://doi.org/10.1109/CEC.2016.7748331>`__

        - "PI Grow now randomly selects a new non-terminal from the
          queue and again a production is done and the resulting
          symbols are added to the queue in the position from which the
          parent node was removed."

        - "As the tree is being expanded, the algorithm checks to see
          if the current symbol is the last recursive symbol remaining
          in the queue. If the depth limit hasn’t been reached, and
          PI Grow currently has the last recursive symbol to expand,
          Pi Grow will only pick a production that results in
          recursive symbols. This process guarantees that at least one
          branch will reach the specified depth limit. This gives the
          initialiser the freedom to generate trees of both a Full and
          Grow nature."

    """
    # Parameter extraction
    md = _get_given_or_default(
        "init_ind_pi_grow_max_depth", parameters, default_parameters
    )

    try:
        # Create the tree with the "pi grow" strategy, mix of "grow" + "full" for a single branch
        dt = _init_tree.grow_one_branch_to_max_depth(grammar, md)

        # Map tree to genotype and phenotype
        ind = _create_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_pi_grow_error()
    return ind


def ptc2_tree(grammar, parameters, default_parameters, representation, mapping):
    """Create an individual from a tree grown with Nicolau's "PTC2".

    The original PTC2 method for growing random trees was invented by
    Sean Luke in 2000. Some slightly modified variants were
    introduced later by other authors.

    This function implements a PTC2 variant described by Miguel Nicolau
    in 2017 in section "3.3 Probabilistic tree-creation 2 (PTC2)" of
    the publication. It restricts tree size not with a maximum tree
    depth but rather with a maximum number of expansions and if
    possible remains strictly below this limit.

    Parameters
    ----------
    grammar : `~alogos.Grammar`
    parameters : `dict` or `~alogos._utilities.parametrization.ParameterCollection`, optional
        Following keyword-value pairs are considered by this function:

        - ``init_ind_ptc2_max_expansions`` (`int`): The maximum number
          of nonterminal expansions that may be used to grow the tree.
    default_parameters : `~alogos._utilities.parametrization.ParameterCollection`
        Default parameters of the system that calls this generic
        function.
    representation : module
        Representation module of the system that calls this generic
        function. This module contains the specific
        `~.representation.Genotype` subclass of the system, which is
        used here to create the genotype for the individual.
    mapping : function
        Mapping module of the system that calls this generic
        function. This module contains the specific `reverse` function
        of the system, which is used here to map the phenotype
        to a genotype and derivation tree.

    Raises
    ------
    InitializationError
        If the initialization of the individual fails.

    References
    ----------
    - 2000, Luke: `Two Fast Tree-Creation Algorithms for Genetic
      Programming <https://doi.org/10.1109/4235.873237>`__

        - "PTC1 is a modification of GROW that allows the user to
          provide probabilities of appearance of functions in the tree,
          plus a desired expected tree size, and guarantees that, on
          average, trees will be of that size."

        - "With PTC2, the user provides a probability distribution of
          requested tree sizes. PTC2 guarantees that, once it has picked
          a random tree size from this distribution, it will generate
          and return a tree of that size or slightly larger."

    - 2010, Harper: `GE, explosive grammars and the lasting legacy of
      bad initialisation
      <https://doi.org/10.1109/CEC.2010.5586336>`__

        - "The PTC2 methodology is extended to GE and found to produce
          a more uniform distribution of parse trees."

        - "If the algorithm is called in a ramped way (i.e. starting
          with a low number of expansions, say 20, and increasing until
          say 240) then a large number of trees of different size and
          shapes will be generated."

    - 2017, Nicolau: `Understanding grammatical evolution:
      initialisation <https://doi.org/10.1007/s10710-017-9309-9>`__:

        - 3.3 Probabilistic tree-creation 2 (PTC2)
        - 3.6 Probabilistic tree-creation 2 with depth limit (PTC2D)

    - 2018, Ryan, O'Neill, Collins: `Handbook of Grammatical Evolution
      <https://doi.org/10.1007/978-3-319-78717-6>`__

        - p. 13: "More recent work on initialisation includes that of
          Nicolau, who demonstrated that across the problems examined
          in their study, a variant of Harper’s PTC2 consistently
          outperforms other initialisations"

    """
    # Parameter extraction
    me = _get_given_or_default(
        "init_ind_ptc2_max_expansions", parameters, default_parameters
    )

    try:
        # Create the tree with the "PTC2" strategy
        dt = _init_tree.ptc2(grammar, me)

        # Map tree to genotype and phenotype
        ind = _create_ind_from_tree(grammar, dt, representation, mapping)
    except Exception:
        _exceptions.raise_init_ind_ptc2_error()
    return ind


def _create_ind_from_tree(grammar, dt, representation, mapping):
    """Given a derivation tree, construct an individual with genotype and phenotype from it."""
    # Genotype can be found by reverse mapping of the derivation tree, reading the decisions in it
    gt = mapping.reverse(grammar, dt)

    # Phenotype can be retrieved by reading the tree leaves, which contain the terminals
    phe = dt.string()

    # Combine data into a single Individual object
    return representation.Individual(gt, phe, details=dict(derivation_tree=dt))
