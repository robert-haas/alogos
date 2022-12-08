"""Data structure for a evolutionary algorithm that uses G3P systems."""

import random as _random
from numbers import Number as _Number

import pylru as _pylru

from ... import Grammar as _Grammar
from ... import _utilities
from ... import exceptions as _exceptions
from ... import systems as _systems
from ... import warnings as _warnings
from ..._utilities import argument_processing as _ap
from ..._utilities.parametrization import ParameterCollection as _ParameterCollection
from ..shared import evaluation as _evaluation
from . import database as _database
from . import operators as _operators
from . import parameters as _parameters
from . import reporting as _reporting
from . import state as _state


class EvolutionaryAlgorithm:
    """Evolutionary algorithm for searching an optimal string in a grammar's language.

    The role of this class is explained by first introducing
    basic concepts from optimization theory and the field of
    grammar-guided genetic programming.
    Then the usage of this evolutionary algorithm is explained.

    1. Concepts from optimization theory

        - Optimization: In general terms, optimization means to search
          for the best item out of many alternatives, or in other words,
          an optimal solution out of many candidate solutions.
        - Search space: The set of all candidate solutions is called
          the search space. It can have a finite or infinite number of
          elements and there can be mathematical relationships between
          them, which can be used to efficiently sample the space.
        - Search goal: Having a best or optimal item in the space
          requires that there is an order on the items. Such an order
          can be established in different ways, but the most common one
          is to define an objective function, which takes an item as
          input and returns a number as output. This number represents
          a quality score or fitness of the item and puts a weak order
          on all items, which means that some of them can be equally
          good and therefore multiple optimal items may exist that
          share the best quality score.
        - Search method: There are many optimization algorithms that
          work very efficiently on specific search spaces with certain
          mathematical structure. Methods that are more general and can
          act on a large variety of search spaces are sometimes called
          metaheuristics. They are usually stochastic optimization
          algorithms, which means they have a random element to them
          and that implies that running them several times can produce
          different results, i.e. find candidate solutions of different
          quality. They don't come with any guarantee to find a good
          or optimal solution, but often can successfully do so in
          hard optimization problems and are therefore often used in
          practice when the search space can not be tackled with exact
          methods. An evolutionary algorithm is a prominent
          example of a population-based metaheuristic, which was
          originally inspired by the process of natural evolution.
          Genetic programming is one subdiscipline in the field of
          evolutionary computation that deals with optimization in
          spaces of programs. Grammar-guided genetic programming (G3P)
          is one flavor of it that allows to define the search space
          with a context-free grammar, which is very often used in
          computer science to define various kinds of formal languages,
          including modern programming languages like Python or Rust.

    2. Concepts from grammar-guided genetic programming

        - A grammar can be used to define a formal language.
        - A formal language is a finite or infinite set of string.
        - A search algorithm can try to find the best item in
          a set according to a given search goal.
        - An objective function can define a search goal by assigning
          a score to each item in a set, so that the item with
          minimum or maximum score are considered the best ones and
          need to be found.

    This class provides a metaheuristic search method known as
    evolutionary algorithm, which was tailored for use with different
    grammar-guided genetic programming systems such as Grammatical
    Evolution (GE) and Context-Free Grammar Genetic
    Programming (CFG-GP). It uses following concepts that are directly
    relevant for the usage of this class:
    
    - Individual: An individual represents a single candidate solution
      and comes with a genotype, phenotype and fitness.
      The genotype can be modified with variation operators to give
      rise to new closely related genotypes and thereby move through
      the search space. It also allows to derive a phenotype, which is
      an element of the search space that in the case of grammar-guided
      genetic programming is simply a string of the grammar's language.
      The fitness is a number that indicates the quality of a candidate
      solution and is
      calculated by a user-provided objective function.
    - Population: A population is a collection of individuals.
    - Generation: A generation is one of several consecutive populations
      used throughout a search with an evolutionary algorithm.
      The search begins with the creation of a random initial population
      and proceeds by generating one population after the other, each
      based on the previous one. The idea is that the average fitness
      of the individuals in the generations can be increased by
      using variation operators to generate new, closely related
      individuals together with selection operators that introduce a
      preference for individuals with higher fitness. This is done by

      1. Parent selection to choose individuals that are allowed to
         create offspring.
      2. Crossover to create new individuals by mixing the genotypes of
         parent individuals.
      3. Mutation to introduce slight random variations into the
         offspring individuals.
      4. Survivor selection to choose individuals that are allowed to
         continue into the next generation.

      All of these steps can be performed by different operators and
      this class provides several well-known ones to modify how the
      search is performed.

    """

    __slots__ = ("database", "parameters", "state", "_report")

    # Initialization and reset
    def __init__(
        self,
        grammar,
        objective_function,
        objective,
        system="cfggpst",
        evaluator=None,
        **kwargs
    ):
        """Create and parameterize an evolutionary algorithm.

        Parameters
        ----------
        grammar : `~alogos.Grammar`
            Context-free grammar that defines the search space.
            A grammar specifies a formal language, which is a finite or
            infinite set of strings. Each item of the search space
            is simply a string of the grammar's language. The aim is
            to find an optimal string according to a user-provided
            objective.
        objective_function : `~collections.abc.Callable`
            A function that gets an item of the search space as input
            and needs to return a numerical value for it as output.
            In other words, it gets a phenotype and returns a fitness
            value for it that represents how good it is.

            Given that the search space is a grammar's language,
            each candidate solution or phenotype is a string. Therefore
            a very simple objective function could just measure the
            length of each string and return that as fitness value::

                def obj_fun(string):
                    fitness = len(string)
                    return fitness

            More realistic objective functions will usually evaluate
            the string in some way and measure how well it performs on
            a task.
        objective : `str`
            Goal of the optimization, which is to find a candidate
            solution that has either the minimum or maximum value
            assigned to it by the objective function.

            Possible values:

            - ``"min"``: Minimization, i.e. looking for a candidate
              solution with the smallest possible fitness value.
            - ``"max"``: Maximization, i.e. looking for a candidate
              solution with the highest possible fitness value.
        system : `str`, optional
            Grammar-guided gentic programming system used by the
            evolutionary algorithm.

            Possible values:

            - ``"cfggp"``: CFG-GP, see `~alogos.systems.cfggp`.
            - ``"cfggpst"``: CFG-GP-ST, see `~alogos.systems.cfggpst`.
            - ``"dsge"``: DSGE, see `~alogos.systems.dsge`.
            - ``"ge"``: GE, see `~alogos.systems.ge`.
            - ``"pige"``: piGE, see `~alogos.systems.pige`.
            - ``"whge"``: WHGE, see `~alogos.systems.whge`.

            The system provides following components to the search
            algorithm:

            - Genotype representation: This is a system-specific,
              indirect representation of a candidate solution,
              e.g. a list of int, a bitarray, or a tree.
            - Genotype-phenotype mapping: This is used to translate a
              genotype to a phenotype. Note that a genotype is
              system-specific, while a phenotype is a string of
              the grammar's language.
            - Random variation operators: These are used to move through
              the search space by generating new genotypes that are
              close or related to the original ones. This is achieved
              by randomly introducing slight changes into a genotype
              (mutation) or recombining multiple genotypes
              into new ones (crossover).
        evaluator : `~collections.abc.Callable`, optional
            A function that gets a function together with a list of
            items as input and needs to return a list of new items as
            output. The task of this function is to evaluate the given
            function for each item in the list, and collect the return
            values in a new list. All function calls are assumed to be
            independent and may be computed serially, in parallel or in
            a distributed fashion.
        **kwargs : `dict`, optional
            Further keyword arguments are forwarded either to the
            evolutionary algorithm or to the chosen `system`. This
            allows to control the search process in great detail.
            Every parameter can also be changed later on via the
            attribute `parameters` of the generated instance.
            This can be done before starting a run with the `run`
            method, but also throughout a run when it is performed
            one step at a time by repeatedly calling the `step` method.

            For a description of available keyword arguments, see

            - Parameters for the evolutionary algorithm itself:
              `~alogos._optimization.ea.parameters.default_parameters`
            - Parameters of the chosen G3P system:

              - CFG-GP: `~alogos.systems.cfggp.default_parameters`
              - CFG-GP-ST: `~alogos.systems.cfggpst.default_parameters`
              - DSGE: `~alogos.systems.dsge.default_parameters`
              - GE: `~alogos.systems.ge.default_parameters`
              - piGE: `~alogos.systems.pige.default_parameters`
              - WHGE: `~alogos.systems.whge.default_parameters`

            For example, when the chosen system is GE, the creation
            and parametrization of the evolutionary algorithm could
            look like this::

                import alogos as al

                grammar = al.Grammar("<S> ::= 1 | 22 | 333")

                def obj_fun(string):
                    return len(string)

                ea = al.EvolutionaryAlgorithm(
                    grammar, obj_fun, "max", system="ge",
                    max_generations=10, max_wraps=2)
                ea.run()

            Note that in this example ``max_generations`` is a parameter
            of the evolutionary algorithm itself and therefore it can be
            used no matter which system is chosen.
            In contrast, ``max_wraps`` is a parameter specific to
            the system GE and would not be accepted if for example the
            system CFG-GP were chosen by passing ``system="cfggp"``
            instead of ``system="ge"``.

        Raises
        ------
        ParameterError
            If a parameter is provided that is not known to the
            evolutionary algorithm or to the chosen system.
            A list of available parameters and their default values
            will be printed.

        """
        # Argument processing
        _ap.check_arg("grammar", grammar, types=(_Grammar,))
        _ap.callable_arg("objective_function", objective_function)
        objective = _ap.str_arg(
            "objective", objective, vals=("min", "max"), to_lower=True
        )
        system = _ap.str_arg(
            "system",
            system,
            vals=("cfggp", "cfggpst", "dsge", "ge", "pige", "whge"),
            to_lower=True,
        )

        system = getattr(_systems, system)
        if evaluator is None:
            evaluator = _evaluation.default_evaluator
        else:
            _ap.callable_arg("evaluator", evaluator)

        # Parameter processing
        self.parameters = self._process_parameters(
            grammar, objective_function, objective, system, evaluator, kwargs
        )

        # State and database (re)initialization
        self.reset()

        # Reporting
        if isinstance(self.parameters.verbose, bool):
            if self.parameters.verbose:
                self._report = _reporting.MinimalReporter()
        elif isinstance(self.parameters.verbose, _Number):
            if self.parameters.verbose == 1:
                self._report = _reporting.MinimalReporter()
            elif self.parameters.verbose > 1:
                self._report = _reporting.VerboseReporter()

    def _process_parameters(
        self, grammar, objective_function, objective, system, evaluator, kwargs
    ):
        """Try to assign keyword arguments to predefined parameters.

        Raises
        ------
        ParameterError
            If a keyword argument is passed whose key is not a known
            parameter name.

        """
        # Dictionary of algorithm parameters and system parameters with all default values
        params = dict(
            grammar=grammar,
            objective_function=objective_function,
            objective=objective,
            system=system,
            evaluator=evaluator,
        )
        params.update(_parameters.default_parameters)
        params.update(system.default_parameters)

        # Overwrite default values with user-provided kwargs, raise error if a name is unknown
        params = _ParameterCollection(params)
        for key, val in kwargs.items():
            if key in params:
                params[key] = val
            else:
                _exceptions.raise_unknown_parameter_error(key, params)

        # Convert values
        if isinstance(params["verbose"], bool):
            params["verbose"] = int(params["verbose"])

        # Check types
        _ap.int_arg("population_size", params["population_size"], min_incl=1)
        _ap.int_arg("offspring_size", params["offspring_size"], min_incl=1)
        _ap.int_arg("verbose", params["verbose"], min_incl=0)
        _ap.check_arg("database_on", params["database_on"], types=(bool,))
        _ap.check_arg("database_location", params["database_location"], types=(str,))
        return params

    # Representations
    def __repr__(self):
        """Compute the "official" string representation of the algorithm.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__repr__

        """
        return "<EvolutionaryAlgorithm object at {}>".format(hex(id(self)))

    def __str__(self):
        """Compute the "informal" string representation of the algorithm.

        References
        ----------
        - https://docs.python.org/3/reference/datamodel.html#object.__str__

        """
        return str(self.state)

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter.

        References
        ----------
        - https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display

        """
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Optimization algorithm
    def step(self):
        """Perform a single step of the evolutionary search.

        Taking a single step means that one generation is added,
        which can happen in one of two ways:

        - If the run has just started, the first generation is created
          by initializating a population.
        - If the run is already ongoing, a new generation is derived
          from the current generation by a process that involves
          1) parent selection that prefers fitter individuals,
          2) generating an offspring population by applying random
          variation operators (mutation, crossover) to the parents and
          3) survivor selection that again prefers fitter individuals.

        Two important notes about the behavior of this method:

        - It contrast to the `run` method, it does not check
          if any stop criterion has been met!
          For example, when the number of generations reaches
          the value provided by the ``max_generations`` parameter,
          calling this method will regardless add another generation.
        - Since this method has to be called repeatedly to perform
          the individual steps of a run, it enables to change the
          parameters of the evolutionary algorithm on the fly by
          modifying the `parameters` attribute between calls.
          This means the search can be adjusted in much greater detail
          than with the `run` method.
          For example, the mutation and crossover rates can be gradually
          modified during a run to reduce the amount of variation when
          getting closer to the goal. It is also possible to change
          the population size or even the objective function.
          Fixed throughout a run, however, are the grammar
          (=search space) and the G3P system (=internal representation).

        Returns
        -------
        best_individual : `~alogos.systems._shared.representation.Individual` of the chosen G3P system
            The individual with the best fitness found so far in the
            entire search. This means the individual may have been
            discovered not in the current step but in an earlier one.

        """
        # Start time
        self._store_start_time()

        # Check parameters
        self._check_parameter_consistency()

        if self.state.population is None:
            # CREATE first generation
            pop = self._initialize_population()

            # Evaluate
            self._evaluate_population(pop)

            # Store and report
            self._finish_generation(survivor_population=pop)
        else:
            # SELECT parents
            parent_population = self._select_parents(self.state.population)

            # CREATE offspring by random variation
            crossed_over_population = self._cross_over(parent_population)
            mutated_population = self._mutate(crossed_over_population)

            # Evaluate
            self._evaluate_population(mutated_population)

            # SELECT survivors
            survivor_population = self._select_survivors(
                self.state.population, mutated_population
            )

            # Store and report
            self._finish_generation(
                parent_population,
                crossed_over_population,
                mutated_population,
                survivor_population,
            )

        # Stop time
        self._store_stop_time()
        return self.state.best_individual

    def run(self):
        """Perform one search step after another until a stop criterion is met.

        A user can set a single or multiple stop criteria. The algorithm
        will halt when one of them is fulfilled. Caution: If the
        criteria are set in such a way that they can never be realized,
        e.g. because an unreachable fitness threshold was chosen,
        then the run will not halt until it is interrupted from the
        outside.

        Note that after a search has halted, it is possible to change
        the stop criteria and call `run` again to continue the search.
        This can also be done multiple times.

        Returns
        -------
        best_individual : `~alogos.systems._shared.representation.Individual` of the chosen G3P system
            The individual with the best fitness found so far in the
            entire search.

        Raises
        ------
        ParameterError
            If no stop criterion was set. This prevents starting a run
            that clearly would not halt.

        Warns
        -----
        OptimizationWarning
            If not a single step was taken, because some stop criterion
            was already fulfilled by the current search state.

        """
        pr = self.parameters

        # Check if a stop criterion was set
        if (
            pr.max_generations is None
            and pr.max_fitness_evaluations is None
            and pr.max_runtime_in_seconds is None
            and pr.max_or_min_fitness is None
        ):
            _exceptions.raise_stop_parameter_error()

        # Run step after step until a stop criterion is met
        gen_before = self.state.generation
        while not self.is_stop_criterion_met():
            self.step()
        gen_after = self.state.generation

        # Warn if not a single step was performed in this call
        if gen_after == gen_before:
            _warnings._warn_no_step_in_ea_performed(self.state.generation)

        # Report
        if pr.verbose:
            self._report.run_end(self.state)
        return self.state.best_individual

    def is_stop_criterion_met(self):
        """Check if the current search state meets any stop criterion.

        This method is used internally by the `run` method.
        It can also be useful when constructing a specialized search
        that repeatedly calls the `step` method, where it is the
        responsibility of the user to ensure a stop.

        Returns
        -------
        stop : bool
            ``True`` if a stop criterion is met, ``False`` otherwise.

        """
        pr = self.parameters
        st = self.state

        # Maximum number of generations
        if pr.max_generations is not None:
            if st.num_generations >= pr.max_generations:
                return True

        # Maximum number of fitness evaluations
        if pr.max_fitness_evaluations is not None:
            if st.num_phe_to_fit_evaluations >= pr.max_fitness_evaluations:
                return True

        # Maximum runtime
        if pr.max_runtime_in_seconds is not None:
            if st.runtime >= pr.max_runtime_in_seconds:
                return True

        # Fitness threshold
        if pr.max_or_min_fitness is not None:
            if st.best_individual is not None:
                if pr.objective == "min":
                    if st.best_individual.fitness <= pr.max_or_min_fitness:
                        return True
                elif pr.objective == "max":
                    if st.best_individual.fitness >= pr.max_or_min_fitness:
                        return True

        # No stop criterion is met
        return False

    def reset(self):
        """Reset the current search state to enable a fresh start.

        Not all aspects of the algorithm object are simply set back to
        their initial state:

        - Parameters are not reset to their default values. If any
          non-default parameters were passed during object-creation or
          if they were set to user-provided values later via the
          `parameters` attribute, then these values are kept and not
          overwritten by default values. The purpose of this behavior
          is to enable repeated runs that are directly comparable.

        - If a database was used, it is reset, but what that means
          depends on its location:

            - If the database location is in memory, the content is
              entirely lost during a reset.

            - If the database location is a file, it is renamed during a
              reset and a new file is created. This means the content of
              a run is not lost but relocated.

        - Cached values are fully deleted. This means previous
          phenotype-to-fitness evaluations will not be reused in a
          follow-up run.

        """
        # State: storage of main data generated during a search in a single object
        self.state = _state.State()

        # Cache: optional storage of least recently used (lru) mapping results for lookup
        if self.parameters.gen_to_phe_cache_lookup_on:
            self.state._gen_to_phe_cache = _pylru.lrucache(
                self.parameters.gen_to_phe_cache_size
            )
        else:
            self.state._gen_to_phe_cache = None
        if self.parameters.phe_to_fit_cache_lookup_on:
            self.state._phe_to_fit_cache = _pylru.lrucache(
                self.parameters.phe_to_fit_cache_size
            )
        else:
            self.state._phe_to_fit_cache = None

        # Database: optional storage of all data generated during a search in an SQL database
        if self.parameters.database_on:
            self.database = _database.Database(
                self.parameters.database_location, self.parameters.system
            )
        else:
            self.database = None

    # 1) Beginning
    def _initialize_population(self):
        """Create the initial population.

        There are different ways this can be done. Which one is used
        is decided by the default or user-provided parameters.

        """
        pr = self.parameters
        st = self.state

        # Report start of initialization
        if pr.verbose:
            self._report.init_start(
                st.generation, st.start_time, _utilities.times.current_time_readable()
            )

        # Generate initial population
        operator = self._get_operator(
            "Population initialization",
            pr.init_pop_operator,
            pr.system.init_population,
        )
        pr["init_pop_size"] = pr["population_size"]
        initial_population = operator(pr.grammar, pr)

        # Set individual ids
        self._assign_new_ids(initial_population)

        # Set state
        st.population = initial_population

        # Report end of initialization
        if pr.verbose:
            self._report.init_end(len(initial_population))
        return initial_population

    # 2) Evaluation
    def _evaluate_population(self, population):
        """Calculate mappings for each individual.

        1) Genotype-to-phenotype mapping: A system-specific genotype
           is translated into a system-independent phenotype.
        2) Phenotype-to-fitness mapping: A phenotype, which is a string
           of the grammar's language, gets a fitness assigned to it
           by evaluating the user-provided objective function on it.
           In realistic problems this can be a costly operation,
           therefore it makes sense to store the result in a limited
           cache or unlimited database and reuse it instead of
           performing the calculation again. The behavior is chosen
           by the default or user-provided parameters.

        """
        pr = self.parameters

        # Genotype-to-phenotype mapping
        if pr.verbose:
            self._report.map_gen_phe()
        self._calculate_phenotypes(population)

        # Phenotype-to-fitness mapping
        if pr.verbose:
            self._report.map_phe_fit()
        self._calculate_fitnesses(population)

    def _calculate_phenotypes(self, population):
        """Perform the genotype-to_phenotype mapping.

        Possible crashes are prevented by wrapping the function such
        that default values are returned in case of an error or invalid
        return value.

        """
        pr = self.parameters
        st = self.state
        cache = st._gen_to_phe_cache
        cache_on = pr.gen_to_phe_cache_lookup_on

        # Lookup or calculate phenotype to fitness & details mapping
        gen_phe_map = dict()

        # 1) Determine unique genotypes
        unique_genotypes = {ind.genotype for ind in population}
        remaining_genotypes = unique_genotypes.copy()
        n_unique = len(unique_genotypes)

        # 2) For each unique genotype, look up if it is available in cache
        if cache_on:
            for gen in unique_genotypes:
                try:
                    gen_phe_map[gen] = cache[gen]
                    remaining_genotypes.remove(gen)
                except KeyError:
                    pass
        n_after_cache = len(remaining_genotypes)

        # 3) For each remaining genotype, calculate the mapping
        if remaining_genotypes:
            # Evaluate
            forward = pr.system.mapping.forward
            for gen in remaining_genotypes:
                try:
                    phe = forward(pr.grammar, gen)
                except _exceptions.MappingError:
                    phe = None
                gen_phe_map[gen] = phe

            # Update state
            st.num_gen_to_phe_evaluations += len(remaining_genotypes)

        # Assign values from lookup or calculation
        for ind in population:
            ind.phenotype = gen_phe_map[ind.genotype]

        # Update cache
        if cache_on:
            for gen in unique_genotypes:
                cache[gen] = gen_phe_map[gen]

        # Report
        if pr.verbose:
            self._report.calc_phe(
                len(population), n_unique, n_unique - n_after_cache, n_after_cache
            )

    def _calculate_fitnesses(self, population):
        """Perform the phenotype-to-fitness mapping.

        Possible crashes are prevented by wrapping the function such
        that default values are returned in case of an error or invalid
        return value.

        """
        pr = self.parameters
        st = self.state
        cache = st._phe_to_fit_cache
        cache_on = pr.phe_to_fit_cache_lookup_on
        db_on = pr.phe_to_fit_database_lookup_on and pr.database_on

        # Lookup or calculate phenotype to fitness & details mapping
        phe_fit_map = dict()

        # 1) Determine unique phenotypes
        unique_phenotypes = []  # list instead set to preserve evaluation order
        for ind in population:
            phe = ind.phenotype
            if phe not in unique_phenotypes:
                unique_phenotypes.append(phe)
        remaining_phenotypes = []
        n_unique = len(unique_phenotypes)

        # 2) For each unique phenotype, look up if it is available in cache
        if cache_on:
            for phe in unique_phenotypes:
                try:
                    phe_fit_map[phe] = cache[phe]
                except KeyError:
                    remaining_phenotypes.append(phe)
        else:
            remaining_phenotypes = unique_phenotypes
        n_after_cache = len(remaining_phenotypes)

        # 3) For each remaining phenotype, look up if it is available in database
        #    because evaluation of the objective function could be very costly
        if db_on and remaining_phenotypes:
            for phe, fit_det in self.database._lookup_phenotype_evaluations(
                remaining_phenotypes
            ):
                phe_fit_map[phe] = fit_det
                remaining_phenotypes.remove(phe)
        n_after_db = len(remaining_phenotypes)

        # 4) For each remaining phenotype, calculate the objective function
        if remaining_phenotypes:
            # Evaluate
            func = _evaluation.get_robust_fitness_function(
                pr.objective_function, pr.objective
            )
            results = pr.evaluator(func, remaining_phenotypes)
            for phe, fit_det in zip(remaining_phenotypes, results):
                phe_fit_map[phe] = fit_det

            # Update state
            st.num_phe_to_fit_evaluations += len(remaining_phenotypes)

        # Assign values from lookup or calculation
        for ind in population:
            ind.fitness, ind.details["evaluation"] = phe_fit_map[ind.phenotype]

        # Update cache
        if cache_on:
            for phe in unique_phenotypes:
                cache[phe] = phe_fit_map[phe]

        # Report
        if pr.verbose:
            self._report.calc_fit(
                len(population),
                n_unique,
                n_unique - n_after_cache,
                n_after_cache - n_after_db,
                n_after_db,
            )

    def _assign_new_ids(self, population):
        """Assign a unique id to each individual created in a search."""
        st = self.state

        for ind in population:
            ind.details["id"] = st.num_individuals
            st.num_individuals += 1

    # 3) Selection
    def _select_parents(self, current_population):
        """Select parent individuals from given population.

        There are different ways this can be done. Which one is used
        is decided by the default or user-provided parameters.

        The population of selected parents contains copies of the
        original individuals, so they have their own ids and can be
        tracked for result analysis.

        """
        pr = self.parameters

        # Select parent individuals
        operator = self._get_operator(
            "Parent selection", pr.parent_selection_operator, _operators.selection
        )
        selected_individuals = operator(
            individuals=current_population.individuals,
            sample_size=pr.offspring_size,
            objective=pr.objective,
            parameters=pr,
            state=self.state,
        )

        # Create parent population
        new_individuals = []
        for ind in selected_individuals:
            new_ind = ind.copy()
            new_ind.details["parent_ids"] = [ind.details["id"]]
            new_individuals.append(new_ind)
        parent_population = pr.system.representation.Population(new_individuals)
        self._assign_new_ids(parent_population)

        # Report
        if pr.verbose:
            self._report.select_par(
                self.state.generation,
                _utilities.times.current_time_readable(),
                len(current_population),
                len(parent_population),
            )
        return parent_population

    def _select_survivors(self, old_population, offspring_population):
        """Select survivor individuals from two given populations.

        There are different ways this can be done. Which one is used
        is decided by the default or user-provided parameters.

        The population of selected survivors contains copies of the
        original individuals, so they have their own ids and can be
        tracked for result analysis.

        """
        pr = self.parameters

        # Create a pool from which survivor individuals will be selected
        operator = self._get_operator(
            "Generation model",
            pr.generation_model,
            _operators.generation_model,
        )
        pool = operator(old_population, offspring_population, pr)

        # Select survivor individuals
        operator = self._get_operator(
            "Survivor selection", pr.survivor_selection_operator, _operators.selection
        )
        selected_individuals = operator(
            individuals=pool,
            sample_size=pr.population_size,
            objective=pr.objective,
            parameters=pr,
            state=self.state,
        )

        # Create survivor population
        new_individuals = []
        for ind in selected_individuals:
            new_ind = ind.copy()
            new_ind.details["parent_ids"] = [ind.details["id"]]
            new_individuals.append(new_ind)
        survivor_population = pr.system.representation.Population(new_individuals)
        self._assign_new_ids(survivor_population)

        # Report
        if pr.verbose:
            self._report.select_sur(
                len(old_population), len(offspring_population), len(survivor_population)
            )
        return survivor_population

    # 4) Variation
    def _check_parameter_consistency(self):
        """Check if the current parameters are consistent.

        This is necessary because the initial parametrization has to
        be checked, but also because a user can change it between
        calls to the step method. An example of an inconsistent
        parametrization is when all variation operators are turned off,
        because in that case no new individuals are generated and
        therefore the search can not progress.

        """
        pr = self.parameters
        if pr.crossover_operator is None and pr.mutation_operator is None:
            _exceptions.raise_missing_variation_error()

    def _cross_over(self, parent_population):
        """Perform crossover events between with given individuals.

        There are different ways this can be done. Which one is used
        is decided by the default or user-provided parameters.

        """
        pr = self.parameters
        st = self.state

        # Optional skip
        if pr.crossover_operator is None:
            return parent_population

        # Cross-over
        operator = self._get_operator(
            "Crossover", pr.crossover_operator, pr.system.crossover
        )
        crossed_over_individuals = []
        n = len(parent_population)
        while len(crossed_over_individuals) < n:
            if n > 2:
                parent0, parent1 = _random.sample(parent_population.individuals, 2)
            elif n == 2:
                parent0, parent1 = parent_population.individuals
            else:
                parent0 = parent1 = parent_population.individuals[0]

            child0_gen, child1_gen = operator(
                pr.grammar, parent0.genotype.copy(), parent1.genotype.copy(), pr
            )
            child0 = pr.system.representation.Individual(
                genotype=child0_gen,
                details=dict(
                    id=st.num_individuals,
                    parent_ids=[parent0.details["id"], parent1.details["id"]],
                ),
            )
            st.num_individuals += 1

            child1 = pr.system.representation.Individual(
                genotype=child1_gen,
                details=dict(
                    id=st.num_individuals,
                    parent_ids=[parent0.details["id"], parent1.details["id"]],
                ),
            )
            st.num_individuals += 1

            crossed_over_individuals.append(child0)
            crossed_over_individuals.append(child1)
        crossed_over_population = pr.system.representation.Population(
            crossed_over_individuals
        )

        # Report
        if pr.verbose:
            self._report.cross_over(
                len(parent_population), len(crossed_over_population)
            )
        return crossed_over_population

    def _mutate(self, crossed_over_population):
        """Perform mutation events on given individuals.

        There are different ways this can be done. Which one is used
        is decided by the default or user-provided parameters.

        """
        pr = self.parameters
        st = self.state

        # Optional skip
        if pr.mutation_operator is None:
            return crossed_over_population

        # Mutation
        operator = self._get_operator(
            "Mutation", pr.mutation_operator, pr.system.mutation
        )

        mutated_individuals = []
        for ind in crossed_over_population:
            new_gen = operator(pr.grammar, ind.genotype, pr)
            new_ind = pr.system.representation.Individual(
                genotype=new_gen,
                details=dict(
                    id=st.num_individuals,
                    parent_ids=[ind.details["id"]],
                ),
            )
            st.num_individuals += 1

            mutated_individuals.append(new_ind)
        mutated_population = pr.system.representation.Population(mutated_individuals)

        # Report
        if pr.verbose:
            self._report.mutate(len(crossed_over_population), len(mutated_population))
        return mutated_population

    # 5) End
    def _finish_generation(
        self,
        parent_population=None,
        crossed_over_population=None,
        mutated_population=None,
        survivor_population=None,
    ):
        """Process and optionally store the given individuals."""
        pr = self.parameters
        st = self.state

        # Optional: Storage to database
        if pr.database_on:
            db = self.database
            if parent_population:
                db._store_population(
                    "parent_selection", parent_population, st.generation
                )
            if crossed_over_population and pr.crossover_operator is not None:
                db._store_population(
                    "crossover", crossed_over_population, st.generation
                )
            if mutated_population and pr.mutation_operator is not None:
                db._store_population("mutation", mutated_population, st.generation)
            if survivor_population:
                db._store_population("main", survivor_population, st.generation)

        # Update state
        if (
            crossed_over_population is not None
            and pr.crossover_operator is not None
            and pr.mutation_operator is None
        ):
            # Crossover population has only been evaluated if it was not followed by mutation
            self._store_best_and_worst_ind(crossed_over_population)
        if mutated_population is not None and pr.mutation_operator is not None:
            self._store_best_and_worst_ind(mutated_population)
        self._store_best_and_worst_ind(survivor_population)
        st.population = survivor_population
        st.generation += 1
        st.num_generations += 1

        # Report
        if pr.verbose:
            self._report.gen_end(self.state)

    # State management
    def _store_start_time(self):
        """Add the start time of the current step to the list of all start times."""
        self.state.start_timestamps.append(_utilities.times.current_time_unix())

    def _store_stop_time(self):
        """Add the stop time of the current step to the list of all stop times."""
        self.state.stop_timestamps.append(_utilities.times.current_time_unix())

    def _store_best_and_worst_ind(self, pop):
        """Remember individuals with min/max/best fitness in an entire run.

        Note that depending on the search strategy, it is possible that
        these individuals get lost throughout a run, i.e. they are no
        longer part of the latest generation. It is important, however,
        that these few special individuals are remembered, so the best
        result can reliably be returned at the end of the run.

        """
        obj = self.parameters.objective
        st = self.state

        # Set min and max individual, not depending on obective
        if st.min_individual is None and pop:
            st.min_individual = pop[0]
        if st.max_individual is None and pop:
            st.max_individual = pop[0]
        for ind in pop:
            if ind.less_than(st.min_individual, obj):
                st.min_individual = ind
            if ind.greater_than(st.max_individual, obj):
                st.max_individual = ind

        # Set best individual, depending on obective
        if obj == "min":
            st.best_individual = st.min_individual
        else:
            st.best_individual = st.max_individual

    @staticmethod
    def _get_operator(description, name, location):
        """Fetch an operator function by its name."""
        try:
            return getattr(location, name)
        except Exception:
            _exceptions.raise_operator_lookup_error(description, name, location)
