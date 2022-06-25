import os as _os
import random as _random

import pylru as _pylru

from . import database as _database
from . import operators as _operators
from . import parameters as _parameters
from . import reporting as _reporting
from . import state as _state
from ..shared import evaluation as _evaluation
from ... import Grammar as _Grammar
from ... import _logging, _utilities
from ... import exceptions as _exceptions
from ... import systems as _systems
from ..._utilities import argument_processing as _ap
from ..._utilities.parametrization import ParameterCollection as _ParameterCollection


class EvolutionaryAlgorithm:
    """Evolutionary algorithm

    This is a population-based metaheuristic search algorithm.
    It starts with a random population, modifies its individuals with random variation,
    chooses fitter ones with non-random selection, and in doing so repeatedly moves
    towards better solutions without getting trapped in local optima.

    """

    __slots__ = ('database', 'parameters', 'state', '_report')
    default_parameters = _parameters.default_parameters

    # Initialization and reset
    def __init__(self, grammar, objective_function, objective, system='cfggpst', evaluator=None,
                 **kwargs):
        """Create an evolutionary algorithm that uses a grammar-based genetic programming system.

        Parameters
        ----------
        grammar : :ref:`Grammar <grammar>`
        system : str
            Grammar-based gentic programming system, which provides a
            genotype representation (e.g. list of int, bitarray),
            a genotype-phenotype mapping (forward, reverse)
            and random variation operators (crossover, mutation).

            Possible values:
            - ``'ge'``: :ref:`Grammatical Evolution <ge>`
            - ``'cfggp'``: :ref:`Context-Free Grammar Genetic Programming <cfggp>`
            - ``'cfggpst'``: :ref:`Context-Free Grammar Genetic Programming on Serialized Tree <cfggpst>`
        objective : str
            Objective of the evolutionary optimization, which is finding either
            a minimum or maximum value of the objective function.

            Possible values: ``'min'``, ``'max'``
        objective_function : Callable
            Function that gets a phenotype (=a string of the grammar's language) as input
            and returns an objective value for it that indicates the quality of the
            candidate solution.
        parameters : dict
            Parameters for the evolutionary algorithm itself (e.g. stop criterion)
            or the chosen grammar-based genetic programming system (e.g. genotype length).
        evaluator : Callable
            TODO

        """
        # Argument processing
        _ap.check_arg('grammar', grammar, types=(_Grammar,))
        _ap.callable_arg('objective_function', objective_function)
        objective = _ap.str_arg('objective', objective, vals=('min', 'max'), to_lower=True)
        system = _ap.str_arg(
            'system', system, vals=('cfggp', 'cfggpst', 'ge'),
            to_lower=True)

        system = getattr(_systems, system)
        if evaluator is None:
            evaluator = _evaluation.default_evaluator
        else:
            _ap.callable_arg('evaluator', evaluator)

        # Parameter processing
        self.parameters = self._process_parameters(
            grammar, objective_function, objective, system, evaluator, kwargs)

        # State and database (re)initialization
        self.reset()

        # Reporting
        self._report = _reporting.MinimalReporter()

    def _process_parameters(self, grammar, objective_function, objective, system, evaluator,
                            kwargs):
        # Dictionary of algorithm parameters and system parameters with all default values
        params = dict(
            grammar=grammar,
            objective_function=objective_function,
            objective=objective,
            system=system,
            evaluator=evaluator,
        )
        params.update(self.default_parameters)
        params.update(system.default_parameters)

        # Overwrite default values with user-provided kwargs, raise error if a name is unknown
        params = _ParameterCollection(params)
        for key, val in kwargs.items():
            if key in params:
                params[key] = val
            else:
                _exceptions.raise_unknown_parameter_error(key, params)

        # Convert values
        if isinstance(params['verbose'], bool):
            params['verbose'] = int(params['verbose'])

        # Check types
        _ap.int_arg('population_size', params['population_size'], min_incl=1)
        _ap.int_arg('offspring_size', params['offspring_size'], min_incl=1)
        _ap.int_arg('verbose', params['verbose'], min_incl=0)
        _ap.check_arg('database_on', params['database_on'], types=(bool,))
        _ap.check_arg('database_location', params['database_location'], types=(str,))
        return params

    def reset(self):
        """Reset the search state and thereby enable a new run.

        Notes
        -----
        - The parameters are not reset.

            - This means, if any non-default parameters were passed during
              object-creation or set later, these values are kept and not
              set back to default values. The purpose of this behavior
              is to enable repeated runs that are comparable.

        - The database is reset.

            - If the database location is in memory, it is lost during reset.

            - If the database location is a file, it is renamed during reset
              and a new file is created.

        - Cached values are deleted.

        """
        # State: storage of main data generated during a search in a single object
        self.state = _state.State()

        # Cache: optional storage of least recently used (lru) mapping results for lookup
        if self.parameters.gen_to_phe_cache_lookup_on:
            self.state._gen_to_phe_cache = _pylru.lrucache(self.parameters.gen_to_phe_cache_size)
        else:
            self.state._gen_to_phe_cache = None
        if self.parameters.phe_to_fit_cache_lookup_on:
            self.state._phe_to_fit_cache = _pylru.lrucache(self.parameters.phe_to_fit_cache_size)
        else:
            self.state._phe_to_fit_cache = None

        # Database: optional storage of all data generated during a search in an SQL database
        if self.parameters.database_on:
            self.database = _database.Database(
                self.parameters.database_location, self.parameters.system)
        else:
            self.database = None

    # Representations
    def __repr__(self):
        return '<EvolutionaryAlgorithm object at {}>'.format(hex(id(self)))

    def __str__(self):
        return str(self.state)

    def _repr_pretty_(self, p, cycle):
        """Provide a rich display representation for IPython interpreters."""
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Optimization algorithm
    def step(self):
        """Perform a single step of the evolutionary search without checking any stop criterion.

        Caution: This method does not check if any stop criterion was met.
        For example, when the maximum generation number was already reached by a
        run, calling this method would add another generation without considering
        whether this fits to the stop criteria defined by the current
        parametrization.

        Note that the parametrization of the evolutionary algorithm can be changed
        between steps, which allows to freely adjust the search. For example,
        the mutation and crossover rates can be gradually modified during a run,
        perhaps in response to search progress as defined by some measure.

        Returns
        -------
        best_individual : :ref:`Individual <individual>`   TODO: depends on system now
            The individual with the best fitness found so far at any point of the search.

        """
        # Start time
        self._store_start_time()

        if self.state.population is None:
            # CREATE first generation
            pop = self._initialize_population()

            # Evaluate
            self._evaluate_population(pop)

            # Store and report
            self._finish_generation(survivor_population=pop)
        else:
            # Check parameters
            self._check_parameter_consistency()

            # SELECT parents
            parent_population = self._select_parents(self.state.population)

            # CREATE offspring by random variation
            crossed_over_population = self._cross_over(parent_population)
            mutated_population = self._mutate(crossed_over_population)

            # Evaluate
            self._evaluate_population(mutated_population)

            # SELECT survivors
            survivor_population = self._select_survivors(
                self.state.population, mutated_population)

            # Store and report
            self._finish_generation(
                parent_population, crossed_over_population, mutated_population,
                survivor_population)

        # Stop time
        self._store_stop_time()
        return self.state.best_individual

    def run(self):
        """Perform the evolutionary search until a stop criterion is met.

        Caution: The current parametrization contains some stop criteria.
        If they are set in a way that can never be reached, the run will
        not halt until interrupted from the outside.

        Returns
        -------
        best_individual : :ref:`Individual <individual>`
            The individual with the best fitness found so far at any point of the search.

        """
        pr = self.parameters

        # Run step after step until a stop criterion is met
        gen_before = self.state.generation
        while not self.is_stop_criterion_met():
            self.step()
        gen_after = self.state.generation

        # Warn if not a single step was performed in this call
        if gen_after == gen_before:
            _logging.warn_no_step_in_ea_performed(self.state.generation)

        # Report
        if pr.verbose:
            self._report.run_end(self.state)
        return self.state.best_individual

    def is_stop_criterion_met(self):
        """Check if any stop criterion in the parameters is met by the current search state.

        Returns
        -------
        stopped : bool
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
                if pr.objective == 'min':
                    if st.best_individual.fitness <= pr.max_or_min_fitness:
                        return True
                elif pr.objective == 'max':
                    if st.best_individual.fitness >= pr.max_or_min_fitness:
                        return True

        # No stop criterion is met
        return False

    # 1) Beginning
    def _initialize_population(self):
        """Create the initial population with the operator chosen in parameters."""
        pr = self.parameters
        st = self.state

        # Report start of initialization
        if pr.verbose:
            self._report.init_start(
                st.generation, st.start_time, _utilities.times.current_time_readable())

        # Generate initial population
        operator = _get_operator(
            'Population initialization', pr.init_pop_operator,
            pr.system.initialization.population)
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
        """Calculate genotype -> phenotype -> fitness mapping for each individual."""
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
        """Perform the genotype-phenotype mapping with default or user-specified function.

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
                len(population), n_unique, n_unique-n_after_cache, n_after_cache)

    def _calculate_fitnesses(self, population):
        """Perform the phenotype-fitness mapping with default or user-specified function.

        Possible crashes are prevented by wrapping the function such that
        default values are returned in case of an error or invalid return value.

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
        if db_on and remaining_phenotypes:
            for phe, fit_det in self.database._lookup_phenotype_evaluations(remaining_phenotypes):
                phe_fit_map[phe] = fit_det
                remaining_phenotypes.remove(phe)
        n_after_db = len(remaining_phenotypes)

        # 4) For each remaining phenotype, calculate the objective function
        if remaining_phenotypes:
            # Evaluate
            func = _evaluation.get_robust_fitness_function(pr.objective_function, pr.objective)
            results = pr.evaluator(func, remaining_phenotypes)
            for phe, fit_det in zip(remaining_phenotypes, results):
                phe_fit_map[phe] = fit_det

            # Update state
            st.num_phe_to_fit_evaluations += len(remaining_phenotypes)

        # Assign values from lookup or calculation
        for ind in population:
            ind.fitness, ind.details['evaluation'] = phe_fit_map[ind.phenotype]

        # Update cache
        if cache_on:
            for phe in unique_phenotypes:
                cache[phe] = phe_fit_map[phe]

        # Report
        if pr.verbose:
            self._report.calc_fit(
                len(population), n_unique, n_unique-n_after_cache, n_after_cache-n_after_db,
                n_after_db)

    def _assign_new_ids(self, population):
        """Assign unique ids to each individual of a population."""
        st = self.state

        for ind in population:
            ind.details['id'] = st.num_individuals
            st.num_individuals += 1

    # 3) Selection
    def _select_parents(self, current_population):
        """Select parents from given individuals with the operator chosen in parameters.

        The population of selected parents is a copy of the original individuals,
        so they have their own ids and can be tracked for result analysis.

        """
        pr = self.parameters

        # Select parent individuals
        operator = _get_operator(
            'Parent selection', pr.parent_selection_operator, _operators.selection)
        selected_individuals = operator(
            individuals=current_population.individuals,
            sample_size=pr.offspring_size,
            objective=pr.objective,
            parameters=pr,
            state=self.state)

        # Create parent population
        new_individuals = []
        for ind in selected_individuals:
            new_ind = ind.copy()
            new_ind.details['parent_ids'] = [ind.details['id']]
            new_individuals.append(new_ind)
        parent_population = pr.system.representation.Population(new_individuals)
        self._assign_new_ids(parent_population)

        # Report
        if pr.verbose:
            self._report.select_par(
                self.state.generation, _utilities.times.current_time_readable(),
                len(current_population), len(parent_population))
        return parent_population

    def _select_survivors(self, old_population, offspring_population):
        """Select survivors from given individuals with the operator chosen in parameters."""
        pr = self.parameters

        # Choose pool from which individuals will be selected
        operator = _get_operator(
            'Survivor selection pooling', pr.survivor_selection_pooling,
            _operators.pooling)
        pool = operator(old_population, offspring_population, pr)

        # Select survivor individuals
        operator = _get_operator(
            'Survivor selection', pr.survivor_selection_operator,
            _operators.selection)
        selected_individuals = operator(
            individuals=pool,
            sample_size=pr.population_size,
            objective=pr.objective,
            parameters=pr,
            state=self.state)

        # Create survivor population
        new_individuals = []
        for ind in selected_individuals:
            new_ind = ind.copy()
            new_ind.details['parent_ids'] = [ind.details['id']]
            new_individuals.append(new_ind)
        survivor_population = pr.system.representation.Population(new_individuals)
        self._assign_new_ids(survivor_population)

        # Optional: Elitism = Guarantee the best known individual survives, ensure it is selected
        if pr.elitism_on:
            survivor_population = _operators.proliferation.elitism(
                old_population, offspring_population, survivor_population, pr, self.state)

        # Report
        if pr.verbose:
            self._report.select_sur(
                len(old_population), len(offspring_population), len(survivor_population))
        return survivor_population

    # 4) Variation
    def _check_parameter_consistency(self):
        """Check if the current user-defined parametrization is consistent for the next step."""
        pr = self.parameters
        if pr.crossover_operator is None and pr.mutation_operator is None:
            _exceptions.raise_missing_variation_error()

    def _cross_over(self, parent_population):
        """Cross-over given individuals with the operator chosen in parameters."""
        pr = self.parameters
        st = self.state

        # Optional skip
        if pr.crossover_operator is None:
            return parent_population

        # Cross-over
        operator = _get_operator('Crossover', pr.crossover_operator, pr.system.crossover)
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
                pr.grammar, parent0.genotype.copy(), parent1.genotype.copy(), pr)
            child0 = pr.system.representation.Individual(
                genotype=child0_gen,
                details=dict(
                    id=st.num_individuals,
                    parent_ids=[parent0.details['id'], parent1.details['id']],
                ),
            )
            st.num_individuals += 1

            child1 = pr.system.representation.Individual(
                genotype=child1_gen,
                details=dict(
                    id=st.num_individuals,
                    parent_ids=[parent0.details['id'], parent1.details['id']],
                ),
            )
            st.num_individuals += 1

            crossed_over_individuals.append(child0)
            crossed_over_individuals.append(child1)
        crossed_over_population = pr.system.representation.Population(crossed_over_individuals)

        # Report
        if pr.verbose:
            self._report.cross_over(len(parent_population), len(crossed_over_population))
        return crossed_over_population

    def _mutate(self, crossed_over_population):
        """Mutate given individuals with the operator chosen in parameters."""
        pr = self.parameters
        st = self.state

        # Optional skip
        if pr.mutation_operator is None:
            return crossed_over_population

        # Mutation
        operator = _get_operator('Mutation', pr.mutation_operator, pr.system.mutation)

        mutated_individuals = []
        for ind in crossed_over_population:
            new_gen = operator(pr.grammar, ind.genotype, pr)
            new_ind = pr.system.representation.Individual(
                genotype=new_gen,
                details=dict(
                    id=st.num_individuals,
                    parent_ids=[ind.details['id']],
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
    def _finish_generation(self, parent_population=None, crossed_over_population=None,
                           mutated_population=None, survivor_population=None):
        """Process and store the given individuals."""
        pr = self.parameters
        st = self.state

        # Optional: Storage to database
        if pr.database_on:
            db = self.database
            if parent_population:
                db._store_population('parent_selection', parent_population, st.generation)
            if crossed_over_population and pr.crossover_operator is not None:
                db._store_population('crossover', crossed_over_population, st.generation)
            if mutated_population and pr.mutation_operator:
                db._store_population('mutation', mutated_population, st.generation)
            if survivor_population:
                db._store_population('main', survivor_population, st.generation)

        # Update state
        if (crossed_over_population is not None
            and pr.crossover_operator is not None
            and pr.mutation_operator is None):
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
        """Remember individuals with min/max/best fitness in an entire run, no matter if lost."""
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
        if obj == 'min':
            st.best_individual = st.min_individual
        else:
            st.best_individual = st.max_individual


def _get_operator(description, name, location):
    try:
        return getattr(location, name)
    except Exception:
        _exceptions.raise_operator_lookup_error(description, name, location)
