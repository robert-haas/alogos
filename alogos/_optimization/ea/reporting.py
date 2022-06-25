class MinimalReporter:
    def init_start(self, gen, time_start, time_now):
        message = '{:<16} {:<16} {:<16} {:<16} {:<16}'.format(
            'Progress', 'Generations', 'Evaluations', 'Runtime (sec)', 'Best fitness')
        self.write(message)

    def init_end(self, n):
        pass

    def map_gen_phe(self):
        pass

    def map_phe_fit(self):
        pass

    def calc_phe(self, n, n_unique, n_cached, n_calc):
        pass

    def calc_fit(self, n, n_unique, n_cached, n_db, n_calc):
        pass

    def select_par(self, gen, time, n, n_par):
        pass

    def select_sur(self, n_old, n_off, n_sel):
        pass

    def cross_over(self, n_par, n_crs):
        pass

    def mutate(self, n_crs, n_mut):
        pass

    def best_ind(self, ind):
        pass

    def gen_end(self, state):
        gen = state.num_generations
        if gen % 10 == 0:
            time = state.runtime
            evals = state.num_phe_to_fit_evaluations
            fit = state.best_individual.fitness
            message = '.      {:<16} {:<16} {:<16.1f} {}'.format(gen, evals, time, fit)
            self.write(message)
        elif gen % 5 == 0:
            self.write('.', end=' ')
        else:
            self.write('.', end='')

    def run_end(self, state):
        gen = state.num_generations
        time = state.runtime
        evals = state.num_phe_to_fit_evaluations
        fit = state.best_individual.fitness
        message = '{:<16} {:<16} {:<16} {:<16.1f} {:<16}'.format(
            'Finished', gen, evals, time, fit)
        self.write('')
        self.write('')
        self.write(message)

    def write(self, message, end=None):
        if end is None:
            print(message)
        else:
            print(message, end=end)


class VerboseReporter:
    def init_start(self, gen, time_start, time_now):
        self.write('╭─ Run started ── {}'.format(time_start))
        self.write('', level=1)
        self.write('├ Generation {} ── {}'.format(gen, time_now))
        self.write('', level=1)
        self.write('Population initialization', level=1)

    def init_end(self, n):
        self.write('Created {} individuals.'.format(n), level=2)

    def map_gen_phe(self):
        self.write('Genotype-to-phenotype evaluation', level=1)

    def map_phe_fit(self):
        self.write('Phenotype-to-fitness evaluation', level=1)

    def calc_phe(self, n, n_unique, n_cached, n_calc):
        message = (
            '{} individuals with {} unique genotypes: '
            '{} found in cache, {} calculated.'.format(n, n_unique, n_cached, n_calc))
        self.write(message, level=2)

    def calc_fit(self, n, n_unique, n_cached, n_db, n_calc):
        message = (
            '{} individuals with {} unique phenotypes: {} found in cache, '
            '{} found in db, {} calculated.'.format(n, n_unique, n_cached, n_db, n_calc))
        self.write(message, level=2)

    def select_par(self, gen, time, n, n_par):
        # Generation
        self.write('', level=1)
        self.write('├ Generation {} ── {}'.format(gen, time))
        # Parent selection
        self.write('', level=1)
        self.write('Parent selection', level=1)
        self.write('Selected {} individuals from a population of {}.'.format(n_par, n),
            level=2)

    def select_sur(self, n_old, n_off, n_sel):
        self.write('Survivor selection', level=1)
        self.write(
            'Selected {} individuals from the old population of {} and the offspring '
            'population of {}.'.format(n_sel, n_old, n_off), level=2)

    def cross_over(self, n_par, n_crs):
        self.write('Crossover', level=1)
        self.write('Created {} crossed-over individuals from {} parents.'.format(n_crs, n_par),
            level=2)

    def mutate(self, n_crs, n_mut):
        self.write('Mutation', level=1)
        self.write('Created {} mutated individuals from {} individuals.'.format(n_mut, n_crs),
        level=2)

    def gen_end(self, state):
        ind = state.best_individual
        self.write('Best individual:', level=1)
        self.write('  Genotype: {}'.format(ind.genotype), level=1)
        self.write('  Phenotype: {}'.format(ind.phenotype), level=1)
        self.write('  Fitness: {}'.format(ind.fitness), level=1)

    def run_end(self, time):
        self.write('', level=1)
        self.write('╰─ Run finished ── {}'.format(time))

    def write(self, message, level=None):
        """Report a message with proper indent."""
        if level is None:
            print(message)
        elif level <= 1:
            prefix = '│ {}'.format('  '*(level-1))
            print('{}{}'.format(prefix, message))
