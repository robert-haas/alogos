from ..._utilities import times as _times
from ..._utilities.operating_system import NEWLINE as _NEWLINE


class State:
    """Provide the current state of the evolutionary search process."""
    __slots__ = (
        'start_timestamps',
        'stop_timestamps',
        'population',
        'generation',
        'num_generations',
        'num_individuals',
        'num_phe_to_fit_evaluations',
        'num_gen_to_phe_evaluations',
        'best_individual',
        'min_individual',
        'max_individual',
        '_gen_to_phe_cache',
        '_phe_to_fit_cache',
    )

    def __init__(self):
        self.start_timestamps = []
        self.stop_timestamps = []
        self.population = None
        self.generation = 0
        self.num_individuals = 0
        self.num_generations = 0
        self.num_gen_to_phe_evaluations = 0
        self.num_phe_to_fit_evaluations = 0
        self.best_individual = None
        self.min_individual = None
        self.max_individual = None
        self._gen_to_phe_cache = None
        self._phe_to_fit_cache = None

    # Representations
    def __repr__(self):
        return '<EvolutionaryAlgorithmState object at {}>'.format(hex(id(self)))

    def __str__(self):
        min_fitness = self.min_individual.fitness if self.min_individual else 'No evaluation yet'
        max_fitness = self.max_individual.fitness if self.max_individual else 'No evaluation yet'
        msg = []
        msg.append('╭─ State of the evolutionary search{}'.format(_NEWLINE))
        msg.append('│ Start time ................................ {}{}'.format(
            self.start_time, _NEWLINE))
        msg.append('│ Stop time ................................. {}{}'.format(
            self.stop_time, _NEWLINE))

        msg.append('│{}'.format(_NEWLINE))
        msg.append('│ Number of generations ..................... {}{}'.format(
            self.num_generations, _NEWLINE))
        msg.append('│ Number of individuals ..................... {}{}'.format(
            self.num_individuals, _NEWLINE))
        msg.append('│ Number of genotype-phenotype evaluations .. {}{}'.format(
            self.num_gen_to_phe_evaluations, _NEWLINE))
        msg.append('│ Number of phenotype-fitness evaluations ... {}{}'.format(
            self.num_phe_to_fit_evaluations, _NEWLINE))

        msg.append('│{}'.format(_NEWLINE))
        msg.append('│ Minimal fitness ........................... {}{}'.format(
            min_fitness, _NEWLINE))
        msg.append('│ Maximal fitness ........................... {}{}'.format(
            max_fitness, _NEWLINE))
        msg.append('╰─')
        text = ''.join(msg)
        return text

    def _repr_pretty_(self, p, cycle):
        """Provide a rich display representation for IPython interpreters."""
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    # Time
    @property
    def start_time(self):
        if self.start_timestamps:
            start_ts = self.start_timestamps[0]
            return _times.unix_timestamp_to_readable(start_ts)
        return 'There is no start time because no run was performed yet.'

    @property
    def stop_time(self):
        if self.stop_timestamps:
            stop_ts = self.stop_timestamps[-1]
            return _times.unix_timestamp_to_readable(stop_ts)
        return 'There is no stop time because no run was performed yet.'

    @property
    def runtime(self):
        rt = 0.0
        for start, stop in zip(self.start_timestamps, self.stop_timestamps):
            rt += stop - start
        return rt
