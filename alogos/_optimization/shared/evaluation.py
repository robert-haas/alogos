import functools as _functools
from numbers import Number as _Number


def default_evaluator(function, args):
    """Evaluate a given function on each item of a list."""
    return [function(x) for x in args]


@_functools.lru_cache(2)
def get_robust_fitness_function(objective_function, objective):
    """Wrap a given function to make it robust."""
    default_fitness = float("+Inf") if objective == "min" else float("-Inf")
    default_details = None
    return RobustCallable(objective_function, default_fitness, default_details)


class RobustCallable:
    """Wrapper to make a function robust against failure and invalid return values.

    It provides following guarantees:

    - The return value is a tuple with two entries:
      ``(fitness, details)``
    - ``fitness`` is ensured to be of type float,
      allowing +Inf and -Inf, but excluding NaN.
    - ``details`` can be any user-chosen type.

    Notes
    -----
    - Simply wrapping the function into another function would cause
      pickling errors.
    - If the robust fitness function cannot be pickled, most parallel
      and distributed evaluation solutions cannot be applied since they
      rely on some pickling function.

    References
    ----------
    - http://gael-varoquaux.info/programming/decoration-in-python-done-right-decorating-and-pickling.html

    """

    def __init__(self, func, def_fit, def_det):
        """Create a robust callable object from a given function."""
        self.func = func
        self.def_fit = def_fit
        self.def_det = def_det

    def __call__(self, phe):
        """Make the object callable."""
        # 1) Invalid phenotype
        if phe is None:
            fit = self.def_fit
            det = self.def_det

        # 2) Valid phenotype
        else:
            try:
                # Objective function evaluation
                result = self.func(phe)

                # Result may be a fitness value or a tuple (fitness, details)
                try:
                    fit, det = result
                except Exception:
                    fit = result
                    det = self.def_det

                # Ensure fitness is a comparable number (includes +Inf and -Inf, excludes NaN)
                if not isinstance(fit, _Number):
                    message = "Returned fitness value is not a number: {}".format(fit)
                    raise ValueError(message)

                if fit != fit:
                    message = (
                        "Returned fitness value is NaN. It is replaced by the "
                        "default fitness value {}".format(self.def_fit)
                    )
                    raise ValueError(message)

                # Ensure fitness is a float (e.g. not Numpy type), can be stored in a database
                fit = float(fit)
            except Exception as excp:
                fit = self.def_fit
                exception_text = "{}: {}".format(type(excp).__name__, str(excp))
                det = exception_text
        return fit, det
