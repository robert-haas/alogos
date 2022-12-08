from .. import exceptions as _exceptions


class ParameterCollection:
    """A collection of parameters with initial values that can be changed by a user.

    Provided features:

    - The set of available parameter names is defined once and for all
      by a dict passed to ``__init__``.

    - The user can get and set parameter values
      by dot notation (via ``__setattr__``, e.g. ``param.a = 4``) and
      bracket notation (via ``__getitem__`` and ``__setitem__``,
      e.g. ``param['a'] = 4``).

    - The user can check if a parameter name is included in the
      collection (via __contains__, e.g. 'a' in param).

    - If a user tries to get or set an unknown parameter (detected via
      ``__getattr__``), an error is raised, which lists all available
      parameters and their current and initial values.
      Note that there is a deliberate asymmetry between ``__setattr__``
      and ``__getattr__`` in Python 3. The former is called on any
      attribute assignment, but the latter is only called when
      attribute lookup fails by all other methods.

    References
    ----------
    - https://docs.python.org/3/reference/datamodel.html
    - https://docs.python.org/3/library/copy.html
    - https://ipython.readthedocs.io/en/stable/config/integrating.html

    """

    def __init__(self, parameter_dict):
        """Create a parameter collection from a dictionary."""
        for name in ["keys", "values", "items"]:
            if name in parameter_dict:
                _exceptions.raise_initial_parameter_error(name)

        self._initial_parameters = parameter_dict
        self.__dict__.update(parameter_dict)

    def __str__(self):
        """Compute the "informal" string representation of the parameter collection."""

        def append_parameters_to_output(lines, pc, indent=0):
            for key in pc:
                original_value = pc._initial_parameters[key]
                current_value = pc[key]
                if isinstance(current_value, ParameterCollection):
                    line = "{}- {}:".format(" " * indent, key)
                    lines.append(line)
                    append_parameters_to_output(lines, current_value, indent + 2)
                elif current_value == original_value:
                    line = "{}- {}: {}".format(" " * indent, key, repr(original_value))
                    lines.append(line)
                else:
                    line = "{}- {}: {} currently, {} originally".format(
                        " " * indent, key, repr(current_value), repr(original_value)
                    )
                    lines.append(line)
            return lines

        lines = []
        lines.append("Parameters:")
        lines = append_parameters_to_output(lines, self)
        return "\n".join(lines)

    def __repr__(self):
        """Compute the "official" string representation of the parameter collection."""
        return "<ParameterCollection object at {}>".format(hex(id(self)))

    def _repr_pretty_(self, p, cycle):
        """Provide rich display representation for IPython and Jupyter."""
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    def __contains__(self, item):
        """Check if a given string is a known parameter name."""
        return item in self._initial_parameters.keys()

    def __getattr__(self, key):
        """Provide a fallback when default attribute access fails.

        Notes
        -----
        Properties starting with '__' are silently ignored, so that
        inspection methods do not raise unnecessary errors, such as
        those used by Python's built-in help system.

        """
        if not key.startswith("__"):
            _exceptions.raise_unknown_parameter_error(key, self)

    def __setattr__(self, key, value):
        """Enable attribute assignment."""
        if key != "_initial_parameters" and key not in self._initial_parameters:
            _exceptions.raise_unknown_parameter_error(key, self)
        self.__dict__[key] = value

    def __getitem__(self, key):
        """Enable item retrieval."""
        if key not in self:
            _exceptions.raise_unknown_parameter_error(key, self)
        return self.__dict__[key]

    def __setitem__(self, key, value):
        """Enable item assignment via object[key]."""
        self.__setattr__(key, value)

    def __iter__(self):
        """Return an iterator that goes over all parameter names."""
        return iter(name for name in self._initial_parameters)

    def __len__(self):
        """Compute the number of parameters."""
        return len(self._initial_parameters)

    def __copy__(self):
        """Create a shallow copy this object."""
        current_dict = {key: val for key, val in self.items()}
        return current_dict

    def keys(self):
        """Get parameter names."""
        return self._initial_parameters.keys()

    def values(self):
        """Get parameter values."""
        return (self.__dict__[key] for key in self._initial_parameters)

    def items(self):
        """Get parameter names and values."""
        return ((key, self.__dict__[key]) for key in self._initial_parameters)


def get_given_or_default(name, given_parameters, default_parameters):
    """Get a parameter from a collection of given parameters or otherwise from defaults."""
    try:
        return given_parameters[name]
    except Exception:
        return default_parameters[name]
