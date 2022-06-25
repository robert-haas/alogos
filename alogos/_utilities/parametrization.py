from .. import exceptions as _exceptions


class ParameterCollection:
    """A collection of parameters with initial values that can be changed by a user.

    Provided features:

    - The set of available parameter names is defined once and for all by a
      dict passed to ``__init__``.

    - The user can get and set parameter values
      by dot notation (via ``__setattr__``, e.g. ``param.a = 4``) and
      bracket notation (via ``__getitem__`` and ``__setitem__``, e.g. ``param['a'] = 4``).

    - The user can check if a parameter name is included in the collection
      (via __contains__, e.g. 'a' in param).

    - If a user tries to get or set an unknown parameter (detected via ``__getattr__``),
      an error is raised, which lists all available parameters and their current
      and initial values.
      Note that there is a deliberate asymmetry between ``__setattr__`` and
      ``__getattr__`` in Python 3. The former is called on any attribute assignment,
      but the latter is only called when attribute lookup fails by all other methods.

    References
    ----------
    - https://docs.python.org/3/reference/datamodel.html
    - https://docs.python.org/3/library/copy.html
    - https://ipython.readthedocs.io/en/stable/config/integrating.html

    """
    def __init__(self, parameter_dict):
        for name in ['keys', 'values', 'items']:
            if name in parameter_dict:
                _exceptions.raise_initial_parameter_error(name)

        self._initial_parameters = parameter_dict
        self.__dict__.update(parameter_dict)

    def __str__(self):
        def append_parameters_to_output(lines, pc, indent=0):
            for key in pc:
                original_value = pc._initial_parameters[key]
                current_value = pc[key]
                if isinstance(current_value, ParameterCollection):
                    line = '{}- {}:'.format(' ' * indent, key)
                    lines.append(line)
                    add_parameters_to_output(lines, current_value, indent+2)
                elif current_value == original_value:
                    line = '{}- {}: {}'.format(' ' * indent, key, repr(original_value))
                    lines.append(line)
                else:
                    line = '{}- {}: {} currently, {} originally'.format(
                        ' ' * indent, key, repr(current_value), repr(original_value))
                    lines.append(line)
            return lines

        lines = []
        lines.append('Parameters:')
        lines = append_parameters_to_output(lines, self)
        return '\n'.join(lines)

    def __repr__(self):
        return '<ParameterCollection object at {}>'.format(hex(id(self)))
    
    def _repr_pretty_(self, p, cycle):
        """Provide a rich display representation for IPython interpreters."""
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    def __contains__(self, item):
        """Called to implement membership test operators, according to docs."""
        return item in self._initial_parameters.keys()

    def __getattr__(self, key):
        """Called when the default attribute access fails, according to docs.

        Notes
        -----
        Properties starting with '__' are silently ignored, so that inspection
        methods do not raise unnecessary errors, such as those used by Python's
        builtin help system.

        """
        if not key.startswith('__'):
            _exceptions.raise_unknown_parameter_error(key, self)

    def __setattr__(self, key, value):
        """Called when an attribute assignment is attempted, according to docs."""
        if key != '_initial_parameters' and key not in self._initial_parameters:
            _exceptions.raise_unknown_parameter_error(key, self)
        self.__dict__[key] = value

    def __getitem__(self, key):
        """Called to implement membership test operators, according to docs."""
        if key not in self:
            _exceptions.raise_unknown_parameter_error(key, self)
        return self.__dict__[key]

    def __setitem__(self, key, value):
        """Called to implement assignment to self[key], according to docs."""
        self.__setattr__(key, value)

    def __iter__(self):
        """Called when an iterator is required for a container, according to docs."""
        return iter(name for name in self._initial_parameters)
    
    def __len__(self):
        """Called to implement the built-in function len(), according to docs."""
        return len(self._initial_parameters)

    def __copy__(self):
        """Create a shallow copy this object, which can be called with copy()."""
        current_dict = {key: val for key, val in self.items()}
        return current_dict

    def keys(self):
        return self._initial_parameters.keys()
    
    def values(self):
        return (self.__dict__[key] for key in self._initial_parameters)
    
    def items(self):
        return ((key, self.__dict__[key]) for key in self._initial_parameters)


def get_given_or_default(name, given_parameters, default_parameters):
    """Get a parameter from a collection of given parameters or otherwise from defaults."""
    try:
        return given_parameters[name]
    except Exception:
        return default_parameters[name]
