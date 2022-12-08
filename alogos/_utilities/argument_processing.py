from collections.abc import Callable as _Callable
from numbers import Number as _Number

from .operating_system import NEWLINE as _NEWLINE


def check_arg(arg_name, arg_val, types=None, vals=None, conv=None):
    """Check an argument by inspecting its type and value."""
    # Check if the argument type is valid
    if types:
        if not any(isinstance(arg_val, typ) for typ in types):
            arg_val_type = type(arg_val).__name__
            type_names = ", ".join(str(typ) for typ in types)
            message = (
                'Argument "{name}" got a value with an invalid type.{nl}'
                "Given value: {val}{nl}"
                "Given type: {typ}{nl}"
                "Possible types: {options}".format(
                    name=arg_name,
                    val=repr(arg_val),
                    typ=arg_val_type,
                    options=type_names,
                    nl=_NEWLINE,
                )
            )
            raise TypeError(message) from None

    # Check if the argument value is valid
    if vals:
        if arg_val not in vals:
            val_names = ", ".join(repr(val) for val in vals)
            message = (
                'Argument "{name}" got an invalid value.{nl}'
                "Given value: {val}{nl}"
                "Possible values: {options}".format(
                    name=arg_name, val=repr(arg_val), options=val_names, nl=_NEWLINE
                )
            )
            raise ValueError(message) from None

    # Convert the argument value
    if conv:
        if arg_val in conv:
            arg_val = conv[arg_val]
    return arg_val


def str_arg(arg_name, arg_val, default=None, vals=None, to_lower=False):
    """Check a string argument."""
    if to_lower:
        try:
            arg_val = arg_val.lower()
        except AttributeError:
            pass
    types = (str,) if default is None else (str, type(None))
    return check_arg(arg_name, arg_val, types=types, vals=vals, conv={None: default})


def int_arg(
    arg_name,
    arg_val,
    default=None,
    vals=None,
    min_incl=None,
    max_incl=None,
    allow_none=False,
):
    """Check an integer argument."""
    if default is not None or allow_none:
        types = (int, type(None))
    else:
        types = (int,)
    value = check_arg(arg_name, arg_val, types=types, vals=vals, conv={None: default})
    if value is not None:
        if min_incl is not None and value < min_incl:
            message = (
                'Argument "{name}" got an invalid value.{nl}'
                "Given value: {val}{nl}"
                "Lowest possible value: {minval}".format(
                    name=arg_name, val=arg_val, minval=min_incl, nl=_NEWLINE
                )
            )
            raise ValueError(message) from None
        if max_incl is not None and value > max_incl:
            message = (
                'Argument "{name}" got an invalid value.{nl}'
                "Given value: {val}{nl}"
                "Highest possible value: {maxval}".format(
                    name=arg_name, val=arg_val, maxval=max_incl, nl=_NEWLINE
                )
            )
            raise ValueError(message) from None
    return value


def num_arg(arg_name, arg_val, default=None, vals=None, min_incl=None, max_incl=None):
    """Check a numerical argument."""
    types = (_Number,) if default is None else (_Number, type(None))
    value = check_arg(arg_name, arg_val, types=types, vals=vals, conv={None: default})
    if value is not None:
        if min_incl is not None and value < min_incl:
            message = (
                'Argument "{name}" got an invalid value.{nl}'
                "Given value: {val}{nl}"
                "Lowest possible value: {minval}".format(
                    name=arg_name, val=arg_val, minval=min_incl, nl=_NEWLINE
                )
            )
            raise ValueError(message) from None
        if max_incl is not None and value > max_incl:
            message = (
                'Argument "{name}" got an invalid value.{nl}'
                "Given value: {val}{nl}"
                "Highest possible value: {maxval}".format(
                    name=arg_name, val=arg_val, maxval=max_incl, nl=_NEWLINE
                )
            )
            raise ValueError(message) from None
    return value


def bool_arg(arg_name, arg_val, default=None, vals=None):
    """Check a boolean argument."""
    types = (bool,) if default is None else (bool, type(None))
    return check_arg(arg_name, arg_val, types=types, vals=vals, conv={None: default})


def callable_arg(arg_name, arg_val, default=None, vals=None):
    """Check a callable argument."""
    types = (_Callable,) if default is None else (_Callable, type(None))
    return check_arg(arg_name, arg_val, types=types, vals=vals, conv={None: default})


def logical_xor(var1, var2):
    """Compute logical XOR between two inputs."""
    return bool(var1) ^ bool(var2)


def ensure_file_extension(filepath, ending):
    """Ensure a filepath ends with a certain extension."""
    if not ending.startswith("."):
        ending = "." + str(ending)
    if not filepath.endswith(ending):
        filepath += ending
    return filepath


def ensure_no_file_extension(filepath, ending):
    """Ensure a filepath ends without a certain extension."""
    if not ending.startswith("."):
        ending = "." + str(ending)
    if filepath.endswith(ending):
        filepath = filepath[: -len(ending)]
    return filepath
