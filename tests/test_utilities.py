import pytest

import alogos as al


def test_utilities_parameters():
    # Initialization
    params = al._utilities.parametrization.ParameterCollection(
        dict(something=42, other="hello")
    )

    # Comparison
    assert params.something == 42
    assert params["something"] == 42
    assert params.other == "hello"
    assert params["other"] == "hello"

    # Assignment
    params.other = params.something
    params["something"] = "there"
    assert params.something == "there"
    assert params.other == 42

    # Reserved parameter names
    with pytest.raises(al.exceptions.ParameterError):
        al._utilities.parametrization.ParameterCollection(dict(keys=42))
    with pytest.raises(al.exceptions.ParameterError):
        al._utilities.parametrization.ParameterCollection(dict(values=42))
    with pytest.raises(al.exceptions.ParameterError):
        al._utilities.parametrization.ParameterCollection(dict(items=42))
