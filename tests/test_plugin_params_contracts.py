from __future__ import annotations

import pytest

from plugins.hmm_fx_daily.config import default_params as hmm_default_params
from plugins.hmm_fx_daily.config import validate_params as validate_hmm_params
from plugins.iris_classification.config import default_params as iris_default_params
from plugins.iris_classification.config import validate_params as validate_iris_params


def test_iris_validate_params_strict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        validate_iris_params({"model": {"C": 2.0}, "unknown": {"x": 1}}, strict=True)


def test_iris_validate_params_non_strict_preserves_unknown_keys() -> None:
    validated = validate_iris_params({"model": {"C": 2.0}, "unknown": {"x": 1}}, strict=False)
    assert validated["model"]["C"] == 2.0
    assert validated["unknown"]["x"] == 1


def test_hmm_default_params_contains_expected_sections() -> None:
    defaults = hmm_default_params()
    assert set(defaults.keys()) == {"model", "train", "preprocess", "eval"}


def test_iris_default_params_contains_expected_sections() -> None:
    defaults = iris_default_params()
    assert set(defaults.keys()) == {"split", "model"}


def test_hmm_validate_params_non_strict_preserves_unknown_keys() -> None:
    validated = validate_hmm_params({"model": {"n_components": 5}, "x_extra": 1}, strict=False)
    assert validated["model"]["n_components"] == 5
    assert validated["x_extra"] == 1
