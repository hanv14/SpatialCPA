"""T01 acceptance tests for the config: round-tripping and the selection gates."""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from spatialcpav25_gen.config import (
    DECODERS,
    EXPR_MODES,
    LAYOUT_MODES,
    PRIOR_MODES,
    TEXT_EMB_MODES,
    Config,
    ConfigError,
)

# The previous version's behaviour, which T09's selector must be able to reach; if this
# tuple ever stops validating, the no-regression guarantee is gone.
V20_CONFIG = {"layout_mode": "resample", "expr_mode": "cross-mix", "prior_mode": "iid"}


def test_shipped_defaults_validate_field_by_field():
    """Every field's declared default must be a value ``validate()`` accepts.

    ``replace()`` validates the **whole** config, so one field whose default its own range check
    rejects kills every run in the project before it does anything. That is what
    ``decoder_theta_floor = 0.0`` did on the day it was added: a **sentinel** default — "0 means
    no floor" — listed in ``_check_positive``, whose dict rejects zero. Three real-data runs died
    in ``replace()`` before loading a file.

    Nothing asserted that the *defaults* were valid. The suite would have caught it, but only as
    a confusing ``pytest.raises(match="ell_xy")`` mismatch two tests away from the cause, so this
    test exists to fail **by name** instead.

    A sentinel default is the shape that invites this — the value is not a value — so a new field
    whose default means "off" belongs in neither the positive nor the fraction registry, and its
    range check is written out in ``_check_relations``.
    """
    base = Config()
    # This line is where the sentinel bug surfaces, and it raises ConfigError directly rather
    # than reaching the loop's message. That is fine: the raised message names the field, the
    # value and the rule, inside a test whose name says what was being asserted.
    base.validate()
    assert base.replace() == base

    for field in dataclasses.fields(Config):
        value = getattr(base, field.name)
        try:
            base.replace(**{field.name: value})
        except ConfigError as exc:
            raise AssertionError(
                f"Config.{field.name}'s own default {value!r} does not validate: {exc}"
            ) from exc


def test_decoder_theta_floor_sentinel_and_range():
    """``0.0`` disables the floor; negatives and a floor at ``zinb_theta_max`` are refused."""
    assert Config().decoder_theta_floor == 0.0
    assert Config().replace(decoder_theta_floor=0.0).decoder_theta_floor == 0.0
    assert Config().replace(decoder_theta_floor=0.25).decoder_theta_floor == 0.25

    with pytest.raises(ConfigError, match="decoder_theta_floor"):
        Config().replace(decoder_theta_floor=-1.0)
    # at or above zinb_theta_max every theta is pinned to one value, which measures the floor
    # rather than the model
    with pytest.raises(ConfigError, match="decoder_theta_floor"):
        Config().replace(decoder_theta_floor=Config().zinb_theta_max)


def test_config_roundtrip(tmp_path):
    """yaml -> Config -> yaml is idempotent, and validate() rejects bad values."""
    path = tmp_path / "cfg.yaml"
    Config().to_yaml(path)
    first = path.read_text()

    cfg = Config.from_yaml(path)
    assert cfg == Config()

    cfg.to_yaml(path)
    assert path.read_text() == first

    again = Config.from_yaml(path)
    assert again == cfg
    assert again.to_dict() == cfg.to_dict()

    # every field survives the round trip, not just the ones with interesting values
    assert set(again.to_dict()) == {f.name for f in dataclasses.fields(Config)}

    # a non-default value survives too
    tuned = cfg.replace(ell_xy=137.5, layout_mode="hybrid", genes_per_step=64)
    tuned.to_yaml(path)
    assert Config.from_yaml(path) == tuned

    with pytest.raises(ConfigError, match="ell_xy"):
        Config().replace(ell_xy=0.0)
    with pytest.raises(ConfigError, match="ema_decay"):
        Config().replace(ema_decay=1.0)
    with pytest.raises(ConfigError, match="section_dropout_p"):
        Config().replace(section_dropout_p=1.5)
    with pytest.raises(ConfigError, match="sefl_min_angle_deg"):
        Config().replace(sefl_min_angle_deg=170.0)
    with pytest.raises(ConfigError, match="holdout_consecutive_k"):
        Config().replace(holdout_consecutive_k=2)
    with pytest.raises(ConfigError, match="w_recon"):
        Config().replace(w_recon=-1.0)


def test_config_rejects_unknown_field(tmp_path):
    """A misspelled key is an error, not a silently ignored line (Convention 6)."""
    path = tmp_path / "cfg.yaml"
    path.write_text("ell_x: 100.0\n")
    with pytest.raises(ConfigError, match="ell_x"):
        Config.from_yaml(path)
    with pytest.raises(ConfigError, match="not_a_field"):
        Config().replace(not_a_field=1)


def test_config_is_frozen():
    """Config is immutable; variants come from replace()."""
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1  # type: ignore[misc]


def test_config_gates_validate():
    """Every gate rejects an out-of-set value; the v20 tuple is reachable."""
    gates = {
        "layout_mode": LAYOUT_MODES,
        "prior_mode": PRIOR_MODES,
        "expr_mode": EXPR_MODES,
        "text_emb_mode": TEXT_EMB_MODES,
        "decoder": DECODERS,
    }
    for name, allowed in gates.items():
        for value in allowed:
            assert Config().replace(**{name: value}) is not None
        with pytest.raises(ConfigError, match=name):
            Config().replace(**{name: "not-a-real-option"})

    # T09's selector must be able to reach the previous version's behaviour, or the
    # no-regression guarantee is not implementable.
    v20 = Config().replace(**V20_CONFIG)
    assert v20.layout_mode == "resample"
    assert v20.expr_mode == "cross-mix"
    assert v20.prior_mode == "iid"
    v20.validate()


def test_every_field_is_documented():
    """Convention 1: a constant with no documentation is a magic number with a name."""
    lines = inspect.getsource(Config).splitlines()
    field_names = {f.name for f in dataclasses.fields(Config)}
    undocumented = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        name = stripped.split(":")[0].strip()
        if name in field_names and stripped.startswith(name):
            following = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if not following.startswith('"""'):
                undocumented.append(name)
    assert undocumented == []
