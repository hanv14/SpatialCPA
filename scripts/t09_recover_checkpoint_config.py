"""Read a saved fit's `Config` out of the file that holds it, and write it into the artifacts.

`specs/10` §4.2a-ii: an artifact must record the arm it describes, or its envelope can never be
matched to it later. The six-metric table's artifacts
(``reports/r11_starmap_layout_modes.json``, ``r11_resample_grid{,_umap}.json``,
``r11_determinism_{a,b}.json``) record ``model``, ``decoder_mu_link``, ``train_steps`` and
``seed`` and **no** ``config_hash``, ``text_emb_mode`` or metric-aware weights — so A9's
three-seed ``bench3`` envelopes cannot be matched to that arm, and five clearance figures are
flagged rather than numbered in ``reports/envelope_correction.md`` §3.

**The information was never lost.** ``scripts/t10_rescore_saved.py`` writes and reads the model
file as ``{"config": <every Config field>, "state_dict": ...}`` and its ``--preflight`` branch
already prints four of the fields. The defect is that the *reporting* scripts never copied the
block into their own output. So this is a retrieval, not a measurement, and it costs one file
read.

Two payload shapes are handled, and they carry different amounts:

``{"config": ..., "state_dict": ...}``
    a **saved fit** (``t10_rescore_saved.py``'s ``--model``). Carries the whole ``Config``;
    everything §4.2a needs is a direct read.
``dataclasses.asdict(FitCheckpoint)``
    a **resumable checkpoint** (``train/checkpoint.py``). Carries ``config_hash`` but *not* the
    config, so the gates are recovered from evidence instead: ``history.terms`` names the loss
    terms a step actually charged, so T08's three metric-aware weights are all zero **iff**
    ``autocorr``/``profile``/``distribution`` are absent (``train_ctfflow`` builds no
    ``LOSOScheduler`` at zero weight, so ``metric_aware_terms`` is never called), and
    ``teacher is None`` **iff** every SEFL weight is zero. Reported as inference, labelled as
    such, and never mixed with the direct reads.

Usage::

    # what the flagged rows are waiting on
    python scripts/t09_recover_checkpoint_config.py runs/pilot/model_exp_2400.pt

    # and repair the artifacts in place, so the next reader does not have to ask
    python scripts/t09_recover_checkpoint_config.py runs/pilot/model_exp_2400.pt \\
        --patch reports/r11_starmap_layout_modes.json \\
        --patch reports/r11_resample_grid.json \\
        --patch reports/r11_resample_grid_umap.json \\
        --patch reports/r11_determinism_a.json \\
        --patch reports/r11_determinism_b.json

    # exercise the payload logic with no torch and no checkpoint
    python scripts/t09_recover_checkpoint_config.py --self-check

``--patch`` adds a ``recovered_config`` block to every arm in the named JSON, recording the
gates plus the provenance of the recovery itself. It never edits a measured value, refuses a
file that already carries one unless ``--force`` is given, and prints a diff-shaped summary of
what it wrote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The gates that decide whether one arm's envelope may be quoted against another's
# (``specs/10`` §4.2a: per-metric, per-arm, per-dataset, per-gate; §4.2a-i adds per-instrument).
# Anything here differing between two arms makes their envelopes non-interchangeable.
ENVELOPE_GATES: tuple[str, ...] = (
    "text_emb_mode",
    "expr_mode",
    "prior_mode",
    "layout_mode",
    "layout_sampler",
    "decoder_mu_link",
    "train_steps",
    "expr_pca_dim",
    "w_autocorr",
    "w_profile",
    "w_distribution",
    "w_cross",
    "w_thick",
    "w_prog",
    "seed",
)

METRIC_AWARE_TERMS = ("autocorr", "profile", "distribution")
SEFL_TERMS = ("cross", "thick", "prog")


class RecoveryError(RuntimeError):
    """The payload is not a shape this script knows how to read."""


def _as_plain(value: Any) -> Any:
    """Return ``value`` as something ``json.dump`` accepts, or its repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    return repr(value)


def describe(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Return what a checkpoint payload says about the arm that wrote it.

    Pure: takes the already-loaded mapping, so it is exercisable without ``torch`` (see
    ``--self-check``). ``source`` is recorded in the result for provenance.

    Returns a mapping with ``kind`` (``"saved_fit"`` or ``"resumable_checkpoint"``), ``direct``
    (fields read straight out of the file), ``inferred`` (fields deduced from evidence, with the
    evidence beside each), and ``unresolved`` (gates this payload cannot settle).
    """
    if not isinstance(payload, dict):
        raise RecoveryError(f"{source}: expected a mapping, got {type(payload).__name__}")

    out: dict[str, Any] = {"source": source, "direct": {}, "inferred": {}, "unresolved": []}

    if "config" in payload and isinstance(payload["config"], dict):
        out["kind"] = "saved_fit"
        cfg = payload["config"]
        out["config"] = {str(k): _as_plain(v) for k, v in cfg.items()}
        out["direct"] = {g: _as_plain(cfg[g]) for g in ENVELOPE_GATES if g in cfg}
        out["unresolved"] = [g for g in ENVELOPE_GATES if g not in cfg]
        return out

    if "config_hash" in payload:
        out["kind"] = "resumable_checkpoint"
        for key in ("step", "steps", "seed", "config_hash", "format"):
            if key in payload:
                out["direct"][key] = _as_plain(payload[key])

        history = payload.get("history")
        terms = history.get("terms") if isinstance(history, dict) else None
        if isinstance(terms, dict):
            charged = sorted(terms)
            out["direct"]["history_terms"] = charged
            metric_on = any(t in terms for t in METRIC_AWARE_TERMS)
            out["inferred"]["w_autocorr/w_profile/w_distribution"] = {
                "value": "at least one > 0" if metric_on else "all three == 0",
                "evidence": (
                    f"history.terms {'contains' if metric_on else 'omits'} "
                    f"{list(METRIC_AWARE_TERMS)}; train_ctfflow builds no LOSOScheduler at zero "
                    "weight, so metric_aware_terms is never called and the keys cannot appear"
                ),
            }
            sefl_on = any(t in terms for t in SEFL_TERMS)
            out["inferred"]["w_cross/w_thick/w_prog"] = {
                "value": "at least one > 0" if sefl_on else "all three == 0",
                "evidence": f"history.terms {'contains' if sefl_on else 'omits'} {list(SEFL_TERMS)}",
            }
        else:
            out["unresolved"].append("history.terms absent — metric-aware and SEFL not inferable")

        if "teacher" in payload:
            teacher_on = payload["teacher"] is not None
            out["inferred"]["SEFL (teacher)"] = {
                "value": "on" if teacher_on else "all SEFL weights == 0",
                "evidence": (
                    "FitCheckpoint.teacher is "
                    f"{'a state_dict' if teacher_on else 'None'}; EMATeacher is built only when "
                    "a SEFL weight exceeds zero"
                ),
            }

        settled = {"seed", "train_steps", "w_autocorr", "w_profile", "w_distribution",
                   "w_cross", "w_thick", "w_prog"}
        out["unresolved"] += [
            g if g != "text_emb_mode" else (
                "text_emb_mode — not stored and not inferable from this payload; identify it by "
                "matching config_hash against a rebuilt Config "
                "(scripts/t09_checkpoint_config.py)"
            )
            for g in ENVELOPE_GATES
            if g not in settled
        ]
        return out

    raise RecoveryError(
        f"{source}: neither a saved fit (no 'config' key) nor a resumable checkpoint "
        f"(no 'config_hash' key); top-level keys are {sorted(payload)[:12]}"
    )


def render(report: dict[str, Any]) -> str:
    """Return the human-readable form of :func:`describe`'s result."""
    lines = [f"## {report['source']}", f"kind: **{report['kind']}**", ""]
    if report["direct"]:
        lines.append("### Direct reads — the file says so")
        for key, value in report["direct"].items():
            lines.append(f"  {key:<24} {value}")
        lines.append("")
    if report["inferred"]:
        lines.append("### Inferred — deduced from evidence, not stored")
        for key, item in report["inferred"].items():
            lines.append(f"  {key}")
            lines.append(f"    value:    {item['value']}")
            lines.append(f"    evidence: {item['evidence']}")
        lines.append("")
    if report["unresolved"]:
        lines.append("### Unresolved — this payload cannot settle these")
        lines += [f"  - {u}" for u in report["unresolved"]]
        lines.append("")
    if report["kind"] == "saved_fit":
        gates = report["direct"]
        lines.append("### Verdict for `specs/10` §4.2a-ii")
        lines.append(
            "  Every envelope-deciding gate is a direct read. Two arms may share an envelope "
            "only if all of the above agree."
        )
        weights = [gates.get(w) for w in ("w_autocorr", "w_profile", "w_distribution")]
        if all(w is not None for w in weights):
            state = "OFF (all zero)" if all(float(w) == 0.0 for w in weights) else "ON"
            lines.append(f"  metric-aware weights: {state} -> {weights}")
        if "text_emb_mode" in gates:
            lines.append(f"  text_emb_mode:        {gates['text_emb_mode']}")
    return "\n".join(lines)


def patch(path: Path, report: dict[str, Any], *, force: bool) -> str:
    """Write ``recovered_config`` into every arm of the JSON at ``path``. Returns a summary."""
    payload = json.loads(path.read_text())
    arms = payload["arms"] if isinstance(payload, dict) and "arms" in payload else payload
    if not isinstance(arms, list):
        raise RecoveryError(f"{path}: expected a list of arms or an 'arms' key")

    existing = [a for a in arms if isinstance(a, dict) and "recovered_config" in a]
    if existing and not force:
        return f"  {path}: already carries recovered_config on {len(existing)} arm(s); --force to overwrite"

    block = {
        "recovered_from": report["source"],
        "recovered_by": "scripts/t09_recover_checkpoint_config.py",
        "why": (
            "specs/10 §4.2a-ii — this artifact recorded no config_hash, text_emb_mode or "
            "metric-aware weights, so its envelope could not be matched to any measured one. "
            "The gates below are a direct read of the file that produced these numbers; no "
            "measured value in this artifact was altered."
        ),
        "gates": report["direct"],
    }
    if "config" in report:
        block["config"] = report["config"]

    touched = 0
    for arm in arms:
        if isinstance(arm, dict):
            arm["recovered_config"] = block
            touched += 1
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return f"  {path}: recovered_config written to {touched} arm(s)"


SELF_CHECK_SAVED_FIT: dict[str, Any] = {
    "config": {
        "text_emb_mode": "lookup",
        "expr_mode": "zinb-flow",
        "prior_mode": "correlated",
        "layout_mode": "resample",
        "layout_sampler": "grid",
        "decoder_mu_link": "exp",
        "train_steps": 2400,
        "expr_pca_dim": 28,
        "w_autocorr": 0.5,
        "w_profile": 0.5,
        "w_distribution": 0.5,
        "w_cross": 0.0,
        "w_thick": 0.0,
        "w_prog": 0.0,
        "seed": 1,
    },
    "state_dict": {},
}

SELF_CHECK_RESUMABLE: dict[str, Any] = {
    "step": 2400,
    "steps": 2400,
    "seed": 3,
    "config_hash": "0123456789abcdef",
    "format": 2,
    "teacher": None,
    "history": {"terms": {"recon": [1.0], "cfm": [0.2], "size": [0.1]}},
}


def self_check() -> int:
    """Exercise :func:`describe` on both payload shapes without ``torch``."""
    failures: list[str] = []

    saved = describe(SELF_CHECK_SAVED_FIT, source="<self-check saved fit>")
    if saved["kind"] != "saved_fit":
        failures.append(f"saved fit misread as {saved['kind']}")
    if saved["direct"].get("text_emb_mode") != "lookup":
        failures.append("saved fit: text_emb_mode not a direct read")
    if saved["unresolved"]:
        failures.append(f"saved fit: unexpected unresolved gates {saved['unresolved']}")

    resumable = describe(SELF_CHECK_RESUMABLE, source="<self-check resumable>")
    if resumable["kind"] != "resumable_checkpoint":
        failures.append(f"resumable misread as {resumable['kind']}")
    metric = resumable["inferred"].get("w_autocorr/w_profile/w_distribution", {})
    if metric.get("value") != "all three == 0":
        failures.append(f"resumable: metric-aware inference wrong -> {metric.get('value')!r}")
    if not any("text_emb_mode" in u for u in resumable["unresolved"]):
        failures.append("resumable: text_emb_mode should be reported unresolved")

    with_terms = dict(SELF_CHECK_RESUMABLE)
    with_terms["history"] = {"terms": {"recon": [1.0], "autocorr": [0.3]}}
    on = describe(with_terms, source="<self-check resumable, terms on>")
    if on["inferred"]["w_autocorr/w_profile/w_distribution"]["value"] != "at least one > 0":
        failures.append("resumable: metric-aware ON not detected from history.terms")

    for bad, why in (({"nope": 1}, "unknown shape"), ([], "not a mapping")):
        try:
            describe(bad, source="<self-check bad>")  # type: ignore[arg-type]
        except RecoveryError:
            pass
        else:
            failures.append(f"{why}: should have raised RecoveryError")

    print(render(saved))
    print()
    print(render(resumable))
    print()
    if failures:
        for f in failures:
            print(f"SELF-CHECK FAIL: {f}")
        return 1
    print("self-check OK — both payload shapes read, both error paths raise")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoints", nargs="*", help="saved fit(s) or resumable checkpoint(s)")
    ap.add_argument(
        "--patch",
        action="append",
        default=[],
        metavar="RESULTS.JSON",
        help="write the recovered gates into this artifact's arms (repeatable)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing recovered_config")
    ap.add_argument("--self-check", action="store_true", help="exercise the reader, no torch")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()
    if not args.checkpoints:
        ap.error("give at least one checkpoint path, or --self-check")

    import torch  # local: the self-check must run without it

    reports = []
    for path in args.checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        report = describe(payload, source=str(path))
        reports.append(report)
        print(render(report))
        print()

    if args.patch:
        saved = [r for r in reports if r["kind"] == "saved_fit"]
        if len(saved) != 1:
            print(
                f"--patch needs exactly one saved fit to copy from, got {len(saved)}; "
                "nothing written",
                file=sys.stderr,
            )
            return 1
        print("### Patched")
        for target in args.patch:
            print(patch(Path(target), saved[0], force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
