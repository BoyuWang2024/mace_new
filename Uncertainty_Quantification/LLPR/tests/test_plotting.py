from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from Uncertainty_Quantification.LLPR.llpr.artifacts import atomic_json_dump
from Uncertainty_Quantification.LLPR.llpr.inference import summarize_variant
from Uncertainty_Quantification.LLPR.llpr.plotting import (
    DEFAULT_SELECTED,
    FIGURE_STEMS,
    PLOTTING_STATISTICS_FIELDS,
    run_plot,
)
from Uncertainty_Quantification.LLPR.tests import (
    test_inference_validation as publication_support,
)


matplotlib.use("Agg", force=True)

EXPECTED_STATISTICS_FIELDS = (
    "variant", "target", "data_level", "unit", "rows", "log_plot_rows",
    "zero_std_rows", "zero_absolute_residual_rows", "mae", "rmse",
    "mean_std", "median_std", "coverage_1sigma", "coverage_2sigma",
    "coverage_3sigma", "standardized_residual_rows",
    "mean_absolute_standardized_residual",
    "undefined_standardized_residual_rows",
    "zero_zero_standardized_residual_rows",
    "uncertainty_residual_correlation",
    "correlation_valid_rows", "correlation_degenerate",
    "shared_axis_min", "shared_axis_max",
)
EXPECTED_FIGURE_STEMS = (
    "selected_uncertainty_residual",
    "energy_comparison",
    "force_component_comparison",
    "force_structure_comparison",
    "reliability",
    "standardized_residual_cdf",
)


def _publication_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return publication_support._evaluated_publication_root(tmp_path, monkeypatch)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(value)

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _rewrite_energy_with_zero_pair(root: Path) -> None:
    for variant in ("he", "hf", "hef"):
        path = root / variant / "energy.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "prediction"] = frame.loc[0, "reference"]
        frame.loc[0, ["residual", "q", "variance", "std"]] = 0.0
        frame.to_csv(path, index=False)
        summary_path = root / variant / "summary.json"
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        atomic_json_dump(
            summary_path,
            summarize_variant(
                root / variant,
                variant=variant,
                ridge_mode=previous["ridge"]["mode"],
                ridge=previous["ridge"]["value"],
                energy_alpha=previous["alpha"]["energy"],
                force_alpha=previous["alpha"]["forces"],
                cholesky_diagnostics=previous["cholesky_diagnostics"],
            ),
        )


def _lock_worker(output: str, events: object) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    with plotting._output_lock(Path(output)):
        events.put(("enter", time.monotonic()))
        time.sleep(0.25)
        events.put(("exit", time.monotonic()))


def test_default_selected_paths() -> None:
    assert DEFAULT_SELECTED == (
        ("he", "energy"),
        ("hf", "forces"),
        ("hef", "energy"),
        ("hef", "forces"),
    )


def test_fixed_statistics_and_figure_contracts() -> None:
    assert PLOTTING_STATISTICS_FIELDS == EXPECTED_STATISTICS_FIELDS
    assert FIGURE_STEMS == EXPECTED_FIGURE_STEMS


def test_energy_plot_uses_per_atom_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    result = run_plot(root, output_dir=tmp_path / "plots")
    stats = pd.read_csv(result / "plotting_statistics.csv")
    row = stats.query("variant == 'he' and target == 'energy'").iloc[0]
    assert row["unit"] == "eV/atom"
    assert row["rows"] == 2


def test_run_plot_generates_real_nonempty_png_pdf_and_fixed_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    result = run_plot(root, output_dir=tmp_path / "plots")
    expected_figures = {
        f"{stem}.{suffix}"
        for stem in EXPECTED_FIGURE_STEMS
        for suffix in ("png", "pdf")
    }
    actual_figures = {
        path.name for path in result.iterdir() if path.suffix in {".png", ".pdf"}
    }
    assert actual_figures == expected_figures
    for stem in EXPECTED_FIGURE_STEMS:
        png = result / f"{stem}.png"
        pdf = result / f"{stem}.pdf"
        assert png.stat().st_size > 1_000
        with Image.open(png) as image:
            image.verify()
            assert image.width >= 600
            assert image.height >= 400
        pdf_bytes = pdf.read_bytes()
        assert len(pdf_bytes) > 1_000
        assert pdf_bytes.startswith(b"%PDF-")
        assert pdf_bytes.rstrip().endswith(b"%%EOF")

    statistics = pd.read_csv(result / "plotting_statistics.csv")
    assert tuple(statistics.columns) == EXPECTED_STATISTICS_FIELDS
    assert list(statistics[["data_level", "variant"]].itertuples(index=False, name=None)) == [
        (level, variant)
        for level in ("energy", "force_component", "force_structure")
        for variant in ("he", "hf", "hef")
    ]
    assert set(statistics.query("data_level == 'energy'")["unit"]) == {"eV/atom"}
    assert set(statistics.query("data_level != 'energy'")["unit"]) == {"eV/\u00c5"}
    for _, group in statistics.groupby("data_level", sort=False):
        assert group["shared_axis_min"].nunique() == 1
        assert group["shared_axis_max"].nunique() == 1
        assert group["shared_axis_min"].iloc[0] < group["shared_axis_max"].iloc[0]


def test_zero_values_are_excluded_only_from_log_rendering_not_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    _rewrite_energy_with_zero_pair(root)
    result = run_plot(root, output_dir=tmp_path / "plots")
    statistics = pd.read_csv(result / "plotting_statistics.csv")
    row = statistics.query("variant == 'he' and data_level == 'energy'").iloc[0]
    source = pd.read_csv(root / "he" / "energy.csv")
    expected_mae = float(np.mean(np.abs(source["residual"].to_numpy())))
    assert row["rows"] == 2
    assert row["log_plot_rows"] == 1
    assert row["zero_std_rows"] == 1
    assert row["zero_absolute_residual_rows"] == 1
    assert row["mae"] == pytest.approx(expected_mae)
    manifest = _strict_json(result / "plotting_manifest.json")
    assert manifest["config"]["zero_strategy"] == (
        "exclude non-positive coordinates from logarithmic rendering only; "
        "retain every validated row in statistics"
    )


def test_plotting_manifest_is_strict_deterministic_and_hashes_every_consumed_input_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    output = tmp_path / "plots"
    first = run_plot(root, output_dir=output)
    first_statistics = (first / "plotting_statistics.csv").read_bytes()
    first_manifest = (first / "plotting_manifest.json").read_bytes()
    first_hashes = {
        path.name: _sha256(path)
        for path in first.iterdir()
        if path.name != "plotting_manifest.json"
    }
    second = run_plot(root, output_dir=output)
    assert (second / "plotting_statistics.csv").read_bytes() == first_statistics
    assert (second / "plotting_manifest.json").read_bytes() == first_manifest
    assert {
        path.name: _sha256(path)
        for path in second.iterdir()
        if path.name != "plotting_manifest.json"
    } == first_hashes

    manifest = _strict_json(second / "plotting_manifest.json")
    assert set(manifest) == {
        "schema_version", "formula_version", "status", "inputs", "outputs",
        "config", "selected", "units", "figures", "shared_axes",
        "standardized_residual_counts",
    }
    expected_inputs = {"manifest.json", "validation.json"} | {
        f"{variant}/{filename}"
        for variant in ("he", "hf", "hef")
        for filename in (
            "energy.csv", "force_components.csv", "force_structure.csv", "summary.json",
        )
    }
    assert set(manifest["inputs"]) == expected_inputs
    assert all(manifest["inputs"][name] == _sha256(root / name) for name in expected_inputs)
    assert "plotting_manifest.json" not in manifest["outputs"]
    expected_outputs = {path.name for path in second.iterdir()} - {"plotting_manifest.json"}
    assert set(manifest["outputs"]) == expected_outputs
    assert all(manifest["outputs"][name] == _sha256(second / name) for name in expected_outputs)
    assert manifest["selected"] == [list(item) for item in DEFAULT_SELECTED]
    assert manifest["units"] == {"energy": "eV/atom", "forces": "eV/\u00c5"}
    assert not any(
        forbidden in first_manifest.lower()
        for forbidden in (b"timestamp", b"created_at", b"updated_at", b"nan", b"infinity")
    )


def test_manifest_figure_paths_follow_configured_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    selected = (
        ("hf", "energy"),
        ("he", "forces"),
        ("hef", "forces"),
        ("hef", "energy"),
    )

    result = run_plot(root, output_dir=tmp_path / "plots", selected=selected)

    manifest = _strict_json(result / "plotting_manifest.json")
    assert manifest["selected"] == [list(item) for item in selected]
    assert manifest["figures"]["selected_uncertainty_residual"]["paths"] == [
        list(item) for item in selected
    ]


def test_constant_and_zero_statistics_are_finite_and_ecdf_counts_zero_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    for variant in ("he", "hf", "hef"):
        path = root / variant / "energy.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "prediction"] = frame.loc[0, "reference"]
        frame.loc[0, "residual"] = 0.0
        frame.loc[1, "residual"] = 1.0
        frame.loc[1, "prediction"] = frame.loc[1, "reference"] - 1.0
        frame.loc[:, ["q", "variance", "std"]] = 0.0
        frame.to_csv(path, index=False)
        summary_path = root / variant / "summary.json"
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        atomic_json_dump(
            summary_path,
            summarize_variant(
                root / variant,
                variant=variant,
                ridge_mode=previous["ridge"]["mode"],
                ridge=previous["ridge"]["value"],
                energy_alpha=previous["alpha"]["energy"],
                force_alpha=previous["alpha"]["forces"],
                cholesky_diagnostics=previous["cholesky_diagnostics"],
            ),
        )

    result = run_plot(root, output_dir=tmp_path / "plots")

    statistics = pd.read_csv(result / "plotting_statistics.csv")
    numeric = statistics.drop(columns=["variant", "target", "data_level", "unit"])
    assert not numeric.isna().any(axis=None)
    assert np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
    row = statistics.query("variant == 'he' and data_level == 'energy'").iloc[0]
    assert row["standardized_residual_rows"] == 1
    assert row["zero_zero_standardized_residual_rows"] == 1
    assert row["undefined_standardized_residual_rows"] == 1
    assert row["mean_absolute_standardized_residual"] == 0.0
    assert row["uncertainty_residual_correlation"] == 0.0
    assert row["correlation_valid_rows"] == 2
    assert row["correlation_degenerate"] == 1

    manifest = _strict_json(result / "plotting_manifest.json")
    counts = manifest["standardized_residual_counts"]["he/energy"]
    assert counts == {
        "ecdf_rows": 1,
        "undefined_rows": 1,
        "zero_zero_rows": 1,
    }

def test_invalid_publication_input_fails_before_replacing_existing_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    output = run_plot(root, output_dir=tmp_path / "plots")
    marker = output / "keep.txt"
    marker.write_text("old plots", encoding="utf-8")
    energy_path = root / "he" / "energy.csv"
    energy_path.write_bytes(energy_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest.*SHA256"):
        run_plot(root, output_dir=output)
    assert marker.read_text(encoding="utf-8") == "old plots"
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_render_failure_preserves_existing_plots_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path, monkeypatch)
    output = tmp_path / "plots"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old plots", encoding="utf-8")

    def fail_render(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(plotting, "_render_all_figures", fail_render)
    with pytest.raises(RuntimeError, match="simulated render failure"):
        run_plot(root, output_dir=output)
    assert marker.read_text(encoding="utf-8") == "old plots"
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


@pytest.mark.parametrize(
    "unsafe", ["root", "ancestor", "canonical_child", "plots_child"]
)
def test_output_directory_cannot_replace_publication_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str
) -> None:
    root = _publication_root(tmp_path / "case", monkeypatch)
    output = {
        "root": root,
        "ancestor": root.parent,
        "canonical_child": root / "he",
        "plots_child": root / "plots",
    }[unsafe]
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="output_dir"):
        run_plot(root, output_dir=output)
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    assert after == before

@pytest.mark.parametrize("case", ["canonical_child", "parent_component"])
def test_output_symlink_escape_is_rejected_before_validation_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    root = _publication_root(tmp_path / "publication", monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    if case == "canonical_child":
        external_data = external / "he"
        (root / "he").rename(external_data)
        sentinel = external_data / "sentinel.txt"
        sentinel.write_text("outside must survive", encoding="utf-8")
        (root / "he").symlink_to(external_data, target_is_directory=True)
        output = root / "he"
    else:
        sentinel = external / "sentinel.txt"
        sentinel.write_text("outside must survive", encoding="utf-8")
        link = tmp_path / "linked-parent"
        link.symlink_to(external, target_is_directory=True)
        output = link / "plots"
    formal_before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    with pytest.raises(ValueError, match="output_dir|symlink"):
        run_plot(root, output_dir=output)

    assert sentinel.read_text(encoding="utf-8") == "outside must survive"
    formal_after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert formal_after == formal_before
    assert not (root / "manifest.json").exists()
    assert not (root / "validation.json").exists()


def test_output_lock_serializes_independent_processes(tmp_path: Path) -> None:
    context = mp.get_context("fork")
    events = context.Queue()
    output = tmp_path / "plots"
    workers = [
        context.Process(target=_lock_worker, args=(str(output), events))
        for _ in range(2)
    ]

    for worker in workers:
        worker.start()
    observed = [events.get(timeout=5.0) for _ in range(4)]
    for worker in workers:
        worker.join(timeout=5.0)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert [name for name, _ in sorted(observed, key=lambda item: item[1])] == [
        "enter", "exit", "enter", "exit"
    ]


def test_pre_exchange_failure_preserves_complete_old_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "publication", monkeypatch)
    output = run_plot(root, output_dir=tmp_path / "plots")
    marker = output / "old-only.txt"
    marker.write_text("old", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail_exchange(first: Path, second: Path) -> None:
        del first, second
        raise RuntimeError("simulated exchange failure")

    monkeypatch.setattr(plotting, "_rename_exchange", fail_exchange, raising=False)
    with pytest.raises(RuntimeError, match="simulated exchange failure"):
        run_plot(root, output_dir=output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not list(output.parent.glob(f".{output.name}.stale-*"))


def test_post_exchange_cleanup_failure_keeps_complete_new_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "publication", monkeypatch)
    output = run_plot(root, output_dir=tmp_path / "plots")
    (output / "old-only.txt").write_text("old", encoding="utf-8")
    real_rmtree = plotting.shutil.rmtree

    def fail_stale_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if f".{output.name}.stale-" in Path(path).name or ".backup-" in Path(path).name:
            raise OSError("simulated stale cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(plotting.shutil, "rmtree", fail_stale_cleanup)
    result = run_plot(root, output_dir=output)

    assert result == output
    assert not (output / "old-only.txt").exists()
    assert (output / "plotting_manifest.json").is_file()
    stale = list(output.parent.glob(f".{output.name}.stale-*"))
    assert len(stale) == 1
    assert (stale[0] / "old-only.txt").read_text(encoding="utf-8") == "old"


def test_input_change_during_render_aborts_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "publication", monkeypatch)
    output = run_plot(root, output_dir=tmp_path / "plots")
    marker = output / "old-only.txt"
    marker.write_text("old", encoding="utf-8")
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    real_render = plotting._render_all_figures

    def mutate_input_after_render(*args: object, **kwargs: object) -> None:
        real_render(*args, **kwargs)
        energy_path = root / "he" / "energy.csv"
        energy_path.write_bytes(energy_path.read_bytes() + b"\n")

    monkeypatch.setattr(plotting, "_render_all_figures", mutate_input_after_render)
    with pytest.raises(ValueError, match="input.*changed"):
        run_plot(root, output_dir=output)

    after = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert marker.read_text(encoding="utf-8") == "old"
    assert not list(output.parent.glob(f".{output.name}.stale-*"))
