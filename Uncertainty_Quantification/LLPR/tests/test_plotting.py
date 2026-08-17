from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing as mp
import shutil
import stat
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from Uncertainty_Quantification.LLPR.llpr.artifacts import atomic_json_dump
from Uncertainty_Quantification.LLPR.llpr import density_plotting
from Uncertainty_Quantification.LLPR.llpr.cli import _load_plot_config
from Uncertainty_Quantification.LLPR.llpr.density_plotting import (
    CARNET_DENSITY_CONFIG,
    CARNET_STATISTICS_FIELDS,
    carnet_shared_limits,
    deterministic_sample_indices,
)
from Uncertainty_Quantification.LLPR.llpr.inference import summarize_variant
from Uncertainty_Quantification.LLPR.llpr.plotting import (
    DEFAULT_SELECTED,
    FIGURE_STEMS,
    PLOTTING_STATISTICS_FIELDS,
    _PanelData,
    run_plot,
)
from Uncertainty_Quantification.LLPR.llpr.validation import validate_publication_root
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

SNAPSHOT_SOURCE_NAMES = (
    "progress.pt",
    "manifest.json",
    *(
        f"{variant}/{filename}"
        for variant in ("he", "hf", "hef")
        for filename in (
            "energy.csv",
            "force_components.csv",
            "force_structure.csv",
            "summary.json",
        )
    ),
)


def _publication_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = publication_support._evaluated_publication_root(tmp_path, monkeypatch)
    validate_publication_root(root)
    return root


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
        frame.loc[0, "residual"] = 0.0
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


def _refresh_publication_validation(root: Path) -> None:
    (root / "manifest.json").unlink(missing_ok=True)
    (root / "validation.json").unlink(missing_ok=True)
    validate_publication_root(root)


def _make_distinct_publication(root: Path) -> None:
    _rewrite_energy_with_zero_pair(root)
    _refresh_publication_validation(root)


def _copy_publication_sources(source: Path, destination: Path) -> None:
    for name in SNAPSHOT_SOURCE_NAMES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)


def _fallback_snapshot_copy(source: Path, snapshot: Path, name: str) -> None:
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / name, target)


def _write_plot_config(
    path: Path,
    *,
    style: str | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    document: dict[str, object] = {
        "publication_root": "publication",
        "output_dir": "plots",
        "selected": [list(item) for item in DEFAULT_SELECTED],
    }
    if style is not None:
        document["style"] = style
    if extra is not None:
        document.update(extra)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _density_panel(
    uncertainty: list[float],
    absolute_residual: list[float],
    *,
    target: str,
    q: list[float] | None = None,
) -> density_plotting.DensityPanel:
    values = np.asarray(absolute_residual, dtype=np.float64)
    return density_plotting.DensityPanel(
        q=np.asarray(q if q is not None else [1.0] * len(values), dtype=np.float64),
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        absolute_residual=values,
        target=target,
        unit="eV/atom" if target == "energy" else "eV/\u00c5",
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


def test_plot_style_defaults_to_existing_suite(tmp_path: Path) -> None:
    config = _load_plot_config(_write_plot_config(tmp_path / "plot.yaml"))
    assert config.style == "diagnostic_suite"


@pytest.mark.parametrize("style", ["unknown", "diagnostic-suite"])
def test_plot_config_rejects_unknown_style(tmp_path: Path, style: str) -> None:
    with pytest.raises(ValueError, match="style"):
        _load_plot_config(_write_plot_config(tmp_path / "plot.yaml", style=style))


def test_plot_config_rejects_extra_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly"):
        _load_plot_config(
            _write_plot_config(tmp_path / "plot.yaml", extra={"unexpected": True})
        )


def test_mace_plotting_has_no_carnet_runtime_dependency() -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    sources = [Path(plotting.__file__), Path(density_plotting.__file__)]
    assert all("carnet_new" not in path.read_text(encoding="utf-8") for path in sources)


def test_pair_limits_are_dataset_local() -> None:
    panels = {
        ("he", "energy"): _density_panel([1.0, 2.0], [2.0, 4.0], target="energy"),
        ("hf", "forces"): _density_panel([10.0, 20.0], [20.0, 40.0], target="forces"),
        ("hef", "energy"): _density_panel([0.5, 4.0], [1.0, 8.0], target="energy"),
        ("hef", "forces"): _density_panel([5.0, 40.0], [10.0, 80.0], target="forces"),
    }
    limits = carnet_shared_limits(panels, margin=0.05)
    assert limits[("he", "energy")] == limits[("hef", "energy")]
    assert limits[("hf", "forces")] == limits[("hef", "forces")]
    assert limits[("he", "energy")] != limits[("hf", "forces")]


def test_sampling_is_deterministic_and_without_replacement() -> None:
    first = deterministic_sample_indices(50_000, 20_000, 20260714)
    second = deterministic_sample_indices(50_000, 20_000, 20260714)
    assert np.array_equal(first, second)
    assert first.size == 20_000
    assert np.unique(first).size == 20_000


def test_carnet_density_configuration_is_fixed() -> None:
    assert CARNET_DENSITY_CONFIG["grid_size"] == 160
    assert CARNET_DENSITY_CONFIG["gaussian_sigma"] == 1.2
    assert CARNET_DENSITY_CONFIG["contour_masses"] == (0.5, 0.7, 0.85, 0.95, 0.99)

def test_carnet_density_requires_exact_selected_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly"):
        run_plot(
            root,
            output_dir=tmp_path / "plots",
            style="carnet_density",
            selected=(
                ("he", "energy"),
                ("hf", "forces"),
                ("hef", "energy"),
                ("hf", "energy"),
            ),
        )


def test_carnet_density_generates_four_figures_and_style_aware_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _publication_root(tmp_path, monkeypatch)
    result = run_plot(root, output_dir=tmp_path / "plots", style="carnet_density")

    expected_stems = {
        "llpr_he_energy_uncertainty_vs_residual",
        "llpr_hf_force_uncertainty_vs_residual",
        "llpr_hef_energy_uncertainty_vs_residual",
        "llpr_hef_force_uncertainty_vs_residual",
    }
    assert {path.stem for path in result.glob("*.png")} == expected_stems
    assert {path.stem for path in result.glob("*.pdf")} == expected_stems
    statistics = pd.read_csv(result / "plotting_statistics.csv")
    assert tuple(statistics.columns) == CARNET_STATISTICS_FIELDS
    assert len(statistics) == 4
    manifest = _strict_json(result / "plotting_manifest.json")
    assert manifest["style"] == "carnet_density"
    assert manifest["config"]["grid_size"] == 160
    assert manifest["config"]["gaussian_sigma"] == 1.2
    assert manifest["config"]["contour_masses"] == [0.5, 0.7, 0.85, 0.95, 0.99]

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
    _refresh_publication_validation(root)
    result = run_plot(root, output_dir=tmp_path / "plots")
    statistics = pd.read_csv(result / "plotting_statistics.csv")
    row = statistics.query("variant == 'he' and data_level == 'energy'").iloc[0]
    source = pd.read_csv(root / "he" / "energy.csv")
    expected_mae = float(np.mean(np.abs(source["residual"].to_numpy())))
    assert row["rows"] == 2
    assert row["log_plot_rows"] == 1
    assert row["zero_std_rows"] == 0
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
    expected_inputs = {"progress.pt", "manifest.json", "validation.json"} | {
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


def test_zero_std_is_excluded_from_standardized_ecdf_but_raw_counts_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "publication").mkdir()
    root, _ = publication_support._policy_bound_publication_root(
        tmp_path / "publication", monkeypatch
    )
    for variant in ("he", "hf", "hef"):
        path = root / variant / "force_components.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "prediction"] = frame.loc[0, "reference"]
        frame.loc[0, "residual"] = 0.0
        frame.to_csv(path, index=False)
        publication_support._refresh_force_aggregates_and_summary(
            root, variant
        )

    _refresh_publication_validation(root)
    result = run_plot(root, output_dir=tmp_path / "plots")

    statistics = pd.read_csv(result / "plotting_statistics.csv")
    numeric = statistics.drop(columns=["variant", "target", "data_level", "unit"])
    assert not numeric.isna().any(axis=None)
    assert np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
    row = statistics.query(
        "variant == 'hf' and data_level == 'force_component'"
    ).iloc[0]
    assert row["rows"] == 6
    assert row["standardized_residual_rows"] == 4
    assert row["zero_zero_standardized_residual_rows"] == 1
    assert row["undefined_standardized_residual_rows"] == 1

    manifest = _strict_json(result / "plotting_manifest.json")
    counts = manifest["standardized_residual_counts"]["hf/force_component"]
    assert counts == {
        "ecdf_rows": 4,
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
    def fail_stale_cleanup(descriptor: int) -> None:
        raise OSError("simulated stale cleanup failure")

    monkeypatch.setattr(
        plotting, "_clear_directory_descriptor", fail_stale_cleanup
    )
    result = run_plot(root, output_dir=output)

    assert result == output
    assert not (output / "old-only.txt").exists()
    assert (output / "plotting_manifest.json").is_file()
    stale = list(output.parent.glob(".mace-llpr-retired-plots.stale-*"))
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


def test_snapshot_copy_validates_replacement_publication_before_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "a", monkeypatch)
    replacement = _publication_root(tmp_path / "b", monkeypatch)
    _make_distinct_publication(replacement)
    expected_hash = _sha256(replacement / "he" / "energy.csv")
    expected_mae = float(
        np.mean(np.abs(pd.read_csv(replacement / "he" / "energy.csv")["residual"]))
    )
    original_mae = float(
        np.mean(np.abs(pd.read_csv(root / "he" / "energy.csv")["residual"]))
    )
    assert expected_mae != original_mae
    real_copy = getattr(
        plotting, "_copy_snapshot_input", _fallback_snapshot_copy
    )
    replaced = False

    def replace_before_first_copy(
        source: Path, snapshot: Path, name: str
    ) -> None:
        nonlocal replaced
        if not replaced:
            _copy_publication_sources(replacement, root)
            replaced = True
        real_copy(source, snapshot, name)

    monkeypatch.setattr(
        plotting, "_copy_snapshot_input", replace_before_first_copy, raising=False
    )
    result = run_plot(root, output_dir=tmp_path / "plots")

    statistics = pd.read_csv(result / "plotting_statistics.csv")
    row = statistics.query("variant == 'he' and data_level == 'energy'").iloc[0]
    manifest = _strict_json(result / "plotting_manifest.json")
    assert row["mae"] == pytest.approx(expected_mae)
    assert manifest["inputs"]["he/energy.csv"] == expected_hash
    assert _sha256(root / "he" / "energy.csv") == expected_hash
    assert not list(tmp_path.glob(".plots.snapshot-*"))


def test_snapshot_render_is_immune_to_live_input_aba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "a", monkeypatch)
    replacement = _publication_root(tmp_path / "b", monkeypatch)
    _make_distinct_publication(replacement)
    validate_publication_root(root)
    saved = tmp_path / "saved-a"
    _copy_publication_sources(root, saved)
    expected_hash = _sha256(root / "he" / "energy.csv")
    expected_mae = float(
        np.mean(np.abs(pd.read_csv(root / "he" / "energy.csv")["residual"]))
    )
    replacement_mae = float(
        np.mean(np.abs(pd.read_csv(replacement / "he" / "energy.csv")["residual"]))
    )
    assert expected_mae != replacement_mae
    real_load = plotting._load_plot_data

    def load_during_live_aba(source: Path) -> object:
        _copy_publication_sources(replacement, root)
        try:
            return real_load(source)
        finally:
            _copy_publication_sources(saved, root)

    monkeypatch.setattr(plotting, "_load_plot_data", load_during_live_aba)
    result = run_plot(root, output_dir=tmp_path / "plots")

    statistics = pd.read_csv(result / "plotting_statistics.csv")
    row = statistics.query("variant == 'he' and data_level == 'energy'").iloc[0]
    manifest = _strict_json(result / "plotting_manifest.json")
    assert row["mae"] == pytest.approx(expected_mae)
    assert manifest["inputs"]["he/energy.csv"] == expected_hash
    assert _sha256(root / "he" / "energy.csv") == expected_hash
    assert not list(tmp_path.glob(".plots.snapshot-*"))


def test_mixed_snapshot_validation_failure_preserves_old_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "a", monkeypatch)
    replacement = _publication_root(tmp_path / "b", monkeypatch)
    _make_distinct_publication(replacement)
    output = run_plot(root, output_dir=tmp_path / "plots")
    (output / "old-only.txt").write_text("old", encoding="utf-8")
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    real_copy = getattr(
        plotting, "_copy_snapshot_input", _fallback_snapshot_copy
    )

    def copy_mixed_snapshot(source: Path, snapshot: Path, name: str) -> None:
        if name == "he/energy.csv":
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(replacement / name, target)
        else:
            real_copy(source, snapshot, name)

    monkeypatch.setattr(
        plotting, "_copy_snapshot_input", copy_mixed_snapshot, raising=False
    )
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        run_plot(root, output_dir=output)

    after = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(tmp_path.glob(".plots.snapshot-*"))
    assert not list(tmp_path.glob(".plots.stale-*"))


def _density_panels(std: list[float], residual: list[float]) -> dict[tuple[str, str], _PanelData]:
    return {key: _density_panel(std, residual, target=key[1]) for key in density_plotting.CARNET_SELECTED}


@pytest.mark.parametrize("style", [[], {}, 1, None, True])
def test_plot_config_rejects_non_string_style(tmp_path: Path, style: object) -> None:
    with pytest.raises(ValueError, match="style"):
        _load_plot_config(_write_plot_config(tmp_path / "plot.yaml", extra={"style": style}))


def test_density_limits_use_paired_valid_mask() -> None:
    panels = _density_panels([1.0, -1.0], [2.0, 1.0e30])
    assert carnet_shared_limits(panels, margin=0.05)[("he", "energy")][1] < 10.0


def test_density_correlations_are_tie_aware_and_undefined_is_null() -> None:
    panels = _density_panels([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    row = density_plotting.carnet_statistics(panels, carnet_shared_limits(panels, margin=0.05))[0]
    assert row["spearman_log"] == pytest.approx(0.8660254037844387)
    assert row["correlation_status"] == "ok"
    for std, residual, count in (
        ([1.0, 1.0], [1.0, 2.0], 2), ([1.0], [1.0], 1),
        ([1.0, 2.0], [0.0, 2.0], 1), ([-1.0, 2.0], [3.0, 2.0], 1),
    ):
        panels = _density_panels(std, residual)
        row = density_plotting.carnet_statistics(panels, carnet_shared_limits(panels, margin=0.05))[0]
        assert row["pearson_log"] is None and row["spearman_log"] is None
        assert row["correlation_rows"] == count
        assert row["correlation_status"].startswith("undefined_")


def test_density_typography_config_is_complete() -> None:
    assert {key: CARNET_DENSITY_CONFIG[key] for key in (
        "title_font_size", "axis_label_font_size", "tick_label_font_size",
        "annotation_font_size", "line_width", "spine_width", "constrained_layout",
    )} == {
        "title_font_size": 26.0, "axis_label_font_size": 22.0,
        "tick_label_font_size": 18.0, "annotation_font_size": 16.0,
        "line_width": 1.5, "spine_width": 1.5, "constrained_layout": True,
    }


def test_density_grid_sigma_rc_and_artists_follow_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panels = _density_panels([1.0, 2.0], [1.0, 2.0])
    seen: dict[str, object] = {}
    real_hist = np.histogram2d
    def hist(*args: object, **kwargs: object) -> object:
        seen["bins"] = kwargs["bins"]; return real_hist(*args, **kwargs)
    def smooth(values: np.ndarray, *, sigma: float) -> np.ndarray:
        seen["shape"], seen["sigma"] = values.shape, sigma; return values
    def save(fig: object, *args: object, **kwargs: object) -> None:
        axis = fig.axes[0]
        ref = next(line for line in axis.lines if line.get_label() == "1:1 reference")
        seen.setdefault("artist", (axis.get_title(), axis.title.get_fontsize(),
            axis.xaxis.label.get_fontsize(), axis.xaxis.get_ticklabels()[0].get_fontsize(),
            axis.texts[0].get_fontsize(), ref.get_linestyle(), ref.get_linewidth(),
            {spine.get_linewidth() for spine in axis.spines.values()},
            type(fig.get_layout_engine()).__name__))
    monkeypatch.setattr(density_plotting.np, "histogram2d", hist)
    monkeypatch.setattr(density_plotting, "gaussian_filter", smooth)
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", save)
    old = matplotlib.rcParams["font.size"]; matplotlib.rcParams["font.size"] = 13.0
    try:
        density_plotting.render_carnet_density(panels, tmp_path, dpi=72)
        assert matplotlib.rcParams["font.size"] == 13.0
    finally:
        matplotlib.rcParams["font.size"] = old
    assert len(seen["bins"][0]) == CARNET_DENSITY_CONFIG["grid_size"] + 1
    assert seen["shape"] == (160, 160) and seen["sigma"] == CARNET_DENSITY_CONFIG["gaussian_sigma"]
    assert seen["artist"] == ("He energy", 26.0, 22.0, 18.0, 16.0, "--", 1.5, {1.5}, "ConstrainedLayoutEngine")


def test_density_loader_reads_only_four_panels_three_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting
    expected = {"he/energy.csv", "hf/force_components.csv", "hef/energy.csv", "hef/force_components.csv"}
    for relative in expected:
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"q": [0.5], "std": [1.0], "residual": [-2.0], "unused": ["x"]}).to_csv(path, index=False)
    calls: list[tuple[str, tuple[str, ...]]] = []; real_read = pd.read_csv
    def read(path: Path, **kwargs: object) -> pd.DataFrame:
        calls.append((str(Path(path).relative_to(tmp_path)), tuple(kwargs.get("usecols", ())))); return real_read(path, **kwargs)
    monkeypatch.setattr(plotting.pd, "read_csv", read)
    panels = plotting._load_carnet_panels(tmp_path)
    assert {name for name, _ in calls} == expected
    assert all(columns == ("q", "std", "residual") for _, columns in calls)
    assert all(not hasattr(panel, "signed_residual") for panel in panels.values())


def test_density_analysis_computed_once_per_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panels = _density_panels([1.0, 2.0], [1.0, 2.0])
    real = density_plotting._analyze_panel; calls: list[tuple[str, str]] = []
    def analyze(*args: object, **kwargs: object) -> object:
        calls.append(args[1]); return real(*args, **kwargs)
    monkeypatch.setattr(density_plotting, "_analyze_panel", analyze)
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", lambda *args, **kwargs: None)
    density_plotting.render_carnet_density(panels, tmp_path, dpi=72)
    assert calls == list(density_plotting.CARNET_SELECTED)


def test_density_manifest_has_stats_and_output_size_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _publication_root(tmp_path / "publication", monkeypatch)
    result = run_plot(root, output_dir=tmp_path / "plots", style="carnet_density")
    rows = pd.read_csv(result / "plotting_statistics.csv")
    manifest = _strict_json(result / "plotting_manifest.json")
    for csv_row in rows.to_dict(orient="records"):
        key = f"{csv_row['variant']}/{csv_row['target']}"
        manifest_row = manifest["statistics"][key]
        assert set(manifest_row) == set(csv_row)
        for field, csv_value in csv_row.items():
            manifest_value = manifest_row[field]
            if isinstance(csv_value, float) and not pd.isna(csv_value):
                assert manifest_value == pytest.approx(csv_value)
            else:
                assert manifest_value == csv_value or (
                    manifest_value is None and pd.isna(csv_value)
                )
    expected = {path.name for path in result.iterdir()} - {"plotting_manifest.json"}
    assert set(manifest["outputs"]) == expected
    for name, record in manifest["outputs"].items():
        assert record == {"size": (result / name).stat().st_size, "sha256": _sha256(result / name)}


@pytest.mark.parametrize("corruption", ["empty_png", "manifest_size", "symlink"])
def test_density_staging_validation_rejects_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting
    root = _publication_root(tmp_path / "publication", monkeypatch)
    result = run_plot(root, output_dir=tmp_path / "plots", style="carnet_density")
    if corruption == "empty_png":
        next(result.glob("*.png")).write_bytes(b"")
    elif corruption == "manifest_size":
        path = result / "plotting_manifest.json"; manifest = _strict_json(path)
        name = next(iter(manifest["outputs"])); manifest["outputs"][name]["size"] += 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        png = next(result.glob("*.png")); target = tmp_path / "outside.png"
        target.write_bytes(png.read_bytes()); png.unlink(); png.symlink_to(target)
        assert stat.S_ISLNK(png.lstat().st_mode)
    with pytest.raises(RuntimeError, match="empty|size|regular|PNG|manifest"):
        plotting._validate_carnet_staging(result)


def test_density_live_change_preserves_existing_output_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _publication_root(tmp_path / "publication", monkeypatch)
    output = run_plot(root, output_dir=tmp_path / "plots", style="carnet_density")
    (output / "old-only.txt").write_text("old", encoding="utf-8")
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    real = density_plotting.render_carnet_density
    def render(*args: object, **kwargs: object) -> object:
        result = real(*args, **kwargs); path = root / "he" / "energy.csv"
        path.write_bytes(path.read_bytes() + b"\n"); return result
    monkeypatch.setattr(density_plotting, "render_carnet_density", render)
    with pytest.raises(ValueError, match="input.*changed"):
        run_plot(root, output_dir=output, style="carnet_density")
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


def test_density_zero_q_categories_retain_raw_rows_but_exclude_zero_std() -> None:
    panels = {
        path: _density_panel(
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 0.0, 3.0],
            target=path[1],
            q=[0.0, 0.0, 1.0, 1.0],
        )
        for path in density_plotting.CARNET_SELECTED
    }

    row = density_plotting.carnet_statistics(
        panels, carnet_shared_limits(panels, margin=0.05)
    )[0]

    assert row["rows"] == 4
    assert row["zero_q_rows"] == 2
    assert row["zero_q_zero_residual_rows"] == 1
    assert row["zero_q_nonzero_residual_rows"] == 1
    assert row["metric_rows"] == 2
    assert row["log_plot_rows"] == 1
    assert row["excluded_from_log_rows"] == 3


def test_density_policy_publication_reports_zero_q_categories_in_csv_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "publication").mkdir()
    root, _ = publication_support._policy_bound_publication_root(
        tmp_path / "publication", monkeypatch
    )
    validate_publication_root(root)

    result = run_plot(
        root, output_dir=tmp_path / "plots", style="carnet_density"
    )

    statistics = pd.read_csv(result / "plotting_statistics.csv")
    manifest = _strict_json(result / "plotting_manifest.json")
    for variant in ("hf", "hef"):
        row = statistics.query(
            "variant == @variant and target == 'forces'"
        ).iloc[0]
        assert row["zero_q_rows"] == 2
        assert row["zero_q_zero_residual_rows"] == 0
        assert row["zero_q_nonzero_residual_rows"] == 2
        assert row["metric_rows"] == row["rows"] - 2
        assert manifest["statistics"][f"{variant}/forces"][
            "zero_q_nonzero_residual_rows"
        ] == 2


def test_density_staging_rejects_inconsistent_zero_q_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "publication", monkeypatch)
    result = run_plot(
        root, output_dir=tmp_path / "plots", style="carnet_density"
    )
    statistics_path = result / "plotting_statistics.csv"
    statistics = pd.read_csv(statistics_path)
    statistics.loc[0, "zero_q_rows"] = 1
    statistics.loc[0, "zero_q_zero_residual_rows"] = 0
    statistics.loc[0, "zero_q_nonzero_residual_rows"] = 0
    statistics.to_csv(statistics_path, index=False)

    with pytest.raises(RuntimeError, match="zero-q.*inconsistent"):
        plotting._validate_carnet_staging(result)


def test_exchange_einval_falls_back_and_publishes_complete_new_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "publication", monkeypatch)
    output = run_plot(
        root, output_dir=tmp_path / "plots", style="carnet_density"
    )
    (output / "old-only.txt").write_text("old", encoding="utf-8")

    def unsupported_exchange(first: Path, second: Path) -> None:
        del first, second
        raise OSError(errno.EINVAL, "exchange unsupported")

    monkeypatch.setattr(plotting, "_rename_exchange", unsupported_exchange)
    result = run_plot(
        root, output_dir=output, style="carnet_density"
    )

    assert result == output
    assert not (output / "old-only.txt").exists()
    plotting._validate_carnet_staging(output)
    assert not list(output.parent.glob(f".{output.name}.stale-*"))
    assert not list(output.parent.glob(f".{output.name}.backup-*"))


@pytest.mark.parametrize(
    "error_number",
    sorted(
        {
            errno.EINVAL,
            errno.ENOSYS,
            errno.EXDEV,
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            getattr(errno, "ENOTSUP", errno.EINVAL),
        }
    ),
)
def test_exchange_capability_errors_use_verified_backup_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    destination = tmp_path / "plots"
    destination.mkdir()
    (destination / "value.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".plots.stale-controlled"
    staging.mkdir()
    (staging / "value.txt").write_text("new", encoding="utf-8")

    monkeypatch.setattr(
        plotting,
        "_rename_exchange",
        lambda first, second: (_ for _ in ()).throw(
            OSError(error_number, "exchange unavailable")
        ),
    )

    plotting._promote_directory(staging, destination)

    assert (destination / "value.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()
    assert not list(tmp_path.glob(".plots.backup-*"))


def test_exchange_permission_error_does_not_fall_back_or_mutate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    destination = tmp_path / "plots"
    destination.mkdir()
    (destination / "value.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".plots.stale-controlled"
    staging.mkdir()
    (staging / "value.txt").write_text("new", encoding="utf-8")

    monkeypatch.setattr(
        plotting,
        "_rename_exchange",
        lambda first, second: (_ for _ in ()).throw(
            OSError(errno.EPERM, "exchange denied")
        ),
    )

    with pytest.raises(OSError) as captured:
        plotting._promote_directory(staging, destination)

    assert captured.value.errno == errno.EPERM
    assert (destination / "value.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "value.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".plots.backup-*"))


def test_fallback_promotion_failure_rolls_back_old_output_without_backup_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    destination = tmp_path / "plots"
    destination.mkdir()
    (destination / "value.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".plots.stale-controlled"
    staging.mkdir()
    (staging / "value.txt").write_text("new", encoding="utf-8")
    real_noreplace = plotting._rename_noreplace

    monkeypatch.setattr(
        plotting,
        "_rename_exchange",
        lambda first, second: (_ for _ in ()).throw(
            OSError(errno.EINVAL, "exchange unsupported")
        ),
    )

    def fail_staging_promotion(source: Path, target: Path) -> None:
        if Path(source) == staging and Path(target) == destination:
            raise OSError(errno.EIO, "simulated staged rename failure")
        real_noreplace(source, target)

    monkeypatch.setattr(plotting, "_rename_noreplace", fail_staging_promotion)

    with pytest.raises(OSError) as captured:
        plotting._promote_directory(staging, destination)

    assert captured.value.errno == errno.EIO
    assert (destination / "value.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "value.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".plots.backup-*"))


def test_exchange_fallback_rejects_destination_replacement_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    destination = tmp_path / "plots"
    destination.mkdir()
    (destination / "value.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".plots.stale-controlled"
    staging.mkdir()
    (staging / "value.txt").write_text("new", encoding="utf-8")
    displaced = tmp_path / "displaced-old"

    def replace_then_report_unsupported(first: Path, second: Path) -> None:
        del first
        Path(second).rename(displaced)
        Path(second).mkdir()
        (Path(second) / "value.txt").write_text("other", encoding="utf-8")
        raise OSError(errno.EINVAL, "exchange unsupported")

    monkeypatch.setattr(plotting, "_rename_exchange", replace_then_report_unsupported)

    with pytest.raises(RuntimeError, match="changed"):
        plotting._promote_directory(staging, destination)

    assert (destination / "value.txt").read_text(encoding="utf-8") == "other"
    assert (displaced / "value.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "value.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".plots.backup-*"))


def test_fallback_never_overwrites_destination_created_during_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    destination = tmp_path / "plots"
    destination.mkdir()
    (destination / "value.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".plots.stale-controlled"
    staging.mkdir()
    (staging / "value.txt").write_text("new", encoding="utf-8")
    real_noreplace = plotting._rename_noreplace
    injected = False

    monkeypatch.setattr(
        plotting,
        "_rename_exchange",
        lambda first, second: (_ for _ in ()).throw(
            OSError(errno.EINVAL, "exchange unsupported")
        ),
    )

    def create_destination_before_promotion(source: Path, target: Path) -> None:
        nonlocal injected
        if Path(source) == staging and Path(target) == destination and not injected:
            injected = True
            destination.mkdir()
            (destination / "value.txt").write_text("other", encoding="utf-8")
        real_noreplace(source, target)

    monkeypatch.setattr(plotting, "_rename_noreplace", create_destination_before_promotion)

    with pytest.raises(RuntimeError, match="could not safely restore"):
        plotting._promote_directory(staging, destination)

    assert (destination / "value.txt").read_text(encoding="utf-8") == "other"
    assert (staging / "value.txt").read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob(".plots.backup-*/previous/value.txt"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old"


def test_plot_failure_does_not_remove_replaced_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    root = _publication_root(tmp_path / "publication", monkeypatch)
    output = tmp_path / "plots"
    replacement: Path | None = None

    def replace_staging_then_fail(staging: Path, destination: Path) -> None:
        nonlocal replacement
        del destination
        owned = staging.with_name(f"{staging.name}.owned")
        staging.rename(owned)
        staging.mkdir()
        (staging / "value.txt").write_text("other", encoding="utf-8")
        replacement = staging
        raise RuntimeError("simulated unsafe rollback")

    monkeypatch.setattr(plotting, "_promote_directory", replace_staging_then_fail)

    with pytest.raises(RuntimeError, match="simulated unsafe rollback"):
        run_plot(root, output_dir=output, style="carnet_density")

    assert replacement is not None
    assert (replacement / "value.txt").read_text(encoding="utf-8") == "other"


def test_owned_cleanup_quarantines_before_removing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    staging = tmp_path / ".plots.stale-controlled"
    staging.mkdir()
    (staging / "value.txt").write_text("owned", encoding="utf-8")
    identity = plotting._directory_path_identity(
        staging, role="plot staging directory"
    )
    real_open = plotting._open_directory_identity
    foreign: Path | None = None
    owned_away: Path | None = None

    def replace_quarantine_before_open(
        path: Path, *, role: str
    ) -> tuple[int, tuple[int, int]]:
        nonlocal foreign, owned_away
        if ".mace-llpr-retired-" in path.name:
            owned_away = path.with_name(f"{path.name}.owned-away")
            path.rename(owned_away)
            path.mkdir()
            (path / "value.txt").write_text("foreign", encoding="utf-8")
            foreign = path
        return real_open(path, role=role)

    monkeypatch.setattr(
        plotting, "_open_directory_identity", replace_quarantine_before_open
    )

    plotting._best_effort_remove_owned_directory(staging, identity)

    assert foreign is not None
    assert owned_away is not None
    assert (foreign / "value.txt").read_text(encoding="utf-8") == "foreign"
    assert (owned_away / "value.txt").read_text(encoding="utf-8") == "owned"
    assert not staging.exists()


def test_owned_cleanup_retries_without_deleting_name_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import plotting

    staging = tmp_path / ".plots.stale-controlled"
    staging.mkdir()
    (staging / "value.txt").write_text("owned", encoding="utf-8")
    identity = plotting._directory_path_identity(
        staging, role="plot staging directory"
    )
    retired_prefix = f".mace-llpr-retired-{staging.name.lstrip('.')}-"
    collision = tmp_path / f"{retired_prefix}collision"
    collision.mkdir()
    (collision / "value.txt").write_text("foreign", encoding="utf-8")
    tokens = iter(("collision", "success"))
    monkeypatch.setattr(plotting.secrets, "token_hex", lambda size: next(tokens))

    plotting._best_effort_remove_owned_directory(staging, identity)

    assert not staging.exists()
    assert (collision / "value.txt").read_text(encoding="utf-8") == "foreign"
    retired = tmp_path / f"{retired_prefix}success"
    assert (retired / "value.txt").read_text(encoding="utf-8") == "owned"
