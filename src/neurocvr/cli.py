from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import typer
from rich.console import Console

from neurocvr.cvr.glm import fit_glm_delay_search, shift_regressor_by_delay
from neurocvr.data.loaders import load_etco2_csv, load_nifti
from neurocvr.data.writers import save_nifti_like
from neurocvr.evaluation.metrics import compute_regression_metrics
from neurocvr.preprocessing.bold import (
    apply_brain_mask,
    compute_global_baseline,
    compute_voxel_baseline,
    create_full_brain_mask,
    reshape_voxel_values_to_volume,
)
from neurocvr.preprocessing.etco2 import prepare_etco2_regressor
from neurocvr.simulation.synthetic import simulate_glm_cvr_dataset

app = typer.Typer(help="NeuroCVR-AI command line interface.")
console = Console()


@app.command()
def inspect_nifti(path: Path) -> None:
    """Inspect a NIfTI image."""
    image = load_nifti(path)
    console.print(f"[bold green]Loaded:[/bold green] {image.path}")
    console.print(f"Shape: {image.shape}")
    console.print(f"4D image: {image.is_4d}")


@app.command()
def inspect_etco2(path: Path) -> None:
    """Inspect an ETCO2 CSV file."""
    df = load_etco2_csv(path)
    console.print(f"[bold green]Loaded:[/bold green] {path}")
    console.print(f"Rows: {len(df)}")
    console.print(f"Columns: {list(df.columns)}")
    console.print(df.head())

@app.command()
def prepare_etco2(
    path: Path,
    n_bold_volumes: int,
    tr_seconds: float = 1.55,
    global_shift_seconds: float = 0.0,
    baseline_end_seconds: float | None = None,
    input_units: str = "mmhg",
) -> None:
    """Prepare a baseline-corrected ETCO2 regressor on the BOLD time axis."""
    df = load_etco2_csv(path)

    trace = prepare_etco2_regressor(
        etco2_df=df,
        n_bold_volumes=n_bold_volumes,
        tr_seconds=tr_seconds,
        global_shift_seconds=global_shift_seconds,
        baseline_end_seconds=baseline_end_seconds,
        input_units=input_units,
    )

    console.print("[bold green]Prepared ETCO2 regressor[/bold green]")
    console.print(f"Input file: {path}")
    console.print(f"Number of BOLD volumes: {n_bold_volumes}")
    console.print(f"TR: {tr_seconds} s")
    console.print(f"Global shift: {global_shift_seconds} s")
    console.print(f"Samples: {trace.n_samples}")
    console.print(f"First 5 regressor values: {trace.etco2_mmhg[:5]}")

@app.command()
def inspect_bold(
    path: Path,
    baseline_volumes: int = 30,
) -> None:
    """
    Inspect a 4D BOLD image and compute simple baseline information.

    This demo command uses a full spatial mask. Real analysis should use a
    brain-extracted binary mask.
    """
    image = load_nifti(path)

    if not image.is_4d:
        raise typer.BadParameter("Expected a 4D BOLD NIfTI image.")

    mask = create_full_brain_mask(image.shape[:3])
    matrix = apply_brain_mask(image.data, mask)

    n_baseline = min(baseline_volumes, matrix.n_timepoints)
    voxel_baseline = compute_voxel_baseline(
        matrix.data,
        n_baseline_volumes=n_baseline,
    )
    global_baseline = compute_global_baseline(voxel_baseline)

    console.print("[bold green]Inspected BOLD image[/bold green]")
    console.print(f"Input file: {path}")
    console.print(f"Image shape: {image.shape}")
    console.print(f"Time points: {matrix.n_timepoints}")
    console.print(f"Voxels in mask: {matrix.n_voxels}")
    console.print(f"Baseline volumes used: {n_baseline}")
    console.print(f"Global baseline: {global_baseline:.4f}")


@app.command()
def glm_demo() -> None:
    """Run a tiny synthetic GLM CVR demo."""
    time = np.arange(10, dtype=float)
    etco2 = np.array([0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 5.0, 0.0, 0.0, 0.0])

    bold_baseline = 100.0
    true_delay = 2.0
    true_cvr = np.array([0.8, 1.2])

    delayed_etco2 = shift_regressor_by_delay(
        time_seconds=time,
        regressor=etco2,
        delay_seconds=true_delay,
    )

    true_beta = true_cvr / 100.0 * bold_baseline
    bold = bold_baseline + np.outer(delayed_etco2, true_beta)

    result = fit_glm_delay_search(
        bold_matrix=bold,
        etco2_regressor=etco2,
        time_seconds=time,
        bold_baseline=bold_baseline,
        delay_candidates_seconds=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    console.print("[bold green]Synthetic GLM CVR demo[/bold green]")
    console.print(f"True CVR values: {true_cvr}")
    console.print(f"Estimated CVR values: {result.cvr_magnitude}")
    console.print(f"True delay: {true_delay}")
    console.print(f"Estimated delays: {result.delay_seconds}")  


@app.command()
def glm_save_demo(output_dir: Path = Path("outputs")) -> None:
    """Run a synthetic GLM CVR demo and save CVR maps as NIfTI files."""
    shape = (2, 2, 1, 10)
    spatial_shape = shape[:3]

    time = np.arange(shape[-1], dtype=float)
    etco2 = np.array([0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 5.0, 0.0, 0.0, 0.0])

    bold_baseline = 100.0
    true_delay = 2.0

    true_cvr_volume = np.array(
        [
            [[0.8], [1.0]],
            [[1.2], [1.4]],
        ],
        dtype=float,
    )

    mask = create_full_brain_mask(spatial_shape)
    true_cvr_voxels = true_cvr_volume[mask]

    delayed_etco2 = shift_regressor_by_delay(
        time_seconds=time,
        regressor=etco2,
        delay_seconds=true_delay,
    )

    true_beta = true_cvr_voxels / 100.0 * bold_baseline
    bold_matrix = bold_baseline + np.outer(delayed_etco2, true_beta)

    bold_4d = np.zeros(shape, dtype=float)
    for time_index in range(shape[-1]):
        volume = reshape_voxel_values_to_volume(
            voxel_values=bold_matrix[time_index, :],
            mask=mask,
        )
        bold_4d[..., time_index] = volume

    reference_image = nib.Nifti1Image(
        bold_4d[..., 0].astype(np.float32),
        affine=np.eye(4),
    )

    result = fit_glm_delay_search(
        bold_matrix=bold_matrix,
        etco2_regressor=etco2,
        time_seconds=time,
        bold_baseline=bold_baseline,
        delay_candidates_seconds=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    cvr_volume = reshape_voxel_values_to_volume(
        voxel_values=result.cvr_magnitude,
        mask=mask,
    )
    delay_volume = reshape_voxel_values_to_volume(
        voxel_values=result.delay_seconds,
        mask=mask,
    )

    cvr_path = save_nifti_like(
        data=cvr_volume,
        reference_image=reference_image,
        output_path=output_dir / "demo_cvr_magnitude.nii.gz",
    )
    delay_path = save_nifti_like(
        data=delay_volume,
        reference_image=reference_image,
        output_path=output_dir / "demo_cvr_delay.nii.gz",
    )

    console.print("[bold green]Saved synthetic GLM CVR maps[/bold green]")
    console.print(f"CVR magnitude map: {cvr_path}")
    console.print(f"CVR delay map: {delay_path}")
    console.print(f"Estimated CVR values: {result.cvr_magnitude}")
    console.print(f"Estimated delays: {result.delay_seconds}")


@app.command()
def eval_demo() -> None:
    """Run a tiny synthetic CVR map evaluation demo."""
    true_cvr = np.array([0.8, 1.0, 1.2, 1.4])
    estimated_cvr = np.array([0.75, 1.05, 1.25, 1.35])

    metrics = compute_regression_metrics(
        y_true=true_cvr,
        y_pred=estimated_cvr,
    )

    console.print("[bold green]Synthetic CVR evaluation demo[/bold green]")
    console.print(f"RMSE: {metrics.rmse:.4f}")
    console.print(f"MAE: {metrics.mae:.4f}")
    console.print(f"Bias: {metrics.bias:.4f}")
    console.print(f"PCC: {metrics.pcc:.4f}")
    console.print(f"Valid voxels: {metrics.n_voxels}")


@app.command()
def simulate_demo(
    output_dir: Path = Path("outputs/simulation"),
    tcnr: float = 5.0,
    seed: int = 42,
) -> None:
    """Generate a small synthetic BOLD-CVR dataset and save it as NIfTI files."""
    dataset = simulate_glm_cvr_dataset(
        spatial_shape=(4, 5, 3),
        n_timepoints=60,
        tr_seconds=1.55,
        tcnr=tcnr,
        seed=seed,
    )

    bold_reference = nib.Nifti1Image(
        dataset.bold_4d.astype(np.float32),
        affine=np.eye(4),
    )
    map_reference = nib.Nifti1Image(
        dataset.bold_4d[..., 0].astype(np.float32),
        affine=np.eye(4),
    )

    bold_path = save_nifti_like(
        data=dataset.bold_4d,
        reference_image=bold_reference,
        output_path=output_dir / "synthetic_bold.nii.gz",
    )
    cvr_path = save_nifti_like(
        data=dataset.cvr_magnitude_map,
        reference_image=map_reference,
        output_path=output_dir / "synthetic_true_cvr_magnitude.nii.gz",
    )
    delay_path = save_nifti_like(
        data=dataset.delay_map,
        reference_image=map_reference,
        output_path=output_dir / "synthetic_true_delay.nii.gz",
    )

    console.print("[bold green]Generated synthetic CVR dataset[/bold green]")
    console.print(f"BOLD image: {bold_path}")
    console.print(f"True CVR map: {cvr_path}")
    console.print(f"True delay map: {delay_path}")
    console.print(f"Shape: {dataset.bold_4d.shape}")
    console.print(f"tCNR: {tcnr}")
    console.print(f"Seed: {seed}")


if __name__ == "__main__":
    app()
