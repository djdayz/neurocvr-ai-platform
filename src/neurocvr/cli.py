from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from neurocvr.data.loaders import load_etco2_csv, load_nifti
from neurocvr.preprocessing.etco2 import prepare_etco2_regressor

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

if __name__ == "__main__":
    app()
