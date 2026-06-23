from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from neurocvr.data.loaders import load_etco2_csv, load_nifti

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
