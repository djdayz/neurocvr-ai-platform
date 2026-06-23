from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NiftiImage:
    """Container for a loaded NIfTI image and its numerical data."""

    path: Path
    image: nib.Nifti1Image
    data: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def is_4d(self) -> bool:
        return self.data.ndim == 4


def load_nifti(path: str | Path) -> NiftiImage:
    """
    Load a NIfTI image from disk.

    Parameters
    ----------
    path:
        Path to a .nii or .nii.gz file.

    Returns
    -------
    NiftiImage
        A small container with the path, nibabel image object, and image data.
    """
    nifti_path = Path(path)

    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    if not (
        nifti_path.name.endswith(".nii") or nifti_path.name.endswith(".nii.gz")
    ):
        raise ValueError(
            f"Expected a .nii or .nii.gz file, got: {nifti_path.name}"
        )

    image = nib.load(str(nifti_path))
    data = np.asarray(image.get_fdata(dtype=np.float32))

    return NiftiImage(path=nifti_path, image=image, data=data)


def load_etco2_csv(
    path: str | Path,
    time_column: str = "time",
    etco2_column: str = "etco2",
) -> pd.DataFrame:
    """
    Load an ETCO2 CSV file.

    Expected minimum columns:
    - time
    - etco2

    Later we will extend this to support your thesis-style gas trace format.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"ETCO2 CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {time_column, etco2_column}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required ETCO2 columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    output = df[[time_column, etco2_column]].copy()
    output = output.rename(
        columns={
            time_column: "time",
            etco2_column: "etco2",
        }
    )

    output["time"] = pd.to_numeric(output["time"], errors="raise")
    output["etco2"] = pd.to_numeric(output["etco2"], errors="raise")

    return output
