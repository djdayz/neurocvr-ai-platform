from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def save_nifti_like(
    data: np.ndarray,
    reference_image: nib.Nifti1Image,
    output_path: str | Path,
) -> Path:
    """
    Save data as a NIfTI image using a reference image geometry.

    The affine and header are copied from the reference image.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output_image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32),
        affine=reference_image.affine,
        header=reference_image.header.copy(),
    )

    nib.save(output_image, str(path))

    return path
