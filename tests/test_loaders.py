from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurocvr.data.loaders import load_etco2_csv, load_nifti


def test_load_nifti_4d(tmp_path: Path) -> None:
    data = np.zeros((4, 5, 6, 7), dtype=np.float32)
    affine = np.eye(4)
    image = nib.Nifti1Image(data, affine)

    path = tmp_path / "sub-01_task-cvr_bold.nii.gz"
    nib.save(image, path)

    loaded = load_nifti(path)

    assert loaded.shape == (4, 5, 6, 7)
    assert loaded.is_4d is True


def test_load_nifti_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.nii.gz"

    with pytest.raises(FileNotFoundError):
        load_nifti(missing_path)


def test_load_etco2_csv(tmp_path: Path) -> None:
    path = tmp_path / "etco2.csv"
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "etco2": [40.0, 41.0, 42.0],
        }
    )
    df.to_csv(path, index=False)

    loaded = load_etco2_csv(path)

    assert list(loaded.columns) == ["time", "etco2"]
    assert loaded.shape == (3, 2)
    assert loaded["etco2"].iloc[-1] == 42.0


def test_load_etco2_csv_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_etco2.csv"
    pd.DataFrame({"seconds": [0, 1], "co2": [40, 41]}).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_etco2_csv(path)
