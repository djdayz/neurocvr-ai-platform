from pathlib import Path

import nibabel as nib
import numpy as np

from neurocvr.data.writers import save_nifti_like


def test_save_nifti_like(tmp_path: Path) -> None:
    reference_data = np.zeros((2, 3, 4), dtype=np.float32)
    affine = np.eye(4)
    reference_image = nib.Nifti1Image(reference_data, affine)

    output_data = np.ones((2, 3, 4), dtype=np.float32)
    output_path = tmp_path / "cvr_magnitude.nii.gz"

    saved_path = save_nifti_like(
        data=output_data,
        reference_image=reference_image,
        output_path=output_path,
    )

    loaded = nib.load(str(saved_path))
    loaded_data = loaded.get_fdata()

    assert saved_path.exists()
    assert loaded.shape == (2, 3, 4)
    np.testing.assert_allclose(loaded.affine, affine)
    np.testing.assert_allclose(loaded_data, output_data)
