from pathlib import Path

import pytest
import yaml

from neurocvr.config.benchmark import (
    SyntheticBenchmarkConfig,
    load_benchmark_config,
    save_default_benchmark_config,
)


def test_default_benchmark_config() -> None:
    config = SyntheticBenchmarkConfig()

    assert config.spatial_shape == (4, 5, 3)
    assert config.n_timepoints == 220
    assert config.tr_seconds == 1.55
    assert config.tcnr == 5.0
    assert config.seed == 42


def test_load_benchmark_config(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"

    with path.open("w") as f:
        yaml.safe_dump(
            {
                "spatial_shape": [2, 3, 1],
                "n_timepoints": 100,
                "tr_seconds": 2.0,
                "tcnr": 0.5,
                "seed": 123,
                "output_dir": "outputs/test",
            },
            f,
        )

    config = load_benchmark_config(path)

    assert config.spatial_shape == (2, 3, 1)
    assert config.n_timepoints == 100
    assert config.tr_seconds == 2.0
    assert config.tcnr == 0.5
    assert config.seed == 123
    assert str(config.output_dir) == "outputs/test"


def test_load_benchmark_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_benchmark_config(tmp_path / "missing.yaml")


def test_benchmark_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        SyntheticBenchmarkConfig(n_timepoints=0)


def test_save_default_benchmark_config(tmp_path: Path) -> None:
    path = tmp_path / "default.yaml"

    saved_path = save_default_benchmark_config(path)

    assert saved_path.exists()

    loaded = load_benchmark_config(saved_path)

    assert loaded.n_timepoints == 220
    assert loaded.spatial_shape == (4, 5, 3)
