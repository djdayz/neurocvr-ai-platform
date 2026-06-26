from fastapi.testclient import TestClient

from neurocvr.api.app import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "neurocvr-ai",
    }


def test_synthetic_glm_benchmark_endpoint() -> None:
    response = client.post(
        "/benchmarks/synthetic-glm",
        json={
            "spatial_shape": [2, 2, 1],
            "n_timepoints": 220,
            "tr_seconds": 1.55,
            "tcnr": 5.0,
            "seed": 42,
            "delay_step_seconds": 2.0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["run_name"] == "synthetic_glm_benchmark_api"
    assert data["spatial_shape"] == [2, 2, 1]
    assert data["n_timepoints"] == 220
    assert data["cvr_magnitude"]["n_voxels"] == 4
    assert data["delay"]["n_voxels"] == 4
    assert data["cvr_magnitude"]["rmse"] >= 0
    assert data["delay"]["rmse"] >= 0


def test_synthetic_glm_benchmark_endpoint_rejects_bad_request() -> None:
    response = client.post(
        "/benchmarks/synthetic-glm",
        json={
            "spatial_shape": [2, 2, 1],
            "n_timepoints": 0,
        },
    )

    assert response.status_code == 422
