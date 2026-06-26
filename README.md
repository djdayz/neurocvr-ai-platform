# NeuroCVR-AI

End-to-end medical imaging ML engineering project for BOLD-fMRI cerebrovascular reactivity analysis.

This project turns a research CVR workflow into a reproducible machine learning engineering pipeline. It is designed to demonstrate skills relevant to R&D Engineer, Machine Learning Engineer, and Medical Imaging AI roles.

This is a research and portfolio prototype. It is not a clinical diagnostic tool and must not be used for patient care.

## Project goal

Cerebrovascular reactivity, or CVR, measures how strongly and how quickly cerebral blood vessels respond to a vasoactive CO2 stimulus.

This project estimates CVR magnitude and delay from BOLD-fMRI and ETCO2 data. It is inspired by my MPhys thesis on simulation-based evaluation of CVR mapping in BOLD MRI using physics-informed neural networks.

The current repository focuses on building the research workflow into a clean engineering system with modular Python code, tests, command line tools, simulation, benchmark evaluation, NIfTI outputs, and metric tracking.

## Current implemented pipeline

1. Generate synthetic ground-truth CVR magnitude and delay maps
2. Generate an ETCO2 block-paradigm regressor
3. Simulate synthetic BOLD-fMRI data
4. Apply BOLD masking and baseline extraction
5. Run voxelwise GLM with delay search
6. Reconstruct CVR magnitude and delay maps
7. Evaluate against ground truth
8. Save NIfTI maps and JSON/CSV metrics

## Implemented features

- Python package structure using src layout
- NIfTI loading and writing with nibabel
- ETCO2 CSV loading
- ETCO2 conversion from percent CO2 to mmHg
- ETCO2 baseline correction and interpolation
- BOLD 4D image validation
- Brain-mask application
- Time-by-voxel BOLD matrix construction
- Voxelwise and global BOLD baseline estimation
- Percent signal change calculation
- Voxelwise GLM fitting
- GLM delay search using minimum sum of squared residuals
- CVR magnitude estimation
- Synthetic CVR and BOLD data generation
- tCNR-controlled Gaussian noise simulation
- RMSE, MAE, bias, PCC, and valid voxel count metrics
- Benchmark metrics saved to JSON and CSV
- CLI commands for demos and benchmarks
- Pytest test suite
- Ruff linting

## Repository structure

src/neurocvr/
  cvr/             CVR estimation models
  data/            data loaders and NIfTI writers
  evaluation/      metrics, benchmark pipeline, and tracking
  models/          future ML and PINN models
  preprocessing/   ETCO2 and BOLD preprocessing
  reporting/       future QC and report generation
  simulation/      synthetic CVR and BOLD dataset generation

tests/             pytest test suite
configs/           future YAML config files
data/              local demo, raw, and processed data
outputs/           generated outputs ignored by Git
docs/              future documentation
notebooks/         exploratory notebooks

## Installation

Create and activate the conda environment:

conda create -n neurocvr python=3.11 -y
conda activate neurocvr

Install the package in editable mode:

pip install -e ".[dev]"

## Quality checks

Run tests:

pytest

Run linting:

ruff check .

Current status:

47 passing tests
Ruff checks passing

## CLI usage

Show available commands:

python -m neurocvr.cli --help

Run a small synthetic GLM demo:

neurocvr glm-demo

Generate synthetic BOLD-CVR data:

neurocvr simulate-demo --tcnr 5.0 --seed 42

Run the full synthetic GLM benchmark:

neurocvr benchmark-demo --tcnr 5.0 --seed 42

The benchmark writes outputs to:

outputs/benchmark/

Expected benchmark files:

- true_cvr_magnitude.nii.gz
- estimated_cvr_magnitude.nii.gz
- true_delay.nii.gz
- estimated_delay.nii.gz
- metrics.json
- metrics.csv

Example benchmark metrics:

CVR magnitude:
- RMSE: 0.0318
- MAE: 0.0228
- Bias: 0.0043
- PCC: 0.9963

Delay:
- RMSE: 0.4151 s
- MAE: 0.3527 s
- Bias: 0.0124 s
- PCC: 0.9825

## Engineering roadmap

Planned upgrades:

1. YAML config support for reproducible experiments
2. GitHub Actions CI for automated tests and linting
3. Docker containerisation
4. FastAPI service for benchmark and inference endpoints
5. Local API testing with curl
6. Cloud deployment prototype using Azure or AWS
7. GenAI-assisted technical QC report generation
8. MLflow experiment tracking
9. Physics-informed neural network or hybrid model module
10. Portfolio documentation and demo screenshots

## GenAI safety scope

The GenAI component will only generate technical QC summaries from metrics and pipeline outputs.

It will not provide diagnosis, treatment advice, or clinical decision-making.

Example intended GenAI output:

The benchmark run passed expected technical QC thresholds. CVR magnitude recovery was strong, with low RMSE and high PCC. Delay recovery showed sub-second RMSE in this synthetic setting. These results support technical pipeline validity on simulated data but do not imply clinical validity.

## Research prototype notice

This project is for research, education, and ML engineering portfolio demonstration only. It is not validated for clinical decision-making and must not be used as a medical device.

## FastAPI service

The project also exposes the synthetic CVR benchmark through a FastAPI web service.

Run the API locally:

uvicorn neurocvr.api.app:app --reload --port 8000

Health check:

curl http://127.0.0.1:8000/health

Expected response:

{"status":"ok","service":"neurocvr-ai"}

Run synthetic GLM benchmark through the API:

curl -X POST http://127.0.0.1:8000/benchmarks/synthetic-glm \
  -H "Content-Type: application/json" \
  -d '{
    "spatial_shape": [2, 2, 1],
    "n_timepoints": 220,
    "tr_seconds": 1.55,
    "tcnr": 5.0,
    "seed": 42,
    "delay_step_seconds": 2.0
  }'

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

The API is intended for technical benchmarking and research workflow demonstration only. It is not for clinical decision-making.

## Cloud deployment

The FastAPI service has been deployed and tested on AWS EC2 as a portfolio cloud deployment.

The deployment used:

- Ubuntu EC2 instance
- Python virtual environment
- FastAPI and Uvicorn
- public REST API endpoint
- inbound security group rule for port 8000

Deployment details are documented in:

- docs/aws_ec2_deployment.md
- docs/cloud_deployment.md
- docs/portfolio_summary.md

Cloud API endpoints:

- GET /health
- POST /benchmarks/synthetic-glm
- POST /reports/synthetic-glm-qc
- GET /docs

## Experiment sweep report

MLflow sweep results can be exported to portfolio-ready CSV tables and plots.

Run the benchmark sweep:

neurocvr benchmark-sweep-mlflow

Export the sweep report:

neurocvr export-mlflow-sweep

Generated files:

- reports/mlflow_sweep/sweep_runs.csv
- reports/mlflow_sweep/sweep_summary.csv
- reports/mlflow_sweep/cvr_rmse_vs_tcnr.png
- reports/mlflow_sweep/delay_rmse_vs_tcnr.png

These reports summarise how CVR magnitude and delay recovery change across tCNR noise levels.

## Automated experiment report

The exported MLflow sweep can be converted into a Markdown experiment report:

neurocvr write-report

Generated report:

- reports/mlflow_sweep/experiment_report.md

The report summarises the benchmark setup, best-performing tCNR conditions, summary metrics, plots, interpretation, engineering relevance, and safety limitations.

## Model card / project card

A safety-aware model card / project card is included at:

- docs/model_card.md

It describes intended use, non-clinical limitations, inputs, outputs, evaluation metrics, safety considerations, and future development.
