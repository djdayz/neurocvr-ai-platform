# NeuroCVR-AI Portfolio Summary

NeuroCVR-AI is an end-to-end medical imaging ML engineering project for BOLD-fMRI cerebrovascular reactivity analysis.

The project turns a research CVR workflow into a reproducible engineering system with a Python package, tests, CLI tools, synthetic benchmarking, FastAPI endpoints, cloud deployment, and technical QC reporting.

## Problem

Cerebrovascular reactivity mapping estimates how strongly and how quickly cerebral blood vessels respond to a vasoactive CO2 stimulus.

This project focuses on estimating:

- CVR magnitude
- CVR delay
- benchmark error against known synthetic ground truth
- technical QC status from model outputs

## Technical stack

- Python
- NumPy
- SciPy
- pandas
- scikit-learn
- nibabel
- pydantic
- Typer CLI
- FastAPI
- Uvicorn
- pytest
- Ruff
- YAML configs
- AWS EC2 deployment
- Dockerfile for containerisation

## Implemented ML/R&D pipeline

1. Generate synthetic CVR magnitude and delay maps
2. Generate ETCO2 block-paradigm stimulus
3. Simulate BOLD-fMRI data
4. Apply BOLD masking and baseline extraction
5. Run voxelwise GLM with delay search
6. Reconstruct estimated CVR magnitude and delay maps
7. Evaluate against ground truth using RMSE, MAE, bias, and PCC
8. Save benchmark metrics to JSON and CSV
9. Generate technical QC report
10. Serve benchmark and QC report through FastAPI

## Engineering features

- Modular src-layout Python package
- Automated pytest suite
- Ruff linting
- YAML-configured benchmark runs
- CLI entrypoints
- FastAPI REST service
- Dockerfile for cloud/container deployment
- AWS EC2 deployment documentation
- Research-safe technical QC report generation

## API endpoints

- GET /health
- POST /benchmarks/synthetic-glm
- POST /reports/synthetic-glm-qc
- GET /docs

## Example benchmark result

CVR magnitude recovery:

- RMSE: 0.0318
- MAE: 0.0228
- Bias: 0.0043
- PCC: 0.9963

Delay recovery:

- RMSE: 0.4151 s
- MAE: 0.3527 s
- Bias: 0.0124 s
- PCC: 0.9825

## Cloud deployment

The FastAPI app was deployed on AWS EC2 using:

- Ubuntu Server
- Python virtual environment
- GitHub repository clone
- editable package installation
- Uvicorn serving on 0.0.0.0:8000
- EC2 security group inbound rule for port 8000

## Relevance to R&D Engineer roles

This project demonstrates:

- scientific modelling
- medical imaging data handling
- simulation-based validation
- numerical methods
- signal processing
- Python package engineering
- technical documentation
- reproducible experimentation

## Relevance to Machine Learning Engineer roles

This project demonstrates:

- production-style Python structure
- tests and linting
- CLI tooling
- API serving
- cloud deployment
- Docker readiness
- benchmark tracking
- model evaluation metrics
- safe AI-assisted reporting design

## Safety scope

This project is a research and portfolio prototype. It is not a clinical diagnostic tool and must not be used for patient care.
