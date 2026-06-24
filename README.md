# NeuroCVR-AI

Research ML engineering pipeline for BOLD-fMRI cerebrovascular reactivity (CVR) analysis.

This project builds an end-to-end medical imaging analysis workflow for:

- BOLD-fMRI and ETCO2 data loading
- ETCO2 preprocessing and interpolation
- BOLD signal preprocessing
- voxelwise GLM CVR magnitude and delay estimation
- synthetic CVR simulation
- benchmark evaluation against ground truth
- NIfTI output generation
- reproducible metric tracking to JSON and CSV
- command line execution

This is a research and portfolio prototype. It is not a clinical diagnostic tool and must not be used for patient care.

## Motivation

Cerebrovascular reactivity mapping estimates how strongly and how quickly cerebral blood vessels respond to a vasoactive CO2 stimulus.

This project is inspired by my MPhys thesis on simulation-based evaluation of CVR mapping in BOLD MRI using physics-informed neural networks. The current repository turns that research workflow into a cleaner ML engineering system with modular Python code, automated tests, CLI tools, synthetic benchmarking, NIfTI outputs, and experiment tracking.

## Current pipeline

Synthetic CVR ground truth
-> ETCO2 block regressor
-> Synthetic BOLD simulation
-> BOLD masking and baseline extraction
-> Voxelwise GLM with delay search
-> CVR magnitude and delay map reconstruction
-> Metric evaluation
-> NIfTI maps plus JSON/CSV metric tracking

## Repository structure

src/neurocvr/
  config/          future config handling
  cvr/             CVR estimation models
  data/            loaders and NIfTI writers
  evaluation/      metrics, benchmark, tracking
  models/          future ML/PINN models
  preprocessing/   ETCO2 and BOLD preprocessing
  reporting/       future report generation
  simulation/      synthetic CVR/BOLD dataset generation

tests/             pytest test suite
configs/           future YAML configs
data/              local demo/raw/processed data
outputs/           generated outputs, ignored by Git
docs/              future documentation
notebooks/         exploratory notebooks

## Installation

Create and activate a conda environment:

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

This produces:

outputs/benchmark/
  true_cvr_magnitude.nii.gz
  estimated_cvr_magnitude.nii.gz
  true_delay.nii.gz
  estimated_delay.nii.gz
  metrics.json
  metrics.csv

Example benchmark output:

CVR magnitude metrics
RMSE: 0.0318
MAE: 0.0228
Bias: 0.0043
PCC: 0.9963

Delay metrics
RMSE: 0.4151 s
MAE: 0.3527 s
Bias: 0.0124 s
PCC: 0.9825

## Implemented components

Data loading:
- NIfTI loading with nibabel
- ETCO2 CSV loading
- NIfTI writing using reference image geometry

ETCO2 preprocessing:
- CO2 percentage to mmHg conversion
- baseline estimation
- temporal shifting
- interpolation to BOLD acquisition time axis
- baseline-corrected ETCO2 regressor generation

BOLD preprocessing:
- 4D BOLD validation
- brain-mask application
- time-by-voxel matrix construction
- voxelwise and global baseline estimation
- percent signal change calculation
- voxel-to-volume reconstruction

CVR estimation:
- voxelwise GLM
- intercept, ETCO2 beta, and linear drift term
- fixed-delay fitting
- delay-search fitting using minimum SSR
- CVR magnitude estimation

Evaluation:
- RMSE
- MAE
- mean bias
- Pearson correlation coefficient
- valid voxel count
- JSON and CSV metric tracking

Simulation:
- synthetic ground-truth CVR magnitude maps
- synthetic ground-truth delay maps
- ETCO2 block-paradigm regressor
- GLM-style BOLD signal simulation
- approximate tCNR-controlled Gaussian noise

## Roadmap

Planned next steps:

- YAML configuration files for reproducible benchmark runs
- MLflow experiment tracking
- Docker containerisation
- GitHub Actions CI
- physics-informed neural network model module
- technical QC report generation

## Research prototype notice

This project is for research, education, and ML engineering portfolio demonstration only. It is not validated for clinical decision-making and must not be used as a medical device.
