# NeuroCVR-AI Model Card / Project Card

## Project overview

NeuroCVR-AI is a medical imaging ML engineering prototype for cerebrovascular reactivity, or CVR, analysis from BOLD-fMRI and ETCO2 data.

The current implementation focuses on a synthetic benchmark pipeline where ground-truth CVR magnitude and delay maps are known. The system simulates BOLD-fMRI data, estimates CVR parameters using a voxelwise GLM with delay search, evaluates recovery against ground truth, logs experiments with MLflow, and generates technical QC reports.

This project is intended as a research engineering and portfolio prototype.

It is not a clinical diagnostic tool and must not be used for patient care.

## Intended use

Intended uses:

- Demonstrate a reproducible medical imaging ML/R&D pipeline
- Benchmark CVR magnitude and delay recovery on synthetic data
- Evaluate model behaviour under different tCNR noise conditions
- Serve benchmark and QC functionality through a FastAPI service
- Demonstrate experiment tracking, cloud deployment, and technical reporting

Not intended for:

- Clinical diagnosis
- Treatment planning
- Patient triage
- Real-time medical decision-making
- Use on patient data without formal validation and governance

## System inputs

The implemented pipeline supports:

- Synthetic CVR magnitude maps
- Synthetic delay maps
- Synthetic ETCO2 stimulus regressors
- Simulated BOLD-fMRI time series
- Optional NIfTI image loading/writing utilities
- Benchmark configuration parameters such as tCNR, seed, TR, and delay search range

## System outputs

The pipeline can produce:

- Estimated CVR magnitude maps
- Estimated delay maps
- Benchmark metrics
- JSON and CSV metric files
- MLflow experiment runs
- Technical QC reports
- Experiment summary plots
- Markdown experiment reports
- FastAPI JSON responses

## Current modelling approach

The current benchmark estimator is a voxelwise general linear model with delay search.

For each candidate delay, the ETCO2 regressor is shifted and fitted to the BOLD signal using ordinary least squares. The delay with the lowest residual error is selected per voxel.

The pipeline estimates:

- CVR magnitude
- Delay
- Residual error
- Benchmark metrics against known synthetic ground truth

## Evaluation metrics

The project uses:

- RMSE: root mean squared error
- MAE: mean absolute error
- Bias: mean signed error
- PCC: Pearson correlation coefficient
- Valid voxel count

Metrics are computed separately for:

- CVR magnitude recovery
- Delay recovery

## Experiment tracking

MLflow is used to track:

- tCNR
- random seed
- simulation configuration
- CVR magnitude metrics
- delay metrics
- run metadata
- safety tags

The project also exports MLflow sweep results into CSV tables and plots for portfolio-ready reporting.

## Data

The current benchmark uses synthetic data with known ground truth.

This is useful for engineering validation because the true CVR magnitude and delay values are known, allowing quantitative error measurement.

Synthetic data does not replace validation on real patient or research datasets.

## Known limitations

Current limitations:

- Synthetic data is simpler than real BOLD-fMRI data
- Motion, registration errors, physiological variability, scanner effects, and preprocessing artefacts are not fully modelled
- The current estimator is GLM-based rather than a trained deep learning model
- Performance on synthetic data does not guarantee performance on real clinical data
- The QC report is technical and should not be interpreted clinically
- No external clinical validation has been performed

## Safety considerations

This project must be treated as research software.

The outputs are not medical diagnoses.

Any future use with real patient data would require:

- ethical approval
- data governance review
- clinical collaborator oversight
- robust validation on real datasets
- uncertainty analysis
- bias and failure-mode analysis
- secure handling of medical data
- appropriate regulatory assessment

## Engineering features

The project demonstrates:

- src-layout Python package
- modular preprocessing, simulation, modelling, evaluation, and reporting code
- pytest test suite
- Ruff linting
- Typer CLI
- YAML configuration
- FastAPI service
- Dockerfile
- AWS EC2 deployment
- MLflow tracking
- automated report generation
- research-safety documentation

## Future development

Potential next steps:

- Add a PINN-based estimator
- Add supervised learning baselines
- Add MLflow model registry support
- Add richer synthetic physiological noise models
- Add real-data preprocessing adapters
- Add uncertainty quantification
- Add authentication for deployed APIs
- Add cloud deployment through container services
- Add CI/CD once GitHub Actions restrictions are resolved

## Ethical and clinical disclaimer

NeuroCVR-AI is a technical research and portfolio project.

It is not validated for clinical use.

It must not be used to diagnose, monitor, or treat any medical condition.
