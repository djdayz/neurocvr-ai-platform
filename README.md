# NeuroCVR-AI

Cloud-native AI-assisted platform for BOLD-fMRI cerebrovascular reactivity (CVR) analysis.

This project builds an end-to-end medical imaging ML engineering pipeline for:
- BIDS/NIfTI/ETCO2 data ingestion
- BOLD-fMRI CVR magnitude and delay estimation
- GLM baseline modelling
- physics-informed / hybrid neural modelling
- simulation-based validation
- experiment tracking and model registry
- FastAPI inference service
- cloud deployment
- AI-assisted technical QC report generation

## Important

This is a research and portfolio prototype. It is not a clinical diagnostic tool and must not be used for patient care.

## Core methods

- BOLD-fMRI preprocessing
- ETCO2 alignment
- voxelwise GLM with lag optimisation
- tissue-wise simulation
- physics-informed neural modelling
- RMSE, MAE, PCC, SSIM, bias and Bland-Altman validation
- domain-shift and QC analysis