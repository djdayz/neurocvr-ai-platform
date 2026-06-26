# AWS EC2 deployment

NeuroCVR-AI was deployed as a FastAPI web service on an AWS EC2 Ubuntu instance.

## Deployed service

The FastAPI app exposes:

- GET /health
- POST /benchmarks/synthetic-glm
- POST /reports/synthetic-glm-qc
- GET /docs

## Deployment summary

The EC2 deployment used:

- Ubuntu Server
- Python virtual environment
- FastAPI
- Uvicorn
- public inbound TCP access on port 8000
- GitHub repository clone
- editable package installation

## App command

uvicorn neurocvr.api.app:app --host 0.0.0.0 --port 8000

## Background command

nohup .venv/bin/uvicorn neurocvr.api.app:app --host 0.0.0.0 --port 8000 > neurocvr.log 2>&1 &

## Health check

curl http://PUBLIC_EC2_IP:8000/health

Expected response:

{"status":"ok","service":"neurocvr-ai"}

## QC report endpoint

curl -X POST http://PUBLIC_EC2_IP:8000/reports/synthetic-glm-qc \
  -H "Content-Type: application/json" \
  -d '{
    "spatial_shape": [2, 2, 1],
    "n_timepoints": 220,
    "tr_seconds": 1.55,
    "tcnr": 5.0,
    "seed": 42,
    "delay_step_seconds": 2.0
  }'

## Portfolio relevance

This deployment demonstrates:

- cloud deployment of a Python ML/R&D service
- REST API serving with FastAPI
- Linux server setup
- inbound security group configuration
- reproducible package installation from GitHub
- technical QC report generation through an API

## Safety notice

This service is a research and portfolio prototype. It is not a clinical diagnostic tool and must not be used for patient care.
