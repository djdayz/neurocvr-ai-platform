# Cloud deployment guide

This document describes how to deploy NeuroCVR-AI as a FastAPI web service using Azure Container Apps.

The deployed service exposes:

- GET /health
- POST /benchmarks/synthetic-glm
- POST /reports/synthetic-glm-qc
- GET /docs

This is a research and portfolio prototype. It is not a clinical diagnostic tool.

## Target architecture

GitHub repository
-> Azure Container Apps cloud build
-> containerised FastAPI service
-> public HTTPS API endpoint
-> benchmark metrics and QC report responses

## Why Azure Container Apps

Azure Container Apps runs containerised applications on a serverless platform without requiring manual Kubernetes management. It is suitable for lightweight API services, demos, and ML engineering portfolio projects.

## Deployment prerequisites

You need:

- an Azure account
- access to Azure Cloud Shell or Azure CLI
- the GitHub repository URL
- the Dockerfile in the project root
- the FastAPI app entrypoint: neurocvr.api.app:app

## Repository

GitHub repository:

https://github.com/djdayz/neurocvr-ai-platform

## Local API command

The local development command is:

uvicorn neurocvr.api.app:app --reload --port 8000

## Container API command

The Dockerfile starts the API with:

uvicorn neurocvr.api.app:app --host 0.0.0.0 --port 8000

The Azure Container App should expose target port 8000.

## Azure deployment using az containerapp up

Suggested Azure resource names:

RESOURCE_GROUP=neurocvr-rg
LOCATION=uksouth
APP_NAME=neurocvr-api
ENV_NAME=neurocvr-env

Create the resource group:

az group create \
  --name neurocvr-rg \
  --location uksouth

Deploy from source:

az containerapp up \
  --name neurocvr-api \
  --resource-group neurocvr-rg \
  --location uksouth \
  --environment neurocvr-env \
  --source . \
  --ingress external \
  --target-port 8000

Get the public URL:

az containerapp show \
  --name neurocvr-api \
  --resource-group neurocvr-rg \
  --query properties.configuration.ingress.fqdn \
  --output tsv

## Test deployed service

Set the URL:

APP_URL=https://YOUR_AZURE_CONTAINER_APP_URL

Health check:

curl $APP_URL/health

Expected response:

{"status":"ok","service":"neurocvr-ai"}

Benchmark endpoint:

curl -X POST $APP_URL/benchmarks/synthetic-glm \
  -H "Content-Type: application/json" \
  -d '{
    "spatial_shape": [2, 2, 1],
    "n_timepoints": 220,
    "tr_seconds": 1.55,
    "tcnr": 5.0,
    "seed": 42,
    "delay_step_seconds": 2.0
  }'

QC report endpoint:

curl -X POST $APP_URL/reports/synthetic-glm-qc \
  -H "Content-Type: application/json" \
  -d '{
    "spatial_shape": [2, 2, 1],
    "n_timepoints": 220,
    "tr_seconds": 1.55,
    "tcnr": 5.0,
    "seed": 42,
    "delay_step_seconds": 2.0
  }'

Interactive API docs:

https://YOUR_AZURE_CONTAINER_APP_URL/docs

## Cost control

For a short portfolio demo, delete the resource group after testing:

az group delete \
  --name neurocvr-rg \
  --yes \
  --no-wait

Deleting the resource group removes the Azure resources created for this deployment.

## Portfolio notes

This deployment demonstrates:

- FastAPI web service
- containerised Python application
- cloud deployment readiness
- REST API endpoints
- reproducible ML benchmark execution
- technical QC report generation
- research-safe GenAI-style reporting scope
