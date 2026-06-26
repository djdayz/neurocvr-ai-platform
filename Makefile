.PHONY: help install test lint format check api benchmark mlflow sweep export-sweep report full-report mlflow-ui clean

help:
	@echo "NeuroCVR-AI commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package in editable mode"
	@echo ""
	@echo "Quality:"
	@echo "  make test           Run pytest"
	@echo "  make lint           Run Ruff linting"
	@echo "  make format         Auto-fix Ruff issues"
	@echo "  make check          Format, test, and lint"
	@echo ""
	@echo "API:"
	@echo "  make api            Run FastAPI app locally"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark      Run one synthetic GLM benchmark"
	@echo "  make mlflow         Run one MLflow-tracked benchmark"
	@echo "  make sweep          Run MLflow benchmark sweep"
	@echo "  make export-sweep   Export MLflow sweep CSVs and plots"
	@echo "  make report         Write Markdown experiment report"
	@echo "  make full-report    Run sweep, export results, and write report"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow-ui      Open local MLflow UI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove local cache files"

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff check . --fix

check: format test lint

api:
	uvicorn neurocvr.api.app:app --reload --port 8000

benchmark:
	neurocvr benchmark-demo --tcnr 5.0 --seed 42

mlflow:
	neurocvr benchmark-mlflow --tcnr 5.0 --seed 42

sweep:
	neurocvr benchmark-sweep-mlflow

export-sweep:
	neurocvr export-mlflow-sweep

report:
	neurocvr write-report

full-report: sweep export-sweep report

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlruns/mlflow.db --port 5000

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
