# ============================================
# Makefile for LGTCN Project
# ============================================

.PHONY: help install install-dev sync lint test mlflow clean extract train evaluate all

# デフォルトはヘルプを表示
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install dependencies using uv"
	@echo "  install-dev   Install dev dependencies using uv"
	@echo "  sync          Sync dependencies using uv"
	@echo ""
	@echo "Development:"
	@echo "  lint          Run linting (ruff, mypy)"
	@echo "  test          Run tests with pytest"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  extract       Extract features from raw images"
	@echo "  train         Train the driving models"
	@echo "  evaluate      Evaluate trained models"
	@echo "  flocking      Run the flocking task"
	@echo ""
	@echo "Tools:"
	@echo "  mlflow        Start MLflow UI"
	@echo ""
	@echo "Utility:"
	@echo "  clean         Remove cache files"
	@echo "  all           Run full pipeline (extract → train)"

# ============================================
# Setup
# ============================================

install:
	uv sync --no-dev

install-dev:
	uv sync

sync:
	uv sync

# ============================================
# Development
# ============================================

lint:
	uv run ruff check src tests scripts
	uv run mypy src

test:
	uv run pytest

# ============================================
# Training & Evaluation
# ============================================

extract:
	uv run python scripts/extract_features.py

train:
	python3 scripts/train_driving.py --model ltcn
	python3 scripts/train_driving.py --model node

evaluate:
	python3 scripts/evaluate_driving.py --model ltcn --data-dir ./data/raw --model-path ./driving_results/LTCN_checkpoint.pth
	python3 scripts/evaluate_driving.py --model node --data-dir ./data/raw --model-path ./driving_results/NODE_checkpoint.pth


flocking:
	uv run python -m src.flocking.run

evaluate-corruption:
	uv run python scripts/evaluate_corruption_robustness.py \
		--data-dir ./data/raw \
		--ltcn-model-path ./driving_results/LTCN_checkpoint.pth \
		--node-model-path ./driving_results/NODE_checkpoint.pth \
		--corruption-type bias \
		--levels 0.0,0.1,0.2,0.3

# ============================================
# Tools
# ============================================

mlflow:
	uv run mlflow ui --port 5001

# ============================================
# Utility
# ============================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

all: train evaluate
