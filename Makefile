SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV  ?= uv
PY  ?= python
RUFF ?= ruff

LABELED_DATASET_PATH ?= ./datasets/CEFlows/mc_labeled/
UNLABELED_DATASET_PATH ?= ./datasets/CEFlows/unlabeled/
LOG_DIR      ?=

TREE_IGNORE := .venv|binary_models|logging|*pyc|tests|datasets|.pytest_cache|.ruff_cache|.git|assets|feature_engineering|.vscode

TARGET_IP ?= 192.168.1.0/24

.PHONY: help sync test clean \
        sim-bin sim-mc xseciot \
        bin-label merge \
        overall-perf overall-scrape

help: 
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make sync"
	@echo "  make test"
	@echo "  make sim-bin"
	@echo "  make sim-mc"
	@echo "  make xseciot"
	@echo "  make bin-label UNLABELED_DATASET_PATH=datasets/CEFlows/CEFlows2_merged.csv"
	@echo "  make merge     LABELED_DATASET_PATH=datasets/CEFlows/indiv"
	@echo "  make overall-perf   LOG_DIR=logs"
	@echo "  make overall-scrape LOG_DIR=logs"

sync: 
	$(UV) sync --group lambda-torch

test: 
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	$(UV) run pytest -q

test.cov:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	$(UV) run pytest --cov=src --cov-report=term-missing --cov-report=xml

bin.test: 
	$(UV) run firce \
		datasets/CETrain/combined_data.csv \
		datasets/CEFlows2/CEFlows2_merged.csv \
	 	--log2File --modelVariant "feedforward" --ceType "approx_tce" --max_rows 100000 \
	 	--useCircularLogger --debug --useMLP --useAC \
		--pipeline simulation

bin.unsw:
	$(UV) run firce \
		datasets/UNSW_NB15/NF-UNSW-NB15-v3.csv \
		datasets/CEFlows2/CEFlows2_merged.csv \
	 	--log2File --modelVariant "feedforward" --ceType "approx_tce" --max_rows 100000 \
	 	--useCircularLogger --debug --useMLP --useAC --unsw \
		--pipeline simulation

label: 
	@if [ -z "$(UNLABELED_DATASET_PATH)" ]; then echo "ERROR: set UNLABELED_DATASET_PATH=..."; exit 1; fi
	$(UV) run scripts/labeling.py --dataset_path "$(UNLABELED_DATASET_PATH)"

merge:
	@if [ -z "$(LABELED_DATASET_PATH)" ]; then echo "ERROR: set LABELED_DATASET_PATH=..."; exit 1; fi
	$(UV) run scripts/merge.py --dataset_path "$(LABELED_DATASET_PATH)"

overall.perf: 
	@if [ -n "$(LOG_DIR)" ]; then \
		$(UV) run scripts/overall_perf_stats.py --log_dir "$(LOG_DIR)"; \
	else \
		$(UV) run scripts/overall_perf_stats.py
	fi

overall.scrape:
	@if [ -n "$(LOG_DIR)" ]; then \
		$(UV) run scripts/overall_stats_scraper.py --log-dir "$(LOG_DIR)"; \
	else \
		$(UV) run scripts/overall_stats_scraper.py --log-dir ./logging/ac; \
	fi

lint: 
	uv run $(RUFF) format .
	uv run $(RUFF) check . --fix

style: fmt lint

tree: ## Print repo tree (ignoring common dirs)
	tree -a --dirsfirst -I "$(TREE_IGNORE)" .

update-cade:
	$(UV) remove cade-firce || true
	$(UV) add "cade-firce @ git+ssh://git@github.com/DFAIR-LAB-Augusta/CADE_FIRCE.git@dev"

build: ## Build sdist/wheel
	$(UV) build

clean: ## Remove build & test artifacts
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + && rm -rf .pytest_cache
	
preflight: ## Build + run twine metadata checks
	$(UV) build
	$(UV) tool run twine check dist/*

deps.check: ## Check for dependency issues
	$(UV) run deptry .

scan: ## Basic Nmap scans of devices
	$(UV) run scripts/scanner.py \
		--targets "$(TARGET_IP)" \
		--exclude 192.168.1.1 \
		--output-dir ./logging/scans \
		--include-closed \
		--stages 1 2 3 \
		--aggressive \
		--sudo