SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV  ?= uv
PY  ?= python
RUFF ?= ruff

LABELED_DATASET_PATH ?= ./datasets/CEFlows/mc_labeled/
UNLABELED_DATASET_PATH ?= ./datasets/CEFlows/unlabeled/
LOG_DIR      ?=

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
	$(UV) sync

test: 
	$(UV) run pytest

clean: 
	rm -rf .pytest_cache **/__pycache__ *.pyc

sim-bin: 
	bash src/core/run_sim_bin.sh

sim-mc: 
	bash src/core/run_sim_mc.sh

xseciot: 
	bash src/core/run_xseciot.sh

bin-test: 
	PYTHONPATH='.' caffeinate $(UV) run $(PY) -m src.core.ce_simulation \
		datasets/CETrain/combined_data.csv datasets/CEFlows/CE_MC_Flows_labeled_merged.csv \
	 	--log2File --modelVariant "feedforward" --ceType "approx_tce" --max_rows 100000 \
	 	--useCircularLogger --debug --useMLP --useAC 

mc-test: 
	PYTHONPATH='.' caffeinate $(UV) run $(PY) -m src.core.ce_simulation \
		datasets/CETrain/combined_data.csv datasets/CEFlows/CE_MC_Flows_labeled_merged.csv \
	 	--log2File --modelVariant "feedforward" --ceType "approx_tce" --max_rows 100000 \
	 	--useCircularLogger --debug --useMLP --useAC --modelType "multi"

bin-label: 
	@if [ -z "$(UNLABELED_DATASET_PATH)" ]; then echo "ERROR: set UNLABELED_DATASET_PATH=..."; exit 1; fi
	$(UV) run $(PY) -m src.utils.bin_labeling --dataset_path "$(UNLABELED_DATASET_PATH)"

mc-label: 
	@if [ -z "$(UNLABELED_DATASET_PATH)" ]; then echo "ERROR: set UNLABELED_DATASET_PATH=..."; exit 1; fi
	$(UV) run $(PY) -m src.utils.mc_labeling --dataset_path "$(UNLABELED_DATASET_PATH)" 

merge:
	@if [ -z "$(LABELED_DATASET_PATH)" ]; then echo "ERROR: set LABELED_DATASET_PATH=..."; exit 1; fi
	$(UV) run $(PY) -m src.utils.merge --dataset_path "$(LABELED_DATASET_PATH)"

overall-perf: 
	@if [ -n "$(LOG_DIR)" ]; then \
		$(UV) run $(PY) -m src.utils.overall_perf_stats --log_dir "$(LOG_DIR)"; \
	else \
		$(UV) run $(PY) -m src.utils.overall_perf_stats; \
	fi

overall-scrape:
	@if [ -n "$(LOG_DIR)" ]; then \
		$(UV) run $(PY) -m src.utils.overall_stats_scraper --log_dir "$(LOG_DIR)"; \
	else \
		$(UV) run $(PY) -m src.utils.overall_stats_scraper; \
	fi

fmt: 
	uv run $(RUFF) format .

lint: 
	uv run $(RUFF) check .

lint-fix: ## Lint with ruff and apply safe auto-fixes
	uv run $(RUFF) check . --fix

lint-fix-unsafe: 
	uv run $(RUFF) check . --fix --unsafe-fixes

style: fmt lint