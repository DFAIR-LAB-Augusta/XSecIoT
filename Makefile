SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV  ?= uv
PY  ?= python

DATASET_PATH ?= ./datasets/CEFlows/unlabeled/
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
	@echo "  make bin-label DATASET_PATH=datasets/CEFlows/CEFlows2_merged.csv"
	@echo "  make merge     DATASET_PATH=datasets/CEFlows/indiv"
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

bin-label: 
	@if [ -z "$(DATASET_PATH)" ]; then echo "ERROR: set DATASET_PATH=..."; exit 1; fi
	$(UV) run $(PY) -m src.utils.bin_labeling --dataset_path "$(DATASET_PATH)"

mc-label: 
	@if [ -z "$(DATASET_PATH)" ]; then echo "ERROR: set DATASET_PATH=..."; exit 1; fi
	$(UV) run $(PY) -m src.utils.mc_labeling --dataset_path "$(DATASET_PATH)" 

merge:
	@if [ -z "$(DATASET_PATH)" ]; then echo "ERROR: set DATASET_PATH=..."; exit 1; fi
	$(UV) run $(PY) -m src.utils.merge --dataset_path "$(DATASET_PATH)"

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
