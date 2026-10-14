.PHONY: data local job jobs fetch census census-fetch ascii-share

# start_jobs.sh parts: machine, throughput, sizes, profile, versions
PARTS   ?= machine,throughput,sizes,profile
FLAVOR  ?= cpu-performance
TIMEOUT ?= 90m
IMAGE   := rust:1-bookworm
DATASET := hf-internal-testing/tokenizers-test-data
REVISION := 5f5c6524084c42a35c28c0184bea37ec2530fd7c
FILES := bert-base-uncased.json fixtures/models/modernbert-base.json fixtures/modalities/code_mixed.txt \
         $(foreach c,eng_Latn rus_Cyrl ell_Grek arb_Arab cmn_Hani,fixtures/lang/$(c).txt)
OVERRIDES := $(foreach v,MAIN BRANCH ROUNDS SECS SIZE_ROUNDS,$(if $($(v)),-e $(v)=$($(v))))
STAMP := $(shell date +%Y%m%d-%H%M%S)

data:
	hf download $(DATASET) $(FILES) --repo-type dataset --revision $(REVISION) --local-dir data

local: data
	mkdir -p local/$(STAMP)
	docker run --rm -e PARTS=$(PARTS) -e FLAVOR=local $(OVERRIDES) -v $(CURDIR)/data:/data:ro \
	  -v $(CURDIR)/benchmark/start_jobs.sh:/start_jobs.sh:ro $(IMAGE) bash /start_jobs.sh | tee local/$(STAMP)/log.txt
	python3 benchmark/analyze.py local

job:
	hf jobs run --detach --flavor $(FLAVOR) --timeout $(TIMEOUT) -e PARTS=$(PARTS) -e FLAVOR=$(FLAVOR) $(OVERRIDES) \
	  -v hf://datasets/$(DATASET):/data:ro $(IMAGE) bash -c "$$(cat benchmark/start_jobs.sh)"

jobs:
	$(MAKE) job FLAVOR=cpu-performance
	$(MAKE) job FLAVOR=cpu-performance PARTS=machine,versions

fetch:
	benchmark/job_logs.sh $(ID)

census:
	hf jobs uv run --detach --flavor cpu-upgrade --secrets HF_TOKEN census/census.py

census-fetch:
	python3 census/analyze.py $(ID)

ascii-share:
	python3 ascii-share/ascii_share.py
