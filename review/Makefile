# SpatialCPA-v18 review — entry points. See README.md for what each one costs.
#
#   make verify        pins + configuration + path + scope tests (seconds, no torch needed)
#   make starmap-row   reproduce the STARmap v18 paper row and check it (~3 min CPU)
#   make selftest      the harness's own oracle/scramble/random check (~4-5 min CPU)
#   make envs          create the five conda environments
#   make tools         fetch SpatialZ (Zenodo) + isoST (GitHub) into benchmark-pbya/tools/
#   make manifest      rewrite MANIFEST.sha256 after an intentional change (then commit it)
#
# PYTHON is the harness interpreter (the bench_eval env, or the one-venv CPU lock).

PYTHON ?= python
V3 := benchmark-pbya-v3

.PHONY: verify starmap-row selftest envs tools manifest clean-reproduced

verify:
	$(PYTHON) -m pytest -q tests

starmap-row:
	PYTHON=$(PYTHON) scripts/reproduce_starmap_v18.sh

selftest:
	cd $(V3) && $(PYTHON) -m src.bench3.prepare_dataset --dataset starmap_visual_cortex
	cd $(V3) && $(PYTHON) -m src.bench3.selftest

envs:
	for e in bench_eval bench_spatialcpa bench_spatialz bench_feast bench_isost; do \
	  conda env create -f envs/$$e.yml || conda env update -f envs/$$e.yml; done

tools:
	scripts/fetch_tools.sh

manifest:
	scripts/write_manifest.sh

clean-reproduced:
	rm -rf reproduced
