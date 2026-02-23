# pipeliner

`pipeliner` is a YAML-driven pipeline multiplexer for experiment variants.

It resolves variation points, expands per-step input/output paths, and builds runnable step contracts.

## Current scope

This scaffold is prepared for packaging and extraction into its own repository.

Included:
- Python package layout (`src/`)
- CLI entrypoint (`pipeliner`)
- Contract model + JSON helpers
- YAML loader + template expansion
- Minimal run-plan builder
- Example `experiment_setup.yaml`
- Conda/dev bootstrap files

## Quickstart

```bash
cd pipeliner
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

pipeliner --help
pipeliner show --setup examples/experiment_setup.yaml --step A40_repair --set dataset_name=kickoff --set dataset_variant=circled
```

## Packaging

Build wheel/sdist:

```bash
python -m build
```

Upload (when ready):

```bash
twine upload dist/*
```

## Conda notes

For `conda-forge`, use `conda.recipe/meta.yaml` as a starting point and create a feedstock.
