# pipeliner

`pipeliner` is a YAML-driven pipeline multiplexer for experiment variants.

It resolves variation points, expands step templates, and builds a runnable contract per step.

## Core idea

You define:
- `variation_points`: what can vary (dataset, algorithm, tiling, etc.)
- `process_steps`: ordered steps with templated `script`, `input`, `output`

Then `pipeliner` resolves one concrete combination and produces:
- fully expanded paths
- a `process_step` payload
- a single `--contract-json` argument for your step script
- a web GUI for selection + run/debug

Step scripts stay simple: parse contract JSON, validate inputs, run, write outputs/logs.

## Example: 3 steps, 2 variation points

`example_setup.yaml`:

```yaml
variation_points:
  dataset: [setA, setB]
  algo: [patchcore, dinomaly]

process_steps:
  S10_prepare:
    script: "steps/S10_prepare.py"
    input: "input/${dataset}"
    output: "pipeline_data/S10_prepare/${dataset}"

  S20_train:
    script: "steps/${algo}/S20_train.py"
    input: "pipeline_data/S10_prepare/${dataset}"
    output: "pipeline_data/${algo}/S20_train/${dataset}"

  S30_eval:
    script: "steps/${algo}/S30_eval.py"
    input: "pipeline_data/${algo}/S20_train/${dataset}"
    output: "pipeline_data/${algo}/S30_eval/${dataset}"
```

If you select `dataset=setA` and `algo=patchcore`, `S20_train` resolves to:
- script: `steps/patchcore/S20_train.py`
- input: `pipeline_data/S10_prepare/setA`
- output: `pipeline_data/patchcore/S20_train/setA`

## Contract model

The runner passes one JSON object:
- `variation_points`: selected concrete values
- `process_step`: expanded config for the current step
- `resolved`: convenience fields (resolved input/output)
- `runtime`: pipeline id, log paths, env metadata

Step entrypoint pattern:
1. parse `--contract-json`
2. read `resolved["input"]` / `resolved["output"]`
3. fail fast if required input is missing
4. create output dir and write artifacts

## Quickstart (this repo)

```bash
cd pipeliner
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

pipeliner --help
pipeliner list --setup examples/experiment_setup.yaml
pipeliner show --setup examples/experiment_setup.yaml --step A40_repair --set dataset_name=kickoff --set dataset_variant=circled --set algo_name=patchcore
pipeliner-web --setup examples/experiment_setup.yaml --host 127.0.0.1 --port 8765
```

## Packaging

Build wheel + sdist:

```bash
python -m build
```

Upload to PyPI:

```bash
twine upload dist/*
```

## Repo layout

```text
pipeliner/
  pyproject.toml
  src/pipeliner/
  tests/
  examples/
  conda.recipe/
```

`pyproject.toml` is the canonical dependency source for the package.
