# pipeliner

`pipeliner` is a YAML-driven pipeline multiplexer for experiment variants.

It resolves variation points, expands step templates, and builds a runnable contract per step.

## Core idea

You define:
- `variation_points.values`: ordered variation definitions (`name`, `values` or `options`)
- `process_steps.values`: ordered step definitions (`name`, templated `script`, `input`, `output`)

Then `pipeliner` resolves one concrete combination and produces:
- fully expanded paths
- a `process_step` payload
- a single `--contract-json` argument for your step script
- a web GUI for selection + run/debug

Step scripts stay simple: parse contract JSON, validate inputs, run, write outputs/logs.

The full web UI (Viewer/Runner/Analysis, run/cancel, runner plan, trees) is now part of the package.
Consumer repos can keep a thin compatibility wrapper at `pipeline/config_webapp.py`.

## Example YAML schema (ordered list style)

`examples/experiment_setup.yaml`:

```yaml
variation_points:
  values:
    - name: dataset_name
      gui:
        display_name: Dataset
      options:
        - name: kickoff
          values: [flat, circled]
        - name: pretest_snaketool_leg4A
          values: [flat]

    - name: algo_name
      values: [patchcore, dinomaly]

process_steps:
  gui:
    enumerate: true
  values:
    - name: A30_mask
      script: "pipeline/manual_steps.py"
      kind: manual
      input: "pipeline_data/A20_cut_out/${dataset_name}/${dataset_variant}"
      output: "pipeline_data/A30_mask/${dataset_name}/${dataset_variant}/${mask_type}"

    - name: A40_repair
      script: "pipeline/${process_step}.py"
      kind: automated
      input_from_previous: true
      previous_step: A30_mask
      output: "pipeline_data/${process_step}/${repair_method}/${mask_type}/${dataset_name}/${dataset_variant}"
```

If you select `dataset_name=kickoff`, `dataset_variant=circled`, `algo_name=patchcore`:
- `script` templates are expanded with selected values
- relative paths are resolved from project root inferred from config path
- each step gets one `--contract-json` payload

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
pipeliner --port 8080 --config /path/to/opticap-qai/pipeline/experiment_setup.yaml

# Optional CLI helper commands
pipeliner-cli list --setup examples/experiment_setup.yaml
pipeliner-cli show --setup examples/experiment_setup.yaml --step A40_repair --set dataset_name=kickoff --set dataset_variant=circled --set algo_name=patchcore
```

Run from any working directory:
- pass absolute/relative `--config`
- project root defaults to config parent (or parent of `pipeline/` when config is `pipeline/experiment_setup.yaml`)

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
