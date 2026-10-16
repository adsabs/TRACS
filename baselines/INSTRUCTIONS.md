# TRACS benchmark baselines

Baseline inference scripts and analysis for the TRACS shared task. Companion to
https://github.com/adsabs/TRACS. Scripts load the `adsabs/TRACS` dataset using Hugging Face Datasets.

## Setup and execution

Use Python 3 in an isolated environment and install `requirements.txt`. Local model
scripts require suitable hardware and model access; API scripts require credentials
in environment variables (`GEMINI_API_KEY` or `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`).
Never commit credential values.

From this directory, a small Gemini smoke run is:

```bash
python tracs_gemini.py --limit 2 --no_wandb --output_dir smoke_outputs
```

This sends dataset text to the configured API and may incur charges. Other examples
and options are in the individual scripts. Use `--model_id` explicitly when
reproducing a run. Several older scripts enable W&B logging by default.

## Analysis and included predictions

Start Jupyter from this directory and open `tracs_analysis.ipynb`. Alternatively,
set `TRACS_BASE_DIR` to this directory before starting Jupyter. Analysis outputs
are generated in `analysis_outputs/`; saved notebook outputs are cleared.

The included CSVs are exactly those referenced by the notebook's model registry,
for GPT, Llama 3.1, Mistral, Zephyr, Llama 4, and Gemini. Historical directory names
are preserved so the notebook resolves its inputs. These are existing predictions,
not newly validated official results. Claude inference code is included, but Claude
is not currently registered in the analysis notebook.

## Reproducibility notes before publication

Record the actual model IDs, API model snapshots/run dates, commands, dependency
versions, and hardware used for reported results. `requirements.txt` lists packages
but does not reconstruct historical environments. The Solar script currently defaults
to a Llama model; supply the actual model ID used. Claude's example and default model
IDs differ. Both Llama 4 implementations are retained because the provenance of the
published run has not yet been established. Do not infer run settings from defaults.

The evaluation currently joins predictions to ground truth by ID using an inner join;
check coverage and duplicate IDs before treating metrics as official benchmark scores.

Add the project's approved license and citation information before publication.
