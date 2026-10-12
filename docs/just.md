# just recipes

Run every recipe from the repository root. You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

`just setup` installs the Kandinsky pipeline into `.venv` (Python 3.13 or 3.14). vLLM-Omni, SGLang, and FastVideo each get a Python 3.12 environment under `.venvs/` and do not replace `.venv`. A generate or serve recipe stops and names the setup recipe when its environment is missing. A second setup reuses a checkout that is already on disk.

`$KANDINSKY_HOME` is the root for weights, logs, and runs. When it is unset, that root is `~/.cache/kandinsky`.

`just --list` prints the recipes.

| Recipe | Guide |
|---|---|
| `just setup`, `just generate`, `just generate-sr`, `just download` | [Quick start](quick_start.md) |
| `just setup-vllm`, `just generate-vllm`, `just serve-vllm` | [vLLM-Omni](quick_start_vllm.md) |
| `just setup-sglang`, `just generate-sglang`, `just serve-sglang` | [SGLang](quick_start_sglang.md) |
| `just setup-fastvideo`, `just generate-fastvideo` | [FastVideo](quick_start_fastvideo.md) |
