# Vast.ai utlis

Find cheap GPU offers on [vast.ai](https://vast.ai) and rent one as a self-hosted, OpenAI-compatible LLM server (vLLM) for coding agents like [opencode](https://github.com/anomalyco/opencode) or [Hermes Agent](https://github.com/NousResearch/hermes-agent).

> **Warning:** Rented GPUs keep running and are billed by the hour until you destroy them, also after `rent` or the terminal was closed (Ctrl+C only closes the ssh tunnel). Always destroy the instance when you are done: `vastai destroy instance <instance id>`, and check the [vast.ai console](https://cloud.vast.ai/instances/) for instances you forgot.

## Installation
```
pipx install .
```

## Setup
Create an API key on vast.ai and set it for the CLI and the SDK:
```
vastai set api-key <KEY>
export VAST_API_KEY=<KEY>
```
Your public SSH key must be added to your vast.ai account, the tunnel to the instance uses it.

## Usage

### show
Lists the cheapest offers (single GPU, >= 40 GB VRAM, Ampere or newer, sorted by price) to get a feeling for current prices:
```
show
```
Output: offer id, GPU, VRAM per GPU, price per hour. Edit the query in `src/vast/list.py` to change the filter.

### rent
Rents the offer, starts a vLLM server with the model, waits until it is ready and forwards the port to `localhost` via an SSH tunnel:
```
rent <offer id>
rent <offer id> --model Qwen/Qwen3.8-27B-FP8 --max-len 131072
```

| Option | Default | |
|---|---|---|
| `--model` | `Qwen/Qwen3.8-27B-FP8` | Hugging Face model id |
| `--name` | `coder` | model name exposed by the API |
| `--max-len` | `131072` | max context length |
| `--disk` | `80` | disk size in GB |
| `--port` | `8000` | local port of the tunnel |
| `--tool-call-parser` | `qwen3_coder` | vLLM tool call parser, depends on the model |
| `--max-seqs` | `64` | max concurrent requests |
| `--kv-cache-dtype` | | e.g. `fp8` to roughly double the context that fits, not tested with this model |
| `--image` | `vllm/vllm-openai:latest` | docker image |
| `--yes` | | skip the confirmation |

For gated models set `HF_TOKEN` before running `rent`, it is passed to the instance.

When the server is up, `rent` prints the connection details:
```
base URL: http://localhost:8000/v1
api key:  <generated>
model:    coder
```
Use them in your agent as an OpenAI-compatible provider. 

Example for [opencode](https://github.com/anomalyco/opencode) (`opencode.json`):
```json
{
  "provider": {
    "vast": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8000/v1", "apiKey": "<generated>" },
      "models": { "coder": { "limit": { "context": 131072, "output": 8192 } } }
    }
  }
}
```
Set `limit.context` to the `--max-len` of the server. Without a limit opencode requests up to 32000 output tokens, which fails with "maximum context length" when the context is 32768 (or similar).

Example for [Hermes Agent](https://github.com/NousResearch/hermes-agent). `hermes setup` (custom OpenAI-compatible endpoint) writes the provider to `~/.hermes/config.yaml` and the key to `~/.hermes/.env`. The result looks like this:
```yaml
model:
  provider: vast
providers:
  vast:
    name: vast
    base_url: http://localhost:8000/v1
    model: coder
    discover_models: true
    models:
      coder:
        context_length: 131072
    key_env: HERMES_CUSTOM_VAST_API_KEY
    context_length: 131072
```
```
# ~/.hermes/.env
HERMES_CUSTOM_VAST_API_KEY=<generated>
```
Hermes requires a context of at least 64K, so the server must run with `--max-len` >= 65536 (the default of `rent` is 131072, which fits a 48 GB card for a single session) and `context_length` must match it.

### restart
Restarts the vLLM server on a running instance with new settings, e.g. a larger context for another agent. Instance, downloaded weights and the api key stay the same, only the model is reloaded (a few minutes):
```
restart <instance id> --max-len 65536
```
Options: `--model`, `--name`, `--max-len`, `--max-seqs`, `--tool-call-parser`, `--kv-cache-dtype` (same defaults as `rent`). Pass the same values as in `rent` for everything you do not want to change. Running requests and the open `rent` tunnel are interrupted only for the duration of the restart.

**Billing:** Ctrl+C only closes the tunnel, the instance keeps running and costing money. Destroy it when you are done:
```
vastai destroy instance <instance id>
```

If tool calls do not work, check the parser for your model. The server log is on the instance in `/var/log/vllm.log`.

## Development 
```
pipx install --force -e . 
```
