# Vast.ai utlis

Find cheap GPU offers on [vast.ai](https://vast.ai) and rent one as a self-hosted, OpenAI-compatible LLM server (vLLM) for coding agents like opencode or Hermes.

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
rent <offer id> --model Qwen/Qwen3.8-27B-FP8 --max-len 65536
```

| Option | Default | |
|---|---|---|
| `--model` | `Qwen/Qwen3.8-27B-FP8` | Hugging Face model id |
| `--name` | `coder` | model name exposed by the API |
| `--max-len` | `32768` | max context length |
| `--disk` | `80` | disk size in GB |
| `--port` | `8000` | local port of the tunnel |
| `--tool-call-parser` | `qwen3_coder` | vLLM tool call parser, depends on the model |
| `--image` | `vllm/vllm-openai:latest` | docker image |
| `--yes` | | skip the confirmation |

For gated models set `HF_TOKEN` before running `rent`, it is passed to the instance.

When the server is up, `rent` prints the connection details:
```
base URL: http://localhost:8000/v1
api key:  <generated>
model:    coder
```
Use them in your agent as an OpenAI-compatible provider. Example for opencode (`opencode.json`):
```json
{
  "provider": {
    "vast": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8000/v1", "apiKey": "<generated>" },
      "models": { "coder": {} }
    }
  }
}
```

**Billing:** Ctrl+C only closes the tunnel, the instance keeps running and costing money. Destroy it when you are done:
```
vastai destroy instance <instance id>
```

If tool calls do not work, check the parser for your model. The server log is on the instance in `/var/log/vllm.log`.

## Development 
```
pipx install --force -e . 
```
