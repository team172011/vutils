# Vast.ai utlis

Find cheap GPU offers on [vast.ai](https://vast.ai) and rent one as a self-hosted, OpenAI-compatible LLM server (vLLM) for coding agents like [opencode](https://github.com/anomalyco/opencode) or [Hermes Agent](https://github.com/NousResearch/hermes-agent).

> **Warning:** Rented GPUs keep running and are billed by the hour until you destroy them, also after `rent` or the terminal was closed (Ctrl+C only closes the ssh tunnel). Always destroy the instance when you are done: `destroy <instance id>`, and check the [vast.ai console](https://cloud.vast.ai/instances/) for instances you forgot.

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
Your public SSH key must be added to your vast.ai [account](https://cloud.vast.ai/manage-keys/), the tunnel to the instance uses it.

## Usage

### show
Lists the cheapest offer of every GPU type (>= 48 GB per GPU, Ampere or newer, >= 1 Gbit/s download, plus the 32 GB RTX 5090), sorted by price, to get a feeling for current prices and speeds:
```
show
```
Output: offer id, number and name of the GPUs, VRAM per GPU and in total, price per hour, memory bandwidth, download speed. The memory bandwidth limits how fast tokens are generated (roughly bandwidth / model size). The RTX 5090 only fits a 4-bit model (e.g. NVFP4). Edit the query in `src/vast/list.py` to change the filter. `rent` without an offer id takes the cheapest offer with >= 48 GB per GPU.

### rent
Rents the offer, starts a vLLM server with the model, waits until it is ready and forwards the port to `localhost` via an SSH tunnel:
```
rent                 # cheapest offer of `show`
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
| `--tensor-parallel-size` | `0` | GPUs the model is split over, `0` = number of GPUs of the instance (counted on the instance) |
| `--reasoning-parser` | `qwen3` | vLLM reasoning parser, puts the thinking into a separate field instead of the answer text, empty = off |
| `--kv-cache-dtype` | | e.g. `fp8` to roughly double the context that fits, not tested with this model |
| `--image` | `vllm/vllm-openai:latest` | docker image |
| `--agent` | | start a docker sandbox on this machine with the agent, preconfigured for this model (`opencode`) |
| `--workspace` | `~/vast-workspace` | directory shared with the `--agent` sandbox, kept when the sandbox is removed |
| `--yes` | | skip the confirmations |

If an instance rented with this tool is already running, `rent` prints its connection details (marked `already running`) and asks `Create another one? [y/N]`. With `N` it opens the ssh tunnel to the existing instance (unless one is already open) and keeps it until Ctrl+C, nothing new is rented.

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

### rent --agent opencode
With `--agent opencode`, `rent` additionally starts a sandbox container with [opencode](https://github.com/anomalyco/opencode) on this machine after the ssh tunnel is established and the model is ready. The container is preconfigured to use the model of the rented instance (through the tunnel), so no local opencode setup is needed. Docker must be installed and running (checked before anything is rented). Other agents may be added later, an unknown value lists the available ones.
```
rent --agent opencode
```
The connection details then include the sandbox:
```
agent:     docker exec -it vast-opencode opencode attach http://127.0.0.1:4096
agent web: http://localhost:4096 (user opencode, password <generated>)
```
Run the first command in a second terminal for the opencode TUI, or open the second URL in a browser and log in with the password. Both share their sessions.

- **Workspace:** the directory `~/vast-workspace` (option `--workspace`) is mounted into the container as `/workspace`, which is also the working directory of the agent. Files the agent writes there stay on this machine. **Everything outside of `/workspace` is lost** when the sandbox is removed, which happens when the tunnel closes (Ctrl+C) or when it is replaced. Put your repository into the workspace to work on existing code.
- **Existing sandbox:** if a sandbox is already running, `rent` asks before replacing it (its open sessions end, the workspace stays). `--yes` replaces it without asking. When the sandbox is kept it still uses the api key of the instance it was started for.
- **Reusing an instance:** running `rent --agent opencode` with an instance that is still running (answer `N` to "Create another one?") starts a new sandbox for it. If the tunnel is already open in another terminal, the sandbox stays after this `rent` exits, remove it with `docker rm -f vast-opencode`.
- **Failures:** if the sandbox cannot be started (docker, image, port 4096 in use), `rent` prints a warning and keeps the instance and the tunnel running.
- **Security:** the agent server only listens on `127.0.0.1` and asks for a generated password. The model's api key is stored in the container's environment, so it is visible with `docker inspect` on this machine. The agent can run commands and reach the network from inside the container, but cannot see the rest of this machine except the workspace. The image is pinned to a version, update it in `src/vast/sandbox.py`.
- **Linux:** the container reaches the tunnel via `host.docker.internal`, which only works with Docker Desktop (macOS, Windows). The tunnel is bound to `localhost`, on Linux with plain Docker the connection is refused. Not tested there.

### restart
Restarts the vLLM server on a running instance with new settings, e.g. a larger context for another agent. Instance, downloaded weights and the api key stay the same, only the model is reloaded (a few minutes):
```
restart <instance id> --max-len 65536
```
Options: `--model`, `--name`, `--max-len`, `--max-seqs`, `--tool-call-parser`, `--tensor-parallel-size`, `--reasoning-parser`, `--kv-cache-dtype` (same defaults as `rent`). Pass the same values as in `rent` for everything you do not want to change. Running requests and the open `rent` tunnel are interrupted only for the duration of the restart. If `--name` changes, the `--agent` sandbox keeps pointing to the old model name until `rent --agent <agent>` is run again.

### destroy
Destroys the instance, which stops the billing. The disk and the downloaded model are deleted with it:
```
destroy <instance id>
```
Asks for confirmation, `--yes` skips it.

**Billing:** Ctrl+C in `rent` only closes the tunnel, the instance keeps running and costing money. Run `destroy` when you are done.

If tool calls do not work, check the parser for your model. The server log is on the instance in `/var/log/vllm.log`.

## Development 
```
pipx install --force -e . 
```
