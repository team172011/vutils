import re
import secrets
import time
from typing import Annotated

import typer

from vast.rent import serve_command, ssh_run, wait_for_ssh


def restart(
    instance_id: Annotated[int, typer.Argument(help="Instance id from `rent`")],
    model: Annotated[str, typer.Option(help="Hugging Face model id")] = "Qwen/Qwen3.8-27B-FP8",
    name: Annotated[str, typer.Option(help="Model name exposed by the API")] = "coder",
    max_len: Annotated[int, typer.Option(help="Max context length")] = 131072,
    max_seqs: Annotated[int, typer.Option(help="Max concurrent sequences")] = 64,
    tool_call_parser: Annotated[str, typer.Option(help="vLLM tool call parser")] = "qwen3_coder",
    kv_cache_dtype: Annotated[str, typer.Option(help="KV cache dtype, e.g. fp8 for ~2x context")] = "",
):
    """Restart the vLLM server on a running instance with new settings (keeps instance, weights and api key)."""
    host, ssh_port = wait_for_ssh(instance_id, timeout=60)

    # keep the api key of the running server so agent configs stay valid
    running = ssh_run(host, ssh_port, "pgrep -af '[v]llm serve'").stdout
    m = re.search(r"--api-key (\S+)", running)
    token = m.group(1) if m else secrets.token_urlsafe(24)

    # kill and start in separate ssh calls, the pkill pattern must not appear in the start command
    ssh_run(host, ssh_port, "pkill -f '[v]llm serve'")
    for _ in range(30):
        if ssh_run(host, ssh_port, "pgrep -f '[v]llm serve'").returncode != 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError("old vLLM process did not stop")

    serve = serve_command(model, name, max_len, max_seqs, tool_call_parser, token, kv_cache_dtype)
    ssh_run(host, ssh_port, f"setsid nohup {serve} > /var/log/vllm.log 2>&1 < /dev/null &")
    print(f"vLLM restarting on instance {instance_id} (max-len {max_len}, max-seqs {max_seqs}) ...")

    check = f"curl -sf -m 5 localhost:8000/v1/models -H 'Authorization: Bearer {token}' > /dev/null"
    misses = 0
    for _ in range(120):
        if ssh_run(host, ssh_port, check).returncode == 0:
            print(f"Ready. api key: {token}" + ("" if m else "  (NEW key, update your agents)"))
            return
        alive = ssh_run(host, ssh_port, "pgrep -f '[v]llm serve'").returncode == 0
        misses = 0 if alive else misses + 1
        if misses >= 3:
            log = ssh_run(host, ssh_port, "tail -n 30 /var/log/vllm.log").stdout
            raise RuntimeError(f"vLLM process died. Last log lines:\n{log}")
        time.sleep(10)
    raise TimeoutError("vLLM server did not come up in time")


def main():
    typer.run(restart)
