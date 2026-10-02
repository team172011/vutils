import json
import os
import secrets
import shlex
import subprocess
import time
import urllib.request
from typing import Annotated
from urllib.parse import urlparse

import typer
from vastai import VastAI

vast = VastAI()  # uses VAST_API_KEY env var


def wait_for_ssh(instance_id: int, timeout: int) -> tuple[str, int]:
    """Wait until the instance is running and return its ssh (host, port)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = vast.show_instance(instance_id)
        if isinstance(inst, list):
            inst = inst[0] if inst else {}
        status = (inst or {}).get("actual_status")
        url = vast.ssh_url(instance_id) if status == "running" else ""
        if url:
            u = urlparse(url)
            return u.hostname, u.port
        print(f"  instance status: {status}")
        time.sleep(10)
    raise TimeoutError("instance did not become ready in time")


def ssh_run(host: str, ssh_port: int, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-p", str(ssh_port), "-o", "ConnectTimeout=10", f"root@{host}", command],
        capture_output=True, text=True,
    )


def wait_for_api(port: int, token: str, host: str, ssh_port: int, timeout: int) -> None:
    """Wait until the vLLM server answers through the tunnel (model download takes a while).

    Fails early with the log tail when the vLLM process on the instance died.
    """
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/models", headers={"Authorization": f"Bearer {token}"}
    )
    deadline = time.time() + timeout
    misses = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                json.load(r)
                return
        except Exception:
            pass
        # [v] keeps pgrep from matching the ssh command line itself
        alive = ssh_run(host, ssh_port, "pgrep -f '[v]llm serve'").returncode == 0
        misses = 0 if alive else misses + 1
        if misses >= 3:  # tolerate the short window before onstart has launched vllm
            log = ssh_run(host, ssh_port, "tail -n 30 /var/log/vllm.log").stdout
            raise RuntimeError(f"vLLM process died. Last log lines:\n{log}")
        time.sleep(15)
    raise TimeoutError("vLLM server did not come up in time")


def serve_command(
    model: str, name: str, max_len: int, max_seqs: int, tool_call_parser: str, token: str, kv_cache_dtype: str = "",
) -> str:
    kv = f"--kv-cache-dtype {kv_cache_dtype} " if kv_cache_dtype else ""
    return (
        f"vllm serve {shlex.quote(model)} --served-model-name {shlex.quote(name)} "
        f"--max-model-len {max_len} --max-num-seqs {max_seqs} --gpu-memory-utilization 0.92 {kv}"
        f"--enable-auto-tool-choice --tool-call-parser {tool_call_parser} "
        f"--enable-prefix-caching --host 127.0.0.1 --port 8000 --api-key {token}"
    )


def rent(
    offer_id: Annotated[int, typer.Argument(help="Offer id from `show`")],
    model: Annotated[str, typer.Option(help="Hugging Face model id")] = "Qwen/Qwen3.8-27B-FP8",
    name: Annotated[str, typer.Option(help="Model name exposed by the API")] = "coder",
    max_len: Annotated[int, typer.Option(help="Max context length")] = 131072,
    max_seqs: Annotated[int, typer.Option(help="Max concurrent sequences")] = 64,
    disk: Annotated[int, typer.Option(help="Disk size in GB")] = 80,
    port: Annotated[int, typer.Option(help="Local port for the tunnel")] = 8000,
    tool_call_parser: Annotated[str, typer.Option(help="vLLM tool call parser")] = "qwen3_coder",
    kv_cache_dtype: Annotated[str, typer.Option(help="KV cache dtype, e.g. fp8 for ~2x context")] = "",
    image: str = "vllm/vllm-openai:latest",
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
):
    """Rent an offer, start a vLLM OpenAI-compatible server and forward its port via ssh."""
    if not yes:
        typer.confirm(f"Rent offer {offer_id} and start billing?", abort=True)

    token = secrets.token_urlsafe(24)
    serve = serve_command(model, name, max_len, max_seqs, tool_call_parser, token, kv_cache_dtype)
    env = {"HF_TOKEN": os.environ["HF_TOKEN"]} if "HF_TOKEN" in os.environ else None
    result = vast.create_instance(
        offer_id,
        image=image,
        disk=disk,
        env=env,
        ssh=True,
        direct=True,
        label="vllm",
        onstart_cmd=f"nohup {serve} > /var/log/vllm.log 2>&1 &",
    )
    instance_id = result["new_contract"]
    print(f"Instance {instance_id} created")

    try:
        host, ssh_port = wait_for_ssh(instance_id, timeout=600)
        tunnel = subprocess.Popen([
            "ssh", "-N", "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-L", f"{port}:localhost:8000",
            f"root@{host}",
        ])
    except BaseException:
        print(f"Aborted. Instance {instance_id} is still running (and billing): "
              f"vastai destroy instance {instance_id}")
        raise

    try:
        print("Waiting for the model to download and the server to start ...")
        wait_for_api(port, token, host, ssh_port, timeout=1800)
        print(f"""
Ready. Instance {instance_id}
  base URL: http://localhost:{port}/v1
  api key:  {token}
  model:    {name}

Stop billing with: vastai destroy instance {instance_id}
Ctrl+C closes the tunnel (the instance keeps running).""")
        tunnel.wait()
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.terminate()
        print(f"Tunnel closed. Instance {instance_id} is still running: "
              f"vastai destroy instance {instance_id}")


def main():
    typer.run(rent)
