import json
import os
import re
import secrets
import shlex
import subprocess
import time
import urllib.request
from typing import Annotated, Optional
from urllib.parse import urlparse

import typer
from vastai import VastAI

from vast.list import find_offers, format_offer

vast = VastAI()  # uses VAST_API_KEY env var
T0 = time.time()


def log(msg: str) -> None:
    """Print a progress line with the time since the start."""
    m, sec = divmod(int(time.time() - T0), 60)
    print(f"[{m:02d}:{sec:02d}] {msg}", flush=True)


def wait_for_ssh(instance_id: int, timeout: int) -> tuple[str, int]:
    """Wait until the instance is running and return its ssh (host, port)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        inst = vast.show_instance(instance_id)
        if isinstance(inst, list):
            inst = inst[0] if inst else {}
        status = (inst or {}).get("actual_status")
        url = vast.ssh_url(instance_id) if status == "running" else ""
        if url:
            u = urlparse(url)
            log(f"Instance is running, ssh at {u.hostname}:{u.port}")
            return u.hostname, u.port
        if status != last:
            log(f"Instance status: {status or 'starting'} (waiting for the host to pull the image and start the container)")
            last = status
        time.sleep(10)
    raise TimeoutError("instance did not become ready in time")


def ssh_run(host: str, ssh_port: int, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh", "-p", str(ssh_port),
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            f"root@{host}", command,
        ],
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
    last_line = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                json.load(r)
                return
        except Exception:
            pass
        # [v] keeps pgrep from matching the ssh command line itself; last log line shows the current step
        out = ssh_run(
            host, ssh_port,
            "pgrep -f '[v]llm serve' > /dev/null && echo ALIVE || echo DEAD; "
            "tail -c 600 /var/log/vllm.log 2>/dev/null | tr '\\r' '\\n' | grep -v '^$' | tail -n 1",
        ).stdout.splitlines()
        alive = bool(out) and out[0] == "ALIVE"
        misses = 0 if alive else misses + 1
        if misses >= 3:  # tolerate the short window before onstart has launched vllm
            tail = ssh_run(host, ssh_port, "tail -n 30 /var/log/vllm.log").stdout
            raise RuntimeError(f"vLLM process died. Last log lines:\n{tail}")
        line = re.sub(r"\x1b\[[0-9;]*m", "", out[1]).strip() if len(out) > 1 else ""
        if not alive:
            line = "waiting for vLLM to be started on the instance"
        if line and line != last_line:
            log(f"vLLM: {line[:140]}")
            last_line = line
        time.sleep(10)
    raise TimeoutError("vLLM server did not come up in time")


def serve_command(
    model: str, name: str, max_len: int, max_seqs: int, tool_call_parser: str, token: str, kv_cache_dtype: str = "",
    tensor_parallel: int = 0, reasoning_parser: str = "",
) -> str:
    # 0 = one shard per GPU of the instance, counted on the instance itself when the command runs
    tp = tensor_parallel or "$(nvidia-smi -L | wc -l)"
    rp = f"--reasoning-parser {reasoning_parser} " if reasoning_parser else ""
    kv = f"--kv-cache-dtype {kv_cache_dtype} " if kv_cache_dtype else ""
    return (
        f"vllm serve {shlex.quote(model)} --served-model-name {shlex.quote(name)} "
        f"--max-model-len {max_len} --max-num-seqs {max_seqs} --gpu-memory-utilization 0.92 --tensor-parallel-size {tp} {kv}"
        f"{rp}--enable-auto-tool-choice --tool-call-parser {tool_call_parser} "
        f"--enable-prefix-caching --host 127.0.0.1 --port 8000 --api-key {token}"
    )


def open_tunnel(host: str, ssh_port: int, port: int) -> subprocess.Popen:
    return subprocess.Popen([
        "ssh", "-N", "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-L", f"{port}:localhost:8000",
        f"root@{host}",
    ])


def api_reachable(port: int, token: str) -> bool:
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/models", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            json.load(r)
            return True
    except Exception:
        return False


def print_ready(instance_id: int, port: int, token: str, name: str, host: str, ssh_port: int, already: bool = False) -> None:
    state = "already running" if already else "ready"
    print(f"""
Ready. Instance {instance_id} {state}
  -> base URL: http://localhost:{port}/v1
  -> api key:  {token}
  -> model:    {name}
  -> ssh:      ssh -p {ssh_port} root@{host}
  -> log:      ssh -p {ssh_port} root@{host} tail -f /var/log/vllm.log

Stop billing with: destroy {instance_id}""")


def hold_tunnel(tunnel: subprocess.Popen, instance_id: int) -> None:
    print("Ctrl+C closes the tunnel (the instance keeps running).")
    try:
        tunnel.wait()
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.terminate()
        print(f"Tunnel closed. Instance {instance_id} is still running: "
              f"destroy {instance_id}")


def show_running(port: int) -> bool:
    """Print the running instances rented by this tool. Returns True and keeps the tunnel open
    when the user does not want another one."""
    running = [i for i in vast.show_instances() if i.get("label") == "vllm" and i.get("actual_status") == "running"]
    if not running:
        return False
    infos = []
    for inst in running:
        host, ssh_port = wait_for_ssh(inst["id"], timeout=60)
        procs = ssh_run(host, ssh_port, "pgrep -af '[v]llm serve'").stdout
        token = re.search(r"--api-key (\S+)", procs)
        name = re.search(r"--served-model-name (\S+)", procs)
        token = token.group(1) if token else "unknown (vLLM is not running on the instance)"
        name = name.group(1) if name else "unknown"
        print_ready(inst["id"], port, token, name, host, ssh_port, already=True)
        infos.append((inst["id"], host, ssh_port, token))
    if typer.confirm("\nCreate another one?", default=False):
        return False
    instance_id, host, ssh_port, token = infos[0]
    if api_reachable(port, token):
        print(f"A tunnel to instance {instance_id} is already open on port {port}.")
        return True
    log(f"Opening ssh tunnel to instance {instance_id} ...")
    hold_tunnel(open_tunnel(host, ssh_port, port), instance_id)
    return True


def rent(
    offer_id: Annotated[Optional[int], typer.Argument(help="Offer id from `show`, default: the cheapest offer of `show`")] = None,
    model: Annotated[str, typer.Option(help="Hugging Face model id")] = "Qwen/Qwen3.8-27B-FP8",
    name: Annotated[str, typer.Option(help="Model name exposed by the API")] = "coder",
    max_len: Annotated[int, typer.Option(help="Max context length")] = 131072,
    max_seqs: Annotated[int, typer.Option(help="Max concurrent sequences")] = 64,
    disk: Annotated[int, typer.Option(help="Disk size in GB")] = 80,
    port: Annotated[int, typer.Option(help="Local port for the tunnel")] = 8000,
    tool_call_parser: Annotated[str, typer.Option(help="vLLM tool call parser")] = "qwen3_coder",
    kv_cache_dtype: Annotated[str, typer.Option(help="KV cache dtype, e.g. fp8 for ~2x context")] = "",
    tensor_parallel: Annotated[int, typer.Option(help="Tensor parallel size, 0 = number of GPUs")] = 0,
    reasoning_parser: Annotated[str, typer.Option(help="vLLM reasoning parser, separates the thinking from the answer, empty = off")] = "qwen3",
    image: str = "vllm/vllm-openai:latest",
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
):
    """Rent an offer, start a vLLM OpenAI-compatible server and forward its port via ssh."""
    if show_running(port):
        return
    if offer_id is None:
        offers = find_offers(limit=1)
        if not offers:
            raise typer.BadParameter("no offer matches the query of `show`")
        print(f"Cheapest offer:\n{format_offer(offers[0])}")
        offer_id = offers[0]["id"]
    if not yes:
        typer.confirm(f"Rent offer {offer_id} and start billing?", abort=True)

    token = secrets.token_urlsafe(24)
    log(f"Renting offer {offer_id} ...")
    serve = serve_command(model, name, max_len, max_seqs, tool_call_parser, token, kv_cache_dtype, tensor_parallel, reasoning_parser)
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
    log(f"Instance {instance_id} created")

    try:
        host, ssh_port = wait_for_ssh(instance_id, timeout=600)
        log("Opening ssh tunnel ...")
        tunnel = open_tunnel(host, ssh_port, port)
    except BaseException:
        print(f"Aborted. Instance {instance_id} is still running (and billing): "
              f"destroy {instance_id}")
        raise

    try:
        log("Waiting for the model download and the vLLM start (shows the last vLLM log line when it changes) ...")
        wait_for_api(port, token, host, ssh_port, timeout=1800)
        print_ready(instance_id, port, token, name, host, ssh_port)
    except BaseException:
        tunnel.terminate()
        print(f"Aborted. Instance {instance_id} is still running (and billing): "
              f"destroy {instance_id}")
        raise
    hold_tunnel(tunnel, instance_id)


def main():
    typer.run(rent)
