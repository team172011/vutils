"""Sandbox containers with a coding agent, preconfigured for the rented model.

The agent runs in a docker container on this machine and reaches the vLLM server on the rented
instance through the ssh tunnel on localhost. Only the workspace directory is shared with the
container, so the agent can write there and the files survive when the container is removed.
Add a new agent by subclassing Agent and registering it in AGENTS.
"""
import base64
import json
import secrets
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from vast.common import log

WORKDIR = "/workspace"  # mount point of the workspace directory inside the container
OUTPUT_LIMIT = 32768    # max output tokens per answer, thinking tokens count too


class Agent:
    name: str = ""            # value of rent --agent
    image: str = ""           # docker image, pinned to a version
    container: str = ""       # container name
    port: int = 0             # agent server port, mapped to localhost
    entrypoint: str = ""      # docker --entrypoint, empty = the image entrypoint
    server_args: tuple[str, ...] = ()  # the container's main command, keeps the container alive
    health_path: str = ""     # answers 200 while the agent server is up
    username: str = ""        # basic auth user of the agent server
    password: str = ""        # basic auth password, generated for every sandbox

    def env(self, base_url: str, token: str, model: str, max_len: int) -> dict[str, str]:
        """Env vars with everything the agent needs to reach the model."""
        return {}

    def use(self) -> str:
        """Command to start the agent in a terminal."""
        raise NotImplementedError


class OpenCode(Agent):
    name = "opencode"
    image = "ghcr.io/anomalyco/opencode:1.18.34"
    container = "vast-opencode"
    port = 4096
    # the alpine image has no git/bash, opencode needs git for its file snapshots
    entrypoint = "sh"
    server_args = ("-c",
                   "command -v apk >/dev/null && { apk add --no-cache git bash procps >/dev/null 2>&1 "
                   "|| echo 'WARNING: could not install git, opencode snapshots will not work' >&2; }; "
                   "exec opencode web --port 4096 --hostname 0.0.0.0")
    health_path = "/global/health"
    username = "opencode"

    def env(self, base_url: str, token: str, model: str, max_len: int) -> dict[str, str]:
        return {
            # the web UI and `opencode attach` (reads the same variable) ask for this password
            "OPENCODE_SERVER_PASSWORD": self.password,
            "OPENCODE_CONFIG_CONTENT": json.dumps({
                "$schema": "https://opencode.ai/config.json",
                "autoupdate": False,
                "model": f"vast/{model}",
                "provider": {
                    "vast": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": base_url, "apiKey": token},
                        "models": {model: {"limit": {"context": max_len,
                                                     "output": min(OUTPUT_LIMIT, max_len // 2)}}},
                    }
                },
            }),
        }

    def use(self) -> str:
        # `docker exec` inherits the container's env, so attach finds OPENCODE_SERVER_PASSWORD itself
        return f"docker exec -it {self.container} opencode attach http://127.0.0.1:{self.port}"


AGENTS: dict[str, Agent] = {"opencode": OpenCode()}


def docker(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kwargs)


def check_agent(name: Optional[str]) -> Optional[Agent]:
    """Validate the agent option (before billing starts) and that docker is usable."""
    if name is None:
        return None
    spec = AGENTS.get(name)
    if spec is None:
        raise typer.BadParameter(f"unknown agent {name!r} (available: {', '.join(sorted(AGENTS))})")
    if shutil.which("docker") is None:
        raise typer.BadParameter(f"--agent {name} needs docker, but it was not found on this machine")
    if docker("info").returncode != 0:
        raise typer.BadParameter(f"--agent {name} needs a running docker daemon, start docker and retry")
    return spec


def is_running(spec: Agent) -> bool:
    return docker("inspect", "-f", "{{.State.Running}}", spec.container).stdout.strip() == "true"


def container_password(spec: Agent) -> str:
    """Password of an already running sandbox, read from its environment."""
    env = docker("inspect", "-f", "{{json .Config.Env}}", spec.container).stdout
    try:
        for item in json.loads(env):
            if item.startswith("OPENCODE_SERVER_PASSWORD="):
                return item.split("=", 1)[1]
    except ValueError:
        pass
    return ""


def start_agent(spec: Agent, base_url: str, token: str, model: str, max_len: int, workspace: Path,
                yes: bool = False) -> Optional[Agent]:
    """Pull the image, start the sandbox container and wait until the agent answers.

    Returns None when the user keeps an already running sandbox. Raises when the start fails.
    """
    if is_running(spec):
        if not yes and not typer.confirm(
                f"\nSandbox {spec.container} is running. Replace it? Its open sessions end, "
                f"the files in {workspace} stay", default=False):
            spec.password = container_password(spec)
            print(f"Keeping the running sandbox, it still uses the api key of the instance it was started for:"
                  f"\n  -> agent:     {spec.use()}")
            return None
    workspace.mkdir(parents=True, exist_ok=True)
    spec.password = secrets.token_urlsafe(16)
    # a pinned image only needs to be pulled once
    if docker("image", "inspect", spec.image).returncode != 0:
        log(f"Pulling {spec.image} ...")
        if subprocess.run(["docker", "pull", spec.image]).returncode != 0:  # progress goes to the terminal
            raise RuntimeError(f"could not pull {spec.image} (is the network up?)")
    docker("rm", "-f", spec.container)  # a new sandbox per rent, the api key changes with every instance
    run = ["run", "-d", "--name", spec.container,
           "-p", f"127.0.0.1:{spec.port}:{spec.port}",
           "--add-host", "host.docker.internal:host-gateway",
           "-v", f"{workspace}:{WORKDIR}", "-w", WORKDIR]
    if spec.entrypoint:
        run += ["--entrypoint", spec.entrypoint]
    for key, value in spec.env(base_url, token, model, max_len).items():
        run += ["-e", f"{key}={value}"]
    run += [spec.image, *spec.server_args]
    log(f"Starting the {spec.name} sandbox container ...")
    try:
        r = docker(*run)
        if r.returncode != 0:
            raise RuntimeError(f"could not start the {spec.name} container:\n{r.stderr.strip()}")
        wait_agent_ready(spec)
    except BaseException:
        stop_agent(spec)
        raise
    return spec


def try_start_agent(spec: Optional[Agent], base_url: str, token: str, model: str, max_len: int,
                    workspace: Path, yes: bool = False) -> Optional[Agent]:
    """start_agent that only warns on failure, the rented instance and the tunnel must stay usable."""
    if spec is None:
        return None
    try:
        return start_agent(spec, base_url, token, model, max_len, workspace, yes)
    except Exception as e:
        print(f"\nWarning: the {spec.name} sandbox could not be started, the instance and the tunnel "
              f"are not affected:\n{e}")
        return None


def stop_agent(spec: Agent) -> None:
    """Remove the sandbox container (safe to call when it does not exist)."""
    docker("rm", "-f", spec.container)


def agent_lines(spec: Agent) -> str:
    return (f"  -> agent:     {spec.use()}\n"
            f"  -> agent web: http://localhost:{spec.port} (user {spec.username}, password {spec.password})")


def wait_agent_ready(spec: Agent, timeout: int = 120) -> None:
    url = f"http://127.0.0.1:{spec.port}{spec.health_path}"
    auth = base64.b64encode(f"{spec.username}:{spec.password}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                r.read()
                return
        except Exception:
            pass
        if not is_running(spec):  # crashed on start, no need to wait for the timeout
            raise RuntimeError(f"the {spec.name} container exited. Last log lines:\n"
                               f"{docker('logs', '--tail', '30', spec.container).stdout}"
                               f"{docker('logs', '--tail', '30', spec.container).stderr}")
        out = docker("logs", "--tail", "1", spec.container).stdout.strip()
        if out and out != last:
            log(f"{spec.name}: {out[:140]}")
            last = out
        time.sleep(2)
    raise TimeoutError(f"{spec.name} server did not come up in time. Last log lines:\n"
                       f"{docker('logs', '--tail', '30', spec.container).stdout}")
