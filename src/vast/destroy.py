from typing import Annotated

import typer
from vastai import VastAI

vast = VastAI()  # uses VAST_API_KEY env var


def destroy(
    instance_id: Annotated[int, typer.Argument(help="Instance id from `rent`")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
):
    """Destroy an instance (stops billing, deletes the instance and its disk)."""
    if not yes:
        typer.confirm(f"Destroy instance {instance_id}? Everything on its disk is lost.", abort=True)
    vast.destroy_instance(instance_id)
    print(f"Instance {instance_id} destroyed")


def main():
    typer.run(destroy)
