from typing import TypedDict

import typer
from vastai import VastAI

vast = VastAI()  # uses VAST_API_KEY env var


class Offer(TypedDict, total=False):
    id: int
    gpu_name: str
    gpu_ram: float  # per GPU, in MB
    num_gpus: int
    dph_total: float  # $/hour
    inet_down: float  # download speed in Mb/s


def find_offers(limit: int = 6) -> list[Offer]:
    """Cheapest matching offers, used by `show` and by `rent` without an offer id."""
    return vast.search_offers(
        query='gpu_ram>=48 num_gpus>=1 reliability>0.90 compute_cap>=800 gpu_name notin [CMP_170HX] inet_down>=1000',
        order='dph_total',
        limit=limit,
    )


def format_offer(e: Offer) -> str:
    n = e['num_gpus']
    per_gpu = e['gpu_ram'] / 1024
    return (
        f"{e['id']:>9}  {n}x {e['gpu_name']:<10} {per_gpu:>3.0f} GB ({n * per_gpu:>3.0f} GB total)"
        f"  {e['dph_total']:.2f} $/h  {e['inet_down']:>5.0f} Mb/s down"
    )


def show():
    for e in find_offers():
        print(format_offer(e))


def main():
    typer.run(show)


if __name__ == "__main__":
    main()