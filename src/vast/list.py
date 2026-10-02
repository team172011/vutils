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


def show():
    offers: list[Offer] = vast.search_offers(
        query='gpu_ram>=40 num_gpus=1 reliability>0.90 compute_cap>=800 gpu_name notin [CMP_170HX]',
        order='dph_total',
        limit=10,
    )
    for e in offers:
        print(f"{e['id']:>9}  {e['gpu_name']:<10} {e['gpu_ram'] / 1024:>3.0f} GB  {e['dph_total']:.2f} $/h")


def main():
    typer.run(show)


if __name__ == "__main__":
    main()