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
    gpu_mem_bw: float  # GPU memory bandwidth in GB/s, limits the generation speed


QUERY = 'num_gpus>=1 reliability>0.90 compute_cap>=800 gpu_name notin [CMP_170HX] inet_down>=1000'


def find_offers(limit: int = 6) -> list[Offer]:
    """Cheapest matching offers, used by `rent` without an offer id. Only cards with >= 48 GB per GPU,
    so that the default FP8 model fits."""
    return vast.search_offers(query=f'gpu_ram>=48 {QUERY}', order='dph_total', limit=limit)


def cheapest_per_gpu_type() -> list[Offer]:
    """Cheapest offer of every GPU type, including the RTX 5090 (32 GB, needs a 4-bit model)."""
    offers = find_offers(limit=100)
    offers += vast.search_offers(query=f'gpu_name=RTX_5090 num_gpus=1 {QUERY}', order='dph_total', limit=5)
    best: dict[tuple, Offer] = {}
    for e in offers:
        key = (e['gpu_name'], e['num_gpus'])
        if key not in best or e['dph_total'] < best[key]['dph_total']:
            best[key] = e
    return sorted(best.values(), key=lambda e: e['dph_total'])


def format_offer(e: Offer) -> str:
    n = e['num_gpus']
    per_gpu = e['gpu_ram'] / 1024
    return (
        f"{e['id']:>9}  {n}x {e['gpu_name']:<10} {per_gpu:>3.0f} GB ({n * per_gpu:>3.0f} GB total)"
        f"  {e['dph_total']:.2f} $/h  {e.get('gpu_mem_bw', 0):>5.0f} GB/s mem  {e['inet_down']:>5.0f} Mb/s down"
    )


def show():
    """Cheapest offer of every GPU type. The memory bandwidth decides how fast tokens are generated."""
    for e in cheapest_per_gpu_type():
        print(format_offer(e))


def main():
    typer.run(show)


if __name__ == "__main__":
    main()