"""Small real GPU broadcast probe, independent of model loading."""
import argparse
from datetime import timedelta
import json
import time

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--port", type=int, default=29984)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"tcp://{args.master}:{args.port}",
                            rank=args.rank, world_size=2, timeout=timedelta(seconds=45))
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=30))
    records = []
    for size in (1024 * 1024, 32 * 1024 * 1024, 256 * 1024 * 1024):
        for repeat in range(4):
            tensor = torch.full((size // 4,), repeat + 1 if args.rank == 0 else 0,
                                dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            dist.barrier()
            started = time.perf_counter()
            dist.broadcast(tensor, src=0, group=group)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            valid = bool(torch.all(tensor == repeat + 1).item())
            row = {"rank": args.rank, "bytes": size, "repeat": repeat,
                   "seconds": elapsed, "valid": valid, "first_communicator_use": size == 1024 * 1024 and repeat == 0}
            records.append(row)
            print(json.dumps(row), flush=True)
            if not valid:
                raise RuntimeError("received GPU payload mismatch")
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
