"""Small two-device probe: default all-reduce plus a second NCCL communicator."""
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29980",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    tensor = torch.full((1024,), rank + 1., device="cuda")
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    assert tensor[0].item() == 3
    print(f"rank {rank}: all-reduce OK", flush=True)
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=30))
    tensor.fill_(rank + 10.)
    dist.broadcast(tensor, src=0, group=group)
    torch.cuda.synchronize()
    assert tensor[0].item() == 10
    print(f"rank {rank}: second communicator broadcast OK", flush=True)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2, join=True)
