"""Opt-in synchronous hybrid-state transfer for the Hygon SGLang 0.5.12 image.

Runs inside the scheduler, only while idle. Both endpoints must use the same
checkpoint and TP layout. Destination must have an empty radix tree. This is
an initial correctness path, not the asynchronous migration implementation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.srt.managers.io_struct import HygonHybridReqOutput
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.utils import init_custom_process_group

logger = logging.getLogger(__name__)

class HybridControl:
    def __init__(self, scheduler):
        self.s = scheduler
        self.groups = {}
        # Scheduler metrics/control handling must not interleave differently
        # typed object collectives on the shared TP CPU communicator.
        self.control_cpu_group = dist.new_group(
            ranks=dist.get_process_group_ranks(scheduler.tp_cpu_group),
            backend="gloo",
        )

    def gather(self, value):
        values = [None] * self.s.tp_size
        dist.all_gather_object(values, value, group=self.control_cpu_group)
        return values

    def validate_runtime(self, require_idle=True):
        s = self.s
        if require_idle and not s.is_fully_idle():
            raise ValueError("hybrid transfer requires an idle scheduler")
        if type(s.tree_cache) is not MambaRadixCache or s.tree_cache.disable:
            raise ValueError("enabled MambaRadixCache required")
        if s.server_args.dp_size != 1 or s.server_args.pp_size != 1:
            raise ValueError("only DP=1, PP=1 supported")
        if s.server_args.speculative_algorithm:
            raise ValueError("speculative decoding not supported")
        if s.page_size != 1 or os.environ.get("SGLANG_KV_LAYOUT_HCU_FA") != "false":
            raise ValueError("page size 1 and standard KV layout required")

    @property
    def pool(self):
        return self.s.token_to_kv_pool_allocator.get_kvcache()

    def buffers(self):
        full = self.pool.full_kv_pool
        state = self.pool.mamba_pool.mamba_cache
        return ([(b, 0) for pair in zip(full.k_buffer, full.v_buffer) for b in pair]
                + [(b, 1) for b in state.conv] + [(state.temporal, 1)])

    def match(self, tokens):
        if not tokens or any(type(t) is not int or t < 0 or t >= self.s.model_config.vocab_size for t in tokens):
            raise ValueError("nonempty, valid token IDs required")
        return self.s.tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))

    def layout(self, count):
        return [{"shape": list(b.shape[:axis]) + [count if axis == 0 else 1] + list(b.shape[axis + 1:]),
                 "dtype": str(b.dtype), "axis": axis}
                for b, axis in self.buffers()]

    def observe_prefix(self, tokens):
        """Read existing nodes only: no match/split, lock, allocation or LRU touch."""
        if not tokens or any(type(t) is not int or t < 0 or t >= self.s.model_config.vocab_size for t in tokens):
            raise ValueError("nonempty, valid token IDs required")
        cache = self.s.tree_cache
        node = cache.root_node
        offset = 0
        boundaries = []
        while offset < len(tokens):
            key = RadixKey(tokens[offset:]).child_key(cache.page_size)
            child = node.children.get(key)
            if child is None or child.value is None:
                break
            edge = child.key.token_ids
            if not edge or len(child.value) != len(edge):
                break
            if len(edge) > len(tokens) - offset or tokens[offset:offset + len(edge)] != edge:
                break
            offset += len(edge)
            if child.mamba_value is not None and child.mamba_value.numel() == 1:
                boundaries.append(offset)
            node = child
        matched = max(boundaries, default=0)
        return {"success": True, "query_tokens": len(tokens), "token_count": len(tokens),
                "matched_tokens": matched, "snapshot_boundaries": boundaries,
                "full_prefix_available": matched == len(tokens), "has_mamba_state": bool(boundaries),
                "available_token_slots": int(self.s.token_to_kv_pool_allocator.available_size()),
                "evictable_token_slots": int(cache.full_evictable_size()),
                "available_snapshot_slots": int(self.pool.mamba_pool.available_size()),
                "evictable_snapshot_slots": int(cache.mamba_evictable_size()),
                "layout": self.layout(len(tokens))}

    def inspect(self, p):
        tokens = p["token_ids"]
        result = self.observe_prefix(tokens)
        if p.get("return_digest", False):
            if not result["full_prefix_available"]:
                raise ValueError("digest requires a complete executable prefix")
            match = self.match(tokens)
            torch.cuda.synchronize()
            digest = hashlib.sha256()
            nbytes = 0
            for buffer, axis in self.buffers():
                slots = match.device_indices if axis == 0 else match.last_device_node.mamba_value
                data = buffer.index_select(axis, slots).contiguous().view(torch.uint8).cpu().numpy().tobytes()
                digest.update(data)
                nbytes += len(data)
            result.update(state_sha256=digest.hexdigest(), state_bytes=nbytes)
        return result

    def init_group(self, p):
        rank, name = p["group_rank"], p["group_name"]
        if rank not in (0, 1) or p["world_size"] != 2 or p["backend"] != "nccl":
            raise ValueError("two-engine NCCL/RCCL groups required")
        if name in self.groups:
            raise ValueError("group name already in use")
        timeout = timedelta(seconds=90)
        port = p["port"] + self.s.tp_rank
        init_error = None
        try:
            store = dist.TCPStore(p["master_address"], port + self.s.tp_size, 2, rank == 0, timeout)
            model_root = Path(self.s.server_args.model_path)
            signature = hashlib.sha256((model_root / "config.json").read_bytes()
                                      + (model_root / "model.safetensors.index.json").read_bytes()).hexdigest()
            identity = {"model_config_and_index": signature, "tp_size": self.s.tp_size,
                        "tp_rank": self.s.tp_rank, "layout": self.layout(1), "name": name}
            store.set(str(rank), json.dumps(identity))
            peer = json.loads(store.get(str(1 - rank)))
        except Exception as exc:
            init_error = repr(exc)
            logger.exception("Hybrid group store/identity initialization failed")
        errors = self.gather(init_error)
        if any(errors):
            raise ValueError(str(errors))
        errors = self.gather(None if identity == peer else "peer model/layout mismatch")
        if any(errors):
            raise ValueError(str(errors))
        pg = init_custom_process_group(backend="nccl", init_method=f"tcp://{p['master_address']}:{port}",
                                       world_size=2, rank=rank, group_name=f"{name}-tp{self.s.tp_rank}",
                                       timeout=timeout)
        self.groups[name] = {"pg": pg, "store": store, "rank": rank, "sequence": 0, "port": p["port"]}
        return {"success": True, "group_name": name, "tp_size": self.s.tp_size}

    def transfer(self, p):
        group = self.groups[p["group_name"]]
        if group.get("failed"):
            raise ValueError("failed group cannot be reused")
        seq = group["sequence"]
        group["sequence"] += 1
        rank, store = group["rank"], group["store"]
        role = "source" if rank == 0 else "destination"
        tokens = p["token_ids"]
        indices = mamba = None
        installed = False
        error = None
        try:
            if p["role"] != role or p["port"] != group["port"]:
                raise ValueError("role/port differs from initialized group")
            match = self.match(tokens)
            if role == "source":
                if len(match.device_indices) != len(tokens) or match.last_device_node.mamba_value is None:
                    raise ValueError("exact completed hybrid boundary unavailable")
                indices, mamba = match.device_indices, match.last_device_node.mamba_value
            else:
                # Opt-in nonempty import. Reserve a complete private copy;
                # vendor insert() frees duplicate prefix indices atomically
                # while the scheduler is idle. No other cache is flushed.
                if self.s.tree_cache.root_node.children and os.environ.get("AGENTSHIFT_NONEMPTY_IMPORT") != "1":
                    raise ValueError("destination cache must be empty; flush before migration")
                indices = self.s.token_to_kv_pool_allocator.alloc(len(tokens))
                mamba = self.pool.mamba_pool.alloc(1)
                if indices is None or mamba is None:
                    raise ValueError("insufficient destination cache capacity")
        except Exception as exc:
            error = str(exc)
        errors = self.gather(error)
        metadata = {"tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    "layout": self.layout(len(tokens)), "errors": errors}
        key = f"{seq}/prepare/"
        try:
            store.set(key + str(rank), json.dumps(metadata))
            peer = json.loads(store.get(key + str(1 - rank)))
            if any(errors) or any(peer["errors"]) or metadata != peer:
                raise ValueError(f"transfer preparation rejected: local={errors}, peer={peer['errors']}")
            torch.cuda.synchronize()
            started = time.perf_counter()
            logger.info("Hybrid %s copying %d tokens on TP rank %d", role, len(tokens), self.s.tp_rank)
            nbytes = 0
            for buffer, axis in self.buffers():
                slots = indices if axis == 0 else mamba
                if role == "source":
                    tensor = buffer.index_select(axis, slots).contiguous()
                else:
                    shape = list(buffer.shape)
                    shape[axis] = len(slots)
                    tensor = torch.empty(shape, dtype=buffer.dtype, device=buffer.device)
                options = dist.BroadcastOptions()
                options.rootRank = 0
                options.rootTensor = 0
                group["pg"].broadcast([tensor], options).wait()
                if role == "destination":
                    buffer.index_copy_(axis, slots, tensor)
                nbytes += tensor.numel() * tensor.element_size()
            torch.cuda.synchronize()
            # All local TP shards must finish copying before any prefix is published.
            self.gather(True)
            if role == "destination":
                inserted = self.s.tree_cache.insert(InsertParams(key=RadixKey(tokens), value=indices, mamba_value=mamba))
                installed = True
                if os.environ.get("AGENTSHIFT_NONEMPTY_IMPORT") == "1":
                    # MambaRadixCache._insert_helper frees duplicate KV slots
                    # but deliberately leaves redundant recurrent state to
                    # its caller. An existing executable snapshot wins.
                    if inserted.mamba_exist:
                        self.pool.mamba_pool.free(mamba)
                    self.s.tree_cache.sanity_check()
                elif inserted.prefix_len != 0 or inserted.mamba_exist:
                    raise RuntimeError("unexpected destination prefix collision")
            ready = self.gather(len(self.match(tokens).device_indices) == len(tokens))
            store.set(f"{seq}/ready/{rank}", json.dumps(ready))
            peer_ready = json.loads(store.get(f"{seq}/ready/{1-rank}"))
            if not all(ready + peer_ready):
                raise RuntimeError("prefix not ready on every shard")
            return {"success": True, "token_count": len(tokens), "bytes_transferred": nbytes,
                    "transfer_seconds": time.perf_counter() - started, "transport": "rccl",
                    "p2p_disabled": os.environ.get("NCCL_P2P_DISABLE") == "1",
                    "shm_disabled": os.environ.get("NCCL_SHM_DISABLE") == "1",
                    "state": "COMPLETE"}
        except Exception:
            group["failed"] = True
            raise
        finally:
            if role == "destination" and not installed:
                if indices is not None:
                    self.s.token_to_kv_pool_allocator.free(indices)
                if mamba is not None:
                    self.pool.mamba_pool.free(mamba)

    def handle(self, request):
        try:
            error = None
            try:
                self.validate_runtime(require_idle=request.operation != "inspect" or bool(request.payload.get("return_digest")))
            except Exception as exc:
                error = str(exc)
            errors = self.gather(error)
            if any(errors):
                raise ValueError(str(errors))
            operations = {"inspect": self.inspect, "direct_group/init": self.init_group,
                          "direct_transfer": self.transfer}
            result = operations[request.operation](request.payload)
        except Exception as exc:
            result = {"success": False, "message": f"{type(exc).__name__}: {exc}"}
        results = self.gather(result)
        failure = next((r for r in results if not r["success"]), None)
        if failure is not None:
            result = failure
        else:
            result = dict(results[0])
            metadata_only = request.operation == "inspect" and not request.payload.get("return_digest")
            fields = ("token_count",) if metadata_only else ("token_count", "matched_tokens", "full_prefix_available", "has_mamba_state", "state")
            for field in fields:
                if field in result and any(r[field] != result[field] for r in results):
                    return HygonHybridReqOutput(result={"success": False, "message": f"TP ranks disagree on {field}", "tp_ranks": results})
            if metadata_only:
                common = sorted(set.intersection(*(set(r["snapshot_boundaries"]) for r in results)))
                result.update(snapshot_boundaries=common, matched_tokens=max(common, default=0),
                              full_prefix_available=bool(common and common[-1] == result["token_count"]),
                              has_mamba_state=bool(common))
                for field in ("available_token_slots", "evictable_token_slots", "available_snapshot_slots", "evictable_snapshot_slots"):
                    result[field] = min(r[field] for r in results)
            for field in ("bytes_transferred", "state_bytes"):
                if field in result:
                    result[field] = sum(r[field] for r in results)
            if "transfer_seconds" in result:
                result["transfer_seconds"] = max(r["transfer_seconds"] for r in results)
            if "state_sha256" in result:
                rank_digests = [{"rank": i, "sha256": r["state_sha256"], "bytes": r["state_bytes"]}
                                for i, r in enumerate(results)]
                result["state_sha256"] = hashlib.sha256(
                    json.dumps(rank_digests, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                result["state_digest_format"] = "sha256 of ordered TP rank digest/byte records"
            result["tp_ranks"] = results
        return HygonHybridReqOutput(result=result)


def handle_hygon_hybrid(scheduler, request):
    if not hasattr(scheduler, "_hygon_hybrid"):
        scheduler._hygon_hybrid = HybridControl(scheduler)
    return scheduler._hygon_hybrid.handle(request)
