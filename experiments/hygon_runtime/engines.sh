#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
overlay=${HYGON_OVERLAY:-/home/zhujianian/hygon-runtime-overlay}
model=${MODEL_PATH:-/home/zhujianian/Qwen3.6-35B-A3B}
cache=${HYGON_TRITON_CACHE:-/home/zhujianian/.cache/hygon-agentshift}
image=harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-sf_b075
vendor=/usr/local/lib/python3.10/dist-packages/sglang/srt
action=${1:-status}
for side in source destination; do
    container=zhujianian-agentshift-hygon-$side
    case "$action" in
        start)
            if docker container inspect "$container" >/dev/null 2>&1; then
                docker start "$container"
                continue
            fi
            if [[ $side == source ]]; then
                devices=${SOURCE_DEVICES:-2,3}; port=31400
            else
                devices=${DESTINATION_DEVICES:-4,5}; port=31401
            fi
            mounts=()
            mkdir -p "$cache/$side"
            for file in managers/io_struct.py managers/tokenizer_control_mixin.py managers/scheduler.py entrypoints/http_server.py managers/hygon_hybrid_control.py; do
                mounts+=(-v "$overlay/$file:$vendor/$file:ro")
            done
            docker run -d --name "$container" --shm-size=16g --network host \
                --device /dev/kfd --device /dev/mkfd --device /dev/dri \
                -v /usr/local/hyhal:/opt/hyhal:ro -v "$model:/model:ro" \
                -v "$cache/$side:/root/.triton" \
                -v "$root/experiments/hygon_runtime/moe_align_block_size.py:$vendor/layers/moe/moe_runner/triton_utils/moe_align_block_size.py:ro" \
                "${mounts[@]}" \
                -e HIP_VISIBLE_DEVICES="$devices" -e SGLANG_KV_LAYOUT_HCU_FA=false \
                -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_DEBUG=INFO \
                -e ENGINE_PORT="$port" \
                --entrypoint bash "$image" -lc \
                'mkdir -p /var/log/hylog; source /opt/dtk/env.sh; exec python -m sglang.launch_server --model-path /model --host 127.0.0.1 --port "$ENGINE_PORT" --tp-size 2 --mem-fraction-static 0.70 --context-length 4096 --max-running-requests 4 --disable-cuda-graph --attention-backend triton --page-size 1 --moe-runner-backend triton --max-total-tokens 8192 --disable-custom-all-reduce --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 32'
            ;;
        stop) docker stop "$container" ;;
        status) docker inspect --format '{{.Name}} {{.State.Status}}' "$container" ;;
        *) echo "Usage: $0 {start|stop|status}" >&2; exit 2 ;;
    esac
done
