"""Build a small, version-checked overlay from an unmodified vendor srt tree."""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"vendor source does not match expected anchor: {old!r}")
    return text.replace(old, new, 1)


def build(source, output):
    edits = {}
    path = "managers/io_struct.py"
    edits[path] = (source / path).read_text() + '''

@dataclass
class HygonHybridReqInput(BaseReq):
    operation: str
    payload: dict

@dataclass
class HygonHybridReqOutput(BaseReq):
    result: dict
'''
    path = "managers/tokenizer_control_mixin.py"
    text = (source / path).read_text()
    text = replace_once(text, "from sglang.srt.managers.io_struct import (", "from sglang.srt.managers.io_struct import (\n    HygonHybridReqOutput,")
    text = replace_once(text, '("get_internal_state", GetInternalStateReqOutput),', '("get_internal_state", GetInternalStateReqOutput),\n    ("hygon_hybrid", HygonHybridReqOutput),')
    edits[path] = text
    path = "managers/scheduler.py"
    text = (source / path).read_text()
    text = replace_once(text, "from sglang.srt.managers.io_struct import (", "from sglang.srt.managers.io_struct import (\n    HygonHybridReqInput,")
    text = replace_once(text, "(GetInternalStateReq, self.get_internal_state),", "(GetInternalStateReq, self.get_internal_state),\n                (HygonHybridReqInput, self.handle_hygon_hybrid),")
    text = replace_once(text, "    def get_internal_state(self, recv_req: GetInternalStateReq):", '''    def handle_hygon_hybrid(self, recv_req):
        from sglang.srt.managers.hygon_hybrid_control import handle_hygon_hybrid
        return handle_hygon_hybrid(self, recv_req)

    def get_internal_state(self, recv_req: GetInternalStateReq):''')
    edits[path] = text
    path = "entrypoints/http_server.py"
    text = (source / path).read_text()
    text += '''

@app.post("/agentshift/hybrid_state/{operation:path}")
async def hygon_hybrid_control(operation: str, request: Request):
    from fastapi.responses import JSONResponse
    from sglang.srt.managers.io_struct import HygonHybridReqInput
    if operation not in {"inspect", "direct_group/init", "direct_transfer"}:
        return JSONResponse({"success": False, "message": "unsupported hybrid operation"}, status_code=404)
    manager = _global_state.tokenizer_manager
    manager.auto_create_handle_loop()
    payload = await request.json()
    results = await manager.hygon_hybrid_communicator(HygonHybridReqInput(operation=operation, payload=payload))
    result = results[0].result
    return JSONResponse(result, status_code=200 if result.get("success") else 409)
'''
    edits[path] = text
    manifest = {}
    for name, text in edits.items():
        ast.parse(text)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        manifest[name] = {"original_sha256": hashlib.sha256((source / name).read_bytes()).hexdigest(),
                          "patched_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    shutil.copyfile(Path(__file__).with_name("hybrid_control.py"), output / "managers/hygon_hybrid_control.py")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
