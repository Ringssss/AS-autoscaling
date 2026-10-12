"""Load the real Qwen3.8 checkpoint on Hygon via isolated upstream Transformers.

Reference inference only: device-map layer sharding is not tensor parallelism
and has no SGLang paging or FluidKV migration endpoints. Never merge these
latencies with the SGLang routing campaign.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp import modeling_qwen4_exp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='/model')
    parser.add_argument('--output',default='/results/qwen38-reference.json')
    args=parser.parse_args()
    torch.set_num_threads(16)
    # Hygon's current grouped GEMM fails for these expert shapes, and its TRSM
    # kernel has an incompatible launch bound. Use upstream eager experts and
    # the mathematically equivalent upstream recurrent GDN scan for prefill.
    # These reference paths are intentionally not serving-performance evidence.
    modeling_qwen4_exp.torch_chunk_gated_delta_rule = modeling_qwen4_exp.torch_recurrent_gated_delta_rule
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    began=time.perf_counter()
    model=Qwen4ExpForConditionalGeneration.from_pretrained(
        args.model,local_files_only=True,dtype=torch.bfloat16,device_map='auto',
        max_memory={**{i:'55GiB' for i in range(torch.cuda.device_count())},'cpu':'180GiB'},
        attn_implementation='sdpa',
        experts_implementation='eager',
    ).eval()
    load_seconds=time.perf_counter()-began
    rows=[]
    prompts=['What is 1 + 1? Answer with just the number.','What is the capital of France? Answer briefly.']
    for prompt in prompts:
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(text,return_tensors='pt').to(model.get_input_embeddings().weight.device)
        began=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=32,do_sample=False)
        for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
        elapsed=time.perf_counter()-began
        ids=output[0,inputs['input_ids'].shape[1]:].tolist()
        row={'prompt':prompt,'output_ids':ids,'text':tokenizer.decode(ids,skip_special_tokens=True),'elapsed_seconds':elapsed}
        rows.append(row);print(json.dumps(row),flush=True)
    report={'scope':'real-weight reference inference, not SGLang serving or FluidKV evaluation',
            'model':args.model,'load_seconds':load_seconds,'torch':torch.__version__,
            'visible_devices':torch.cuda.device_count(),'device_map':{k:str(v) for k,v in model.hf_device_map.items()},
            'reference_fallbacks':['upstream eager MoE','upstream recurrent GDN for prefill'],
            'rows':rows,'memory_allocated_bytes':[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
