"""Tiny random-weight architecture smoke; not a checkpoint/quality benchmark."""
import json
import torch
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

torch.manual_seed(7)
config=Qwen4ExpTextConfig(
    vocab_size=1024,hidden_size=128,num_hidden_layers=4,
    num_attention_heads=2,num_key_value_heads=1,head_dim=64,
    layer_types=['linear_attention']*3+['full_attention'],
    linear_num_key_heads=1,linear_num_value_heads=2,
    linear_key_head_dim=16,linear_value_head_dim=16,
    num_experts=4,num_experts_per_tok=2,moe_intermediate_size=64,
    shared_expert_intermediate_size=64,hc_lowrank=32,
    heads_per_ngram=2,ngram_vocab_size_base=128,split_ngram_parts=8,
    ple_embed_dim=128,ple_layer_ids=[2],indexer_n_heads=1,
    indexer_kv_heads=1,indexer_head_dim=16,indexer_budget=16,indexer_compress_ratio=4,
    bos_token_id=1,eos_token_id=2,pad_token_id=0,
    rope_parameters={'rope_type':'default','rope_theta':10000000,'partial_rotary_factor':.25,
                     'mrope_section':[3,3,2],'mrope_interleaved':True},
)
model=Qwen4ExpForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
tokens=torch.tensor([[10,11,12,13,14,15,16,17]],device='cuda')
with torch.inference_mode():
    output=model.generate(tokens,max_new_tokens=4,do_sample=False)
torch.cuda.synchronize()
print(json.dumps({'scope':'tiny random-weight architecture smoke only','torch':torch.__version__,
                  'device':torch.cuda.get_device_name(0),'tokens':output.tolist()}))
