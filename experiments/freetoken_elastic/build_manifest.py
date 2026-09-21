"""Freeze real source-text prefix fixtures using the serving tokenizer's version."""
import argparse,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer


def sha(text):return hashlib.sha256(text.encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix-lengths',nargs='+',type=int,default=[4096,16384]);p.add_argument('--source-root',type=Path,default=Path('agentshift'));a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    paths=sorted(a.source_root.rglob('*.py'));parts=[];sources=[]
    for path in paths:
        text=path.read_text();parts.append(f'\nFile {path.as_posix()}:\n{text}');sources.append({'path':str(path),'sha256':sha(text),'bytes':len(text.encode())})
    text='Review this Python repository and retain its details for a follow-up question.\n'+''.join(parts)
    ids=tokenizer.encode(text,add_special_tokens=True)
    pairs=[]
    for n in a.prefix_lengths:
        if len(ids)<n:raise ValueError('not enough real source text; do not repeat synthetic filler')
        # Roundtrip is measured, not assumed; reported lengths use the exact text API input.
        parent=tokenizer.decode(ids[:n],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        child=parent+'\n\nFollow-up: identify one concrete state-management invariant in the code above. Explain it briefly.\nAnswer:'
        pi=tokenizer.encode(parent,add_special_tokens=True);ci=tokenizer.encode(child,add_special_tokens=True)
        lcp=0
        for x,y in zip(pi,ci):
            if x!=y:break
            lcp+=1
        pairs.append({'id':f'prefix-{n}','parent_text':parent,'child_text':child,'expected_parent_tokens':len(pi),'expected_child_tokens':len(ci),
                      'requested_prefix_tokens':n,'observed_token_lcp':lcp,'parent_sha256':sha(parent),'child_sha256':sha(child),
                      'parent_token_sha256':sha(json.dumps(pi)),'child_token_sha256':sha(json.dumps(ci))})
    import transformers
    out={'scope':'Constructed repeated-prefix source-text mechanism fixtures; not autonomous-agent task-quality or native timing trace.',
         'model':a.model,'transformers_version':transformers.__version__,'tokenizer_class':type(tokenizer).__name__,'add_special_tokens':True,
         'sources':sources,'pairs':pairs}
    if a.output.exists():raise ValueError('output exists')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in pair.items() if not k.endswith('_text')} for pair in pairs]))
if __name__=='__main__':main()
