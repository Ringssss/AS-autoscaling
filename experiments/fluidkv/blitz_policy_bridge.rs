//! Execute the pinned upstream LMetric policy, its radix matcher and admission
//! update. Only engine observations and pre-tokenized inputs cross this bridge.
//! The native Blitz backend transport is replaced by a shared SGLang adapter.
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Duration;
use serde::Deserialize;
use tokio::sync::{mpsc, Mutex};
use tokio::time::Instant;
use crate::gateway::validation::ValidGenerateRequest;
use crate::scheduler::kvcache::{BlockHash, BlockHashState, PrefixBlockHash};
use crate::scheduler::policies::{Entry, LmetricQ};
use crate::scheduler::policies::policy_trait::Policy;
use crate::{LMetric, ScheduleContext};
use crate::types::GenerationParams;

#[derive(Deserialize)]
struct Observed {
    bs: usize,
    waiting: usize,
    queued_tokens: isize,
    all_tokens: usize,
    hit_tokens: usize,
}

#[derive(Deserialize)]
struct Input {
    request_id: u64,
    tokens: Vec<u32>,
    max_new_tokens: u32,
    observations: Vec<Observed>,
}

pub fn run() {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let mut context = <LmetricQ as Policy>::GlobalContext::default();
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let input: Input = serde_json::from_str(&line.unwrap()).unwrap();
        let result = rt.block_on(async {
            let block_size = 16;
            let state = BlockHashState::new(&input.tokens, block_size);
            let hashes = state.get_hashes();
            let contexts: Vec<_> = input.observations.iter().map(|o| {
                let count = (o.hit_tokens / block_size).min(hashes.len());
                let mut matcher = PrefixBlockHash::new(hashes.len()+64);
                if count > 0 {
                    matcher.insert(&hashes[..count], (0..count as u64).collect());
                }
                let mut lmetric = LMetric::default();
                lmetric.bs = o.bs;
                lmetric.waiting_reqs = o.waiting;
                lmetric.prefill_tokens = o.queued_tokens;
                lmetric.all_tokens = o.all_tokens;
                Arc::new(Mutex::new(ScheduleContext { lmetric, block_hash:matcher }))
            }).collect();
            let n = input.tokens.len() as u32;
            let request = ValidGenerateRequest { request_id:input.request_id,
                messages:Vec::new(), inputs:String::new(), input_length:n,
                truncate:n, decoder_input_details:false, top_n_tokens:0,
                params:GenerationParams { max_new_tokens:input.max_new_tokens,
                    ignore_eos_token:true, ..GenerationParams::default() },
                input_tokens:input.tokens,
            };
            let (tx, _rx) = mpsc::unbounded_channel();
            let entry = Entry { request, block_hash_state:state, response_tx:tx,
                span:tracing::Span::none(), temp_span:None, queue_time:Instant::now(),
                batch_time:None, generated_token_cnt:0, prev_token_time:None,
                time_of_per_token:None, max_time_between_tokens:Duration::ZERO };
            let chosen = LmetricQ::schedule(&entry,&contexts,&mut context).await;
            let admitted = if let Some(i)=chosen {
                let c=contexts[i].lock().await;
                serde_json::json!({"bs":c.lmetric.bs,"waiting":c.lmetric.waiting_reqs,
                    "queued_tokens":c.lmetric.prefill_tokens,"all_tokens":c.lmetric.all_tokens})
            } else { serde_json::Value::Null };
            serde_json::json!({"request_id":entry.request.request_id,"target":chosen,
                "after_admission":admitted,"implementation":"upstream LmetricQ::schedule"})
        });
        writeln!(stdout,"{}",result).unwrap();
        stdout.flush().unwrap();
    }
}
