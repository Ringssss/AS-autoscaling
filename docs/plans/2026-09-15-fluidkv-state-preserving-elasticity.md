# FluidKV state-preserving elasticity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:subagent-dev to implement this plan task-by-task.

**Goal:** On an actual offloaded MoE engine under explicit24/32GiB-class memory envelopes, establish whether destructive state-pool repartition causes material user-visible recovery cost; if so compare simple legal-state preservation and determine the needed FluidKV component changes.

**Hypothesis:** KV/recurrent/window pool resize can destroy subsequently reused legal hybrid checkpoints. Preserving useful state may reduce full-cycle pause+resume latency, but restore traffic and lost expert residency can erase that benefit.

**Validation scope:** User approved discovery first, bottleneck attribution second, upgrade only when supported. L0 real offload/resources/backend/observer checks and L3 real-engine prefix/continuation/rebuild smoke enabled; CPU deterministic tests for state schema and measurement. Training gradients/overfit inapplicable. No PP, no system-driver changes, no claim H100 equals4090/5090. Consumer hardware validation remains separate.

**Architecture:** Pin FreeToken afd99cb3b7c118dce4a9835359bcbc33b70c3cb8; isolated env/build/research checkout with source archived. Existing SGLang env untouched. New external experiment code under experiments/freetoken_elastic. Initial baseline is actual FreeToken offload with limited expert residency and bounded KV/recurrent pools, not resident-weight H100 with tiny KV. No external contributions/messages/commits.

## Shared scaffold and environment gate

Driver575.57.08, H100/A100; installedCUDA12.4/12.9, originalTorch2.9.1cu128. PinnedFreeToken requiresTorch2.11cu130/CUDA13/r580+. Check whether isolated official user-mode CUDA forward-compatibility library can safely support this datacenter GPU; otherwise record concrete inability and evaluate a documented compatible baseline only if feature/semantic parity can be established. Never silently present a patched/older engine as currentFreeToken. Keep download/cache/build off scarce root/tmp filesystem; workspace has more space. Useuv as repository requires; bootstrap standaloneuv if needed.

## Subtask1 baseline interface and state adapter feasibility

Audit actual resize APIs, cache-report/raw input support, rollback/generation gating, all recurrent sibling tensors and shared KV tree ownership. Identify exact state to preserve, minimal snapshot selection/readout/install seams, and mandatory graph recapture. PureCPU tests for independent schema/size/compatibility accounting. Do not modify runtime before a working baseline and evidence of actual cache loss.

## Subtask2 actual offload mechanism gate

Use available Qwen3.5-35B-A3B same weights initially if supported, explicit offload strategy. Set expert slots so full weight pool is host-resident and only subsetGPU; record realexpertcache hits/misses/fetches/host-bank bytes and device processmemory. Calibrate stable nonexpert+graph+KV/recurrent/expert total budget; include startup and temporarybuffer peaks.24/32GiB target is capacity emulation, not consumer performance. Reject configurations that only limitKV or exceed claimed budget. GPUCPUbandwidth/threads/NUMA fixed and recorded.

## Subtask3 bounded paired destructive-repartition experiment

Warm same lawful prefixes then compare: noresize, expert-onlyresize (cache-preserving if source says so), KV/recurrent-poolresize with caches dropped, explicitcacheflush at samefinalgeometry for attribution. The noresize/expert-only controls may use different finalgeometry: label diagnostic controls, not equal-budget performance winners. Main fair comparisons use identical finalpools/expertbudget and same actual continuations. At least two prefix/context lengths and three paired repeats. Freeze feasible sizes after smoke, before treatment outcomes. Capture before/aftercachedtokens/snapshotcount, initialcontextbuild, rebuildpause/graphcapture, firstresumedTTFT/E2E/tokenoutputs and decode, allfailures, host/GPUresidency. Requests arrivingduringmaintenance counted/gatedexplicitly, not removed.

## Subtask4 conditional simple preserve/restore

Only after baseline confirms consumed-state loss and nontrivial cost: implement small researchadapter preserving complete legal KV+recurrent+conv/sibling state at exactprefix and model/schema identity. Back up beforedestructivefree, restore into freshlyallocatedtarget slots and publishcachehandles only when complete; stale indices never reused. Compare unchanged drop/recompute against simple fullbackup/restore and tokenreplay aftersamefinalgeometry. Include D2H/H2D, CPU staging, graphrecapture and per-requestpause. Boundedhostmemory; nofreeextraGPU. Correctness compares sourcecachedcontinuation and coldrecompute separately. If simplebackup captures benefit, report engineeringresult ratherthaninventing3contributions.

## Subtask5 decision and component mapping

Decide whether issueexists, how much full-cycle latency it explains, and whether preservation stillwins afterallcosts. Candidate components only if justified:1legalstate-selection/materialization;2expert/KV/host-budget-aware restoration scheduling;3safe cache-generation remapping/publication. Distributedowner-epoch orMILP notautomatic singleengine requirements. Compare strongest simplebaseline and acknowledge overlapFreeToken/HiCache/WiSP/eLLM. Produce JSON/rawlogs/report/plots, independentaudit, sourceprovenance, appropriateCPUchecks, dedicatedresourcecleanup. No publication or desktop-speed claims without actual evidence.

## Completion

Completed60formaltrials:24destructivebaseline24GiB +24matchedpreservation24GiB +12KV-only32GiBcrosschecks;180formalcompletions allinput/outputcountsverified.5separatesmokeGPUstatebytehashroundtrips pass,18formalpreservechildrenconsume4032/16320 restoredboundaries. SharedupstreamoverlapN-1lengthbug fixedbeforeformaltests andnotcreditedasFluidKVbenefit. Actual60GiB-classBF16hostexpertoffload with2210/3371of10240GPUslots; noPP, no systemdriverchange. OfficialuserlandCUDA13compat andisolatedenvfunctionalvalidated.

Long16k statepoolresizecoldrecoverydominatesadditionaldelay; simplepreservation reducesresize-to-first-responsecycle~44% at24GiB and46%at32GiB;short4k regresses2-4%. Expert-onlyshrinkremainsstrongnear-equal-reclaimalternative; no globalallocatorwinclaim. Outputtextparity isnotestablished despitebytepreservation. Longrepeatcache0 isalsoaffectedbycheckpointcarryacrossshortlastchunk, notjustslotcapacity. Allrawfailures/smokes/observationgapspreserved.24NVMLpeakcoverageincomplete;32coveragecomplete. BF16RSS~63GiBpreventsconsumer-RAMclaims.387CPUtests pass. Allownedengine/ballast/observerprocessesstopped;GPUmemoryzero.

Functional3layers(planning,resource/recoverydecision,safepublication) canorganizeimplementation, butthreeindependentresearchcontributionsnotproved. Furtherinvestmentshouldtargetrealresizeevents,strongpool-choicebaseline,quantizedconsumerhardwareandtaskquality—notimmediateMILP/distributedprotocol expansion. Results: experiments/freetoken_elastic/results/20260915/RESULTS.md.
