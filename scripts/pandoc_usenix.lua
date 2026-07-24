local function raw(s)
  return pandoc.RawBlock("latex", s)
end

local figures = {
  ["fig:workload"] = {
    env = "figure*",
    width = "0.98\\textwidth",
    path = "figures/fig_workload_characterization.pdf",
    caption = "Workload evidence. Kimi contains substantial long-context demand; FlowPrefill sessions accumulate context across turns; and child-arrival proxies cluster in short windows. FlowPrefill deltas are not labeled tool durations."
  },
  ["fig:architecture"] = {
    env = "figure*",
    width = "0.97\\textwidth",
    path = "figures/fig_architecture.pdf",
    caption = "AgentShift separates durable continuation from warm KV state. It copies an immutable completed prefix during the tool gap, publishes it only after all ranks are ready, commits ownership with an epoch CAS, and releases the source shadow after first-token acknowledgement."
  },
  ["fig:context"] = {
    env = "figure",
    width = "0.98\\columnwidth",
    path = "figures/fig_context_latency.pdf",
    caption = "Post-tool latency as context grows. Reroute reconstructs a growing history, while AgentShift changes placement and remains near Sticky."
  },
  ["fig:gap"] = {
    env = "figure",
    width = "0.98\\columnwidth",
    path = "figures/fig_gap_overlap.pdf",
    caption = "Post-tool latency versus blocked interval at 32K. AgentShift removes transfer from the critical path once the interval covers migration; On-return cannot use the interval."
  },
  ["fig:hotspot"] = {
    env = "figure*",
    width = "0.94\\textwidth",
    path = "figures/fig_hotspot.pdf",
    caption = "Eight-agent 32K return burst. AgentShift moves 50\\% of owners with full hits and reaches the low-latency, high-relief region. Source-only methods do not change future placement."
  },
  ["fig:elasticity"] = {
    env = "figure*",
    width = "0.94\\textwidth",
    path = "figures/fig_elasticity.pdf",
    caption = "Warm-pool elasticity with eight 16K agents. AgentShift activates half of a model-ready target during scale-out and reaches zero source owners during scale-in without re-prefill. Sticky latency is low because it performs neither operation."
  },
  ["fig:interference"] = {
    env = "figure*",
    width = "0.94\\textwidth",
    path = "figures/fig_interference.pdf",
    caption = "Foreground impact of one 32K migration. Async copy leaves throughput and steady TPOT within noise, while reducing excess arrival TTFT and maximum token gap relative to synchronous copy."
  },
  ["fig:control"] = {
    env = "figure",
    width = "0.98\\columnwidth",
    path = "figures/fig_control_plane.pdf",
    caption = "Control-plane behavior as registered agents and client concurrency increase. Aggregate throughput remains adequate for the prototype, but concurrent durable writes create a long p99 tail."
  }
}

function Figure(el)
  local f = figures[el.identifier]
  if not f then
    return nil
  end
  local latex = string.format([[
\begin{%s}[t]
  \centering
  \includegraphics[width=%s]{%s}
  \caption{%s}
  \label{%s}
\end{%s}
]], f.env, f.width, f.path, f.caption, el.identifier, f.env)
  return raw(latex)
end

local tables = {
  ["tab:capabilities"] = [[
\begin{table*}[t]
\centering
\small
\caption{Capability comparison. Literature-named baselines in our experiments are mechanism-equivalent implementations, not official artifact reproductions.}
\label{tab:capabilities}
\begin{tabular}{lcccc}
\toprule
Design & Moves next turn & Full prefix hit & Prepares in gap & Durable owner transfer \\
\midrule
Sticky & No & Yes & N/A & No \\
Agentix-style routing & Optional & Only on source & No & No \\
Continuum-style TTL & No & Until expiry & No & No \\
TokenCake-style source reload & No & Yes & Yes & No \\
Stateless reroute & Yes & No & N/A & Routing only \\
Symphony-style shared prefetch & Yes & Yes & Yes & No \\
Llumnix active migration & Yes & Yes & N/A & Request scoped \\
On-return handoff & Yes & Yes & No & Yes \\
\textbf{AgentShift} & \textbf{Yes} & \textbf{Yes} & \textbf{Yes} & \textbf{Yes} \\
\bottomrule
\end{tabular}
\end{table*}
]],
  ["tab:headline"] = [[
\begin{table}[t]
\centering
\small
\caption{Next-turn latency for Qwen3-8B, TP=1, 32K prefix, 500 ms gap, and five repetitions.}
\label{tab:headline}
\begin{tabular}{lrrr}
\toprule
Strategy & Mean (ms) $\downarrow$ & Full hit & Owner moved \\
\midrule
Sticky & 54.4 & Yes & No \\
Reroute & 1260.5 & No & Routing only \\
On-return & 126.1 & Yes & Yes \\
\textbf{AgentShift} & \textbf{52.4} & \textbf{Yes} & \textbf{Yes} \\
Oracle & 54.5 & Yes & Yes \\
\bottomrule
\end{tabular}
\end{table}
]],
  ["tab:faults"] = [[
\begin{table*}[t]
\centering
\small
\caption{Fault-injection matrix. Claims apply within this evaluated fault model.}
\label{tab:faults}
\begin{tabular}{lll}
\toprule
Fault point & Expected result & Observed \\
\midrule
Destination fails after commit & Recover source shadow at new epoch & Pass \\
Destination fails after release & Cold reconstruct at new epoch & Pass \\
First-token ACK lost & Retry idempotent release & Pass \\
Restart at \texttt{DEST\_READY} & Abort; source remains owner & Pass \\
Tool result races CAS & New owner consumes once & Pass \\
Managed effect retry & One external row/submission & Pass \\
No-fencing ablation & Full system admits one executor & Pass \\
Flush after terminal transfer & No stale prefix reference & Pass \\
\bottomrule
\end{tabular}
\end{table*}
]]
}

function Table(el)
  local value = tables[el.identifier]
  if value then
    return raw(value)
  end
  return nil
end
