# Motivation Options After Research

## Option A: Agent Execution Mobility (Recommended)

Long-lived agents create an execution state absent from request-oriented
serving: a completed prefix remains warm while the program waits on an external
interrupt. Locality pins the next turn to one engine; stateless rerouting loses
the accumulated prefix. AgentShift makes this suspended execution mobile and
uses durable ownership to make the relocation safe.

This option is supported by the strongest existing evidence and keeps all three
components necessary.

## Option B: Stateful Live Autoscaling

Model-ready capacity is not agent-ready capacity because a new engine lacks
session KV and authority. AgentShift can activate or drain warm capacity by
moving state and ownership.

This is a strong application, but the current prototype has no cold model
loading or scale trigger. It should remain a secondary evaluation until those
mechanisms exist.

## Option C: Networked State Handoff

The protocol can extend across nodes and different failure domains. This option
would strengthen NSDI scope, but no cross-node RDMA evidence is currently
available. It belongs in future work, not the title or abstract.
