---
name: cold-start
description: >-
  Investigate a slow scale-from-zero cold start on the ECS inference pipeline: break the latency
  from a request's enqueue to its first token into node-provisioning, image-pull, and
  weights-load stages and find which one dominated. Use when first-request latency after an idle
  period is too high or a gpu_inference.cold_start record shows a long stage.
---

# Cold-start investigation

## When to use

- The first request after an idle period (scale-from-zero) took far longer than steady-state.
- A `gpu_inference.cold_start` record shows a long gap between stages.
- You are deciding whether to keep a tier at `MinTaskCount=1` to eliminate cold starts, and need
  the measured breakdown to justify the cost.

## Contract

- **Event:** `gpu_inference.cold_start`
- **Schema:** [`telemetry/contract/v1/cold_start.schema.json`](../../../telemetry/contract/v1/cold_start.schema.json)
- **Supported versions:** `>=1.0.0 <2.0.0`. Ignore fields you do not recognize.

Emitted once per task after its first successful inference, as a series of five stage
**timestamps**. Durations are deliberately not stored — derive them yourself so there is no second
copy that can disagree. Stage order is guaranteed non-decreasing (conformance rule CR-4):
`enqueued ≤ node_ready ≤ image_pulled ≤ weights_loaded ≤ first_token`.

## Where the records are

| Source | Location |
|---|---|
| Contract record (JSON line) | ECS task log group, plus OTLP to CloudWatch (Omni) |

```bash
aws logs filter-log-events \
  --log-group-name "<task-log-group>" \
  --filter-pattern '{ $.event_name = "gpu_inference.cold_start" }' \
  --start-time "$(( ($(date +%s) - 86400) * 1000 ))"
```

## Investigate

1. **Derive the stage durations** from `attributes."cold_start.stages"`:

   | Segment | From → to | What it measures |
   |---|---|---|
   | Queue wait | `enqueued` → `node_ready` | Time until a GPU instance was registered and ready |
   | Node provisioning | part of the above when cold | EC2 launch + ECS registration for a fresh instance |
   | Image pull | `node_ready` → `image_pulled` | Pulling the ~8–9 GB vLLM image (ECS `pullStoppedAt`) |
   | Weights load | `image_pulled` → `weights_loaded` | Model into VRAM; first successful vLLM `/health` |
   | First token | `weights_loaded` → `first_token` | vLLM warm-up to the first completion |

   The dominant segment is where to focus. On this stack, image pull + weights load usually
   dominate — the slow part is getting the container and the model onto a fresh host, not ECS's
   scheduling.

2. **Check whether the node was even cold — `attributes."cold_start.node_reused"`.**
   `true` means the task landed on an already-registered instance, so node provisioning did not
   contribute and `node_ready` was clamped to `enqueued`. A long total with `node_reused: true`
   means the cost was image pull or weights load, not capacity.

3. **Attribute the dominant segment.**
   - **Node provisioning dominates** (`enqueued` → `node_ready` large, `node_reused: false`):
     capacity/launch latency. Expected for scale-from-zero. If it is far beyond the tier's norm,
     check for `InsufficientInstanceCapacity` on the capacity provider and the instance types the
     tier is allowed to launch.
   - **Image pull dominates** (`node_ready` → `image_pulled` large): the image is being pulled over
     the network each time. This is the usual target for optimization — see "minimizing model
     re-pull time" in the repo's GPU auto-repair notes.
   - **Weights load dominates** (`image_pulled` → `weights_loaded` large): model download/load.
     Read `attributes."inference.model.source"`: `huggingface` means weights are pulled from the
     Hub at runtime (slower, network-bound); `s3` means from a bucket. A `huggingface` source on a
     large model is a prime candidate for staging weights in S3 or baking them into the image.

4. **Put it in context** — `resource."inference.tier"`, `resource."host.type"`, and
   `attributes."inference.model.name"`. A `large` tier (g6e.48xlarge, Llama-2-70B) legitimately
   cold-starts in the 15–20 minute range; a `small` tier (g6e.xlarge, Mistral-7B) in ~7 minutes.
   Compare against the right baseline before calling a cold start "slow".

## Blind spots

Absence of a record is not absence of a cold start. From the contract's producer limits:

- **`cold_start` is skipped when the triggering request was enqueued *after* the task's image
  pull** — the task was already warm for that request, so there is nothing to measure.
- **It is skipped when a stage cannot be resolved** — for example, the task metadata endpoint or
  `DescribeContainerInstances` was unavailable. A genuinely slow cold start can therefore leave no
  record. If users saw slow first-request latency but no `cold_start` event exists, reconstruct the
  stages from ECS task timestamps (`createdAt`, `pullStartedAt`, `pullStoppedAt`) and the vLLM
  readiness log directly.
- **The event is emitted only after the *first successful* inference.** A task that cold-started
  but then failed before serving (and dead-lettered) emits no cold-start record — pivot to
  [`dead-letter`](../dead-letter/SKILL.md).
