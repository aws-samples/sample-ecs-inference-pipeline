---
name: gpu-impairment
description: >-
  Investigate a GPU container instance that Amazon ECS marked IMPAIRED on the ECS inference
  pipeline: identify the XID fault, confirm whether auto-repair replaced the host, and account for
  the tasks that were running on it. Use when the container-instance health EventBridge rule fires,
  when a gpu_inference.gpu_impairment record appears, or when a GPU task stops with no app error.
---

# GPU impairment investigation

## When to use

- The `${STACK_NAME}-container-instance-health` EventBridge rule fired, or the SNS alarm topic
  reported an `ECS Container Instance Health Change`.
- A `gpu_inference.gpu_impairment` record was emitted.
- A GPU inference task stopped and the application logs show no error — the host under it may have
  gone `IMPAIRED` and been drained by auto-repair (a stop during repair can be normal).

## Contract

- **Event:** `gpu_inference.gpu_impairment`
- **Schema:** [`telemetry/contract/v1/gpu_impairment.schema.json`](../../../telemetry/contract/v1/gpu_impairment.schema.json)
- **Supported versions:** `>=1.0.0 <2.0.0`. Ignore fields you do not recognize.

This event is normalized from the `ECS Container Instance Health Change` EventBridge event and,
**before** auto-repair drains and terminates the instance, enriched with the tasks that were
running on it (forensic capture). Read the fields; do not re-derive them from the raw event.

## Where the records are

| Source | Location |
|---|---|
| Contract record (JSON line) | CloudWatch Logs `/aws/lambda/${STACK_NAME}-gpu-impairment-normalizer` |
| Raw trigger | EventBridge rule `${STACK_NAME}-container-instance-health` |
| Live instance health | `aws ecs describe-container-instances --cluster <cluster> --container-instances <arn>` |

Pull recent records:

```bash
aws logs filter-log-events \
  --log-group-name "/aws/lambda/${STACK_NAME}-gpu-impairment-normalizer" \
  --filter-pattern '{ $.event_name = "gpu_inference.gpu_impairment" }' \
  --start-time "$(( ($(date +%s) - 86400) * 1000 ))"
```

## Investigate

1. **Classify the fault — `attributes."gpu.impairment.cause"`.**
   - `xid`: read `attributes."gpu.xid.code"`. It is one of the 18 codes ECS auto-repair acts on
     (46, 48, 54, 62, 64, 74, 79, 95, 109, 110, 136, 140, 142, 143, 151, 155, 156, 158). Look the
     code up in the
     [NVIDIA XID reference](https://docs.nvidia.com/deploy/xid-errors/index.html). Common ones
     here: 48 double-bit ECC, 74 NVLink, 79 GPU fell off the bus.
   - `other`: a non-XID impairment ECS documents, such as an NVIDIA MPS daemon failure. There is no
     `gpu.xid.code`.
   - A code **not** in the enum is recorded as `cause: other` (see Blind spots). That is a signal
     the XID list needs a MINOR contract bump, not that the fault was minor.

2. **Confirm the host identity — `resource."host.id"`,
   `resource."aws.ecs.container_instance.arn"`, `resource."aws.ecs.capacity_provider.name"`.**
   These pin the EC2 instance, its ECS container instance, and the capacity provider whose
   auto-repair rate limit applies.

3. **Confirm auto-repair ran — `attributes."remediation.auto_repair_enabled"`.**
   - `true`: ECS uses start-before-stop — it set the instance `DRAINING`, provisioned a
     replacement, let tasks stop within their stop timeout, then terminated the impaired host.
     Verify the replacement actually came up (a capacity shortage can stall it):
     ```bash
     aws ecs describe-container-instances --cluster <cluster> \
       --container-instances <new-arn> --query 'containerInstances[].status'
     aws ecs describe-capacity-providers --capacity-providers <capacity-provider-name>
     ```
     If `describe-container-instances` on the impaired host returns `InstanceNotFound` or
     `shutting-down`, repair completed. If no replacement registered, check for
     `InsufficientInstanceCapacity` in the autoscaling/ECS events.
   - `false` or absent: auto-repair did not replace the host. ECS still monitors health but will
     not replace it automatically — this needs manual intervention. Do not re-enable or change the
     capacity provider's auto-repair setting as part of an investigation; that is a production
     change (confirm with the operator first).

4. **Account for the workload — `attributes."forensics.task_arns"` (and
   `forensics.task_families`).** These are the tasks that were on the instance at
   `forensics.captured_at`. For each task, decide whether the work was recovered:
   - **ECS service task:** the service reschedules it on healthy capacity. Confirm the service
     reached its desired count.
   - **Standalone task:** it is *not* rescheduled. If one of these was mid-inference, its message
     either completed, redrove to the DLQ, or was lost — follow with the
     [`dead-letter`](../dead-letter/SKILL.md) skill using the task's request IDs.
   - An **empty** `forensics.task_arns` means no tasks were running (benign). An **absent** list
     means capture was skipped — see Blind spots.

5. **Trace back to the raw event if needed — `attributes."source.event_id"`** is the EventBridge
   event `id`, and `gpu.health_check.status_reason` is the verbatim `statusReason` (e.g. `XID_79`).

## Blind spots

Absence of a record is not absence of the failure. From the contract's producer limits:

- **`forensics.task_arns` is absent when `ListTasks` failed.** Per CR-1 that record does not
  conform, which marks the forensic capture as *missed*, not as "no tasks". Reconstruct the
  workload from the service's task history instead
  (`aws ecs list-tasks --cluster <cluster> --container-instance <arn> --desired-status STOPPED`).
- **A non-enum XID is recorded as `cause: other`.** The GPU fault may be real and serious; it just
  is not one ECS auto-repair acts on yet. File a MINOR contract bump to add the code.
- **RESTART_APP XIDs (e.g. 13, 31) never produce this event** — they are not impairments. They
  surface as the `TaskGPURestartAppXidCount` metric instead; see the
  [`cold-start`](../cold-start/SKILL.md) and metrics notes, and do not expect an impairment record
  for them.
