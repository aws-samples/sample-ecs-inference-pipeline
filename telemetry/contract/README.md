# GPU inference telemetry contract

A versioned schema shared by the two sides of observability for this pipeline:

- **Producers** (day-one instrumentation): the SQS worker, the router, and the normalizers that
  turn AWS events into contract records. They emit records that must validate.
- **Consumers** (day-two investigation): the GPU-impairment, dead-letter, and cold-start
  investigation skills. They read fields by the names defined here and nowhere else.

If a producer renames a field, the change fails contract validation in CI instead of breaking a
runbook silently during an incident.

| | |
|---|---|
| Contract version | **1.0.0** (`contract.json`) |
| Status | Draft: schemas defined, producers not yet instrumented (see [Producer status](#producer-status)) |
| Schema dialect | JSON Schema 2020-12 |
| Schemas | [`v1/`](v1/) |

## Layout

```
telemetry/contract/
├── contract.json                 # manifest: version, event -> schema, conformance rules
├── CHANGELOG.md
└── v1/                           # one directory per MAJOR version
    ├── common.schema.json        # envelope, resource attributes, shared types (XID enum, tier, ARNs)
    ├── gpu_impairment.schema.json
    ├── dead_letter.schema.json
    ├── cold_start.schema.json
    └── examples/                 # one valid record per event (two for dead_letter)
```

## Record shape

Every event is one JSON object with the same envelope:

```json
{
  "contract_version": "1.0.0",
  "event_name": "gpu_inference.dead_letter",
  "event_id": "<uuid>",
  "timestamp": "2026-10-27T18:02:41.530Z",
  "resource":   { "service.name": "sqs-worker", "cloud.region": "us-west-2", "aws.ecs.cluster.arn": "...", "...": "..." },
  "attributes": { "inference.dlq.reason": "malformed_input", "...": "..." }
}
```

- **`resource`** identifies the emitter. Keys follow
  [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/resource/cloud-provider/aws/ecs/)
  (`cloud.region`, `aws.ecs.task.arn`, `host.id`, `host.type`, ...) so the same names come out of
  OpenTelemetry instrumentation without remapping. Contract-specific keys are
  `aws.ecs.container_instance.arn`, `aws.ecs.capacity_provider.name`, and `inference.tier`.
- **`attributes`** carry the event-specific fields, with dotted, snake_case, namespaced keys.
- **Timestamps** are RFC 3339 UTC with a `Z` suffix. `timestamp` is when the record was emitted.
  Times of the underlying condition live in `attributes`.

## Events

### `gpu_inference.gpu_impairment`

Amazon ECS marked a GPU container instance `IMPAIRED`. This event is normalized from the
`ECS Container Instance Health Change` EventBridge event. **Before** auto-repair drains and
terminates the instance, it is enriched with the tasks running on that instance.

| Field | Req | Notes |
|---|---|---|
| `resource.host.id`, `resource.aws.ecs.container_instance.arn`, `resource.aws.ecs.capacity_provider.name` | ✓ | Instance identity, from `detail.ec2InstanceId`, `detail.containerInstanceArn`, `detail.capacityProviderName` |
| `gpu.impairment.cause` | ✓ | `xid` or `other` (ECS also impairs on NVIDIA MPS daemon failure, which has no XID) |
| `gpu.xid.code` | if `xid` | Integer. Restricted to the **18 XIDs ECS auto-repair acts on**: 46, 48, 54, 62, 64, 74, 79, 95, 109, 110, 136, 140, 142, 143, 151, 155, 156, 158 ([source](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu-auto-repair.html)). RESTART_APP XIDs such as 13 and 31 are *not* impairments and are rejected. |
| `gpu.health_check.type` / `.status` | ✓ | `ACCELERATED_COMPUTE` / `IMPAIRED` |
| `gpu.health_check.status_reason` | ✓ | Verbatim `statusReason`, e.g. `XID_79` |
| `gpu.health_check.last_status_change` | ✓ | From the health check |
| `forensics.captured_at`, `forensics.task_arns` | ✓ | Tasks on the instance at capture time. The list may be empty but must be present, because a missing list means capture was skipped. |
| `forensics.task_families`, `remediation.auto_repair_enabled`, `gpu.index` | | Optional context |
| `source.event_id` | ✓ | EventBridge event `id`, for tracing back to the raw event |

### `gpu_inference.dead_letter`

A request landed in the DLQ. The **`inference.dlq.reason` discriminator** separates the two
paths that today share one queue:

| `reason` | Meaning | How it gets there today | Allowed `reason_code` |
|---|---|---|---|
| `malformed_input` | Customer error | Worker validation fails, then explicit `send_message` to the DLQ | `body_null`, `body_not_string`, `invalid_json`, `not_json_object`, `missing_request_id`, `invalid_request_id`, `missing_prompt`, `prompt_not_string`, `prompt_empty`, `max_tokens_not_integer`, `max_tokens_out_of_range`, `temperature_not_number`, `temperature_out_of_range`, `top_p_not_number`, `top_p_out_of_range` |
| `processing_exhausted` | System error | The worker doesn't delete the message, so SQS redrives it after `maxReceiveCount` (3) | `inference_failed`, `output_write_failed`, `unknown` |

The `malformed_input` codes map one-to-one onto the error strings `validate_message()` returns in
`container/sqs_worker.py`.

| Field | Req | Notes |
|---|---|---|
| `inference.dlq.reason`, `inference.dlq.reason_code` | ✓ | See above. Codes are validated against the reason. |
| `inference.dlq.source_tier` | ✓ | `small` or `large` |
| `inference.dlq.source_queue` | ✓ | Queue URL (`OriginalQueue`) or ARN (`DeadLetterQueueSourceArn`) |
| `messaging.message.id` | ✓ | SQS `MessageId` |
| `inference.dlq.dead_lettered_at` | ✓ | |
| `inference.request.id` | `processing_exhausted` | Optional for malformed input, because the body may not parse |
| `inference.dlq.receive_count` | `processing_exhausted` | `ApproximateReceiveCount` |
| `inference.request.enqueued_at`, `inference.dlq.error_message` | | `error_message` is human-readable only. Match on `reason_code`. |

### `gpu_inference.cold_start`

One scale-from-zero cold start, emitted once per task after its first successful inference.

| Stage (`cold_start.stages.*`) | Source of truth |
|---|---|
| `enqueued` | SQS `SentTimestamp` of the triggering request |
| `node_ready` | Container instance `registeredAt`. If the task reused a running instance, clamp to `enqueued` and set `cold_start.node_reused: true`. |
| `image_pulled` | ECS task `pullStoppedAt` |
| `weights_loaded` | First successful vLLM `/health` (weights in VRAM) |
| `first_token` | First vLLM completion response for the triggering request |

The record also requires `resource.aws.ecs.task.arn`, `resource.host.id`,
`resource.inference.tier`, `inference.model.name`, `inference.model.source` (`s3` or
`huggingface`), and `inference.request.id`. Stage durations are left out on purpose. Consumers
derive them from the timestamps, so there's no second copy that can disagree.

## Conformance rules

These are the narrative's Appendix B rules, numbered so the harness and skills can cite them:

| ID | Rule | Enforced by |
|---|---|---|
| CR-1 | Required fields are present on every event | Schema |
| CR-2 | XID codes match the defined enum | Schema |
| CR-3 | Dead-letter events carry a reason discriminator and source tier | Schema |
| CR-4 | Cold-start stages are non-decreasing: `enqueued ≤ node_ready ≤ image_pulled ≤ weights_loaded ≤ first_token` | Harness (not expressible in JSON Schema) |
| CR-5 | For cause `xid`, `status_reason == "XID_" + gpu.xid.code` | Harness |
| CR-6 | `contract_version` major matches the schema directory (`v1` → `1.x.y`) | Schema |

Producer records are validated **strictly**: `additionalProperties` / `unevaluatedProperties`
are `false`, so a renamed or misspelled field fails CR-1 or the closed-object check instead of
passing silently. Consumers must **ignore** unknown fields, so a newer-minor producer doesn't
break an older consumer.

## Versioning policy

The contract has **one** semantic version for all events (`contract.json` → `version`). Every record
carries the version it was produced against in `contract_version`.

| Change | Bump | Examples |
|---|---|---|
| Breaking for producers or consumers | **MAJOR** | Remove or rename a field. Change a type, unit, or meaning. Make an optional field required. Remove an enum value. Change an event name. |
| Backward-compatible addition | **MINOR** | Add an optional field. Add a new event type. Add an enum value (for example, ECS starts acting on a new XID). |
| No validation change | **PATCH** | Descriptions, docs, examples |

Mechanics:

1. **MINOR / PATCH** changes edit the schemas in `v1/` in place. Bump `version` in
   `contract.json` and add a `CHANGELOG.md` entry. The v1 schemas accept any `1.x.y`, so older
   producers keep validating.
2. **MAJOR** changes add a new `v2/` directory with new `$id`s (`.../contract/v2/...`). Keep
   `v1/` until every producer and consumer has moved. During the overlap, producers may dual-emit.
3. Consumers declare the version range they support (for example, `>=1.0.0 <2.0.0`) and **must
   ignore unknown fields**. A skill that relies on a field added in 1.2.0 requires `>=1.2.0`.
4. Adding an enum value is MINOR, so consumers must handle values they don't recognize as
   "unclassified" rather than failing. In particular, the XID list tracks the AWS documentation.
5. Releases are tagged `telemetry-contract/v<version>` in git.

## Producer status

| Event | Producer | Where records land |
|---|---|---|
| `gpu_impairment` | `infrastructure/gpu_impairment_normalizer.py`, a Lambda targeted by the `${STACK_NAME}-container-instance-health` rule. It ignores non-`ACCELERATED_COMPUTE` impairments and runs `ListTasks` on the impaired instance before the drain. | `/aws/lambda/${STACK_NAME}-gpu-impairment-normalizer` (JSON line). Active tracing for Omni. |
| `dead_letter` (`malformed_input`) | `container/sqs_worker.py`, after the explicit DLQ send. The existing `ErrorReason` / `OriginalQueue` attributes are unchanged. | Task log group (JSON line), plus OTLP to CloudWatch (Omni) |
| `dead_letter` (`processing_exhausted`) | `container/sqs_worker.py`, on the failed attempt where `ApproximateReceiveCount` reaches `MAX_RECEIVE_COUNT` (3, matching the RedrivePolicy) | Same |
| `cold_start` | `container/sqs_worker.py`, once per task after its first successful inference. Stages come from SQS `SentTimestamp`, `registeredAt`, `PullStoppedAt` (task metadata), `VLLM_READY_AT` (stamped by `entrypoint.sh`), and the vLLM response time. | Same |

The worker runs under `opentelemetry-instrument` with the CloudWatch plugin, in its own virtualenv
(`/opt/worker-venv`, isolated from vLLM's own OpenTelemetry dependencies). It exports OTLP to a
CloudWatch agent sidecar on `localhost:4318`. `tests/test_telemetry_instrumentation.py`
validates every producer's records against these schemas.

Known producer limits:

- **`processing_exhausted` is emitted by the worker, so a worker that dies mid-request emits
  nothing.** Examples are an OOM, or the instance being impaired. SQS still redrives the message,
  so the DLQ depth alarm catches it, but no contract record is written.
- **`cold_start` is skipped when the triggering request was enqueued after the task's image pull.**
  The task was already warm for that request. It is also skipped when a stage can't be resolved,
  for example if the task metadata endpoint or `DescribeContainerInstances` is unavailable.
- **`gpu_impairment` omits `forensics.task_arns` when `ListTasks` fails.** Per CR-1, that
  record doesn't conform, which marks the capture as missed. An XID outside the enum is
  recorded as `cause: other`, so that XID needs a MINOR contract bump.

## Validating a record

```python
import json, pathlib
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

v1 = pathlib.Path("telemetry/contract/v1")
schemas = [json.loads(p.read_text()) for p in v1.glob("*.schema.json")]
registry = Registry().with_resources((s["$id"], Resource.from_contents(s)) for s in schemas)
schema = json.loads((v1 / "dead_letter.schema.json").read_text())

validator = Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
record = json.loads((v1 / "examples/dead_letter_malformed_input.json").read_text())
for error in validator.iter_errors(record):
    print(error.json_path, error.message)
```

Needs `jsonschema>=4.18` (for `referencing`). CR-4 and CR-5 are semantic checks that belong
to the conformance harness, which is next step 5.
