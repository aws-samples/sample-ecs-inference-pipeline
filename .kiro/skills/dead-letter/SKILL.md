---
name: dead-letter
description: >-
  Investigate inference requests that landed in the dead-letter queue on the ECS inference
  pipeline: split customer errors (malformed input) from system errors (processing exhausted),
  identify the exact reason code, and find the source tier and request. Use when the DLQ depth
  alarm fires or a gpu_inference.dead_letter record appears.
---

# Dead-letter investigation

## When to use

- The DLQ depth alarm (`ApproximateNumberOfMessagesVisible` on the dead-letter queue) fired.
- A `gpu_inference.dead_letter` record was emitted.
- A caller reports a request that never produced a result in S3.

## Contract

- **Event:** `gpu_inference.dead_letter`
- **Schema:** [`telemetry/contract/v1/dead_letter.schema.json`](../../../telemetry/contract/v1/dead_letter.schema.json)
- **Supported versions:** `>=1.0.0 <2.0.0`. Ignore fields you do not recognize.

One DLQ today carries two failure paths. The **`inference.dlq.reason` discriminator** separates
them, so read it first and branch — never treat the DLQ as one undifferentiated pile.

## Where the records are

| Source | Location |
|---|---|
| Contract record (JSON line) | ECS task log group, plus OTLP to CloudWatch (Omni) |
| Live DLQ | `aws sqs get-queue-attributes --queue-url <dlq-url> --attribute-names ApproximateNumberOfMessagesVisible` |

```bash
aws logs filter-log-events \
  --log-group-name "<task-log-group>" \
  --filter-pattern '{ $.event_name = "gpu_inference.dead_letter" }' \
  --start-time "$(( ($(date +%s) - 86400) * 1000 ))"
```

## Investigate

1. **Branch on `attributes."inference.dlq.reason"`.**

   - **`malformed_input` — customer error.** The worker rejected the body and sent it to the DLQ
     explicitly. Read `attributes."inference.dlq.reason_code"`; it maps one-to-one onto the error
     `validate_message()` returns in `container/sqs_worker.py`:
     `body_null`, `body_not_string`, `invalid_json`, `not_json_object`, `missing_request_id`,
     `invalid_request_id`, `missing_prompt`, `prompt_not_string`, `prompt_empty`,
     `max_tokens_not_integer`, `max_tokens_out_of_range`, `temperature_not_number`,
     `temperature_out_of_range`, `top_p_not_number`, `top_p_out_of_range`.
     The fix is on the **caller's** side — the request shape is wrong. `inference.request.id` may
     be absent here, because the body may not have parsed far enough to yield one. A spike of one
     reason_code across many callers points at a client regression, not the pipeline.

   - **`processing_exhausted` — system error.** The worker did not delete the message, so SQS
     redrove it after `maxReceiveCount` (3) failed attempts. `inference.request.id` and
     `inference.dlq.receive_count` are always present. Read
     `attributes."inference.dlq.reason_code"`:
     - `inference_failed`: vLLM rejected or errored on the request (read
       `inference.dlq.error_message` for the last exception — human-readable only, do **not** match
       on it). A read timeout to `localhost:8000` points at an overloaded or stuck model server;
       correlate with the tier's cold start via [`cold-start`](../cold-start/SKILL.md) and with GPU
       health via [`gpu-impairment`](../gpu-impairment/SKILL.md).
     - `output_write_failed`: inference succeeded but the result could not be written to S3. Check
       the bucket policy, the task role's `s3:PutObject`, and the bucket's existence/region.
     - `unknown`: an unclassified failure. Fall back to `error_message` and the task logs around
       `inference.dlq.dead_lettered_at`.

2. **Locate the source — `attributes."inference.dlq.source_tier"` (`small` / `large`) and
   `inference.dlq.source_queue`** (the `OriginalQueue` URL on the explicit path, or the
   `DeadLetterQueueSourceArn` on the redrive path). A failure confined to one tier narrows the
   search to that tier's tasks and model.

3. **Pull the request trail.** With `inference.request.id`, find the result or its absence in S3
   (`results/{requestId}.json`), and with `messaging.message.id` and
   `inference.request.enqueued_at` measure how long the request sat before dead-lettering.

4. **Decide the action.**
   - `malformed_input` → report the bad request shape back to the caller; no pipeline change.
   - `processing_exhausted` → a pipeline problem. After root-causing, a verified-good message can
     be redriven from the DLQ. Redrive is a write action on production data: confirm with the
     operator and target only the messages you have validated, never a blanket redrive of the
     whole queue.

## Blind spots

Absence of a record is not absence of the failure. From the contract's producer limits:

- **`processing_exhausted` is emitted by the worker, so a worker that dies mid-request emits
  nothing** (e.g. an OOM, or the host going `IMPAIRED`). SQS still redrives the message and the DLQ
  depth alarm still fires, but **no contract record is written**. If the alarm fired with fewer (or
  zero) matching records than messages in the DLQ, suspect this: check for impaired hosts with
  [`gpu-impairment`](../gpu-impairment/SKILL.md) and inspect the raw DLQ messages directly
  (`aws sqs receive-message`, read-only — do not delete).
- **Match on `reason_code`, not `error_message`.** The error message is free text and will drift;
  the code is the stable contract field.
