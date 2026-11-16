# GPU inference investigation skills

Day-two investigation skills, one per failure mode the pipeline emits telemetry for. Each is the
**consumer** side of the [telemetry contract](../../telemetry/contract/README.md): it reads fields
by the names the contract defines and nowhere else, so a producer rename fails contract validation
in CI instead of breaking a runbook silently during an incident.

| Skill | Investigates | Contract event | Schema |
|---|---|---|---|
| [`gpu-impairment`](gpu-impairment/SKILL.md) | A GPU instance marked `IMPAIRED`, before/after auto-repair | `gpu_inference.gpu_impairment` | [`v1/gpu_impairment.schema.json`](../../telemetry/contract/v1/gpu_impairment.schema.json) |
| [`dead-letter`](dead-letter/SKILL.md) | Requests that landed in the DLQ | `gpu_inference.dead_letter` | [`v1/dead_letter.schema.json`](../../telemetry/contract/v1/dead_letter.schema.json) |
| [`cold-start`](cold-start/SKILL.md) | A scale-from-zero cold start that was too slow | `gpu_inference.cold_start` | [`v1/cold_start.schema.json`](../../telemetry/contract/v1/cold_start.schema.json) |

The `consumer_skill` key in [`contract.json`](../../telemetry/contract/contract.json) maps each
event to the skill here that reads it.

## Contract version

These skills read contract **major version 1** (`>=1.0.0 <2.0.0`). Per the contract's versioning
policy, a consumer **must ignore fields it does not recognize** so that a newer-minor producer does
not break it, and **must treat enum values it does not recognize as unclassified** rather than
failing (the XID list, for example, grows by MINOR bump as AWS documents new codes). When the
contract goes to `v2`, each skill's "Contract" section is where the supported range is bumped.

## How a skill is structured

Every `SKILL.md` follows the same shape so they read consistently under pressure:

1. **When to use** — the symptom that sends you here (an alarm, an event, a user report).
2. **Contract** — the event and schema it consumes, and the supported version range.
3. **Where the records are** — the exact log groups / OTLP source from the contract's
   [Producer status](../../telemetry/contract/README.md#producer-status) table.
4. **Investigate** — an ordered decision procedure keyed off contract fields.
5. **Blind spots** — the producer limits (from the same table) that mean *absence of a record is
   not absence of the failure*, so you don't conclude "no problem" from an empty query.

These are plain Markdown runbooks. They name AWS CLI read calls (`describe-*`, `filter-log-events`)
but do not run them for you; copy the command, fill the placeholders, and keep to read-only calls
during an investigation.
