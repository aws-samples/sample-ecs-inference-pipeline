# Changelog: GPU inference telemetry contract

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: see the
[versioning policy](README.md#versioning-policy).

## [1.0.0] - 2026-11-08 (draft)

### Added

- Envelope shared by all events: `contract_version`, `event_name`, `event_id`, `timestamp`,
  `resource` (OpenTelemetry resource attribute names), `attributes`.
- `gpu_inference.gpu_impairment`: normalized ECS `ACCELERATED_COMPUTE` impairment. XID restricted
  to the 18 codes ECS auto-repair acts on, plus a forensic capture of the tasks on the instance
  before replacement.
- `gpu_inference.dead_letter`: `reason` discriminator (`malformed_input` |
  `processing_exhausted`), per-reason `reason_code` enums, and `source_tier`.
- `gpu_inference.cold_start`: five-stage timestamp series (`enqueued`, `node_ready`,
  `image_pulled`, `weights_loaded`, `first_token`).
- Conformance rules CR-1 through CR-6. CR-4 (stage monotonicity) and CR-5 (XID/status_reason
  consistency) are harness-enforced.
