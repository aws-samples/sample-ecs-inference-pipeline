"""Emit records that conform to the GPU inference telemetry contract.

Contract: telemetry/contract (v1). Each record is one JSON object with the envelope
contract_version / event_name / event_id / timestamp / resource / attributes.

Records leave the worker two ways, carrying the same JSON:

* stdout, as a pure-JSON line (no asctime prefix). The awslogs driver ships it to the
  task's log group, where CloudWatch Logs Insights parses the fields directly.
* OpenTelemetry logs, when the worker runs under ``opentelemetry-instrument`` with
  ``OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true``. The OTel handler on the
  root logger exports the record (body = the JSON, attribute ``event.name``) over OTLP to
  the CloudWatch agent sidecar, which forwards it to CloudWatch for Omni.

This module imports only the standard library, ``requests``, and (lazily) ``boto3``,
so the worker and its unit tests run without the OpenTelemetry packages installed.
Telemetry must never break request processing: callers wrap emission in try/except.
"""

import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone

CONTRACT_VERSION = "1.0.0"
TELEMETRY_LOGGER_NAME = "gpu_inference.telemetry"
DEFAULT_SERVICE_NAME = "sqs-worker"

EVENT_DEAD_LETTER = "gpu_inference.dead_letter"
EVENT_COLD_START = "gpu_inference.cold_start"

# Resource keys each event requires (mirrors the contract schemas). A record missing
# any of these is dropped with a warning instead of being emitted invalid.
_COMMON_RESOURCE_KEYS = ("service.name", "cloud.region", "aws.ecs.cluster.arn")
REQUIRED_RESOURCE_KEYS = {
    EVENT_DEAD_LETTER: _COMMON_RESOURCE_KEYS,
    EVENT_COLD_START: _COMMON_RESOURCE_KEYS + ("aws.ecs.task.arn", "host.id", "inference.tier"),
}

# validate_message() error prefix -> contract reason_code (malformed_input).
# Order matters only where one prefix extends another; none do today.
MALFORMED_REASON_CODES = (
    ("Message body is null", "body_null"),
    ("Message body is not a string", "body_not_string"),
    ("Invalid JSON", "invalid_json"),
    ("Message is not a JSON object", "not_json_object"),
    ("Missing required field: requestId", "missing_request_id"),
    ("Invalid requestId", "invalid_request_id"),
    ("Missing required field: prompt", "missing_prompt"),
    ("Field 'prompt' must be a string", "prompt_not_string"),
    ("Field 'prompt' must be non-empty", "prompt_empty"),
    ("Field 'maxTokens' must be an integer", "max_tokens_not_integer"),
    ("Field 'maxTokens' must be between", "max_tokens_out_of_range"),
    ("Field 'temperature' must be a number", "temperature_not_number"),
    ("Field 'temperature' must be between", "temperature_out_of_range"),
    ("Field 'topP' must be a number", "top_p_not_number"),
    ("Field 'topP' must be between", "top_p_out_of_range"),
)

_MODEL_S3_NONE = ("", "none", "s3://none/none/")
_UUID_V4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE
)
_ISO_TS = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$"
)

logger = logging.getLogger(__name__)
_telemetry_logger = logging.getLogger(TELEMETRY_LOGGER_NAME)


# --- logging setup -----------------------------------------------------------------

class _ExcludeTelemetry(logging.Filter):
    """Keep contract records out of the human-readable stdout stream."""

    def filter(self, record):
        return not record.name.startswith(TELEMETRY_LOGGER_NAME)


def configure_logging(level=logging.INFO, stream=None):
    """Install the plain-text and pure-JSON stdout handlers. Idempotent.

    Replaces ``logging.basicConfig``: basicConfig is a no-op once the OpenTelemetry
    logging handler is already on the root logger, which would silently drop the
    worker's stdout logs under ``opentelemetry-instrument``.
    """
    stream = stream or sys.stdout
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, "_gpu_inference_plain", False) for h in root.handlers):
        plain = logging.StreamHandler(stream)
        plain.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        plain.addFilter(_ExcludeTelemetry())
        plain._gpu_inference_plain = True
        root.addHandler(plain)
    if not any(getattr(h, "_gpu_inference_json", False) for h in _telemetry_logger.handlers):
        as_json = logging.StreamHandler(stream)
        as_json.setFormatter(logging.Formatter("%(message)s"))
        as_json._gpu_inference_json = True
        _telemetry_logger.addHandler(as_json)
    _telemetry_logger.setLevel(logging.INFO)
    _telemetry_logger.propagate = True  # reach the OTel handler on the root logger


# --- timestamps --------------------------------------------------------------------

def parse_ts(value):
    """Parse a datetime, epoch milliseconds (int or digit string), or ISO 8601 string.

    Returns an aware UTC datetime, or None if the value is empty or unparseable.
    Handles the 9-digit fractional seconds the ECS task metadata endpoint returns.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    if not isinstance(value, str):
        return None
    m = _ISO_TS.match(value.strip())
    if not m:
        return None
    year, month, day, hour, minute, second, frac, offset = m.groups()
    micros = int((frac or "0")[:6].ljust(6, "0"))
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), micros)
    if offset and offset != "Z":
        sign = 1 if offset[0] == "+" else -1
        digits = offset[1:].replace(":", "")
        dt = dt - sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
    return dt.replace(tzinfo=timezone.utc)


def fmt_ts(dt):
    """Format an aware datetime as RFC 3339 UTC with millisecond precision and Z."""
    dt = parse_ts(dt)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (dt.microsecond // 1000)


def now_utc():
    return datetime.now(timezone.utc)


# --- ECS context -------------------------------------------------------------------

_context_cache = None


def reset_context_cache():
    """Forget cached ECS context (tests)."""
    global _context_cache
    _context_cache = None


def _task_metadata():
    uri = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
    if not uri:
        return {}
    import requests

    resp = requests.get(uri.rstrip("/") + "/task", timeout=2)
    resp.raise_for_status()
    return resp.json()


def _instance_details(cluster_arn, task_arn, region):
    import boto3

    ecs = boto3.client("ecs", region_name=region)
    tasks = ecs.describe_tasks(cluster=cluster_arn, tasks=[task_arn]).get("tasks", [])
    ci_arn = tasks[0].get("containerInstanceArn") if tasks else None
    if not ci_arn:
        return {}
    cis = ecs.describe_container_instances(
        cluster=cluster_arn, containerInstances=[ci_arn]
    ).get("containerInstances", [])
    if not cis:
        return {"container_instance_arn": ci_arn}
    ci = cis[0]
    attrs = {a.get("name"): a.get("value") for a in ci.get("attributes", [])}
    return {
        "container_instance_arn": ci_arn,
        "ec2_instance_id": ci.get("ec2InstanceId"),
        "instance_registered_at": ci.get("registeredAt"),
        "instance_type": attrs.get("ecs.instance-type"),
        "capacity_provider": ci.get("capacityProviderName"),
    }


def _region_from_arn(arn):
    parts = (arn or "").split(":")
    return parts[3] if len(parts) > 3 and parts[3] else None


def ecs_context():
    """Return identity of this task and its instance, cached for the process lifetime.

    Sources: the ECS task metadata endpoint v4 (task ARN, cluster, family, AZ,
    pullStoppedAt), then DescribeTasks / DescribeContainerInstances for the instance.
    Any lookup that fails leaves its keys out; emit() then drops records that need them.
    """
    global _context_cache
    if _context_cache is not None:
        return _context_cache
    ctx = {}
    try:
        meta = _task_metadata()
        ctx.update({
            "task_arn": meta.get("TaskARN"),
            "cluster_arn": meta.get("Cluster"),
            "task_family": meta.get("Family"),
            "task_revision": meta.get("Revision"),
            "availability_zone": meta.get("AvailabilityZone"),
            "pull_stopped_at": meta.get("PullStoppedAt"),
        })
    except Exception as e:  # noqa: BLE001 - telemetry must not fail the worker
        logger.warning("telemetry: task metadata unavailable: %s", e)
    ctx["region"] = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or _region_from_arn(ctx.get("cluster_arn"))
    )
    if ctx.get("cluster_arn") and ctx.get("task_arn"):
        try:
            ctx.update(_instance_details(ctx["cluster_arn"], ctx["task_arn"], ctx["region"]))
        except Exception as e:  # noqa: BLE001
            logger.warning("telemetry: container instance lookup failed: %s", e)
    _context_cache = {k: v for k, v in ctx.items() if v not in (None, "")}
    return _context_cache


# --- record building ---------------------------------------------------------------

def tier():
    """Inference tier from TIER, falling back to the request queue name."""
    value = os.environ.get("TIER", "").strip().lower()
    if value in ("small", "large"):
        return value
    return "large" if "large-request-queue" in os.environ.get("REQUEST_QUEUE_URL", "") else "small"


def model_source():
    return "huggingface" if os.environ.get("MODEL_S3_PATH", "").strip() in _MODEL_S3_NONE else "s3"


def build_resource(ctx, tier_value=None):
    """Contract resource block from ECS context. Unknown keys are omitted."""
    revision = ctx.get("task_revision")
    candidates = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME,
        "cloud.region": ctx.get("region"),
        "cloud.availability_zone": ctx.get("availability_zone"),
        "aws.ecs.cluster.arn": ctx.get("cluster_arn"),
        "aws.ecs.task.arn": ctx.get("task_arn"),
        "aws.ecs.task.family": ctx.get("task_family"),
        "aws.ecs.task.revision": str(revision) if revision not in (None, "") else None,
        "aws.ecs.container_instance.arn": ctx.get("container_instance_arn"),
        "aws.ecs.capacity_provider.name": ctx.get("capacity_provider"),
        "host.id": ctx.get("ec2_instance_id"),
        "host.type": ctx.get("instance_type"),
        "inference.tier": tier_value,
    }
    return {k: v for k, v in candidates.items() if v not in (None, "")}


def envelope(event_name, resource, attributes, timestamp=None):
    return {
        "contract_version": CONTRACT_VERSION,
        "event_name": event_name,
        "event_id": str(uuid.uuid4()),
        "timestamp": fmt_ts(timestamp or now_utc()),
        "resource": resource,
        "attributes": attributes,
    }


def malformed_reason_code(error):
    """Map a validate_message() error string to its contract reason_code, or None."""
    for prefix, code in MALFORMED_REASON_CODES:
        if error.startswith(prefix):
            return code
    return None


def _request_id_from_body(body):
    """Best-effort requestId from a body that failed validation."""
    try:
        value = json.loads(body).get("requestId")
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, str) and _UUID_V4.match(value) else None


def _sqs_attr(message, name):
    return (message.get("Attributes") or {}).get(name)


def dead_letter_record(ctx, message, source_queue, reason, reason_code, error_message,
                       request_id=None, now=None):
    """Build a gpu_inference.dead_letter record for an SQS message."""
    now = now or now_utc()
    attributes = {
        "inference.dlq.reason": reason,
        "inference.dlq.reason_code": reason_code,
        "inference.dlq.source_tier": tier(),
        "inference.dlq.source_queue": source_queue,
        "messaging.message.id": message.get("MessageId") or "unknown",
        "inference.dlq.dead_lettered_at": fmt_ts(now),
    }
    if request_id is None and reason == "malformed_input":
        request_id = _request_id_from_body(message.get("Body"))
    if request_id:
        attributes["inference.request.id"] = request_id
    receive_count = _sqs_attr(message, "ApproximateReceiveCount")
    if receive_count is not None and str(receive_count).isdigit():
        attributes["inference.dlq.receive_count"] = int(receive_count)
    enqueued = parse_ts(_sqs_attr(message, "SentTimestamp"))
    if enqueued:
        attributes["inference.request.enqueued_at"] = fmt_ts(enqueued)
    if error_message:
        attributes["inference.dlq.error_message"] = str(error_message)[:1024]
    return envelope(EVENT_DEAD_LETTER, build_resource(ctx, tier()), attributes, now)


class ColdStartTracker:
    """Emit one gpu_inference.cold_start per task, after its first successful inference.

    Stages: enqueued (SQS SentTimestamp), node_ready (container instance registeredAt),
    image_pulled (task pullStoppedAt), weights_loaded (VLLM_READY_AT, stamped by
    entrypoint.sh at the first good /health), first_token (vLLM response received).

    A task whose first request was enqueued after its image was pulled did not cold
    start for that request (the task was already warm), so no record is produced.
    """

    def __init__(self, weights_loaded_at=None):
        self.pending = True
        self.weights_loaded_at = parse_ts(
            weights_loaded_at if weights_loaded_at is not None else os.environ.get("VLLM_READY_AT")
        )

    def observe(self, ctx, message, request_id, first_token_at):
        """Return a cold_start record for the first inference, else None."""
        if not self.pending:
            return None
        self.pending = False
        enqueued = parse_ts(_sqs_attr(message, "SentTimestamp"))
        image_pulled = parse_ts(ctx.get("pull_stopped_at"))
        node_ready = parse_ts(ctx.get("instance_registered_at"))
        weights_loaded = self.weights_loaded_at
        first_token = parse_ts(first_token_at)
        missing = [n for n, v in (("enqueued", enqueued), ("image_pulled", image_pulled),
                                  ("node_ready", node_ready), ("weights_loaded", weights_loaded),
                                  ("first_token", first_token)) if v is None]
        if missing:
            logger.warning("telemetry: cold_start skipped, missing stages %s", missing)
            return None
        if enqueued > image_pulled:
            logger.info("telemetry: warm start (request enqueued after image pull), no cold_start")
            return None
        node_reused = node_ready < enqueued
        if node_reused:
            node_ready = enqueued  # instance pre-dated the request: provisioning cost 0
        attributes = {
            "inference.model.name": os.environ.get("MODEL_NAME") or "unknown",
            "inference.model.source": model_source(),
            "inference.request.id": request_id,
            "cold_start.stages": {
                "enqueued": fmt_ts(enqueued),
                "node_ready": fmt_ts(node_ready),
                "image_pulled": fmt_ts(image_pulled),
                "weights_loaded": fmt_ts(weights_loaded),
                "first_token": fmt_ts(first_token),
            },
            "cold_start.node_reused": node_reused,
        }
        return envelope(EVENT_COLD_START, build_resource(ctx, tier()), attributes)


# --- emission ----------------------------------------------------------------------

def emit(record):
    """Log a contract record; drop it (with a warning) if required resource keys are absent."""
    required = REQUIRED_RESOURCE_KEYS.get(record.get("event_name"), _COMMON_RESOURCE_KEYS)
    missing = [k for k in required if k not in record.get("resource", {})]
    if missing:
        logger.warning("telemetry: dropped %s record, missing resource %s",
                       record.get("event_name"), missing)
        return False
    _telemetry_logger.info(
        json.dumps(record, separators=(",", ":"), sort_keys=True),
        extra={"event.name": record["event_name"], "contract.version": CONTRACT_VERSION},
    )
    return True
