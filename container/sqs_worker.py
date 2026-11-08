"""SQS Worker for GPU Inference Pipeline.

Polls SQS request queue for inference requests, validates messages,
forwards valid requests to the local vLLM endpoint, and routes
malformed messages to the dead-letter queue.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone

import boto3
import requests

import telemetry

# --- Configuration ---
REQUEST_QUEUE_URL = os.environ.get("REQUEST_QUEUE_URL", "")
DLQ_URL = os.environ.get("DLQ_URL", "")
OUTPUT_DESTINATION = os.environ.get("OUTPUT_DESTINATION", "")

VLLM_ENDPOINT = "http://localhost:8000/v1/completions"
ECS_TASK_ARN = os.environ.get("ECS_TASK_ARN", "unknown")

# Long-polling wait time in seconds (MUST be > 0)
LONG_POLL_WAIT_SECONDS = 20

# Retry configuration for output writes
MAX_WRITE_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1

# SQS RedrivePolicy maxReceiveCount for the request queue. A processing failure on
# this receive is the last one before SQS moves the message to the DLQ.
MAX_RECEIVE_COUNT = int(os.environ.get("MAX_RECEIVE_COUNT", "3"))

# SQS system attributes the worker needs for telemetry (enqueue time, attempt count).
SQS_SYSTEM_ATTRIBUTES = ["ApproximateReceiveCount", "SentTimestamp"]

# UUID v4 regex pattern
UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

telemetry.configure_logging(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# One cold_start record per task, after the first successful inference.
_cold_start = telemetry.ColdStartTracker()


def _task_arn():
    """Task ARN for result payloads: ECS_TASK_ARN if set, else the task metadata endpoint."""
    if ECS_TASK_ARN and ECS_TASK_ARN != "unknown":
        return ECS_TASK_ARN
    try:
        return telemetry.ecs_context().get("task_arn", "unknown")
    except Exception:  # noqa: BLE001 - never fail a write over telemetry
        return "unknown"


def _emit_dead_letter(message, reason, reason_code, error, request_id=None):
    """Emit a gpu_inference.dead_letter contract record. Never raises."""
    try:
        record = telemetry.dead_letter_record(
            telemetry.ecs_context(), message, REQUEST_QUEUE_URL, reason, reason_code,
            error, request_id=request_id,
        )
        telemetry.emit(record)
    except Exception as e:  # noqa: BLE001
        logger.warning("telemetry: dead_letter emission failed: %s", e)


def _emit_if_exhausted(message, request_id, reason_code, error):
    """On the final permitted receive, record that SQS will dead-letter this message."""
    attrs = message.get("Attributes") or {}
    try:
        receive_count = int(attrs.get("ApproximateReceiveCount", "0"))
    except (TypeError, ValueError):
        return
    if receive_count >= MAX_RECEIVE_COUNT:
        _emit_dead_letter(message, "processing_exhausted", reason_code, error, request_id)


def _emit_cold_start(message, request_id, first_token_at):
    """Emit the task's gpu_inference.cold_start record once. Never raises."""
    if not _cold_start.pending:
        return
    try:
        record = _cold_start.observe(telemetry.ecs_context(), message, request_id, first_token_at)
        if record:
            telemetry.emit(record)
    except Exception as e:  # noqa: BLE001
        logger.warning("telemetry: cold_start emission failed: %s", e)


def validate_message(body):
    """Validate an SQS message body against the inference request schema.

    Args:
        body: Raw message body string.

    Returns:
        Tuple of (parsed_message, None) on success,
        or (None, error_string) on validation failure.
    """
    # Parse JSON
    if body is None:
        return None, "Message body is null"

    if not isinstance(body, str):
        return None, "Message body is not a string"

    try:
        message = json.loads(body)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"Invalid JSON: {e}"

    if not isinstance(message, dict):
        return None, "Message is not a JSON object"

    # Validate requestId: required, valid UUID v4
    if "requestId" not in message:
        return None, "Missing required field: requestId"

    request_id = message["requestId"]
    if not isinstance(request_id, str) or not UUID_V4_PATTERN.match(request_id):
        return None, f"Invalid requestId: must be a valid UUID v4, got '{request_id}'"

    # Validate prompt: required, non-empty string
    if "prompt" not in message:
        return None, "Missing required field: prompt"

    prompt = message["prompt"]
    if not isinstance(prompt, str):
        return None, "Field 'prompt' must be a string"

    if len(prompt.strip()) == 0:
        return None, "Field 'prompt' must be non-empty"

    # Validate optional maxTokens: integer in range 1-2048
    if "maxTokens" in message:
        max_tokens = message["maxTokens"]
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            return None, "Field 'maxTokens' must be an integer"
        if max_tokens < 1 or max_tokens > 2048:
            return None, f"Field 'maxTokens' must be between 1 and 2048, got {max_tokens}"

    # Validate optional temperature: float/int in range 0.0-2.0
    if "temperature" in message:
        temperature = message["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            return None, "Field 'temperature' must be a number"
        if temperature < 0.0 or temperature > 2.0:
            return None, f"Field 'temperature' must be between 0.0 and 2.0, got {temperature}"

    # Validate optional topP: float/int in range 0.0-1.0
    if "topP" in message:
        top_p = message["topP"]
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            return None, "Field 'topP' must be a number"
        if top_p < 0.0 or top_p > 1.0:
            return None, f"Field 'topP' must be between 0.0 and 1.0, got {top_p}"

    return message, None


def route_to_dlq(sqs_client, dlq_url, original_message, error_reason):
    """Send a malformed message to the dead-letter queue with error metadata.

    Args:
        sqs_client: Boto3 SQS client.
        dlq_url: URL of the dead-letter queue.
        original_message: The original message body string.
        error_reason: Description of why the message was rejected.
    """
    logger.warning("Routing message to DLQ: %s", error_reason)

    sqs_client.send_message(
        QueueUrl=dlq_url,
        MessageBody=original_message if original_message is not None else "",
        MessageAttributes={
            "ErrorReason": {
                "DataType": "String",
                "StringValue": error_reason,
            },
            "OriginalQueue": {
                "DataType": "String",
                "StringValue": REQUEST_QUEUE_URL,
            },
        },
    )


def check_idempotency(request_id):
    """Check if a result already exists for this requestId in the output destination.

    Args:
        request_id: The UUID v4 request identifier.

    Returns:
        True if a result already exists (skip processing), False otherwise.
    """
    if not OUTPUT_DESTINATION:
        return False

    try:
        if OUTPUT_DESTINATION.startswith("s3://"):
            s3_client = boto3.client("s3")
            # Parse bucket and prefix from s3://bucket/prefix/
            parts = OUTPUT_DESTINATION.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
            key = f"{prefix}/{request_id}.json" if prefix else f"{request_id}.json"

            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                logger.info("Idempotency check: result already exists for requestId=%s", request_id)
                return True
            except s3_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                raise
        else:
            # For SQS response queue, we cannot check idempotency easily
            # so we skip the check
            return False
    except Exception as e:
        logger.warning("Idempotency check failed for requestId=%s: %s", request_id, e)
        return False


def forward_to_vllm(message):
    """Forward a validated inference request to the local vLLM endpoint.

    Args:
        message: Parsed and validated message dict with prompt and optional params.

    Returns:
        Tuple of (response_dict, processing_time_ms) on success,
        or raises an exception on failure.
    """
    payload = {
        "model": os.environ.get("MODEL_NAME", "default"),
        "prompt": message["prompt"],
        "max_tokens": message.get("maxTokens", 256),
        "temperature": message.get("temperature", 0.7),
        "top_p": message.get("topP", 1.0),
    }

    start_time = time.monotonic()
    response = requests.post(VLLM_ENDPOINT, json=payload, timeout=30)
    processing_time_ms = int((time.monotonic() - start_time) * 1000)

    response.raise_for_status()
    result = response.json()

    # Extract vLLM response fields
    choice = result.get("choices", [{}])[0]
    usage = result.get("usage", {})

    return {
        "text": choice.get("text", ""),
        "usage": {
            "promptTokens": usage.get("prompt_tokens", 0),
            "completionTokens": usage.get("completion_tokens", 0),
            "totalTokens": usage.get("total_tokens", 0),
        },
        "finishReason": choice.get("finish_reason", "stop"),
    }, processing_time_ms


def write_result(result, request_id, processing_time_ms):
    """Write inference result to the output destination with exponential backoff retry.

    Args:
        result: The inference result dict (text, usage, finishReason).
        request_id: The UUID v4 request identifier.
        processing_time_ms: Time taken to process the inference in milliseconds.

    Raises:
        Exception: If all retry attempts fail.
    """
    output_payload = json.dumps({
        "requestId": request_id,
        "status": "success",
        "result": result,
        "processingTimeMs": processing_time_ms,
        "taskArn": _task_arn(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    last_exception = None
    for attempt in range(MAX_WRITE_RETRIES):
        try:
            if OUTPUT_DESTINATION.startswith("s3://"):
                s3_client = boto3.client("s3")
                parts = OUTPUT_DESTINATION.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
                key = f"{prefix}/{request_id}.json" if prefix else f"{request_id}.json"

                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=output_payload,
                    ContentType="application/json",
                )
            else:
                # Assume SQS response queue URL
                sqs_client = boto3.client("sqs")
                sqs_client.send_message(
                    QueueUrl=OUTPUT_DESTINATION,
                    MessageBody=output_payload,
                )

            logger.info("Result written for requestId=%s (attempt %d)", request_id, attempt + 1)
            return
        except Exception as e:
            last_exception = e
            if attempt < MAX_WRITE_RETRIES - 1:
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "Write failed for requestId=%s (attempt %d/%d), retrying in %ds: %s",
                    request_id, attempt + 1, MAX_WRITE_RETRIES, backoff, e,
                )
                time.sleep(backoff)  # noqa: S322 — intentional exponential backoff between S3 write retries

    logger.error("All %d write attempts failed for requestId=%s", MAX_WRITE_RETRIES, request_id)
    raise last_exception


def process_message(sqs_client, message):
    """Orchestrate processing of a single SQS message.

    Flow: validate → check idempotency → forward to vLLM → write result → delete message.

    Args:
        sqs_client: Boto3 SQS client.
        message: SQS message dict from ReceiveMessage response.
    """
    receipt_handle = message["ReceiptHandle"]
    body = message.get("Body", "")
    message_id = message.get("MessageId", "unknown")

    logger.info("Processing message: %s", message_id)

    # Step 1: Validate
    parsed, error = validate_message(body)
    if error:
        route_to_dlq(sqs_client, DLQ_URL, body, error)
        sqs_client.delete_message(
            QueueUrl=REQUEST_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )
        reason_code = telemetry.malformed_reason_code(error)
        if reason_code:
            _emit_dead_letter(message, "malformed_input", reason_code, error)
        else:
            logger.warning("telemetry: no contract reason_code for validation error: %s", error)
        return

    request_id = parsed["requestId"]

    # Step 2: Check idempotency
    if check_idempotency(request_id):
        logger.info("Skipping duplicate requestId=%s", request_id)
        sqs_client.delete_message(
            QueueUrl=REQUEST_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )
        return

    # Step 3: Forward to vLLM
    try:
        result, processing_time_ms = forward_to_vllm(parsed)
    except Exception as e:
        logger.error("vLLM inference failed for requestId=%s: %s", request_id, e)
        _emit_if_exhausted(message, request_id, "inference_failed", e)
        # Let visibility timeout return message to queue for retry
        return
    _emit_cold_start(message, request_id, telemetry.now_utc())

    # Step 4: Write result
    try:
        write_result(result, request_id, processing_time_ms)
    except Exception as e:
        logger.error("Output write failed for requestId=%s after retries: %s", request_id, e)
        _emit_if_exhausted(message, request_id, "output_write_failed", e)
        # Let visibility timeout return message to queue for retry
        return

    # Step 5: Delete processed message
    try:
        sqs_client.delete_message(
            QueueUrl=REQUEST_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )
        logger.info("Successfully processed requestId=%s", request_id)
    except Exception as e:
        logger.error("Failed to delete message for requestId=%s: %s", request_id, e)


def poll_queue(sqs_client):
    """Long-poll the SQS request queue for messages and process them.

    Uses WaitTimeSeconds > 0 for long-polling to reduce empty responses
    and API call costs.

    Args:
        sqs_client: Boto3 SQS client.
    """
    logger.info("Starting SQS long-polling loop (WaitTimeSeconds=%d)", LONG_POLL_WAIT_SECONDS)

    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=REQUEST_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=LONG_POLL_WAIT_SECONDS,
                MessageAttributeNames=["All"],
                AttributeNames=SQS_SYSTEM_ATTRIBUTES,
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            for msg in messages:
                process_message(sqs_client, msg)

        except KeyboardInterrupt:
            logger.info("Polling stopped by user")
            break
        except Exception as e:
            logger.error("Error during polling: %s", e)
            time.sleep(5)  # noqa: S322 — intentional pause to avoid tight crash-loop on polling error


def main():
    """Entry point: create boto3 SQS client and start polling."""
    if not REQUEST_QUEUE_URL:
        logger.error("REQUEST_QUEUE_URL environment variable is not set")
        sys.exit(1)

    if not DLQ_URL:
        logger.error("DLQ_URL environment variable is not set")
        sys.exit(1)

    if not OUTPUT_DESTINATION:
        logger.warning("OUTPUT_DESTINATION not set; results will not be persisted")

    logger.info("SQS Worker starting — queue=%s, output=%s", REQUEST_QUEUE_URL, OUTPUT_DESTINATION)

    sqs_client = boto3.client("sqs")
    poll_queue(sqs_client)


if __name__ == "__main__":
    main()
