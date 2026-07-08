"""
Tests for the Router Lambda classification and handler logic.

The Router Lambda code lives inline in the CloudFormation template ZipFile.
This module defines the same logic locally so it can be unit-tested without
deploying the stack.

Validates: Requirements FR-1
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Router Lambda source (mirrors the ZipFile inline code in template.yaml)
# ---------------------------------------------------------------------------
# This is intentionally duplicated here so we can unit-test it without
# deploying the stack. If the inline source changes, this must be kept in sync.

SMALL_QUEUE = "https://sqs.us-east-1.amazonaws.com/123/small-queue"
LARGE_QUEUE = "https://sqs.us-east-1.amazonaws.com/123/large-queue"
PROMPT_THRESH = 2000
TOKENS_THRESH = 1024
BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"

AMBIGUOUS_MIN = 500
AMBIGUOUS_MAX = PROMPT_THRESH


def _classify_with_bedrock(bedrock_client, prompt, model_id=BEDROCK_MODEL_ID):
    """Ask a small Bedrock model whether the prompt needs the large tier."""
    payload = {
        "messages": [{
            "role": "user",
            "content": (
                "Classify this AI inference request as 'small' (simple factual lookup, "
                "short answer, basic summary) or 'large' (complex reasoning, multi-step "
                "analysis, code generation, creative writing, or nuanced judgment).\n\n"
                f"Request: {prompt[:500]}\n\n"
                "Reply with exactly one word: small or large."
            )
        }],
        "inferenceConfig": {"maxTokens": 10, "temperature": 0},
    }
    try:
        resp = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        answer = result["output"]["message"]["content"][0]["text"].strip().lower()
        tier = "large" if "large" in answer else "small"
        return tier
    except Exception:
        return "small"


def classify(body, sqs_client, bedrock_client, cw_client,
             small_queue=SMALL_QUEUE, large_queue=LARGE_QUEUE,
             prompt_thresh=PROMPT_THRESH, tokens_thresh=TOKENS_THRESH):
    """Classify and route a request body dict."""
    prompt = body.get("prompt", "")
    max_tokens = body.get("maxTokens", 256)
    route = body.get("route", "")

    # Explicit overrides always win
    if route == "large":
        return "large"
    if route == "small":
        return "small"

    # Clear-cut cases
    if len(prompt) > prompt_thresh:
        return "large"
    if max_tokens > tokens_thresh:
        return "large"

    # Ambiguous zone: use Bedrock
    if AMBIGUOUS_MIN < len(prompt) <= AMBIGUOUS_MAX:
        return _classify_with_bedrock(bedrock_client, prompt)

    return "small"


def handler(event, context, sqs_client, bedrock_client, cw_client,
            small_queue=SMALL_QUEUE, large_queue=LARGE_QUEUE,
            prompt_thresh=PROMPT_THRESH, tokens_thresh=TOKENS_THRESH):
    """Lambda handler — parse body, classify, enqueue."""
    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON"})}

    tier = classify(body, sqs_client, bedrock_client, cw_client,
                    small_queue, large_queue, prompt_thresh, tokens_thresh)
    queue_url = large_queue if tier == "large" else small_queue

    sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(body))

    try:
        cw_client.put_metric_data(
            Namespace="Custom/ECSInference",
            MetricData=[{
                "MetricName": "RoutedRequests",
                "Value": 1,
                "Unit": "Count",
                "Dimensions": [{"Name": "Tier", "Value": tier}],
            }],
        )
    except Exception:
        pass  # Metric publishing failure must not block routing

    return {
        "statusCode": 202,
        "body": json.dumps({"requestId": body.get("requestId"), "tier": tier}),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(body_dict=None, body_str=None):
    """Build a Lambda event with a JSON body."""
    if body_dict is not None:
        return {"body": json.dumps(body_dict)}
    return {"body": body_str}


def make_clients():
    """Return mock SQS, Bedrock, and CloudWatch clients."""
    sqs = MagicMock()
    bedrock = MagicMock()
    cw = MagicMock()
    return sqs, bedrock, cw


def make_bedrock_response(answer: str):
    """Build a mock Bedrock invoke_model response."""
    import io
    payload = {"output": {"message": {"content": [{"text": answer}]}}}
    mock_resp = MagicMock()
    mock_resp.__getitem__ = lambda self, key: (
        io.BytesIO(json.dumps(payload).encode()) if key == "body" else MagicMock()
    )
    return mock_resp


# ---------------------------------------------------------------------------
# TestRouterLambdaClassify
# ---------------------------------------------------------------------------

class TestRouterLambdaClassify:
    """
    Unit tests for the classify() routing logic.

    Validates: Requirements FR-1.1 through FR-1.7
    """

    def test_short_prompt_routes_to_small(self):
        """A short prompt with low maxTokens must be classified as 'small'. (FR-1.1)"""
        sqs, bedrock, cw = make_clients()
        body = {"prompt": "What is the capital of France?", "maxTokens": 64}
        result = classify(body, sqs, bedrock, cw)
        assert result == "small", f"Expected 'small', got '{result}'"

    def test_long_prompt_routes_to_large(self):
        """A prompt longer than PROMPT_THRESH must be classified as 'large'. (FR-1.2)"""
        sqs, bedrock, cw = make_clients()
        long_prompt = "x" * (PROMPT_THRESH + 1)
        body = {"prompt": long_prompt, "maxTokens": 64}
        result = classify(body, sqs, bedrock, cw)
        assert result == "large", (
            f"Expected 'large' for prompt length {len(long_prompt)}, got '{result}'"
        )

    def test_high_max_tokens_routes_to_large(self):
        """maxTokens above TOKENS_THRESH must route to large regardless of prompt length. (FR-1.3)"""
        sqs, bedrock, cw = make_clients()
        body = {"prompt": "short", "maxTokens": TOKENS_THRESH + 1}
        result = classify(body, sqs, bedrock, cw)
        assert result == "large", (
            f"Expected 'large' for maxTokens={TOKENS_THRESH + 1}, got '{result}'"
        )

    def test_explicit_route_large_override(self):
        """route='large' must route to large tier even when heuristics would choose small. (FR-1.4)"""
        sqs, bedrock, cw = make_clients()
        body = {"prompt": "short", "maxTokens": 64, "route": "large"}
        result = classify(body, sqs, bedrock, cw)
        assert result == "large", (
            f"Explicit route='large' must override heuristics, got '{result}'"
        )

    def test_explicit_route_small_override(self):
        """route='small' must route to small tier even when prompt is very long. (FR-1.5)"""
        sqs, bedrock, cw = make_clients()
        long_prompt = "x" * (PROMPT_THRESH + 100)
        body = {"prompt": long_prompt, "maxTokens": 64, "route": "small"}
        result = classify(body, sqs, bedrock, cw)
        assert result == "small", (
            f"Explicit route='small' must override heuristics, got '{result}'"
        )

    def test_bedrock_classifies_large(self):
        """When Bedrock returns 'large', classify must return 'large'. (FR-1.6)"""
        sqs, bedrock, cw = make_clients()
        # Prompt in ambiguous zone: 501-2000 chars
        ambiguous_prompt = "y" * 600
        body = {"prompt": ambiguous_prompt, "maxTokens": 256}

        bedrock.invoke_model.return_value = make_bedrock_response("large")

        result = classify(body, sqs, bedrock, cw)
        assert result == "large", (
            f"Bedrock classification 'large' must be respected, got '{result}'"
        )
        bedrock.invoke_model.assert_called_once()

    def test_bedrock_classifies_small(self):
        """When Bedrock returns 'small', classify must return 'small'. (FR-1.6)"""
        sqs, bedrock, cw = make_clients()
        ambiguous_prompt = "y" * 600
        body = {"prompt": ambiguous_prompt, "maxTokens": 256}

        bedrock.invoke_model.return_value = make_bedrock_response("small")

        result = classify(body, sqs, bedrock, cw)
        assert result == "small", (
            f"Bedrock classification 'small' must be respected, got '{result}'"
        )

    def test_bedrock_exception_falls_back_to_small(self):
        """When Bedrock raises an exception, classify must fall back to 'small'. (FR-1.7)"""
        sqs, bedrock, cw = make_clients()
        ambiguous_prompt = "y" * 600
        body = {"prompt": ambiguous_prompt, "maxTokens": 256}

        bedrock.invoke_model.side_effect = Exception("Bedrock unavailable")

        result = classify(body, sqs, bedrock, cw)
        assert result == "small", (
            f"Bedrock failure must fall back to 'small', got '{result}'"
        )

    def test_prompt_at_ambiguous_lower_boundary_not_bedrock(self):
        """Prompt exactly at AMBIGUOUS_MIN (500) must NOT trigger Bedrock (boundary is exclusive)."""
        sqs, bedrock, cw = make_clients()
        boundary_prompt = "z" * AMBIGUOUS_MIN  # exactly 500 — not in ambiguous zone
        body = {"prompt": boundary_prompt, "maxTokens": 256}
        classify(body, sqs, bedrock, cw)
        bedrock.invoke_model.assert_not_called()

    def test_prompt_just_above_ambiguous_lower_boundary_triggers_bedrock(self):
        """Prompt just above AMBIGUOUS_MIN (501 chars) must trigger Bedrock classification."""
        sqs, bedrock, cw = make_clients()
        bedrock.invoke_model.return_value = make_bedrock_response("small")
        boundary_prompt = "z" * (AMBIGUOUS_MIN + 1)  # 501 chars — in ambiguous zone
        body = {"prompt": boundary_prompt, "maxTokens": 256}
        classify(body, sqs, bedrock, cw)
        bedrock.invoke_model.assert_called_once()


# ---------------------------------------------------------------------------
# TestRouterLambdaHandler
# ---------------------------------------------------------------------------

class TestRouterLambdaHandler:
    """
    Unit tests for the handler() function.

    Validates: Requirements FR-1.8, FR-1.9, FR-1.10
    """

    def test_invalid_json_body_returns_400(self):
        """handler() must return HTTP 400 when the body is not valid JSON. (FR-1.8)"""
        sqs, bedrock, cw = make_clients()
        event = make_event(body_str="NOT VALID JSON {{{")
        result = handler(event, None, sqs, bedrock, cw)
        assert result["statusCode"] == 400, (
            f"Expected 400 for invalid JSON, got {result['statusCode']}"
        )
        body = json.loads(result["body"])
        assert "error" in body
        assert "Invalid JSON" in body["error"]

    def test_empty_body_returns_400(self):
        """handler() must return HTTP 400 when body is None or empty string."""
        sqs, bedrock, cw = make_clients()
        event = {"body": None}
        result = handler(event, None, sqs, bedrock, cw)
        # None body → json.loads(None) raises TypeError → 400
        assert result["statusCode"] == 400

    def test_valid_small_request_returns_202(self):
        """handler() must return HTTP 202 with tier='small' for a short prompt. (FR-1.9)"""
        sqs, bedrock, cw = make_clients()
        body_dict = {
            "requestId": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "Hello world",
            "maxTokens": 64,
        }
        event = make_event(body_dict=body_dict)
        result = handler(event, None, sqs, bedrock, cw)

        assert result["statusCode"] == 202, (
            f"Expected 202 for valid small request, got {result['statusCode']}"
        )
        resp_body = json.loads(result["body"])
        assert resp_body["tier"] == "small"
        assert resp_body["requestId"] == body_dict["requestId"]

        # Verify the message was sent to the small queue
        sqs.send_message.assert_called_once()
        call_kwargs = sqs.send_message.call_args[1]
        assert call_kwargs["QueueUrl"] == SMALL_QUEUE

    def test_valid_large_request_returns_202(self):
        """handler() must return HTTP 202 with tier='large' for a long prompt. (FR-1.10)"""
        sqs, bedrock, cw = make_clients()
        body_dict = {
            "requestId": "550e8400-e29b-41d4-a716-446655440001",
            "prompt": "x" * (PROMPT_THRESH + 10),
            "maxTokens": 64,
        }
        event = make_event(body_dict=body_dict)
        result = handler(event, None, sqs, bedrock, cw)

        assert result["statusCode"] == 202, (
            f"Expected 202 for valid large request, got {result['statusCode']}"
        )
        resp_body = json.loads(result["body"])
        assert resp_body["tier"] == "large"

        # Verify message sent to the large queue
        sqs.send_message.assert_called_once()
        call_kwargs = sqs.send_message.call_args[1]
        assert call_kwargs["QueueUrl"] == LARGE_QUEUE

    def test_cloudwatch_failure_does_not_break_routing(self):
        """handler() must still return 202 even if CloudWatch PutMetricData fails."""
        sqs, bedrock, cw = make_clients()
        cw.put_metric_data.side_effect = Exception("CW unavailable")
        body_dict = {"requestId": "abc", "prompt": "Hello", "maxTokens": 16}
        event = make_event(body_dict=body_dict)
        result = handler(event, None, sqs, bedrock, cw)
        # CloudWatch failure must be swallowed — routing should still succeed
        assert result["statusCode"] == 202

    def test_handler_enqueues_full_body(self):
        """handler() must enqueue the full original request body to SQS."""
        sqs, bedrock, cw = make_clients()
        body_dict = {
            "requestId": "550e8400-e29b-41d4-a716-446655440002",
            "prompt": "Test prompt",
            "maxTokens": 128,
            "temperature": 0.7,
        }
        event = make_event(body_dict=body_dict)
        handler(event, None, sqs, bedrock, cw)

        sqs.send_message.assert_called_once()
        enqueued = json.loads(sqs.send_message.call_args[1]["MessageBody"])
        assert enqueued["requestId"] == body_dict["requestId"]
        assert enqueued["prompt"] == body_dict["prompt"]
        assert enqueued["maxTokens"] == body_dict["maxTokens"]
        assert enqueued["temperature"] == body_dict["temperature"]
