"""Unit tests for SQS worker message validation and DLQ routing."""

import json
import uuid
from unittest.mock import MagicMock

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "container"))

from sqs_worker import validate_message, route_to_dlq


# --- Helper ---
def make_valid_message(**overrides):
    msg = {
        "requestId": str(uuid.uuid4()),
        "prompt": "Tell me a joke",
    }
    msg.update(overrides)
    return json.dumps(msg)


# --- validate_message: valid inputs ---

class TestValidateMessageValid:
    def test_valid_required_fields_only(self):
        body = make_valid_message()
        result, error = validate_message(body)
        assert error is None
        assert result["prompt"] == "Tell me a joke"

    def test_valid_all_optional_fields(self):
        body = make_valid_message(maxTokens=256, temperature=0.7, topP=0.9)
        result, error = validate_message(body)
        assert error is None
        assert result["maxTokens"] == 256

    def test_valid_max_tokens_boundary_low(self):
        body = make_valid_message(maxTokens=1)
        result, error = validate_message(body)
        assert error is None

    def test_valid_max_tokens_boundary_high(self):
        body = make_valid_message(maxTokens=2048)
        result, error = validate_message(body)
        assert error is None

    def test_valid_temperature_boundary_low(self):
        body = make_valid_message(temperature=0.0)
        result, error = validate_message(body)
        assert error is None

    def test_valid_temperature_boundary_high(self):
        body = make_valid_message(temperature=2.0)
        result, error = validate_message(body)
        assert error is None

    def test_valid_top_p_boundary_low(self):
        body = make_valid_message(topP=0.0)
        result, error = validate_message(body)
        assert error is None

    def test_valid_top_p_boundary_high(self):
        body = make_valid_message(topP=1.0)
        result, error = validate_message(body)
        assert error is None

    def test_valid_with_callback_url(self):
        body = make_valid_message(callbackUrl="https://example.com/callback")
        result, error = validate_message(body)
        assert error is None

    def test_valid_integer_temperature(self):
        body = make_valid_message(temperature=1)
        result, error = validate_message(body)
        assert error is None


# --- validate_message: invalid inputs ---

class TestValidateMessageInvalid:
    def test_null_body(self):
        result, error = validate_message(None)
        assert result is None
        assert "null" in error.lower()

    def test_non_string_body(self):
        result, error = validate_message(12345)
        assert result is None
        assert error is not None

    def test_invalid_json(self):
        result, error = validate_message("not json at all")
        assert result is None
        assert "Invalid JSON" in error

    def test_empty_string(self):
        result, error = validate_message("")
        assert result is None
        assert error is not None

    def test_json_array_not_object(self):
        result, error = validate_message("[1, 2, 3]")
        assert result is None
        assert "not a JSON object" in error

    def test_missing_request_id(self):
        body = json.dumps({"prompt": "hello"})
        result, error = validate_message(body)
        assert result is None
        assert "requestId" in error

    def test_invalid_request_id_not_uuid(self):
        body = json.dumps({"requestId": "not-a-uuid", "prompt": "hello"})
        result, error = validate_message(body)
        assert result is None
        assert "UUID v4" in error

    def test_invalid_request_id_uuid_v1(self):
        # UUID v1 has version digit '1' in the third group
        body = json.dumps({"requestId": "550e8400-e29b-11d4-a716-446655440000", "prompt": "hello"})
        result, error = validate_message(body)
        assert result is None
        assert "UUID v4" in error

    def test_missing_prompt(self):
        body = json.dumps({"requestId": str(uuid.uuid4())})
        result, error = validate_message(body)
        assert result is None
        assert "prompt" in error

    def test_empty_prompt(self):
        body = json.dumps({"requestId": str(uuid.uuid4()), "prompt": ""})
        result, error = validate_message(body)
        assert result is None
        assert "non-empty" in error

    def test_whitespace_only_prompt(self):
        body = json.dumps({"requestId": str(uuid.uuid4()), "prompt": "   "})
        result, error = validate_message(body)
        assert result is None
        assert "non-empty" in error

    def test_prompt_not_string(self):
        body = json.dumps({"requestId": str(uuid.uuid4()), "prompt": 123})
        result, error = validate_message(body)
        assert result is None
        assert "string" in error

    def test_max_tokens_zero(self):
        body = make_valid_message(maxTokens=0)
        result, error = validate_message(body)
        assert result is None
        assert "maxTokens" in error

    def test_max_tokens_too_high(self):
        body = make_valid_message(maxTokens=2049)
        result, error = validate_message(body)
        assert result is None
        assert "maxTokens" in error

    def test_max_tokens_negative(self):
        body = make_valid_message(maxTokens=-1)
        result, error = validate_message(body)
        assert result is None
        assert "maxTokens" in error

    def test_max_tokens_not_integer(self):
        body = make_valid_message(maxTokens=3.5)
        result, error = validate_message(body)
        assert result is None
        assert "integer" in error

    def test_max_tokens_boolean(self):
        body = make_valid_message(maxTokens=True)
        result, error = validate_message(body)
        assert result is None
        assert "integer" in error

    def test_temperature_negative(self):
        body = make_valid_message(temperature=-0.1)
        result, error = validate_message(body)
        assert result is None
        assert "temperature" in error

    def test_temperature_too_high(self):
        body = make_valid_message(temperature=2.1)
        result, error = validate_message(body)
        assert result is None
        assert "temperature" in error

    def test_temperature_boolean(self):
        body = make_valid_message(temperature=True)
        result, error = validate_message(body)
        assert result is None
        assert "number" in error

    def test_top_p_negative(self):
        body = make_valid_message(topP=-0.1)
        result, error = validate_message(body)
        assert result is None
        assert "topP" in error

    def test_top_p_too_high(self):
        body = make_valid_message(topP=1.1)
        result, error = validate_message(body)
        assert result is None
        assert "topP" in error

    def test_top_p_boolean(self):
        body = make_valid_message(topP=False)
        result, error = validate_message(body)
        assert result is None
        assert "number" in error


# --- route_to_dlq ---

class TestRouteToDlq:
    def test_sends_message_to_dlq(self):
        sqs_client = MagicMock()
        route_to_dlq(sqs_client, "https://sqs.us-east-1.amazonaws.com/123/dlq", "bad msg", "Invalid JSON")

        sqs_client.send_message.assert_called_once()
        call_kwargs = sqs_client.send_message.call_args[1]
        assert call_kwargs["QueueUrl"] == "https://sqs.us-east-1.amazonaws.com/123/dlq"
        assert call_kwargs["MessageBody"] == "bad msg"
        assert "ErrorReason" in call_kwargs["MessageAttributes"]
        assert call_kwargs["MessageAttributes"]["ErrorReason"]["StringValue"] == "Invalid JSON"

    def test_handles_none_original_message(self):
        sqs_client = MagicMock()
        route_to_dlq(sqs_client, "https://sqs.us-east-1.amazonaws.com/123/dlq", None, "null body")

        call_kwargs = sqs_client.send_message.call_args[1]
        assert call_kwargs["MessageBody"] == ""

    def test_includes_original_queue_attribute(self):
        sqs_client = MagicMock()
        route_to_dlq(sqs_client, "https://sqs.us-east-1.amazonaws.com/123/dlq", "msg", "error")

        call_kwargs = sqs_client.send_message.call_args[1]
        assert "OriginalQueue" in call_kwargs["MessageAttributes"]
