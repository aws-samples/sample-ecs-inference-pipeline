"""
Property 7: Malformed request DLQ routing

For any SQS message that fails schema validation (missing requestId, missing
prompt, empty prompt, invalid UUID format, maxTokens outside 1-2048,
temperature outside 0.0-2.0, topP outside 0.0-1.0, invalid JSON, or
non-object JSON), the worker SHALL route the message to the dead-letter queue
and SHALL NOT attempt inference processing.

Validates: Requirements 6.3
"""

import json
import os
import sys
import uuid
from unittest.mock import MagicMock, patch, call

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "container"))

import sqs_worker


# --- Helpers ---

def make_sqs_message(body_str):
    """Create a mock SQS message dict from a raw body string."""
    return {
        "MessageId": f"msg-{uuid.uuid4().hex[:8]}",
        "ReceiptHandle": f"receipt-{uuid.uuid4().hex[:12]}",
        "Body": body_str,
    }


# --- Strategies for generating invalid messages ---

def valid_uuid_v4():
    """Generate a valid UUID v4 string."""
    return st.uuids(version=4).map(str)


def invalid_uuid_strategy():
    """Generate strings that are NOT valid UUID v4."""
    return st.one_of(
        st.just("not-a-uuid"),
        st.just("12345"),
        st.just(""),
        # UUID v1 format (version digit '1' in third group)
        st.just("550e8400-e29b-11d4-a716-446655440000"),
        # Random short strings
        st.text(min_size=1, max_size=30).filter(
            lambda s: not sqs_worker.UUID_V4_PATTERN.match(s)
        ),
        # Integers/numbers as strings
        st.integers(min_value=0, max_value=999999).map(str),
    )


def non_empty_prompt():
    """Generate a non-empty prompt string."""
    return st.text(min_size=1, max_size=100).filter(lambda s: len(s.strip()) > 0)


@st.composite
def missing_request_id_strategy(draw):
    """Generate a message dict missing the requestId field."""
    prompt = draw(non_empty_prompt())
    msg = {"prompt": prompt}
    # Optionally add valid optional fields
    if draw(st.booleans()):
        msg["maxTokens"] = draw(st.integers(min_value=1, max_value=2048))
    return json.dumps(msg)


@st.composite
def missing_prompt_strategy(draw):
    """Generate a message dict missing the prompt field."""
    request_id = draw(valid_uuid_v4())
    msg = {"requestId": request_id}
    if draw(st.booleans()):
        msg["maxTokens"] = draw(st.integers(min_value=1, max_value=2048))
    return json.dumps(msg)


@st.composite
def empty_prompt_strategy(draw):
    """Generate a message with an empty or whitespace-only prompt."""
    request_id = draw(valid_uuid_v4())
    prompt = draw(st.one_of(
        st.just(""),
        st.just("   "),
        st.just("\t"),
        st.just("\n"),
        st.from_regex(r"^\s+$", fullmatch=True),
    ))
    return json.dumps({"requestId": request_id, "prompt": prompt})


@st.composite
def invalid_uuid_request_id_strategy(draw):
    """Generate a message with an invalid UUID format for requestId."""
    request_id = draw(invalid_uuid_strategy())
    prompt = draw(non_empty_prompt())
    return json.dumps({"requestId": request_id, "prompt": prompt})


@st.composite
def max_tokens_out_of_range_strategy(draw):
    """Generate a message with maxTokens outside 1-2048."""
    request_id = draw(valid_uuid_v4())
    prompt = draw(non_empty_prompt())
    max_tokens = draw(st.one_of(
        st.integers(max_value=0),
        st.integers(min_value=2049, max_value=100000),
    ))
    return json.dumps({"requestId": request_id, "prompt": prompt, "maxTokens": max_tokens})


@st.composite
def temperature_out_of_range_strategy(draw):
    """Generate a message with temperature outside 0.0-2.0."""
    request_id = draw(valid_uuid_v4())
    prompt = draw(non_empty_prompt())
    temperature = draw(st.one_of(
        st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.floats(min_value=2.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    ))
    return json.dumps({"requestId": request_id, "prompt": prompt, "temperature": temperature})


@st.composite
def top_p_out_of_range_strategy(draw):
    """Generate a message with topP outside 0.0-1.0."""
    request_id = draw(valid_uuid_v4())
    prompt = draw(non_empty_prompt())
    top_p = draw(st.one_of(
        st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    ))
    return json.dumps({"requestId": request_id, "prompt": prompt, "topP": top_p})


def invalid_json_strategy():
    """Generate strings that are not valid JSON."""
    return st.one_of(
        st.just("{bad json"),
        st.just("not json at all"),
        st.just("{\"unclosed\": "),
        st.just("{'single_quotes': 'bad'}"),
        st.text(min_size=1, max_size=50).filter(lambda s: _is_invalid_json(s)),
    )


def _is_invalid_json(s):
    """Return True if s is not valid JSON."""
    try:
        json.loads(s)
        return False
    except (json.JSONDecodeError, ValueError):
        return True


def non_object_json_strategy():
    """Generate valid JSON that is not an object (arrays, strings, numbers, etc.)."""
    return st.one_of(
        st.lists(st.integers(), min_size=0, max_size=5).map(json.dumps),
        st.text(min_size=0, max_size=20).map(json.dumps),
        st.integers().map(json.dumps),
        st.floats(allow_nan=False, allow_infinity=False).map(json.dumps),
        st.just("null"),
        st.just("true"),
        st.just("false"),
    )


@st.composite
def any_malformed_message_strategy(draw):
    """Generate any type of malformed message."""
    return draw(st.one_of(
        missing_request_id_strategy(),
        missing_prompt_strategy(),
        empty_prompt_strategy(),
        invalid_uuid_request_id_strategy(),
        max_tokens_out_of_range_strategy(),
        temperature_out_of_range_strategy(),
        top_p_out_of_range_strategy(),
        invalid_json_strategy(),
        non_object_json_strategy(),
    ))


# --- Property-Based Tests ---

class TestMalformedRequestDlqRoutingProperty:
    """
    Property-based test for malformed request DLQ routing.

    **Validates: Requirements 6.3**
    """

    @given(body=any_malformed_message_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_malformed_messages_routed_to_dlq_without_inference(self, body):
        """
        Property 7: Malformed request DLQ routing

        For any message that fails schema validation, the worker SHALL route
        the message to the dead-letter queue and SHALL NOT attempt inference
        processing.

        **Validates: Requirements 6.3**
        """
        sqs_message = make_sqs_message(body)
        sqs_client = MagicMock()

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123/req-queue"), \
             patch.object(sqs_worker, "DLQ_URL", "https://sqs.us-east-1.amazonaws.com/123/dlq"), \
             patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://output-bucket/results/"), \
             patch("sqs_worker.forward_to_vllm") as mock_vllm, \
             patch("sqs_worker.check_idempotency", return_value=False):

            sqs_worker.process_message(sqs_client, sqs_message)

            # 1. Message MUST be routed to DLQ
            sqs_client.send_message.assert_called_once()
            dlq_call_kwargs = sqs_client.send_message.call_args[1]
            assert dlq_call_kwargs["QueueUrl"] == "https://sqs.us-east-1.amazonaws.com/123/dlq", (
                "Malformed message must be sent to the DLQ"
            )

            # 2. No inference processing must occur
            mock_vllm.assert_not_called(), (
                "forward_to_vllm must NOT be called for malformed messages"
            )

            # 3. Source message must be deleted from the queue (cleanup after DLQ routing)
            sqs_client.delete_message.assert_called_once()
            delete_kwargs = sqs_client.delete_message.call_args[1]
            assert delete_kwargs["ReceiptHandle"] == sqs_message["ReceiptHandle"], (
                "Source message must be deleted after DLQ routing"
            )
