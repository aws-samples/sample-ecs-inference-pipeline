"""
Property 6: Valid request processing round-trip

For any valid inference request message (containing a valid UUID requestId,
non-empty prompt, and optional parameters within their allowed ranges), the
SQS worker SHALL produce a response with the same requestId, a status of
"success", a non-empty result.text, and SHALL delete the source message
from the request queue.

Validates: Requirements 6.1, 6.4
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


# --- Strategies ---

def uuid_v4_strategy():
    """Generate valid UUID v4 strings."""
    return st.uuids(version=4).map(str)


def non_empty_prompt_strategy():
    """Generate non-empty prompt strings."""
    return st.text(min_size=1, max_size=200).filter(lambda s: len(s.strip()) > 0)


def optional_max_tokens_strategy():
    """Generate optional maxTokens values in range 1-2048."""
    return st.one_of(st.none(), st.integers(min_value=1, max_value=2048))


def optional_temperature_strategy():
    """Generate optional temperature values in range 0.0-2.0."""
    return st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))


def optional_top_p_strategy():
    """Generate optional topP values in range 0.0-1.0."""
    return st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_request_strategy(draw):
    """Generate a valid inference request message dict."""
    request_id = draw(uuid_v4_strategy())
    prompt = draw(non_empty_prompt_strategy())
    max_tokens = draw(optional_max_tokens_strategy())
    temperature = draw(optional_temperature_strategy())
    top_p = draw(optional_top_p_strategy())

    msg = {
        "requestId": request_id,
        "prompt": prompt,
    }
    if max_tokens is not None:
        msg["maxTokens"] = max_tokens
    if temperature is not None:
        msg["temperature"] = temperature
    if top_p is not None:
        msg["topP"] = top_p

    return msg


def make_sqs_message(body_dict):
    """Create a mock SQS message dict from a body dict."""
    return {
        "MessageId": f"msg-{uuid.uuid4().hex[:8]}",
        "ReceiptHandle": f"receipt-{uuid.uuid4().hex[:12]}",
        "Body": json.dumps(body_dict),
    }


def make_vllm_response(generated_text="Generated response text."):
    """Create a mock vLLM response dict."""
    return {
        "choices": [{"text": generated_text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }


# --- Unit Tests ---

class TestValidRequestProcessingUnit:
    """Unit tests for valid request processing round-trip."""

    @patch("sqs_worker.write_result")
    @patch("sqs_worker.forward_to_vllm")
    @patch("sqs_worker.check_idempotency", return_value=False)
    def test_response_has_matching_request_id(self, mock_idemp, mock_vllm, mock_write):
        """Response written to output must have the same requestId as the request."""
        mock_vllm.return_value = (
            {"text": "hello", "usage": {"promptTokens": 1, "completionTokens": 2, "totalTokens": 3}, "finishReason": "stop"},
            100,
        )
        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message({"requestId": req_id, "prompt": "test prompt"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            sqs_worker.process_message(sqs_client, msg)

        mock_write.assert_called_once()
        written_request_id = mock_write.call_args[0][1]
        assert written_request_id == req_id

    @patch("sqs_worker.boto3")
    @patch("sqs_worker.forward_to_vllm")
    @patch("sqs_worker.check_idempotency", return_value=False)
    def test_output_payload_has_success_status(self, mock_idemp, mock_vllm, mock_boto3):
        """The output payload written to S3 must have status 'success'."""
        mock_vllm.return_value = (
            {"text": "result", "usage": {"promptTokens": 1, "completionTokens": 2, "totalTokens": 3}, "finishReason": "stop"},
            200,
        )
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message({"requestId": req_id, "prompt": "test"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            with patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://bucket/output/"):
                sqs_worker.process_message(sqs_client, msg)

        body = json.loads(mock_s3.put_object.call_args[1]["Body"])
        assert body["status"] == "success"

    @patch("sqs_worker.boto3")
    @patch("sqs_worker.forward_to_vllm")
    @patch("sqs_worker.check_idempotency", return_value=False)
    def test_output_payload_has_non_empty_result_text(self, mock_idemp, mock_vllm, mock_boto3):
        """The output payload must have a non-empty result.text."""
        mock_vllm.return_value = (
            {"text": "Some generated text", "usage": {"promptTokens": 1, "completionTokens": 5, "totalTokens": 6}, "finishReason": "stop"},
            150,
        )
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message({"requestId": req_id, "prompt": "hello"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            with patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://bucket/output/"):
                sqs_worker.process_message(sqs_client, msg)

        body = json.loads(mock_s3.put_object.call_args[1]["Body"])
        assert len(body["result"]["text"]) > 0

    @patch("sqs_worker.write_result")
    @patch("sqs_worker.forward_to_vllm")
    @patch("sqs_worker.check_idempotency", return_value=False)
    def test_source_message_deleted_after_processing(self, mock_idemp, mock_vllm, mock_write):
        """The source SQS message must be deleted after successful processing."""
        mock_vllm.return_value = (
            {"text": "ok", "usage": {"promptTokens": 1, "completionTokens": 1, "totalTokens": 2}, "finishReason": "stop"},
            50,
        )
        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message({"requestId": req_id, "prompt": "test"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            sqs_worker.process_message(sqs_client, msg)

        sqs_client.delete_message.assert_called_once()


# --- Property-Based Tests ---

class TestValidRequestProcessingProperty:
    """
    Property-based test for valid request processing round-trip.

    **Validates: Requirements 6.1, 6.4**
    """

    @given(request=valid_request_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_response_has_matching_request_id_and_success_status(self, request):
        """
        Property 6: Valid request processing round-trip

        For any valid request message, the response written to output must have
        the same requestId, status "success", non-empty result.text, and the
        source message must be deleted from the queue.

        **Validates: Requirements 6.1, 6.4**
        """
        generated_text = "Model generated output text."
        sqs_message = make_sqs_message(request)

        sqs_client = MagicMock()
        mock_s3 = MagicMock()

        mock_vllm_response = MagicMock()
        mock_vllm_response.status_code = 200
        mock_vllm_response.raise_for_status = MagicMock()
        mock_vllm_response.json.return_value = make_vllm_response(generated_text)

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123/req-queue"), \
             patch.object(sqs_worker, "DLQ_URL", "https://sqs.us-east-1.amazonaws.com/123/dlq"), \
             patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://output-bucket/results/"), \
             patch("sqs_worker.check_idempotency", return_value=False), \
             patch("sqs_worker.requests") as mock_requests, \
             patch("sqs_worker.boto3") as mock_boto3:

            mock_requests.post.return_value = mock_vllm_response
            mock_boto3.client.return_value = mock_s3

            sqs_worker.process_message(sqs_client, sqs_message)

        # 1. Verify source message was deleted
        sqs_client.delete_message.assert_called_once()
        delete_kwargs = sqs_client.delete_message.call_args[1]
        assert delete_kwargs["ReceiptHandle"] == sqs_message["ReceiptHandle"]

        # 2. Verify output was written to S3
        mock_s3.put_object.assert_called_once()
        put_kwargs = mock_s3.put_object.call_args[1]
        output_body = json.loads(put_kwargs["Body"])

        # 3. Verify requestId matches
        assert output_body["requestId"] == request["requestId"], (
            f"Response requestId '{output_body['requestId']}' does not match "
            f"request requestId '{request['requestId']}'"
        )

        # 4. Verify status is "success"
        assert output_body["status"] == "success", (
            f"Response status should be 'success', got '{output_body['status']}'"
        )

        # 5. Verify result.text is non-empty
        assert len(output_body["result"]["text"]) > 0, (
            "Response result.text must be non-empty"
        )
