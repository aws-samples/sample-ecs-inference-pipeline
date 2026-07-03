"""Unit tests for SQS worker message processing, vLLM forwarding, and result writing."""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "container"))

import sqs_worker


# --- Helpers ---

def make_sqs_message(body_dict=None, body_str=None):
    """Create a mock SQS message dict."""
    if body_dict is not None:
        body = json.dumps(body_dict)
    elif body_str is not None:
        body = body_str
    else:
        body = json.dumps({
            "requestId": str(uuid.uuid4()),
            "prompt": "Tell me a joke",
        })
    return {
        "MessageId": "msg-123",
        "ReceiptHandle": "receipt-abc",
        "Body": body,
    }


def make_valid_request(**overrides):
    msg = {
        "requestId": str(uuid.uuid4()),
        "prompt": "Hello world",
    }
    msg.update(overrides)
    return msg


# --- check_idempotency ---

class TestCheckIdempotency:
    def test_returns_false_when_no_output_destination(self):
        with patch.object(sqs_worker, "OUTPUT_DESTINATION", ""):
            assert sqs_worker.check_idempotency("some-id") is False

    @patch("sqs_worker.boto3")
    def test_returns_true_when_s3_object_exists(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_object.return_value = {}

        with patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://my-bucket/results/"):
            result = sqs_worker.check_idempotency("test-request-id")

        assert result is True
        mock_s3.head_object.assert_called_once_with(
            Bucket="my-bucket", Key="results/test-request-id.json"
        )

    @patch("sqs_worker.boto3")
    def test_returns_false_when_s3_object_not_found(self, mock_boto3):
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        # Use real ClientError so the except clause matches properly
        mock_s3.exceptions.ClientError = ClientError
        error_response = {"Error": {"Code": "404"}}
        mock_s3.head_object.side_effect = ClientError(error_response, "HeadObject")

        with patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://my-bucket/results/"):
            result = sqs_worker.check_idempotency("test-request-id")

        assert result is False

    def test_returns_false_for_sqs_destination(self):
        with patch.object(sqs_worker, "OUTPUT_DESTINATION", "https://sqs.us-east-1.amazonaws.com/123/queue"):
            assert sqs_worker.check_idempotency("test-id") is False


# --- forward_to_vllm ---

class TestForwardToVllm:
    @patch("sqs_worker.requests")
    def test_forwards_prompt_with_defaults(self, mock_requests):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"text": "A funny joke", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        mock_requests.post.return_value = mock_response

        message = {"prompt": "Tell me a joke", "requestId": str(uuid.uuid4())}
        result, processing_time = sqs_worker.forward_to_vllm(message)

        assert result["text"] == "A funny joke"
        assert result["usage"]["promptTokens"] == 5
        assert result["usage"]["completionTokens"] == 10
        assert result["usage"]["totalTokens"] == 15
        assert result["finishReason"] == "stop"
        assert isinstance(processing_time, int)
        assert processing_time >= 0

    @patch("sqs_worker.requests")
    def test_forwards_optional_params(self, mock_requests):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"text": "result", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 100, "total_tokens": 103},
        }
        mock_requests.post.return_value = mock_response

        message = {
            "requestId": str(uuid.uuid4()),
            "prompt": "Hello",
            "maxTokens": 100,
            "temperature": 0.5,
            "topP": 0.9,
        }
        sqs_worker.forward_to_vllm(message)

        call_kwargs = mock_requests.post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9

    @patch("sqs_worker.requests")
    def test_raises_on_http_error(self, mock_requests):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500 Server Error")
        mock_requests.post.return_value = mock_response

        with pytest.raises(Exception, match="500 Server Error"):
            sqs_worker.forward_to_vllm({"prompt": "test", "requestId": "id"})


# --- write_result ---

class TestWriteResult:
    @patch("sqs_worker.boto3")
    def test_writes_to_s3(self, mock_boto3):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        result = {"text": "hello", "usage": {}, "finishReason": "stop"}
        with patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://bucket/output/"):
            sqs_worker.write_result(result, "req-123", 500)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "bucket"
        assert call_kwargs["Key"] == "output/req-123.json"
        body = json.loads(call_kwargs["Body"])
        assert body["requestId"] == "req-123"
        assert body["status"] == "success"
        assert body["processingTimeMs"] == 500

    @patch("sqs_worker.boto3")
    def test_writes_to_sqs_queue(self, mock_boto3):
        mock_sqs = MagicMock()
        mock_boto3.client.return_value = mock_sqs

        result = {"text": "world", "usage": {}, "finishReason": "stop"}
        with patch.object(sqs_worker, "OUTPUT_DESTINATION", "https://sqs.us-east-1.amazonaws.com/123/resp"):
            sqs_worker.write_result(result, "req-456", 200)

        mock_sqs.send_message.assert_called_once()

    @patch("sqs_worker.time")
    @patch("sqs_worker.boto3")
    def test_retries_with_exponential_backoff(self, mock_boto3, mock_time):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.put_object.side_effect = [Exception("fail1"), Exception("fail2"), None]

        result = {"text": "ok", "usage": {}, "finishReason": "stop"}
        with patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://bucket/out/"):
            sqs_worker.write_result(result, "req-789", 100)

        assert mock_s3.put_object.call_count == 3
        # Verify backoff sleeps: 1s, 2s
        assert mock_time.sleep.call_count == 2
        mock_time.sleep.assert_any_call(1)
        mock_time.sleep.assert_any_call(2)

    @patch("sqs_worker.time")
    @patch("sqs_worker.boto3")
    def test_raises_after_max_retries(self, mock_boto3, mock_time):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.put_object.side_effect = Exception("persistent failure")

        result = {"text": "ok", "usage": {}, "finishReason": "stop"}
        with patch.object(sqs_worker, "OUTPUT_DESTINATION", "s3://bucket/out/"):
            with pytest.raises(Exception, match="persistent failure"):
                sqs_worker.write_result(result, "req-fail", 100)

        assert mock_s3.put_object.call_count == 3


# --- process_message ---

class TestProcessMessage:
    @patch("sqs_worker.write_result")
    @patch("sqs_worker.forward_to_vllm")
    @patch("sqs_worker.check_idempotency", return_value=False)
    def test_happy_path(self, mock_idemp, mock_vllm, mock_write):
        mock_vllm.return_value = ({"text": "hi", "usage": {}, "finishReason": "stop"}, 100)
        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message(body_dict={"requestId": req_id, "prompt": "hello"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            sqs_worker.process_message(sqs_client, msg)

        mock_vllm.assert_called_once()
        mock_write.assert_called_once()
        sqs_client.delete_message.assert_called_once()

    @patch("sqs_worker.route_to_dlq")
    def test_routes_invalid_message_to_dlq(self, mock_dlq):
        sqs_client = MagicMock()
        msg = make_sqs_message(body_str="not json")

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            with patch.object(sqs_worker, "DLQ_URL", "https://dlq-url"):
                sqs_worker.process_message(sqs_client, msg)

        mock_dlq.assert_called_once()
        sqs_client.delete_message.assert_called_once()

    @patch("sqs_worker.forward_to_vllm")
    @patch("sqs_worker.check_idempotency", return_value=True)
    def test_skips_duplicate_request(self, mock_idemp, mock_vllm):
        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message(body_dict={"requestId": req_id, "prompt": "hello"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            sqs_worker.process_message(sqs_client, msg)

        mock_vllm.assert_not_called()
        sqs_client.delete_message.assert_called_once()

    @patch("sqs_worker.write_result")
    @patch("sqs_worker.forward_to_vllm", side_effect=Exception("vLLM down"))
    @patch("sqs_worker.check_idempotency", return_value=False)
    def test_does_not_delete_on_vllm_failure(self, mock_idemp, mock_vllm, mock_write):
        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message(body_dict={"requestId": req_id, "prompt": "hello"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            sqs_worker.process_message(sqs_client, msg)

        mock_write.assert_not_called()
        sqs_client.delete_message.assert_not_called()

    @patch("sqs_worker.write_result", side_effect=Exception("write failed"))
    @patch("sqs_worker.forward_to_vllm")
    @patch("sqs_worker.check_idempotency", return_value=False)
    def test_does_not_delete_on_write_failure(self, mock_idemp, mock_vllm, mock_write):
        mock_vllm.return_value = ({"text": "hi", "usage": {}, "finishReason": "stop"}, 100)
        sqs_client = MagicMock()
        req_id = str(uuid.uuid4())
        msg = make_sqs_message(body_dict={"requestId": req_id, "prompt": "hello"})

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue-url"):
            sqs_worker.process_message(sqs_client, msg)

        sqs_client.delete_message.assert_not_called()


# --- poll_queue ---

class TestPollQueue:
    def test_uses_long_polling_wait_time_greater_than_zero(self):
        """Verify WaitTimeSeconds > 0 in ReceiveMessage calls."""
        assert sqs_worker.LONG_POLL_WAIT_SECONDS > 0

        sqs_client = MagicMock()
        # Return one message then raise KeyboardInterrupt to exit loop
        sqs_client.receive_message.side_effect = [
            {"Messages": []},
            KeyboardInterrupt(),
        ]

        sqs_worker.poll_queue(sqs_client)

        call_kwargs = sqs_client.receive_message.call_args_list[0][1]
        assert call_kwargs["WaitTimeSeconds"] == sqs_worker.LONG_POLL_WAIT_SECONDS
        assert call_kwargs["WaitTimeSeconds"] > 0


# --- main ---

class TestMain:
    @patch("sqs_worker.poll_queue")
    @patch("sqs_worker.boto3")
    def test_exits_if_no_queue_url(self, mock_boto3, mock_poll):
        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", ""):
            with pytest.raises(SystemExit) as exc_info:
                sqs_worker.main()
            assert exc_info.value.code == 1

    @patch("sqs_worker.poll_queue")
    @patch("sqs_worker.boto3")
    def test_exits_if_no_dlq_url(self, mock_boto3, mock_poll):
        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue"):
            with patch.object(sqs_worker, "DLQ_URL", ""):
                with pytest.raises(SystemExit) as exc_info:
                    sqs_worker.main()
                assert exc_info.value.code == 1

    @patch("sqs_worker.poll_queue")
    @patch("sqs_worker.boto3")
    def test_starts_polling_with_valid_config(self, mock_boto3, mock_poll):
        mock_sqs = MagicMock()
        mock_boto3.client.return_value = mock_sqs

        with patch.object(sqs_worker, "REQUEST_QUEUE_URL", "https://queue"):
            with patch.object(sqs_worker, "DLQ_URL", "https://dlq"):
                sqs_worker.main()

        mock_poll.assert_called_once_with(mock_sqs)
