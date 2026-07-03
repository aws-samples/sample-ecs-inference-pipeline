"""
Property 4: SQS long-polling consumption

For any invocation of the SQS ReceiveMessage API by the worker, the WaitTimeSeconds
parameter SHALL be greater than 0, ensuring pull-based long-polling rather than
short-polling.

Validates: Requirements 4.2
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "container"))

import sqs_worker


class TestSQSLongPollingUnit:
    """Unit tests for SQS long-polling configuration."""

    def test_long_poll_wait_seconds_constant_is_positive(self):
        """LONG_POLL_WAIT_SECONDS must be greater than 0."""
        assert sqs_worker.LONG_POLL_WAIT_SECONDS > 0

    def test_long_poll_wait_seconds_is_integer(self):
        """LONG_POLL_WAIT_SECONDS must be an integer."""
        assert isinstance(sqs_worker.LONG_POLL_WAIT_SECONDS, int)

    def test_poll_queue_passes_wait_time_to_receive_message(self):
        """poll_queue must pass WaitTimeSeconds to every receive_message call."""
        sqs_client = MagicMock()
        sqs_client.receive_message.side_effect = [
            {"Messages": []},
            KeyboardInterrupt(),
        ]

        sqs_worker.poll_queue(sqs_client)

        call_kwargs = sqs_client.receive_message.call_args_list[0][1]
        assert "WaitTimeSeconds" in call_kwargs
        assert call_kwargs["WaitTimeSeconds"] > 0


class TestSQSLongPollingProperty:
    """
    Property-based test for SQS long-polling consumption.

    **Validates: Requirements 4.2**

    We generate random numbers of polling iterations and verify that every
    ReceiveMessage call uses WaitTimeSeconds > 0.
    """

    @given(num_iterations=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_all_receive_message_calls_use_long_polling(self, num_iterations):
        """
        Property 4: SQS long-polling consumption

        For any number of polling iterations, every ReceiveMessage call
        must have WaitTimeSeconds > 0.

        **Validates: Requirements 4.2**
        """
        sqs_client = MagicMock()

        # Return empty messages for num_iterations, then stop
        side_effects = [{"Messages": []} for _ in range(num_iterations)]
        side_effects.append(KeyboardInterrupt())
        sqs_client.receive_message.side_effect = side_effects

        sqs_worker.poll_queue(sqs_client)

        # Verify every receive_message call used WaitTimeSeconds > 0
        assert sqs_client.receive_message.call_count == num_iterations + 1 or \
               sqs_client.receive_message.call_count == num_iterations

        for call_obj in sqs_client.receive_message.call_args_list:
            kwargs = call_obj[1]
            assert "WaitTimeSeconds" in kwargs, (
                f"ReceiveMessage call missing WaitTimeSeconds parameter"
            )
            assert kwargs["WaitTimeSeconds"] > 0, (
                f"WaitTimeSeconds must be > 0, got {kwargs['WaitTimeSeconds']}"
            )

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_long_poll_constant_always_positive(self, data):
        """
        Property 4: SQS long-polling consumption (constant check)

        The LONG_POLL_WAIT_SECONDS constant must always be > 0.

        **Validates: Requirements 4.2**
        """
        # Verify the constant is positive regardless of any generated context
        assert sqs_worker.LONG_POLL_WAIT_SECONDS > 0, (
            f"LONG_POLL_WAIT_SECONDS must be > 0, got {sqs_worker.LONG_POLL_WAIT_SECONDS}"
        )
