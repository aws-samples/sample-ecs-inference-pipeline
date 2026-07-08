"""
Property 5: Composite scaling metric calculation

For any pair of non-negative values (queueDepth, gpuMemoryUtilization) and a
positive queueDepthThreshold, the composite scaling metric SHALL equal
max(queueDepth / queueDepthThreshold, gpuMemoryUtilization / 80), and the
result SHALL always be non-negative.

Validates: Requirements 4.3
"""

import os
import sys

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "infrastructure"))

from scaling_metric_lambda import compute_metric


# --- Unit Tests ---

class TestCompositeScalingMetricUnit:
    """Unit tests for composite scaling metric calculation from the design doc."""

    def test_all_zeros(self):
        """compute_metric(0, 0, 5) returns 0.0"""
        assert compute_metric(0, 0, 5) == 0.0

    def test_queue_dominates(self):
        """compute_metric(10, 40, 5) returns 2.0 — queue depth ratio dominates."""
        assert compute_metric(10, 40, 5) == 2.0

    def test_gpu_dominates(self):
        """compute_metric(1, 90, 5) returns 1.125 — GPU memory ratio dominates."""
        assert compute_metric(1, 90, 5) == 1.125


# --- Property-Based Tests ---

class TestCompositeScalingMetricProperty:
    """
    Property-based test for composite scaling metric calculation.

    **Validates: Requirements 4.3**
    """

    @given(
        queue_depth=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        gpu_memory_utilization=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        queue_depth_threshold=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_metric_equals_max_formula_and_non_negative(
        self, queue_depth, gpu_memory_utilization, queue_depth_threshold
    ):
        """
        Property 5: Composite scaling metric calculation

        For any random (queueDepth, gpuMemoryUtilization, queueDepthThreshold)
        tuple, the metric must equal max(qd/threshold, gpu/80) and the result
        must always be non-negative.

        **Validates: Requirements 4.3**
        """
        result = compute_metric(queue_depth, gpu_memory_utilization, queue_depth_threshold)

        expected = max(queue_depth / queue_depth_threshold, gpu_memory_utilization / 80)

        assert result == pytest.approx(expected), (
            f"compute_metric({queue_depth}, {gpu_memory_utilization}, {queue_depth_threshold}) "
            f"returned {result}, expected {expected}"
        )

        assert result >= 0.0, (
            f"compute_metric({queue_depth}, {gpu_memory_utilization}, {queue_depth_threshold}) "
            f"returned {result}, which is negative"
        )


# --- Scaling Lambda Handler Tests ---

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import importlib

import scaling_metric_lambda


class TestScalingLambdaHandler:
    """
    Tests for the scaling_metric_lambda handler, GPU metric fetching, and publishing.

    Validates: Requirements FR-2
    """

    def _make_env(self):
        return {
            "QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/123/req-queue",
            "QUEUE_DEPTH_THRESHOLD": "5",
            "CLUSTER_NAME": "test-cluster",
            "SERVICE_NAME": "test-service",
        }

    @patch("scaling_metric_lambda.cloudwatch_client")
    @patch("scaling_metric_lambda.sqs_client")
    def test_handler_returns_metric_value(self, mock_sqs, mock_cw):
        """handler() must return a dict with metricValue key."""
        mock_sqs.get_queue_attributes.return_value = {
            "Attributes": {"ApproximateNumberOfMessages": "10"}
        }
        mock_cw.get_metric_statistics.return_value = {"Datapoints": []}
        mock_cw.put_metric_data.return_value = {}

        with patch.dict("os.environ", self._make_env()):
            result = scaling_metric_lambda.handler({}, None)

        assert "metricValue" in result
        assert isinstance(result["metricValue"], float)
        assert result["metricValue"] >= 0.0

    @patch("scaling_metric_lambda.cloudwatch_client")
    @patch("scaling_metric_lambda.sqs_client")
    def test_handler_calls_publish_metric(self, mock_sqs, mock_cw):
        """handler() must call put_metric_data exactly once."""
        mock_sqs.get_queue_attributes.return_value = {
            "Attributes": {"ApproximateNumberOfMessages": "3"}
        }
        mock_cw.get_metric_statistics.return_value = {"Datapoints": []}
        mock_cw.put_metric_data.return_value = {}

        with patch.dict("os.environ", self._make_env()):
            scaling_metric_lambda.handler({}, None)

        mock_cw.put_metric_data.assert_called_once()
        call_kwargs = mock_cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == "Custom/ECSInference"

    @patch("scaling_metric_lambda.cloudwatch_client")
    def test_get_gpu_memory_utilization_no_datapoints_returns_zero(self, mock_cw):
        """_get_gpu_memory_utilization() must return 0.0 when Datapoints is empty."""
        mock_cw.get_metric_statistics.return_value = {"Datapoints": []}

        result = scaling_metric_lambda._get_gpu_memory_utilization("cluster", "service")

        assert result == 0.0

    @patch("scaling_metric_lambda.cloudwatch_client")
    def test_get_gpu_memory_utilization_returns_latest_datapoint(self, mock_cw):
        """_get_gpu_memory_utilization() must return the most recent datapoint, not the first."""
        t1 = datetime(2026, 7, 8, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 7, 8, 0, 1, 0, tzinfo=timezone.utc)  # later
        t3 = datetime(2026, 7, 8, 0, 0, 30, tzinfo=timezone.utc)

        mock_cw.get_metric_statistics.return_value = {
            "Datapoints": [
                {"Timestamp": t1, "Average": 30.0},
                {"Timestamp": t2, "Average": 75.0},  # latest — should be returned
                {"Timestamp": t3, "Average": 50.0},
            ]
        }

        result = scaling_metric_lambda._get_gpu_memory_utilization("cluster", "service")

        assert result == 75.0, f"Expected 75.0 (latest datapoint), got {result}"

    @patch("scaling_metric_lambda.cloudwatch_client")
    @patch("scaling_metric_lambda.sqs_client")
    def test_handler_missing_env_var_raises(self, mock_sqs, mock_cw):
        """handler() must raise KeyError when QUEUE_URL env var is missing."""
        import os
        env_without_queue = {k: v for k, v in self._make_env().items() if k != "QUEUE_URL"}

        with patch.dict("os.environ", env_without_queue, clear=False):
            # Remove QUEUE_URL if present from environment
            with patch.object(os, "environ", {**os.environ, **env_without_queue}):
                # Directly test that missing key raises
                import os as real_os
                saved = real_os.environ.pop("QUEUE_URL", None)
                try:
                    with pytest.raises(KeyError):
                        scaling_metric_lambda.handler({}, None)
                finally:
                    if saved is not None:
                        real_os.environ["QUEUE_URL"] = saved

    @patch("scaling_metric_lambda.cloudwatch_client")
    def test_get_gpu_uses_correct_metric_name(self, mock_cw):
        """_get_gpu_memory_utilization() must query TaskGPUMemoryUtilization (not GPUMemoryUtilization)."""
        mock_cw.get_metric_statistics.return_value = {"Datapoints": []}

        scaling_metric_lambda._get_gpu_memory_utilization("cluster", "service")

        call_kwargs = mock_cw.get_metric_statistics.call_args[1]
        assert call_kwargs["MetricName"] == "TaskGPUMemoryUtilization", (
            f"Expected MetricName='TaskGPUMemoryUtilization', got '{call_kwargs['MetricName']}'. "
            "This metric name must match ECS Container Insights Enhanced."
        )
