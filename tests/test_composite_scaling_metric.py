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
