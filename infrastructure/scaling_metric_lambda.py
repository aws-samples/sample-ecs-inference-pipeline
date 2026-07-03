"""
Lambda function that computes a composite scaling metric for the ECS GPU
inference pipeline and publishes it to CloudWatch every 60 seconds.

Metric formula:
    max(queueDepth / queueDepthThreshold, gpuMemoryUtilization / 80)

Environment variables:
    QUEUE_URL               – SQS request queue URL
    QUEUE_DEPTH_THRESHOLD   – queue depth divisor (positive integer)
    CLUSTER_NAME            – ECS cluster name (for Container Insights lookup)
    SERVICE_NAME            – ECS service name (for Container Insights lookup)
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sqs_client = boto3.client("sqs")
cloudwatch_client = boto3.client("cloudwatch")

METRIC_NAMESPACE = "Custom/ECSInference"
METRIC_NAME = "InferenceScalingMetric"


def compute_metric(
    queue_depth: float,
    gpu_memory_utilization: float,
    queue_depth_threshold: float,
) -> float:
    """Pure function – compute the composite scaling metric.

    Returns the maximum of queue-depth ratio and GPU-memory ratio.
    The result is guaranteed to be non-negative.
    """
    value = max(
        queue_depth / queue_depth_threshold,
        gpu_memory_utilization / 80,
    )
    return max(value, 0.0)


def _get_queue_depth(queue_url: str) -> int:
    """Return the approximate number of visible messages in the SQS queue."""
    response = sqs_client.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages"],
    )
    return int(response["Attributes"]["ApproximateNumberOfMessages"])


def _get_gpu_memory_utilization(cluster_name: str, service_name: str) -> float:
    """Return the average GPU memory utilization (%) from Container Insights.

    Queries the last 5 minutes and returns the most recent data-point.
    Falls back to 0.0 when no data is available (e.g. no running tasks).
    """
    now = datetime.now(timezone.utc)
    response = cloudwatch_client.get_metric_statistics(
        Namespace="ECS/ContainerInsights",
        MetricName="GPUMemoryUtilization",
        Dimensions=[
            {"Name": "ClusterName", "Value": cluster_name},
            {"Name": "ServiceName", "Value": service_name},
        ],
        StartTime=now - timedelta(minutes=5),
        EndTime=now,
        Period=60,
        Statistics=["Average"],
    )

    datapoints = response.get("Datapoints", [])
    if not datapoints:
        logger.info("No GPU memory utilization data available – defaulting to 0.0")
        return 0.0

    # Return the most recent data-point
    latest = max(datapoints, key=lambda dp: dp["Timestamp"])
    return latest["Average"]


def _publish_metric(value: float, cluster_name: str, service_name: str) -> None:
    """Publish the composite scaling metric to CloudWatch."""
    cloudwatch_client.put_metric_data(
        Namespace=METRIC_NAMESPACE,
        MetricData=[
            {
                "MetricName": METRIC_NAME,
                "Value": value,
                "Unit": "None",
                "Dimensions": [
                    {"Name": "ClusterName", "Value": cluster_name},
                    {"Name": "ServiceName", "Value": service_name},
                ],
            }
        ],
    )


def handler(event, context):
    """Lambda entry-point – orchestrates AWS calls and metric publication."""
    queue_url = os.environ["QUEUE_URL"]
    queue_depth_threshold = int(os.environ["QUEUE_DEPTH_THRESHOLD"])
    cluster_name = os.environ["CLUSTER_NAME"]
    service_name = os.environ["SERVICE_NAME"]

    queue_depth = _get_queue_depth(queue_url)
    gpu_utilization = _get_gpu_memory_utilization(cluster_name, service_name)

    metric_value = compute_metric(queue_depth, gpu_utilization, queue_depth_threshold)

    logger.info(
        "queue_depth=%d gpu_utilization=%.2f threshold=%d metric=%.4f",
        queue_depth,
        gpu_utilization,
        queue_depth_threshold,
        metric_value,
    )

    _publish_metric(metric_value, cluster_name, service_name)

    return {"metricValue": metric_value}
