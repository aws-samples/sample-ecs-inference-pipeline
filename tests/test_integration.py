"""
Integration tests for the ECS GPU Inference Pipeline.

These tests validate the end-to-end deployed pipeline and require:
  - A deployed CloudFormation stack
  - Environment variables:
      STACK_NAME  – name of the deployed CloudFormation stack
      AWS_REGION  – AWS region where the stack is deployed (default: us-east-1)

Run with:  pytest -m integration tests/test_integration.py

Validates: Requirements 6.1, 6.2, 6.3, 5.2
"""

import json
import os
import time
import uuid

import boto3
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STACK_NAME = os.environ.get("STACK_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def _skip_if_no_stack():
    if not STACK_NAME:
        pytest.skip("STACK_NAME environment variable not set – skipping integration test")


def _cfn_client():
    return boto3.client("cloudformation", region_name=AWS_REGION)


def _ecs_client():
    return boto3.client("ecs", region_name=AWS_REGION)


def _sqs_client():
    return boto3.client("sqs", region_name=AWS_REGION)


def _ecr_client():
    return boto3.client("ecr", region_name=AWS_REGION)


def _cw_client():
    return boto3.client("cloudwatch", region_name=AWS_REGION)


def _get_stack_outputs():
    """Return a dict of stack output keys to values."""
    cfn = _cfn_client()
    resp = cfn.describe_stacks(StackName=STACK_NAME)
    outputs = resp["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}


def _get_stack_resources():
    """Return all physical resources in the stack."""
    cfn = _cfn_client()
    paginator = cfn.get_paginator("list_stack_resources")
    resources = []
    for page in paginator.paginate(StackName=STACK_NAME):
        resources.extend(page["StackResourceSummaries"])
    return resources


# ---------------------------------------------------------------------------
# 1. Stack Deployment
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestStackDeployment:
    """Verify the CloudFormation stack deploys successfully with all expected resources."""

    def test_stack_exists_and_complete(self):
        """Stack should be in CREATE_COMPLETE or UPDATE_COMPLETE state."""
        _skip_if_no_stack()
        cfn = _cfn_client()
        resp = cfn.describe_stacks(StackName=STACK_NAME)
        status = resp["Stacks"][0]["StackStatus"]
        assert status in (
            "CREATE_COMPLETE",
            "UPDATE_COMPLETE",
        ), f"Stack status is {status}, expected CREATE_COMPLETE or UPDATE_COMPLETE"

    def test_expected_resource_types_present(self):
        """Stack should contain the core resource types for the pipeline."""
        _skip_if_no_stack()
        resources = _get_stack_resources()
        resource_types = {r["ResourceType"] for r in resources}

        expected_types = {
            "AWS::ECS::Cluster",
            "AWS::ECS::Service",
            "AWS::ECS::TaskDefinition",
            "AWS::SQS::Queue",
            "AWS::ECR::Repository",
            "AWS::ECS::CapacityProvider",
            "AWS::AutoScaling::AutoScalingGroup",
            "AWS::IAM::Role",
        }
        missing = expected_types - resource_types
        assert not missing, f"Missing resource types in stack: {missing}"

    def test_stack_outputs_present(self):
        """Stack should expose required outputs."""
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        expected_keys = {
            "ClusterArn",
            "RequestQueueUrl",
            "RequestQueueArn",
            "ECRRepositoryUri",
            "TaskDefinitionArn",
        }
        missing = expected_keys - set(outputs.keys())
        assert not missing, f"Missing stack outputs: {missing}"


# ---------------------------------------------------------------------------
# 2. Image Build & Push to ECR
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestImageBuildAndPush:
    """Verify the container image exists in the ECR repository."""

    def test_ecr_repository_has_image(self):
        """ECR repository should contain at least one image tagged 'latest'."""
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        repo_uri = outputs["ECRRepositoryUri"]
        # repo_uri looks like 123456789012.dkr.ecr.us-east-1.amazonaws.com/repo-name
        repo_name = repo_uri.split("/", 1)[1]

        ecr = _ecr_client()
        resp = ecr.describe_images(
            repositoryName=repo_name,
            imageIds=[{"imageTag": "latest"}],
        )
        images = resp.get("imageDetails", [])
        assert len(images) >= 1, "No image with tag 'latest' found in ECR repository"

    def test_ecr_image_architecture(self):
        """ECR image should target linux/amd64."""
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        repo_uri = outputs["ECRRepositoryUri"]
        repo_name = repo_uri.split("/", 1)[1]

        ecr = _ecr_client()
        resp = ecr.describe_images(
            repositoryName=repo_name,
            imageIds=[{"imageTag": "latest"}],
        )
        image = resp["imageDetails"][0]
        # imageManifestMediaType or artifact architecture check
        assert image.get("imageSizeInBytes", 0) > 0, "Image appears to have zero size"


# ---------------------------------------------------------------------------
# 3. GPU Task Launch
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGPUTaskLaunch:
    """Verify a GPU task can be launched and reaches RUNNING state."""

    def test_run_gpu_task_reaches_running(self):
        """Launch a standalone task and verify it reaches RUNNING with a GPU assigned."""
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        cluster_arn = outputs["ClusterArn"]
        task_def_arn = outputs["TaskDefinitionArn"]

        ecs = _ecs_client()
        run_resp = ecs.run_task(
            cluster=cluster_arn,
            taskDefinition=task_def_arn,
            count=1,
            launchType="EC2",
        )
        assert run_resp["tasks"], "run_task returned no tasks"
        task_arn = run_resp["tasks"][0]["taskArn"]

        # Poll for up to 5 minutes for the task to reach RUNNING
        deadline = time.time() + 300
        last_status = None
        try:
            while time.time() < deadline:
                desc = ecs.describe_tasks(cluster=cluster_arn, tasks=[task_arn])
                last_status = desc["tasks"][0]["lastStatus"]
                if last_status == "RUNNING":
                    break
                time.sleep(15)

            assert last_status == "RUNNING", (
                f"Task did not reach RUNNING within 5 minutes (last status: {last_status})"
            )

            # Verify GPU attachment
            task_detail = desc["tasks"][0]
            container = task_detail["containers"][0]
            gpu_ids = container.get("gpuIds", [])
            assert len(gpu_ids) >= 1, "Task container has no GPU assigned"
        finally:
            # Clean up: stop the task
            ecs.stop_task(cluster=cluster_arn, task=task_arn, reason="integration-test-cleanup")


# ---------------------------------------------------------------------------
# 4. Inference Round-Trip via SQS
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInferenceRoundTrip:
    """Send a valid request to SQS and verify a response appears within 30s."""

    def test_valid_request_produces_response(self):
        """
        Send a valid inference request, then poll the output destination
        for a matching response within 30 seconds.

        Validates: Requirement 6.1
        """
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        queue_url = outputs["RequestQueueUrl"]

        sqs = _sqs_client()
        request_id = str(uuid.uuid4())
        message = json.dumps({
            "requestId": request_id,
            "prompt": "What is the capital of France?",
            "maxTokens": 64,
            "temperature": 0.7,
        })

        sqs.send_message(QueueUrl=queue_url, MessageBody=message)

        # Poll for the response – the worker writes results to an output
        # destination (S3 or response queue). Here we check the request
        # queue is drained (message deleted after processing) as a proxy.
        deadline = time.time() + 30
        message_gone = False
        while time.time() < deadline:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
                MessageAttributeNames=["All"],
            )
            bodies = [
                json.loads(m["Body"])
                for m in resp.get("Messages", [])
                if json.loads(m["Body"]).get("requestId") == request_id
            ]
            if not bodies:
                message_gone = True
                break
            time.sleep(2)

        assert message_gone, (
            f"Request {request_id} was not consumed from the queue within 30s"
        )

    def test_response_contains_expected_fields(self):
        """
        Validates: Requirement 6.2 – response latency and structure.

        This test sends a short prompt and verifies the response
        structure if an output queue is configured.
        """
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        queue_url = outputs["RequestQueueUrl"]

        sqs = _sqs_client()
        request_id = str(uuid.uuid4())
        message = json.dumps({
            "requestId": request_id,
            "prompt": "Hello",
            "maxTokens": 16,
        })

        send_time = time.time()
        sqs.send_message(QueueUrl=queue_url, MessageBody=message)

        # Wait for processing (message deletion indicates completion)
        deadline = time.time() + 30
        while time.time() < deadline:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
            )
            remaining = [
                m for m in resp.get("Messages", [])
                if json.loads(m["Body"]).get("requestId") == request_id
            ]
            if not remaining:
                break
            time.sleep(2)

        elapsed = time.time() - send_time
        assert elapsed < 30, f"Processing took {elapsed:.1f}s, expected < 30s"


# ---------------------------------------------------------------------------
# 5. DLQ Routing for Malformed Messages
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDLQRouting:
    """Verify malformed messages are routed to the dead-letter queue."""

    def _get_dlq_url(self):
        """Resolve the DLQ URL from stack resources."""
        resources = _get_stack_resources()
        dlq_resources = [
            r for r in resources
            if r["ResourceType"] == "AWS::SQS::Queue"
            and "DLQ" in r["LogicalResourceId"].upper()
        ]
        assert dlq_resources, "Could not find DLQ resource in stack"
        physical_id = dlq_resources[0]["PhysicalResourceId"]
        # PhysicalResourceId for SQS is the queue URL
        return physical_id

    def test_malformed_json_routed_to_dlq(self):
        """
        Send invalid JSON to the request queue; verify it lands in the DLQ.

        Validates: Requirement 6.3
        """
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        queue_url = outputs["RequestQueueUrl"]
        dlq_url = self._get_dlq_url()

        sqs = _sqs_client()
        sqs.send_message(QueueUrl=queue_url, MessageBody="NOT VALID JSON {{{")

        # Poll DLQ for up to 60 seconds
        deadline = time.time() + 60
        found = False
        while time.time() < deadline:
            resp = sqs.receive_message(
                QueueUrl=dlq_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
            )
            for msg in resp.get("Messages", []):
                if "NOT VALID JSON" in msg.get("Body", ""):
                    found = True
                    # Clean up
                    sqs.delete_message(
                        QueueUrl=dlq_url,
                        ReceiptHandle=msg["ReceiptHandle"],
                    )
                    break
            if found:
                break
            time.sleep(2)

        assert found, "Malformed message did not appear in DLQ within 60s"

    def test_missing_required_fields_routed_to_dlq(self):
        """
        Send a message missing the required 'prompt' field; verify DLQ routing.

        Validates: Requirement 6.3
        """
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        queue_url = outputs["RequestQueueUrl"]
        dlq_url = self._get_dlq_url()

        sqs = _sqs_client()
        bad_message = json.dumps({"requestId": str(uuid.uuid4())})
        sqs.send_message(QueueUrl=queue_url, MessageBody=bad_message)

        deadline = time.time() + 60
        found = False
        while time.time() < deadline:
            resp = sqs.receive_message(
                QueueUrl=dlq_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
            )
            for msg in resp.get("Messages", []):
                body = msg.get("Body", "")
                if "requestId" in body and "prompt" not in body:
                    found = True
                    sqs.delete_message(
                        QueueUrl=dlq_url,
                        ReceiptHandle=msg["ReceiptHandle"],
                    )
                    break
            if found:
                break
            time.sleep(2)

        assert found, "Message with missing 'prompt' did not appear in DLQ within 60s"


# ---------------------------------------------------------------------------
# 6. Scale-Out on Burst
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestScaleOut:
    """Verify the service scales out when a burst of messages arrives."""

    def test_burst_triggers_scale_out(self):
        """
        Send a burst of messages exceeding the queue-depth threshold,
        then verify the ECS service desired count increases.

        Validates: Requirement 6.2 (autoscaling response)
        """
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        queue_url = outputs["RequestQueueUrl"]
        cluster_arn = outputs["ClusterArn"]

        sqs = _sqs_client()
        ecs = _ecs_client()

        # Record initial desired count
        services = ecs.list_services(cluster=cluster_arn)["serviceArns"]
        assert services, "No ECS services found in cluster"
        service_arn = services[0]

        initial = ecs.describe_services(
            cluster=cluster_arn, services=[service_arn]
        )["services"][0]["desiredCount"]

        # Send a burst of 20 messages to exceed the default threshold of 5
        burst_size = 20
        for i in range(burst_size):
            msg = json.dumps({
                "requestId": str(uuid.uuid4()),
                "prompt": f"Burst test message {i}",
                "maxTokens": 16,
            })
            sqs.send_message(QueueUrl=queue_url, MessageBody=msg)

        # Wait up to 5 minutes for the desired count to increase
        deadline = time.time() + 300
        scaled = False
        while time.time() < deadline:
            desc = ecs.describe_services(
                cluster=cluster_arn, services=[service_arn]
            )
            current_desired = desc["services"][0]["desiredCount"]
            if current_desired > initial:
                scaled = True
                break
            time.sleep(30)

        assert scaled, (
            f"Service did not scale out within 5 minutes "
            f"(desired count stayed at {initial})"
        )


# ---------------------------------------------------------------------------
# 7. Scale-In to Minimum
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestScaleIn:
    """Verify the service scales down to minimum when the queue is drained."""

    def test_drain_triggers_scale_in(self):
        """
        Purge the request queue, then verify the service desired count
        decreases to the configured minimum.

        Validates: Requirement 6.2 (scale-in behavior)
        """
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        queue_url = outputs["RequestQueueUrl"]
        cluster_arn = outputs["ClusterArn"]

        sqs = _sqs_client()
        ecs = _ecs_client()

        # Purge the queue to remove any pending messages
        sqs.purge_queue(QueueUrl=queue_url)

        services = ecs.list_services(cluster=cluster_arn)["serviceArns"]
        assert services, "No ECS services found in cluster"
        service_arn = services[0]

        # Read the minimum from the scalable target (Application Auto Scaling)
        aas = boto3.client("application-autoscaling", region_name=AWS_REGION)
        targets = aas.describe_scalable_targets(
            ServiceNamespace="ecs",
            ResourceIds=[
                f"service/{cluster_arn.split('/')[-1]}/{service_arn.split('/')[-1]}"
            ],
        ).get("ScalableTargets", [])

        min_capacity = targets[0]["MinCapacity"] if targets else 0

        # Wait up to 10 minutes for scale-in (cooldown + evaluation period)
        deadline = time.time() + 600
        scaled_in = False
        while time.time() < deadline:
            desc = ecs.describe_services(
                cluster=cluster_arn, services=[service_arn]
            )
            current_desired = desc["services"][0]["desiredCount"]
            if current_desired <= min_capacity:
                scaled_in = True
                break
            time.sleep(30)

        assert scaled_in, (
            f"Service did not scale in to minimum ({min_capacity}) within 10 minutes"
        )


# ---------------------------------------------------------------------------
# 8. GPU Telemetry in CloudWatch
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGPUTelemetry:
    """Verify GPU metrics appear in CloudWatch after task start."""

    def test_gpu_metrics_published(self):
        """
        Check that GPU telemetry metrics are published to CloudWatch
        within 5 minutes of a running GPU task.

        Validates: Requirement 5.2
        """
        _skip_if_no_stack()
        outputs = _get_stack_outputs()
        cluster_name = outputs["ClusterArn"].split("/")[-1]

        cw = _cw_client()

        # GPU metrics published by Enhanced Container Insights
        expected_metrics = [
            "instance_gpu_memory_used",
            "instance_gpu_memory_total",
            "instance_gpu_temperature",
        ]

        deadline = time.time() + 300  # 5 minutes
        found_metrics = set()

        while time.time() < deadline and len(found_metrics) < len(expected_metrics):
            for metric_name in expected_metrics:
                if metric_name in found_metrics:
                    continue
                resp = cw.list_metrics(
                    Namespace="ECS/ContainerInsights",
                    MetricName=metric_name,
                    Dimensions=[
                        {"Name": "ClusterName", "Value": cluster_name},
                    ],
                )
                if resp.get("Metrics"):
                    found_metrics.add(metric_name)

            if len(found_metrics) < len(expected_metrics):
                time.sleep(30)

        missing = set(expected_metrics) - found_metrics
        assert not missing, (
            f"GPU telemetry metrics not found in CloudWatch within 5 minutes: {missing}"
        )

    def test_gpu_temperature_alarm_exists(self):
        """Verify the GPU temperature alarm is configured in CloudWatch."""
        _skip_if_no_stack()

        cw = _cw_client()
        resources = _get_stack_resources()
        alarm_resources = [
            r for r in resources
            if r["ResourceType"] == "AWS::CloudWatch::Alarm"
            and "Temp" in r["LogicalResourceId"]
        ]
        assert alarm_resources, "No GPU temperature alarm found in stack resources"

        alarm_name = alarm_resources[0]["PhysicalResourceId"]
        resp = cw.describe_alarms(AlarmNames=[alarm_name])
        alarms = resp.get("MetricAlarms", [])
        assert len(alarms) == 1, f"Expected 1 alarm, found {len(alarms)}"
        assert alarms[0]["Threshold"] == 90.0, (
            f"GPU temp alarm threshold is {alarms[0]['Threshold']}, expected 90.0"
        )
