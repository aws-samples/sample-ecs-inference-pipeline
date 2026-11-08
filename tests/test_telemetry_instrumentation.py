"""
Telemetry contract instrumentation: every record a producer emits conforms to
telemetry/contract/v1, and the template wires the producers and the CloudWatch
agent (CloudWatch Omni) the way the code expects.

Producers covered:
- container/telemetry.py + sqs_worker.py: gpu_inference.dead_letter, gpu_inference.cold_start
- infrastructure/gpu_impairment_normalizer.py: gpu_inference.gpu_impairment
"""

import copy
import json
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "container"))
sys.path.insert(0, str(ROOT / "infrastructure"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")  # normalizer creates an ECS client at import

import sqs_worker  # noqa: E402
import telemetry  # noqa: E402
import gpu_impairment_normalizer as normalizer  # noqa: E402

CONTRACT_DIR = ROOT / "telemetry" / "contract" / "v1"
TEMPLATE_PATH = ROOT / "infrastructure" / "template.yaml"

ACCOUNT = "111122223333"
CLUSTER_ARN = f"arn:aws:ecs:us-west-2:{ACCOUNT}:cluster/ecs-gpu-inference-cluster"
TASK_ARN = f"arn:aws:ecs:us-west-2:{ACCOUNT}:task/ecs-gpu-inference-cluster/{'a' * 32}"
CI_ARN = f"arn:aws:ecs:us-west-2:{ACCOUNT}:container-instance/ecs-gpu-inference-cluster/{'b' * 32}"
QUEUE_URL = f"https://sqs.us-west-2.amazonaws.com/{ACCOUNT}/ecs-gpu-inference-request-queue"
REQUEST_ID = "3f2b8c1a-9d4e-4f6a-8b7c-1e2d3c4b5a69"

FAKE_CTX = {
    "region": "us-west-2",
    "cluster_arn": CLUSTER_ARN,
    "task_arn": TASK_ARN,
    "task_family": "gpu-inference-task",
    "task_revision": "12",
    "availability_zone": "us-west-2b",
    "pull_stopped_at": "2026-10-27T19:54:12.370123456Z",  # TMDE v4 nanosecond format
    "container_instance_arn": CI_ARN,
    "ec2_instance_id": "i-0fedcba9876543210",
    "instance_type": "g6e.xlarge",
    "capacity_provider": "gpu-inference-ecs-gpu-inference-capacity",
    "instance_registered_at": datetime(2026, 10, 27, 19, 51, 40, 600000, tzinfo=timezone.utc),
}

ENQUEUED_MS = str(int(datetime(2026, 10, 27, 19, 49, 2, 115000, tzinfo=timezone.utc).timestamp() * 1000))
STAGE_ORDER = ["enqueued", "node_ready", "image_pulled", "weights_loaded", "first_token"]


# --- contract helpers --------------------------------------------------------------

def _validators():
    schemas = {p.name: json.loads(p.read_text()) for p in CONTRACT_DIR.glob("*.schema.json")}
    registry = Registry().with_resources(
        [(s["$id"], Resource.from_contents(s)) for s in schemas.values()]
    )
    by_event = {}
    for name, schema in schemas.items():
        if name == "common.schema.json":
            continue
        by_event[schema["title"]] = Draft202012Validator(
            schema, registry=registry, format_checker=FormatChecker()
        )
    return by_event, schemas


VALIDATORS, SCHEMAS = _validators()


def assert_conforms(record):
    """CR-1/2/3/6 via schema, CR-4/5 semantically."""
    validator = VALIDATORS[record["event_name"]]
    errors = sorted(validator.iter_errors(record), key=lambda e: e.json_path)
    assert not errors, "\n".join(f"{e.json_path}: {e.message}" for e in errors)
    attrs = record["attributes"]
    if record["event_name"] == "gpu_inference.cold_start":  # CR-4
        stages = attrs["cold_start.stages"]
        times = [telemetry.parse_ts(stages[s]) for s in STAGE_ORDER]
        assert times == sorted(times), f"CR-4: stages not non-decreasing: {stages}"
    if attrs.get("gpu.impairment.cause") == "xid":  # CR-5
        assert attrs["gpu.health_check.status_reason"] == f"XID_{attrs['gpu.xid.code']}"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(json.loads(record.getMessage()))


@pytest.fixture
def captured(monkeypatch):
    """Capture emitted contract records; fake the ECS context and worker config."""
    handler = _Capture()
    tlog = logging.getLogger(telemetry.TELEMETRY_LOGGER_NAME)
    tlog.addHandler(handler)
    monkeypatch.setattr(telemetry, "ecs_context", lambda: dict(FAKE_CTX))
    monkeypatch.setattr(sqs_worker, "REQUEST_QUEUE_URL", QUEUE_URL)
    monkeypatch.setattr(sqs_worker, "DLQ_URL", QUEUE_URL.replace("request-queue", "dlq"))
    monkeypatch.setattr(sqs_worker, "OUTPUT_DESTINATION", "")
    monkeypatch.setattr(sqs_worker, "_cold_start", telemetry.ColdStartTracker(weights_loaded_at=""))
    monkeypatch.setenv("TIER", "small")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "sqs-worker-small")
    yield handler.records
    tlog.removeHandler(handler)


def _message(body, receive_count="1"):
    return {
        "MessageId": "5fea7756-0ea4-451a-a703-a558b933e274",
        "ReceiptHandle": "rh-1",
        "Body": body,
        "Attributes": {"ApproximateReceiveCount": receive_count, "SentTimestamp": ENQUEUED_MS},
    }


def _valid_body():
    return json.dumps({"requestId": REQUEST_ID, "prompt": "hello", "maxTokens": 16})


# --- dead_letter: malformed_input --------------------------------------------------

# One body per validate_message() rejection branch.
MALFORMED_BODIES = [
    None,
    123,
    "not json",
    "[1, 2]",
    "{}",
    json.dumps({"requestId": "not-a-uuid"}),
    json.dumps({"requestId": REQUEST_ID}),
    json.dumps({"requestId": REQUEST_ID, "prompt": 1}),
    json.dumps({"requestId": REQUEST_ID, "prompt": "   "}),
    json.dumps({"requestId": REQUEST_ID, "prompt": "p", "maxTokens": "x"}),
    json.dumps({"requestId": REQUEST_ID, "prompt": "p", "maxTokens": 4096}),
    json.dumps({"requestId": REQUEST_ID, "prompt": "p", "temperature": "x"}),
    json.dumps({"requestId": REQUEST_ID, "prompt": "p", "temperature": 3}),
    json.dumps({"requestId": REQUEST_ID, "prompt": "p", "topP": "x"}),
    json.dumps({"requestId": REQUEST_ID, "prompt": "p", "topP": 2}),
]


class TestMalformedReasonCodes:
    def test_every_validation_error_maps_to_a_contract_code(self):
        codes = []
        for body in MALFORMED_BODIES:
            _, error = sqs_worker.validate_message(body)
            assert error, f"expected a validation error for {body!r}"
            code = telemetry.malformed_reason_code(error)
            assert code, f"no reason_code for validation error {error!r}"
            codes.append(code)
        assert len(set(codes)) == len(MALFORMED_BODIES), "each branch must map to its own code"

    def test_mapping_matches_contract_enum_exactly(self):
        schema = SCHEMAS["dead_letter.schema.json"]
        enum = schema["properties"]["attributes"]["then"]["properties"]["inference.dlq.reason_code"]["enum"]
        assert sorted(code for _, code in telemetry.MALFORMED_REASON_CODES) == sorted(enum)

    @pytest.mark.parametrize("body", [b for b in MALFORMED_BODIES if isinstance(b, str)])
    def test_worker_emits_conformant_record(self, captured, body):
        sqs_client = MagicMock()
        sqs_worker.process_message(sqs_client, _message(body))

        # existing DLQ behaviour is unchanged
        sqs_client.send_message.assert_called_once()
        sqs_client.delete_message.assert_called_once()

        assert len(captured) == 1
        record = captured[0]
        assert_conforms(record)
        attrs = record["attributes"]
        assert attrs["inference.dlq.reason"] == "malformed_input"
        assert attrs["inference.dlq.source_tier"] == "small"
        assert attrs["inference.dlq.source_queue"] == QUEUE_URL
        assert attrs["inference.request.enqueued_at"] == "2026-10-27T19:49:02.115Z"

    def test_request_id_kept_when_body_parses(self, captured):
        body = json.dumps({"requestId": REQUEST_ID, "prompt": "p", "maxTokens": 4096})
        sqs_worker.process_message(MagicMock(), _message(body))
        assert captured[0]["attributes"]["inference.request.id"] == REQUEST_ID
        assert captured[0]["attributes"]["inference.dlq.reason_code"] == "max_tokens_out_of_range"


# --- dead_letter: processing_exhausted ---------------------------------------------

class TestProcessingExhausted:
    def test_final_failed_attempt_emits_record(self, captured):
        with patch.object(sqs_worker, "forward_to_vllm", side_effect=RuntimeError("read timeout")):
            sqs_worker.process_message(MagicMock(), _message(_valid_body(), receive_count="3"))
        assert len(captured) == 1
        record = captured[0]
        assert_conforms(record)
        attrs = record["attributes"]
        assert attrs["inference.dlq.reason"] == "processing_exhausted"
        assert attrs["inference.dlq.reason_code"] == "inference_failed"
        assert attrs["inference.dlq.receive_count"] == 3
        assert attrs["inference.request.id"] == REQUEST_ID

    def test_earlier_attempts_emit_nothing(self, captured):
        with patch.object(sqs_worker, "forward_to_vllm", side_effect=RuntimeError("boom")):
            for count in ("1", "2"):
                sqs_worker.process_message(MagicMock(), _message(_valid_body(), receive_count=count))
        assert captured == []

    def test_output_write_failure_code(self, captured):
        ok = ({"text": "t", "usage": {}, "finishReason": "stop"}, 5)
        with patch.object(sqs_worker, "forward_to_vllm", return_value=ok), \
             patch.object(sqs_worker, "write_result", side_effect=RuntimeError("s3 down")):
            sqs_worker.process_message(MagicMock(), _message(_valid_body(), receive_count="3"))
        dead = [r for r in captured if r["event_name"] == "gpu_inference.dead_letter"]
        assert len(dead) == 1
        assert_conforms(dead[0])
        assert dead[0]["attributes"]["inference.dlq.reason_code"] == "output_write_failed"

    def test_telemetry_failure_does_not_break_processing(self, captured, monkeypatch):
        monkeypatch.setattr(telemetry, "ecs_context", MagicMock(side_effect=RuntimeError("no tmde")))
        sqs_client = MagicMock()
        sqs_worker.process_message(sqs_client, _message("not json"))
        sqs_client.send_message.assert_called_once()
        sqs_client.delete_message.assert_called_once()


# --- cold_start --------------------------------------------------------------------

class TestColdStart:
    def _tracker(self):
        return telemetry.ColdStartTracker(weights_loaded_at="2026-10-27T19:58:20.040Z")

    def test_scale_from_zero_record_conforms(self, monkeypatch):
        monkeypatch.setenv("TIER", "small")
        monkeypatch.setenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
        monkeypatch.setenv("MODEL_S3_PATH", "none")
        first_token = datetime(2026, 10, 27, 19, 58, 31, 610000, tzinfo=timezone.utc)
        record = self._tracker().observe(FAKE_CTX, _message(_valid_body()), REQUEST_ID, first_token)
        assert_conforms(record)
        attrs = record["attributes"]
        assert attrs["cold_start.node_reused"] is False
        assert attrs["inference.model.source"] == "huggingface"
        assert attrs["cold_start.stages"]["image_pulled"] == "2026-10-27T19:54:12.370Z"
        assert record["resource"]["host.id"] == FAKE_CTX["ec2_instance_id"]

    def test_reused_instance_clamps_node_ready(self):
        ctx = dict(FAKE_CTX, instance_registered_at="2026-10-27T10:00:00Z")
        first_token = datetime(2026, 10, 27, 19, 58, 31, tzinfo=timezone.utc)
        record = self._tracker().observe(ctx, _message(_valid_body()), REQUEST_ID, first_token)
        assert_conforms(record)
        stages = record["attributes"]["cold_start.stages"]
        assert stages["node_ready"] == stages["enqueued"]
        assert record["attributes"]["cold_start.node_reused"] is True

    def test_warm_task_emits_nothing(self):
        late_ms = str(int(datetime(2026, 10, 27, 20, 30, tzinfo=timezone.utc).timestamp() * 1000))
        msg = _message(_valid_body())
        msg["Attributes"]["SentTimestamp"] = late_ms
        assert self._tracker().observe(FAKE_CTX, msg, REQUEST_ID, datetime.now(timezone.utc)) is None

    def test_only_first_inference_counts(self):
        tracker = self._tracker()
        first_token = datetime(2026, 10, 27, 19, 58, 31, tzinfo=timezone.utc)
        assert tracker.observe(FAKE_CTX, _message(_valid_body()), REQUEST_ID, first_token)
        assert tracker.observe(FAKE_CTX, _message(_valid_body()), REQUEST_ID, first_token) is None

    def test_missing_stage_emits_nothing(self):
        ctx = {k: v for k, v in FAKE_CTX.items() if k != "pull_stopped_at"}
        assert self._tracker().observe(ctx, _message(_valid_body()), REQUEST_ID,
                                       datetime.now(timezone.utc)) is None

    def test_emit_drops_record_without_required_resource(self, caplog):
        record = telemetry.envelope("gpu_inference.cold_start", {"service.name": "x"}, {})
        with caplog.at_level(logging.WARNING):
            assert telemetry.emit(record) is False
        assert "dropped" in caplog.text


class TestTimestamps:
    @pytest.mark.parametrize("value,expected", [
        ("2026-10-27T19:54:12.370123456Z", "2026-10-27T19:54:12.370Z"),
        ("2026-10-27T19:54:12Z", "2026-10-27T19:54:12.000Z"),
        ("2026-10-27T21:54:12.5+02:00", "2026-10-27T19:54:12.500Z"),
        ("1793130852370", "2026-10-27T19:54:12.370Z"),
    ])
    def test_parse_and_format(self, value, expected):
        assert telemetry.fmt_ts(value) == expected


# --- gpu_impairment normalizer -----------------------------------------------------

def _health_event(status_reason="XID_79", check_type="ACCELERATED_COMPUTE"):
    return {
        "version": "0",
        "id": "a1b2c3d4-0000-1111-2222-333344445555",
        "detail-type": "ECS Container Instance Health Change",
        "source": "aws.ecs",
        "account": ACCOUNT,
        "time": "2026-10-27T19:39:12Z",
        "region": "us-west-2",
        "resources": [CI_ARN],
        "detail": {
            "clusterArn": CLUSTER_ARN,
            "containerInstanceArn": CI_ARN,
            "capacityProviderName": "gpu-inference-ecs-gpu-inference-capacity",
            "ec2InstanceId": "i-0123456789abcdef0",
            "overallStatus": "IMPAIRED",
            "healthChecks": [{
                "type": check_type,
                "status": "IMPAIRED",
                "lastStatusChange": "2026-10-27T19:39:12.204Z",
                "lastUpdated": "2026-10-27T19:39:12.204Z",
                "statusReason": status_reason,
            }],
        },
    }


TASKS = [{
    "taskArn": TASK_ARN,
    "taskDefinitionArn": f"arn:aws:ecs:us-west-2:{ACCOUNT}:task-definition/gpu-inference-task:12",
}]
CONTAINER_INSTANCE = {"attributes": [
    {"name": "ecs.instance-type", "value": "g6.2xlarge"},
    {"name": "ecs.availability-zone", "value": "us-west-2a"},
]}
NOW = datetime(2026, 10, 27, 19, 39, 14, 101000, tzinfo=timezone.utc)


class TestGpuImpairmentNormalizer:
    def test_xid_event_conforms(self):
        record = normalizer.build_record(_health_event(), TASKS, CONTAINER_INSTANCE, NOW)
        assert_conforms(record)
        attrs = record["attributes"]
        assert attrs["gpu.xid.code"] == 79
        assert attrs["forensics.task_arns"] == [TASK_ARN]
        assert attrs["forensics.task_families"] == ["gpu-inference-task"]
        assert record["resource"]["host.type"] == "g6.2xlarge"

    def test_no_tasks_running_is_an_empty_capture(self):
        record = normalizer.build_record(_health_event(), [], CONTAINER_INSTANCE, NOW)
        assert_conforms(record)
        assert record["attributes"]["forensics.task_arns"] == []

    def test_failed_capture_omits_task_list(self):
        record = normalizer.build_record(_health_event(), None, CONTAINER_INSTANCE, NOW)
        assert "forensics.task_arns" not in record["attributes"]
        assert list(VALIDATORS["gpu_inference.gpu_impairment"].iter_errors(record)), \
            "a record without a forensic capture must not pass CR-1"

    def test_xid_outside_enum_is_cause_other(self):
        record = normalizer.build_record(_health_event("XID_13"), TASKS, CONTAINER_INSTANCE, NOW)
        assert_conforms(record)
        assert record["attributes"]["gpu.impairment.cause"] == "other"
        assert "gpu.xid.code" not in record["attributes"]

    def test_non_xid_impairment_is_cause_other(self):
        record = normalizer.build_record(_health_event("MPS daemon unhealthy"), TASKS, {}, NOW)
        assert_conforms(record)

    def test_non_accelerated_compute_event_is_ignored(self):
        assert normalizer.build_record(_health_event(check_type="CONTAINER_RUNTIME"), TASKS, {}, NOW) is None

    def test_handler_captures_tasks_and_prints_record(self, capsys):
        ecs = MagicMock()
        ecs.list_tasks.return_value = {"taskArns": [TASK_ARN]}
        ecs.describe_tasks.return_value = {"tasks": TASKS}
        ecs.describe_container_instances.return_value = {"containerInstances": [CONTAINER_INSTANCE]}
        with patch.object(normalizer, "ecs", ecs):
            assert normalizer.handler(_health_event(), None) == {"emitted": True}
        ecs.list_tasks.assert_called_once_with(cluster=CLUSTER_ARN, containerInstance=CI_ARN)
        assert_conforms(json.loads(capsys.readouterr().out.strip()))


# --- template wiring ---------------------------------------------------------------

def _cfn(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


class _CfnLoader(yaml.SafeLoader):
    pass


_CfnLoader.add_multi_constructor("!", _cfn)


@pytest.fixture(scope="module")
def resources():
    return yaml.load(TEMPLATE_PATH.read_text(), Loader=_CfnLoader)["Resources"]  # noqa: S506 - SafeLoader subclass


TASK_DEFS = [
    ("GPUInferenceTaskDefinition", "vllm-inference", "small"),
    ("LargeGPUInferenceTaskDefinition", "vllm-large-inference", "large"),
]


class TestTemplateWiring:
    def test_inline_normalizer_matches_source_file(self, resources):
        inline = resources["GpuImpairmentNormalizerFunction"]["Properties"]["Code"]["ZipFile"]
        source = (ROOT / "infrastructure" / "gpu_impairment_normalizer.py").read_text()
        assert inline.strip() == source.strip(), "re-embed gpu_impairment_normalizer.py into template.yaml"
        assert len(inline) <= 4096, "CloudFormation inline ZipFile limit is 4096 characters"

    @pytest.mark.parametrize("logical_id,app_name,tier", TASK_DEFS)
    def test_cloudwatch_agent_sidecar(self, resources, logical_id, app_name, tier):
        containers = resources[logical_id]["Properties"]["ContainerDefinitions"]
        app, agent = containers[0], containers[1]
        assert app["Name"] == app_name
        assert agent["Name"] == "cloudwatch-agent"
        assert agent["Essential"] is False
        assert agent["Image"].startswith("public.ecr.aws/cloudwatch-agent/cloudwatch-agent:")
        assert not agent["Image"].endswith(":latest"), "pin the agent image"
        assert json.loads(agent["Environment"][0]["Value"]) == {"opentelemetry": {"collect": {"otlp": {}}}}
        # START, not HEALTHY: the agent image has no health check
        assert app["DependsOn"] == [{"ContainerName": "cloudwatch-agent", "Condition": "START"}]

    @pytest.mark.parametrize("logical_id,app_name,tier", TASK_DEFS)
    def test_app_otel_and_contract_env(self, resources, logical_id, app_name, tier):
        app = resources[logical_id]["Properties"]["ContainerDefinitions"][0]
        env = {e["Name"]: e["Value"] for e in app["Environment"]}
        assert env["TIER"] == tier
        assert env["MAX_RECEIVE_COUNT"] == "3"
        assert env["OTEL_SERVICE_NAME"] == f"sqs-worker-{tier}"
        assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:4318"
        assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
        assert env["OTEL_LOGS_EXPORTER"] == "otlp"
        assert env["OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED"] == "true"

    def test_max_receive_count_matches_redrive_policy(self, resources):
        for queue in ("RequestQueue", "LargeModelRequestQueue"):
            assert resources[queue]["Properties"]["RedrivePolicy"]["maxReceiveCount"] == 3

    @pytest.mark.parametrize("role", ["ECSTaskRole", "LargeModelTaskRole"])
    def test_task_roles_allow_agent_and_cluster_scoped_introspection(self, resources, role):
        props = resources[role]["Properties"]
        assert "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy" in props["ManagedPolicyArns"]
        stmts = [s for p in props["Policies"] for s in p["PolicyDocument"]["Statement"]]
        intro = next(s for s in stmts if s.get("Sid") == "EcsTaskIntrospection")
        assert set(intro["Action"]) == {"ecs:DescribeTasks", "ecs:DescribeContainerInstances"}
        assert "ecs:cluster" in intro["Condition"]["ArnEquals"]

    def test_health_rule_targets_normalizer(self, resources):
        targets = resources["ContainerInstanceHealthEventRule"]["Properties"]["Targets"]
        assert any(t["Id"] == "gpu-impairment-normalizer" for t in targets)
        perm = resources["GpuImpairmentNormalizerPermission"]["Properties"]
        assert perm["Principal"] == "events.amazonaws.com"
        fn = resources["GpuImpairmentNormalizerFunction"]["Properties"]
        assert fn["TracingConfig"]["Mode"] == "Active"
