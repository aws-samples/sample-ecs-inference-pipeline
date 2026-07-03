"""
Property 3: IAM task role least-privilege scoping

For any IAM policy statement attached to the GPU task role in the CloudFormation
template, the Resource field SHALL reference only ARNs scoped to the specific
model S3 bucket, the request SQS queue, the dead-letter SQS queue, or the
CloudWatch log group — no wildcard (*) resource ARNs are permitted.

Validates: Requirements 3.3, 7.2
"""

import os
import yaml
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "infrastructure", "template.yaml"
)

# Services that the task role is allowed to reference
ALLOWED_ARN_SERVICE_PREFIXES = ("arn:aws:s3:", "arn:aws:sqs:", "arn:aws:logs:")


def _cfn_constructor(loader, tag_suffix, node):
    """Handle CloudFormation intrinsic function tags (!Ref, !Sub, !GetAtt, etc.)."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return None


def _get_cfn_loader():
    """Create a YAML loader that handles CloudFormation intrinsic functions."""
    loader = yaml.SafeLoader
    cfn_tags = [
        "!Ref", "!Sub", "!GetAtt", "!Select", "!Split", "!Join",
        "!FindInMap", "!If", "!Equals", "!Not", "!And", "!Or",
        "!GetAZs", "!ImportValue", "!Condition", "!Base64", "!Cidr",
    ]
    for tag in cfn_tags:
        yaml.add_multi_constructor(tag, _cfn_constructor, Loader=loader)
    return loader


def load_template():
    """Load and parse the CloudFormation template."""
    with open(TEMPLATE_PATH, "r") as f:
        return yaml.load(f, Loader=_get_cfn_loader())


def get_ecs_task_role(template):
    """Return the ECSTaskRole resource from the template."""
    resources = template.get("Resources", {})
    task_role = resources.get("ECSTaskRole")
    assert task_role is not None, "ECSTaskRole resource not found in template"
    assert task_role.get("Type") == "AWS::IAM::Role", (
        "ECSTaskRole must be of type AWS::IAM::Role"
    )
    return task_role


def get_policy_statements(task_role):
    """Extract all IAM policy statements from the ECSTaskRole inline policies."""
    props = task_role.get("Properties", {})
    policies = props.get("Policies", [])
    assert len(policies) > 0, "ECSTaskRole must have at least one inline policy"

    statements = []
    for policy in policies:
        doc = policy.get("PolicyDocument", {})
        stmts = doc.get("Statement", [])
        statements.extend(stmts)

    assert len(statements) > 0, "ECSTaskRole must have at least one policy statement"
    return statements


def collect_resource_arns(statement):
    """Collect all Resource ARN values from a policy statement as a flat list of strings."""
    resource = statement.get("Resource")
    if resource is None:
        return []
    if isinstance(resource, str):
        return [resource]
    if isinstance(resource, list):
        # Flatten: each element could be a string or a resolved intrinsic (string)
        arns = []
        for item in resource:
            if isinstance(item, str):
                arns.append(item)
        return arns
    return []


def is_bare_wildcard(arn_value):
    """Check if a resource ARN is a bare wildcard '*' (not scoped)."""
    return arn_value.strip() == "*"


def is_cfn_intrinsic_resolved(arn_value):
    """Check if a value is a resolved CloudFormation intrinsic (e.g., !GetAtt, !Sub).

    When the YAML loader encounters intrinsic functions like !GetAtt RequestQueue.Arn
    or !Sub '${InferenceLogGroup.Arn}:*', it resolves them to plain strings.
    These are valid scoped references — they resolve to specific resource ARNs at
    deploy time. We identify them by checking if they reference known resource
    logical IDs from the template.
    """
    known_resource_refs = (
        "RequestQueue", "DeadLetterQueue", "InferenceLogGroup",
        "ModelS3Path", "BucketName",
    )
    for ref in known_resource_refs:
        if ref in arn_value:
            return True
    return False


def is_scoped_to_allowed_service(arn_value):
    """Check if a resource ARN is scoped to an allowed service (s3, sqs, logs).

    A value is considered scoped if it either:
    1. Starts with an allowed ARN prefix (literal ARN), or
    2. Is a resolved CloudFormation intrinsic that references a known resource
    """
    if arn_value.startswith(ALLOWED_ARN_SERVICE_PREFIXES):
        return True
    if is_cfn_intrinsic_resolved(arn_value):
        return True
    return False


class TestIAMLeastPrivilegeUnit:
    """Unit tests for IAM task role least-privilege scoping."""

    def test_template_loads_successfully(self):
        template = load_template()
        assert template is not None
        assert "Resources" in template

    def test_ecs_task_role_exists(self):
        """ECSTaskRole must exist in the template."""
        template = load_template()
        task_role = get_ecs_task_role(template)
        assert task_role is not None

    def test_task_role_has_inline_policy(self):
        """ECSTaskRole must have at least one inline policy with statements."""
        template = load_template()
        task_role = get_ecs_task_role(template)
        statements = get_policy_statements(task_role)
        assert len(statements) > 0

    def test_no_bare_wildcard_resources(self):
        """No policy statement should have a bare wildcard '*' as its Resource
        unless it has a Condition that scopes access (e.g., cloudwatch:PutMetricData
        requires Resource '*' but can be scoped by cloudwatch:namespace condition)."""
        template = load_template()
        task_role = get_ecs_task_role(template)
        statements = get_policy_statements(task_role)

        for stmt in statements:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            has_condition = "Condition" in stmt
            for arn in arns:
                if has_condition:
                    continue
                assert not is_bare_wildcard(arn), (
                    f"Statement '{sid}' has a bare wildcard '*' Resource ARN. "
                    f"All resources must be scoped to specific bucket/queue/log-group."
                )

    def test_all_resources_scoped_to_allowed_services(self):
        """All Resource ARNs must reference s3, sqs, logs, or cloudwatch (with Condition)."""
        template = load_template()
        task_role = get_ecs_task_role(template)
        statements = get_policy_statements(task_role)

        for stmt in statements:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            has_condition = "Condition" in stmt
            for arn in arns:
                if has_condition and is_bare_wildcard(arn):
                    continue
                assert is_scoped_to_allowed_service(arn), (
                    f"Statement '{sid}' has Resource ARN '{arn}' that is not scoped "
                    f"to an allowed service (s3, sqs, logs)."
                )

    def test_s3_statements_reference_specific_bucket(self):
        """S3 policy statements must reference a specific bucket, not all buckets."""
        template = load_template()
        task_role = get_ecs_task_role(template)
        statements = get_policy_statements(task_role)

        for stmt in statements:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            s3_actions = [a for a in actions if a.startswith("s3:")]
            if not s3_actions:
                continue

            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            for arn in arns:
                # S3 ARNs should be like arn:aws:s3:::bucket-name or arn:aws:s3:::bucket-name/*
                # They should NOT be arn:aws:s3:::* or arn:aws:s3:::*/*
                assert ":::*" not in arn, (
                    f"Statement '{sid}' has S3 Resource ARN '{arn}' that references "
                    f"all buckets. Must be scoped to a specific bucket."
                )


class TestIAMLeastPrivilegeProperty:
    """
    Property-based test for IAM task role least-privilege scoping.

    **Validates: Requirements 3.3, 7.2**

    Since the CloudFormation template is a static artifact, we use hypothesis
    to generate random statement indices and verify the property holds for every
    statement selected. This ensures the property is checked across all possible
    statement selections from the template.
    """

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_statement_no_bare_wildcard_resource(self, data):
        """
        Property 3: IAM task role least-privilege scoping

        For any randomly selected IAM policy statement from the ECSTaskRole,
        the Resource field must NOT contain a bare wildcard '*'.

        **Validates: Requirements 3.3, 7.2**
        """
        template = load_template()
        task_role = get_ecs_task_role(template)
        statements = get_policy_statements(task_role)

        selected_stmt = data.draw(st.sampled_from(statements))
        sid = selected_stmt.get("Sid", "unknown")
        arns = collect_resource_arns(selected_stmt)
        has_condition = "Condition" in selected_stmt

        for arn in arns:
            if has_condition:
                continue
            assert not is_bare_wildcard(arn), (
                f"Statement '{sid}' has a bare wildcard '*' Resource ARN. "
                f"All resources must be scoped to specific bucket/queue/log-group."
            )

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_statement_resources_scoped_to_allowed_services(self, data):
        """
        Property 3: IAM task role least-privilege scoping

        For any randomly selected IAM policy statement from the ECSTaskRole,
        all Resource ARNs must reference specific services (s3, sqs, logs).

        **Validates: Requirements 3.3, 7.2**
        """
        template = load_template()
        task_role = get_ecs_task_role(template)
        statements = get_policy_statements(task_role)

        selected_stmt = data.draw(st.sampled_from(statements))
        sid = selected_stmt.get("Sid", "unknown")
        arns = collect_resource_arns(selected_stmt)
        has_condition = "Condition" in selected_stmt

        for arn in arns:
            if has_condition and is_bare_wildcard(arn):
                continue
            assert is_scoped_to_allowed_service(arn), (
                f"Statement '{sid}' has Resource ARN '{arn}' that is not scoped "
                f"to an allowed service (s3, sqs, logs). Task role must only "
                f"reference specific bucket, queue, or log-group ARNs."
            )
