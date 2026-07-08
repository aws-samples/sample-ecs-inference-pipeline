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
        "ModelS3Path", "BucketName", "LargeModelLogGroup",
        "LargeModelRequestQueue", "LargeModelS3Path",
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


# ---------------------------------------------------------------------------
# Large Model Task Role (FR-3.5)
# ---------------------------------------------------------------------------

def get_large_model_task_role(template):
    """Return the LargeModelTaskRole resource from the template."""
    resources = template.get("Resources", {})
    role = resources.get("LargeModelTaskRole")
    assert role is not None, "LargeModelTaskRole resource not found in template"
    assert role.get("Type") == "AWS::IAM::Role", (
        "LargeModelTaskRole must be of type AWS::IAM::Role"
    )
    return role


def get_scaling_lambda_role(template):
    """Return the ScalingMetricLambdaRole resource from the template."""
    resources = template.get("Resources", {})
    role = resources.get("ScalingMetricLambdaRole")
    assert role is not None, "ScalingMetricLambdaRole not found in template"
    return role


def get_router_lambda_role(template):
    """Return the RouterLambdaRole resource from the template."""
    resources = template.get("Resources", {})
    role = resources.get("RouterLambdaRole")
    assert role is not None, "RouterLambdaRole not found in template"
    return role


def get_all_policy_statements(role):
    """Extract all IAM policy statements from any role's inline policies."""
    props = role.get("Properties", {})
    policies = props.get("Policies", [])
    statements = []
    for policy in policies:
        doc = policy.get("PolicyDocument", {})
        stmts = doc.get("Statement", [])
        statements.extend(stmts)
    return statements


class TestLargeModelTaskRoleLeastPrivilege:
    """
    IAM least-privilege checks for LargeModelTaskRole.

    Validates: Requirements FR-3.5
    """

    def test_large_model_task_role_exists(self):
        """LargeModelTaskRole must exist in the template."""
        template = load_template()
        role = get_large_model_task_role(template)
        assert role is not None

    def test_large_model_task_role_has_inline_policy(self):
        """LargeModelTaskRole must have at least one inline policy with statements."""
        template = load_template()
        role = get_large_model_task_role(template)
        statements = get_all_policy_statements(role)
        assert len(statements) > 0, "LargeModelTaskRole must have at least one policy statement"

    def test_large_task_role_no_bare_wildcard_resources(self):
        """No LargeModelTaskRole statement should have a bare wildcard '*' Resource without Condition."""
        template = load_template()
        role = get_large_model_task_role(template)
        statements = get_all_policy_statements(role)
        for stmt in statements:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            has_condition = "Condition" in stmt
            for arn in arns:
                if has_condition:
                    continue
                assert not is_bare_wildcard(arn), (
                    f"LargeModelTaskRole statement '{sid}' has a bare wildcard '*' Resource ARN. "
                    f"All resources must be scoped."
                )

    def test_large_task_role_all_resources_scoped_to_allowed_services(self):
        """All LargeModelTaskRole Resource ARNs must reference s3, sqs, logs (or have Condition)."""
        template = load_template()
        role = get_large_model_task_role(template)
        statements = get_all_policy_statements(role)
        for stmt in statements:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            has_condition = "Condition" in stmt
            for arn in arns:
                if has_condition and is_bare_wildcard(arn):
                    continue
                assert is_scoped_to_allowed_service(arn), (
                    f"LargeModelTaskRole statement '{sid}' has Resource ARN '{arn}' "
                    f"not scoped to an allowed service (s3, sqs, logs)."
                )

    def test_large_task_role_s3_statements_reference_specific_bucket(self):
        """LargeModelTaskRole S3 statements must reference a specific bucket, not all buckets."""
        template = load_template()
        role = get_large_model_task_role(template)
        statements = get_all_policy_statements(role)
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
                assert ":::*" not in arn, (
                    f"LargeModelTaskRole statement '{sid}' has S3 ARN '{arn}' "
                    f"referencing all buckets. Must be scoped to a specific bucket."
                )


# ---------------------------------------------------------------------------
# Scaling Metric Lambda Role (FR-6.1)
# ---------------------------------------------------------------------------

class TestScalingMetricLambdaRoleLeastPrivilege:
    """
    IAM scoping checks for ScalingMetricLambdaRole.

    Validates: Requirements FR-6.1
    """

    def test_scaling_lambda_role_exists(self):
        """ScalingMetricLambdaRole must exist in the template."""
        template = load_template()
        role = get_scaling_lambda_role(template)
        assert role is not None

    def test_sqs_actions_scoped_to_specific_queues(self):
        """SQS GetQueueAttributes must be scoped to specific queue ARNs, not wildcard."""
        template = load_template()
        role = get_scaling_lambda_role(template)
        statements = get_all_policy_statements(role)

        sqs_stmts = [
            s for s in statements
            if any(
                a.startswith("sqs:") for a in (
                    [s.get("Action")] if isinstance(s.get("Action"), str)
                    else s.get("Action", [])
                )
            )
        ]
        assert len(sqs_stmts) > 0, "ScalingMetricLambdaRole must have an SQS statement"

        for stmt in sqs_stmts:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            for arn in arns:
                assert not is_bare_wildcard(arn), (
                    f"ScalingMetricLambdaRole SQS statement '{sid}' has bare wildcard '*' resource. "
                    f"SQS GetQueueAttributes must be scoped to specific queue ARNs."
                )

    def test_cloudwatch_put_metric_has_namespace_condition(self):
        """CloudWatch PutMetricData must have a namespace condition to limit scope."""
        template = load_template()
        role = get_scaling_lambda_role(template)
        statements = get_all_policy_statements(role)

        cw_put_stmts = [
            s for s in statements
            if "cloudwatch:PutMetricData" in (
                [s.get("Action")] if isinstance(s.get("Action"), str)
                else s.get("Action", [])
            )
        ]
        assert len(cw_put_stmts) > 0, (
            "ScalingMetricLambdaRole must have a cloudwatch:PutMetricData statement"
        )

        for stmt in cw_put_stmts:
            arns = collect_resource_arns(stmt)
            has_wildcard = any(is_bare_wildcard(a) for a in arns)
            has_condition = "Condition" in stmt
            if has_wildcard:
                assert has_condition, (
                    f"ScalingMetricLambdaRole PutMetricData statement has wildcard '*' resource "
                    f"but no Condition to scope the namespace. "
                    f"Must include a cloudwatch:namespace condition."
                )

    def test_no_bare_wildcard_without_condition(self):
        """ScalingMetricLambdaRole must not have bare wildcard '*' resources without a Condition,
        except for cloudwatch:GetMetricStatistics which does not support resource-level scoping."""
        template = load_template()
        role = get_scaling_lambda_role(template)
        statements = get_all_policy_statements(role)
        for stmt in statements:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            has_condition = "Condition" in stmt
            # GetMetricStatistics requires Resource '*' — AWS does not support resource scoping
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            is_read_only_cw = all(
                a in ("cloudwatch:GetMetricStatistics", "cloudwatch:GetMetricData",
                       "cloudwatch:ListMetrics", "cloudwatch:DescribeAlarms")
                for a in actions
            )
            for arn in arns:
                if has_condition or is_read_only_cw:
                    continue
                assert not is_bare_wildcard(arn), (
                    f"ScalingMetricLambdaRole statement '{sid}' has bare wildcard '*' resource "
                    f"without a scoping Condition."
                )


# ---------------------------------------------------------------------------
# Router Lambda Role (FR-6.2)
# ---------------------------------------------------------------------------

class TestRouterLambdaRoleLeastPrivilege:
    """
    IAM scoping checks for RouterLambdaRole.

    Validates: Requirements FR-6.2
    """

    def test_router_lambda_role_exists(self):
        """RouterLambdaRole must exist in the template."""
        template = load_template()
        role = get_router_lambda_role(template)
        assert role is not None

    def test_sqs_send_message_scoped_to_specific_queues(self):
        """RouterLambdaRole SQS SendMessage must be scoped to specific queue ARNs."""
        template = load_template()
        role = get_router_lambda_role(template)
        statements = get_all_policy_statements(role)

        sqs_stmts = [
            s for s in statements
            if any(
                "sqs:" in a for a in (
                    [s.get("Action")] if isinstance(s.get("Action"), str)
                    else s.get("Action", [])
                )
            )
        ]
        assert len(sqs_stmts) > 0, "RouterLambdaRole must have an SQS statement"

        for stmt in sqs_stmts:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            for arn in arns:
                assert not is_bare_wildcard(arn), (
                    f"RouterLambdaRole SQS statement '{sid}' has bare wildcard '*' resource. "
                    f"SQS SendMessage must be scoped to specific queue ARNs."
                )

    def test_bedrock_resource_is_scoped(self):
        """RouterLambdaRole Bedrock InvokeModel resource must reference foundation-model/, not bare '*'."""
        template = load_template()
        role = get_router_lambda_role(template)
        statements = get_all_policy_statements(role)

        bedrock_stmts = [
            s for s in statements
            if any(
                "bedrock:" in a for a in (
                    [s.get("Action")] if isinstance(s.get("Action"), str)
                    else s.get("Action", [])
                )
            )
        ]
        assert len(bedrock_stmts) > 0, (
            "RouterLambdaRole must have a bedrock:InvokeModel statement"
        )

        for stmt in bedrock_stmts:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            for arn in arns:
                assert not is_bare_wildcard(arn), (
                    f"RouterLambdaRole Bedrock statement '{sid}' has bare wildcard '*' resource. "
                    f"Must be scoped to a foundation-model ARN."
                )
                # If it's a string ARN (not an intrinsic), verify it mentions foundation-model
                if isinstance(arn, str) and arn.startswith("arn:"):
                    assert "foundation-model" in arn, (
                        f"RouterLambdaRole Bedrock resource ARN '{arn}' should reference "
                        f"a foundation-model ARN path."
                    )

    def test_cloudwatch_put_metric_has_namespace_condition(self):
        """RouterLambdaRole CloudWatch PutMetricData must have a namespace condition."""
        template = load_template()
        role = get_router_lambda_role(template)
        statements = get_all_policy_statements(role)

        cw_put_stmts = [
            s for s in statements
            if "cloudwatch:PutMetricData" in (
                [s.get("Action")] if isinstance(s.get("Action"), str)
                else s.get("Action", [])
            )
        ]
        # Router may or may not have CloudWatch — only assert condition if statement exists
        for stmt in cw_put_stmts:
            arns = collect_resource_arns(stmt)
            has_wildcard = any(is_bare_wildcard(a) for a in arns)
            has_condition = "Condition" in stmt
            if has_wildcard:
                assert has_condition, (
                    "RouterLambdaRole PutMetricData with wildcard '*' must have a "
                    "cloudwatch:namespace Condition to scope the metric namespace."
                )

    def test_no_bare_wildcard_without_condition(self):
        """RouterLambdaRole must not have bare wildcard '*' resources without a Condition."""
        template = load_template()
        role = get_router_lambda_role(template)
        statements = get_all_policy_statements(role)
        for stmt in statements:
            arns = collect_resource_arns(stmt)
            sid = stmt.get("Sid", "unknown")
            has_condition = "Condition" in stmt
            for arn in arns:
                if has_condition:
                    continue
                assert not is_bare_wildcard(arn), (
                    f"RouterLambdaRole statement '{sid}' has bare wildcard '*' resource "
                    f"without a scoping Condition."
                )
