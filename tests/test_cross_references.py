"""
Task 8.1: Verify all CloudFormation resource cross-references are correct.

Ensures no orphaned or disconnected resources in the template. Validates:
1. ECS Service -> Task Definition, Capacity Provider, Subnets, Security Groups
2. Task Definition -> ECR Repo URI, SQS Queue URLs, Log Group, IAM Roles
3. Scaling Policies -> Scalable Target -> ECS Service
4. Lambda -> SQS Queue URL, Cluster Name, Service Name
5. Alarms -> Scaling Policies, SNS Topic
6. Capacity Provider -> ASG
7. ASG -> Launch Template, Subnets

Validates: Requirements 1.2, 4.1, 3.1
"""

import os
import re
import yaml
import pytest

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "infrastructure", "template.yaml"
)


class CfnTag:
    """Represents a CloudFormation intrinsic function value."""
    def __init__(self, tag_name, value):
        self.tag = tag_name   # e.g. "Ref", "GetAtt", "Sub"
        self.value = value

    def __repr__(self):
        return f"CfnTag({self.tag}, {self.value!r})"


def _make_cfn_constructor(tag_name):
    """Create a YAML constructor for a specific CFN tag."""
    def constructor(loader, node):
        if isinstance(node, yaml.ScalarNode):
            return CfnTag(tag_name, loader.construct_scalar(node))
        elif isinstance(node, yaml.SequenceNode):
            return CfnTag(tag_name, loader.construct_sequence(node))
        elif isinstance(node, yaml.MappingNode):
            return CfnTag(tag_name, loader.construct_mapping(node))
        return None
    return constructor


def _get_cfn_loader():
    """Create a YAML loader that handles CloudFormation intrinsic functions."""
    loader = type("CfnLoader", (yaml.SafeLoader,), {})
    cfn_tags = {
        "!Ref": "Ref",
        "!Sub": "Sub",
        "!GetAtt": "GetAtt",
        "!Select": "Select",
        "!Split": "Split",
        "!Join": "Join",
        "!FindInMap": "FindInMap",
        "!If": "If",
        "!Equals": "Equals",
        "!Not": "Not",
        "!And": "And",
        "!Or": "Or",
        "!GetAZs": "GetAZs",
        "!ImportValue": "ImportValue",
        "!Condition": "Condition",
        "!Base64": "Base64",
        "!Cidr": "Cidr",
    }
    for yaml_tag, name in cfn_tags.items():
        loader.add_constructor(yaml_tag, _make_cfn_constructor(name))
    return loader


def load_template():
    """Load and parse the CloudFormation template."""
    with open(TEMPLATE_PATH, "r") as f:
        return yaml.load(f, Loader=_get_cfn_loader())


def get_resource(template, logical_id):
    """Get a resource by logical ID."""
    return template.get("Resources", {}).get(logical_id)


def get_all_resource_ids(template):
    """Get all logical resource IDs defined in the template."""
    return set(template.get("Resources", {}).keys())


def get_all_parameter_names(template):
    """Get all parameter names defined in the template."""
    return set(template.get("Parameters", {}).keys())


def is_ref(value, target=None):
    """Check if a value is a !Ref, optionally to a specific target."""
    if isinstance(value, CfnTag) and value.tag == "Ref":
        if target is None:
            return True
        return value.value == target
    return False


def get_ref_target(value):
    """Extract the target of a !Ref."""
    if isinstance(value, CfnTag) and value.tag == "Ref":
        return value.value
    return None


def is_getatt(value, target_resource=None, target_attr=None):
    """Check if a value is a !GetAtt, optionally to a specific resource.attribute."""
    if isinstance(value, CfnTag) and value.tag == "GetAtt":
        val = value.value
        if isinstance(val, str) and "." in val:
            resource, attr = val.split(".", 1)
            if target_resource and resource != target_resource:
                return False
            if target_attr and attr != target_attr:
                return False
            return True
    return False


def get_getatt_resource(value):
    """Extract the resource name from a !GetAtt."""
    if isinstance(value, CfnTag) and value.tag == "GetAtt":
        val = value.value
        if isinstance(val, str) and "." in val:
            return val.split(".", 1)[0]
    return None


def collect_refs_recursive(obj):
    """Recursively collect all !Ref and !GetAtt targets from a nested structure."""
    refs = set()
    getatts = set()
    if isinstance(obj, CfnTag):
        if obj.tag == "Ref":
            refs.add(obj.value)
        elif obj.tag == "GetAtt":
            val = obj.value
            if isinstance(val, str) and "." in val:
                getatts.add(val.split(".", 1)[0])
        elif obj.tag == "Sub":
            # Parse ${Resource} and ${Resource.Attr} from !Sub strings
            val = obj.value
            local_vars = set()
            if isinstance(val, str):
                template_str = val
            elif isinstance(val, list) and len(val) >= 1:
                # !Sub with variable map: [template_string, {var: value}]
                template_str = val[0] if isinstance(val[0], str) else ""
                # Collect local variable names from the map
                if len(val) > 1 and isinstance(val[1], dict):
                    local_vars = set(val[1].keys())
                    # Recurse into the variable map values
                    for v in val[1].values():
                        r, g = collect_refs_recursive(v)
                        refs.update(r)
                        getatts.update(g)
            else:
                template_str = ""

            for match in re.findall(r'\$\{([^}]+)\}', template_str):
                # Skip local variables defined in the !Sub variable map
                if match in local_vars:
                    continue
                if "." in match and "::" not in match:
                    getatts.add(match.split(".", 1)[0])
                elif "::" not in match:
                    refs.add(match)
        else:
            # For other tags, recurse into the value
            r, g = collect_refs_recursive(obj.value)
            refs.update(r)
            getatts.update(g)
    elif isinstance(obj, dict):
        for v in obj.values():
            r, g = collect_refs_recursive(v)
            refs.update(r)
            getatts.update(g)
    elif isinstance(obj, list):
        for item in obj:
            r, g = collect_refs_recursive(item)
            refs.update(r)
            getatts.update(g)
    return refs, getatts


# --- Helpers to extract specific resource properties ---


def get_ecs_service_props(template):
    """Get the GPUInferenceService properties."""
    svc = get_resource(template, "GPUInferenceService")
    assert svc is not None, "GPUInferenceService not found in template"
    return svc.get("Properties", {})


def get_task_def_props(template):
    """Get the GPUInferenceTaskDefinition properties."""
    td = get_resource(template, "GPUInferenceTaskDefinition")
    assert td is not None, "GPUInferenceTaskDefinition not found in template"
    return td.get("Properties", {})


def get_container_def(template):
    """Get the first container definition from the task definition."""
    props = get_task_def_props(template)
    containers = props.get("ContainerDefinitions", [])
    assert len(containers) > 0, "No container definitions found"
    return containers[0]


def get_env_var(container_def, name):
    """Get an environment variable value from a container definition."""
    for env in container_def.get("Environment", []):
        if env.get("Name") == name:
            return env.get("Value")
    return None


# --- Test Classes ---


class TestECSServiceCrossReferences:
    """1. ECS Service -> Task Definition, Capacity Provider, Subnets, Security Groups"""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)
        self.params = get_all_parameter_names(self.template)
        self.svc_props = get_ecs_service_props(self.template)

    def test_service_references_task_definition(self):
        """ECS Service TaskDefinition must reference GPUInferenceTaskDefinition."""
        td_ref = self.svc_props.get("TaskDefinition")
        assert is_ref(td_ref, "GPUInferenceTaskDefinition"), (
            f"ECS Service TaskDefinition should !Ref GPUInferenceTaskDefinition, got {td_ref}"
        )
        assert "GPUInferenceTaskDefinition" in self.resources

    def test_service_references_capacity_provider(self):
        """ECS Service CapacityProviderStrategy must reference GPUCapacityProvider."""
        strategy = self.svc_props.get("CapacityProviderStrategy", [])
        assert len(strategy) > 0, "ECS Service must have a CapacityProviderStrategy"
        cp_ref = strategy[0].get("CapacityProvider")
        assert is_ref(cp_ref, "GPUCapacityProvider"), (
            f"CapacityProviderStrategy should !Ref GPUCapacityProvider, got {cp_ref}"
        )
        assert "GPUCapacityProvider" in self.resources

    def test_service_references_cluster(self):
        """ECS Service Cluster must reference ECSCluster."""
        cluster_ref = self.svc_props.get("Cluster")
        assert is_ref(cluster_ref, "ECSCluster"), (
            f"ECS Service Cluster should !Ref ECSCluster, got {cluster_ref}"
        )
        assert "ECSCluster" in self.resources

    def test_service_references_private_subnets(self):
        """ECS Service network config must reference both private subnets."""
        net_config = self.svc_props.get("NetworkConfiguration", {})
        awsvpc = net_config.get("AwsvpcConfiguration", {})
        subnets = awsvpc.get("Subnets", [])
        subnet_targets = {get_ref_target(s) for s in subnets if is_ref(s)}
        assert "PrivateSubnetA" in subnet_targets, "Service must reference PrivateSubnetA"
        assert "PrivateSubnetB" in subnet_targets, "Service must reference PrivateSubnetB"
        for target in subnet_targets:
            assert target in self.resources, f"Subnet {target} not found in resources"

    def test_service_references_security_group(self):
        """ECS Service network config must reference a valid security group."""
        net_config = self.svc_props.get("NetworkConfiguration", {})
        awsvpc = net_config.get("AwsvpcConfiguration", {})
        sgs = awsvpc.get("SecurityGroups", [])
        assert len(sgs) > 0, "ECS Service must have at least one security group"
        sg_targets = {get_ref_target(s) for s in sgs if is_ref(s)}
        for target in sg_targets:
            assert target in self.resources, f"Security group {target} not found in resources"


class TestTaskDefinitionCrossReferences:
    """2. Task Definition -> ECR Repo URI, SQS Queue URLs, Log Group, IAM Roles"""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)
        self.td_props = get_task_def_props(self.template)
        self.container = get_container_def(self.template)

    def test_task_def_references_execution_role(self):
        """Task Definition ExecutionRoleArn must reference ECSTaskExecutionRole."""
        exec_role = self.td_props.get("ExecutionRoleArn")
        assert is_getatt(exec_role, "ECSTaskExecutionRole", "Arn"), (
            f"ExecutionRoleArn should !GetAtt ECSTaskExecutionRole.Arn, got {exec_role}"
        )
        assert "ECSTaskExecutionRole" in self.resources

    def test_task_def_references_task_role(self):
        """Task Definition TaskRoleArn must reference ECSTaskRole."""
        task_role = self.td_props.get("TaskRoleArn")
        assert is_getatt(task_role, "ECSTaskRole", "Arn"), (
            f"TaskRoleArn should !GetAtt ECSTaskRole.Arn, got {task_role}"
        )
        assert "ECSTaskRole" in self.resources

    def test_container_image_references_ecr_repo(self):
        """Container image must reference InferenceECRRepository URI."""
        image = self.container.get("Image")
        assert image is not None, "Container Image must be defined"
        refs, getatts = collect_refs_recursive(image)
        assert "InferenceECRRepository" in getatts, (
            f"Container Image should reference InferenceECRRepository, "
            f"found refs={refs}, getatts={getatts}"
        )
        assert "InferenceECRRepository" in self.resources

    def test_env_request_queue_url_references_queue(self):
        """REQUEST_QUEUE_URL env var must reference RequestQueue."""
        val = get_env_var(self.container, "REQUEST_QUEUE_URL")
        assert val is not None, "REQUEST_QUEUE_URL env var must be defined"
        assert is_ref(val, "RequestQueue"), (
            f"REQUEST_QUEUE_URL should !Ref RequestQueue, got {val}"
        )
        assert "RequestQueue" in self.resources

    def test_env_dlq_url_references_dlq(self):
        """DLQ_URL env var must reference DeadLetterQueue."""
        val = get_env_var(self.container, "DLQ_URL")
        assert val is not None, "DLQ_URL env var must be defined"
        assert is_ref(val, "DeadLetterQueue"), (
            f"DLQ_URL should !Ref DeadLetterQueue, got {val}"
        )
        assert "DeadLetterQueue" in self.resources

    def test_log_config_references_log_group(self):
        """Log configuration must reference InferenceLogGroup."""
        log_config = self.container.get("LogConfiguration", {})
        options = log_config.get("Options", {})
        log_group = options.get("awslogs-group")
        assert log_group is not None, "awslogs-group must be defined"
        assert is_ref(log_group, "InferenceLogGroup"), (
            f"awslogs-group should !Ref InferenceLogGroup, got {log_group}"
        )
        assert "InferenceLogGroup" in self.resources


class TestScalingPolicyCrossReferences:
    """3. Scaling Policies -> Scalable Target -> ECS Service"""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)

    def test_scalable_target_references_ecs_service(self):
        """ScalableTarget ResourceId must reference ECSCluster and GPUInferenceService."""
        target = get_resource(self.template, "ECSScalableTarget")
        assert target is not None, "ECSScalableTarget not found"
        props = target.get("Properties", {})
        resource_id = props.get("ResourceId")
        refs, getatts = collect_refs_recursive(resource_id)
        all_targets = refs | getatts
        assert "ECSCluster" in all_targets or "GPUInferenceService" in all_targets, (
            f"ScalableTarget ResourceId should reference ECS service resources, "
            f"found refs={refs}, getatts={getatts}"
        )

    def test_scale_out_policy_references_scalable_target(self):
        """ScaleOutPolicy must reference ECSScalableTarget."""
        policy = get_resource(self.template, "ScaleOutPolicy")
        assert policy is not None, "ScaleOutPolicy not found"
        props = policy.get("Properties", {})
        target_id = props.get("ScalingTargetId")
        assert is_ref(target_id, "ECSScalableTarget"), (
            f"ScaleOutPolicy ScalingTargetId should !Ref ECSScalableTarget, got {target_id}"
        )
        assert "ECSScalableTarget" in self.resources

    def test_scale_in_policy_references_scalable_target(self):
        """ScaleInPolicy must reference ECSScalableTarget."""
        policy = get_resource(self.template, "ScaleInPolicy")
        assert policy is not None, "ScaleInPolicy not found"
        props = policy.get("Properties", {})
        target_id = props.get("ScalingTargetId")
        assert is_ref(target_id, "ECSScalableTarget"), (
            f"ScaleInPolicy ScalingTargetId should !Ref ECSScalableTarget, got {target_id}"
        )
        assert "ECSScalableTarget" in self.resources


class TestLambdaCrossReferences:
    """4. Lambda -> SQS Queue URL, Cluster Name, Service Name"""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)

    def test_lambda_role_references_request_queue(self):
        """Lambda IAM role must reference RequestQueue ARN for SQS access."""
        role = get_resource(self.template, "ScalingMetricLambdaRole")
        assert role is not None, "ScalingMetricLambdaRole not found"
        refs, getatts = collect_refs_recursive(role)
        assert "RequestQueue" in getatts or "RequestQueue" in refs, (
            f"Lambda role should reference RequestQueue, found refs={refs}, getatts={getatts}"
        )

    def test_lambda_function_references_role(self):
        """Lambda function must reference its IAM role."""
        fn = get_resource(self.template, "ScalingMetricLambdaFunction")
        assert fn is not None, "ScalingMetricLambdaFunction not found"
        props = fn.get("Properties", {})
        role = props.get("Role")
        assert is_getatt(role, "ScalingMetricLambdaRole", "Arn"), (
            f"Lambda Role should !GetAtt ScalingMetricLambdaRole.Arn, got {role}"
        )
        assert "ScalingMetricLambdaRole" in self.resources

    def test_lambda_env_references_queue_url(self):
        """Lambda QUEUE_URL env var must reference RequestQueue."""
        fn = get_resource(self.template, "ScalingMetricLambdaFunction")
        assert fn is not None
        props = fn.get("Properties", {})
        env_vars = props.get("Environment", {}).get("Variables", {})
        queue_url = env_vars.get("QUEUE_URL")
        assert is_ref(queue_url, "RequestQueue"), (
            f"Lambda QUEUE_URL should !Ref RequestQueue, got {queue_url}"
        )

    def test_lambda_permission_references_function_and_rule(self):
        """Lambda permission must reference the function and the schedule rule."""
        perm = get_resource(self.template, "ScalingMetricLambdaPermission")
        assert perm is not None, "ScalingMetricLambdaPermission not found"
        props = perm.get("Properties", {})
        fn_name = props.get("FunctionName")
        assert is_ref(fn_name, "ScalingMetricLambdaFunction"), (
            f"Permission FunctionName should !Ref ScalingMetricLambdaFunction, got {fn_name}"
        )
        source_arn = props.get("SourceArn")
        assert is_getatt(source_arn, "ScalingMetricScheduleRule", "Arn"), (
            f"Permission SourceArn should !GetAtt ScalingMetricScheduleRule.Arn, got {source_arn}"
        )

    def test_schedule_rule_targets_lambda(self):
        """EventBridge schedule rule must target the Lambda function."""
        rule = get_resource(self.template, "ScalingMetricScheduleRule")
        assert rule is not None, "ScalingMetricScheduleRule not found"
        props = rule.get("Properties", {})
        targets = props.get("Targets", [])
        assert len(targets) > 0, "Schedule rule must have at least one target"
        target_arn = targets[0].get("Arn")
        assert is_getatt(target_arn, "ScalingMetricLambdaFunction", "Arn"), (
            f"Schedule target Arn should !GetAtt ScalingMetricLambdaFunction.Arn, got {target_arn}"
        )


class TestAlarmCrossReferences:
    """5. Alarms -> Scaling Policies, SNS Topic"""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)

    def test_scale_out_alarm_references_scale_out_policy(self):
        """ScaleOutAlarm AlarmActions must reference ScaleOutPolicy."""
        alarm = get_resource(self.template, "ScaleOutAlarm")
        assert alarm is not None, "ScaleOutAlarm not found"
        props = alarm.get("Properties", {})
        actions = props.get("AlarmActions", [])
        action_targets = {get_ref_target(a) for a in actions if is_ref(a)}
        assert "ScaleOutPolicy" in action_targets, (
            f"ScaleOutAlarm should reference ScaleOutPolicy, found {action_targets}"
        )
        assert "ScaleOutPolicy" in self.resources

    def test_scale_in_alarm_references_scale_in_policy(self):
        """ScaleInAlarm AlarmActions must reference ScaleInPolicy."""
        alarm = get_resource(self.template, "ScaleInAlarm")
        assert alarm is not None, "ScaleInAlarm not found"
        props = alarm.get("Properties", {})
        actions = props.get("AlarmActions", [])
        action_targets = {get_ref_target(a) for a in actions if is_ref(a)}
        assert "ScaleInPolicy" in action_targets, (
            f"ScaleInAlarm should reference ScaleInPolicy, found {action_targets}"
        )
        assert "ScaleInPolicy" in self.resources

    def test_gpu_temp_alarm_references_sns_topic(self):
        """GPUTemperatureAlarm AlarmActions must reference AlarmNotificationTopic."""
        alarm = get_resource(self.template, "GPUTemperatureAlarm")
        assert alarm is not None, "GPUTemperatureAlarm not found"
        props = alarm.get("Properties", {})
        actions = props.get("AlarmActions", [])
        action_targets = {get_ref_target(a) for a in actions if is_ref(a)}
        assert "AlarmNotificationTopic" in action_targets, (
            f"GPUTemperatureAlarm should reference AlarmNotificationTopic, found {action_targets}"
        )
        assert "AlarmNotificationTopic" in self.resources

    def test_dlq_depth_alarm_references_sns_topic(self):
        """DLQDepthAlarm AlarmActions must reference AlarmNotificationTopic."""
        alarm = get_resource(self.template, "DLQDepthAlarm")
        assert alarm is not None, "DLQDepthAlarm not found"
        props = alarm.get("Properties", {})
        actions = props.get("AlarmActions", [])
        action_targets = {get_ref_target(a) for a in actions if is_ref(a)}
        assert "AlarmNotificationTopic" in action_targets, (
            f"DLQDepthAlarm should reference AlarmNotificationTopic, found {action_targets}"
        )

    def test_dlq_depth_alarm_references_dlq(self):
        """DLQDepthAlarm Dimensions must reference DeadLetterQueue."""
        alarm = get_resource(self.template, "DLQDepthAlarm")
        assert alarm is not None
        refs, getatts = collect_refs_recursive(alarm)
        assert "DeadLetterQueue" in getatts or "DeadLetterQueue" in refs, (
            f"DLQDepthAlarm should reference DeadLetterQueue, found refs={refs}, getatts={getatts}"
        )


class TestCapacityProviderCrossReferences:
    """6. Capacity Provider -> Managed Instances"""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)

    def test_capacity_provider_uses_managed_instances(self):
        """GPUCapacityProvider must use ManagedInstancesProvider."""
        cp = get_resource(self.template, "GPUCapacityProvider")
        assert cp is not None, "GPUCapacityProvider not found"
        props = cp.get("Properties", {})
        mi_provider = props.get("ManagedInstancesProvider")
        assert mi_provider is not None, (
            "GPUCapacityProvider must have ManagedInstancesProvider"
        )

    def test_capacity_provider_references_infrastructure_role(self):
        """GPUCapacityProvider must reference ECSInfrastructureRole."""
        cp = get_resource(self.template, "GPUCapacityProvider")
        props = cp.get("Properties", {})
        mi_provider = props.get("ManagedInstancesProvider", {})
        infra_role_arn = mi_provider.get("InfrastructureRoleArn")
        assert is_getatt(infra_role_arn, "ECSInfrastructureRole", "Arn"), (
            f"InfrastructureRoleArn should !GetAtt ECSInfrastructureRole.Arn, got {infra_role_arn}"
        )
        assert "ECSInfrastructureRole" in self.resources

    def test_capacity_provider_references_instance_profile(self):
        """GPUCapacityProvider must reference ECSInstanceProfile."""
        cp = get_resource(self.template, "GPUCapacityProvider")
        props = cp.get("Properties", {})
        mi_provider = props.get("ManagedInstancesProvider", {})
        launch_template = mi_provider.get("InstanceLaunchTemplate", {})
        profile_arn = launch_template.get("Ec2InstanceProfileArn")
        assert is_getatt(profile_arn, "ECSInstanceProfile", "Arn"), (
            f"Ec2InstanceProfileArn should !GetAtt ECSInstanceProfile.Arn, got {profile_arn}"
        )
        assert "ECSInstanceProfile" in self.resources

    def test_capacity_provider_references_private_subnets(self):
        """GPUCapacityProvider network config must reference both private subnets."""
        cp = get_resource(self.template, "GPUCapacityProvider")
        props = cp.get("Properties", {})
        mi_provider = props.get("ManagedInstancesProvider", {})
        launch_template = mi_provider.get("InstanceLaunchTemplate", {})
        net_config = launch_template.get("NetworkConfiguration", {})
        subnets = net_config.get("Subnets", [])
        subnet_targets = {get_ref_target(s) for s in subnets if is_ref(s)}
        assert "PrivateSubnetA" in subnet_targets, "Must reference PrivateSubnetA"
        assert "PrivateSubnetB" in subnet_targets, "Must reference PrivateSubnetB"

    def test_capacity_provider_references_security_group(self):
        """GPUCapacityProvider network config must reference ECSInstanceSecurityGroup."""
        cp = get_resource(self.template, "GPUCapacityProvider")
        props = cp.get("Properties", {})
        mi_provider = props.get("ManagedInstancesProvider", {})
        launch_template = mi_provider.get("InstanceLaunchTemplate", {})
        net_config = launch_template.get("NetworkConfiguration", {})
        sgs = net_config.get("SecurityGroups", [])
        sg_targets = {get_ref_target(s) for s in sgs if is_ref(s)}
        assert "ECSInstanceSecurityGroup" in sg_targets, (
            f"Must reference ECSInstanceSecurityGroup, found {sg_targets}"
        )

    def test_capacity_provider_gpu_auto_repair_enabled(self):
        """GPUCapacityProvider must have GPU auto-repair enabled."""
        cp = get_resource(self.template, "GPUCapacityProvider")
        props = cp.get("Properties", {})
        mi_provider = props.get("ManagedInstancesProvider", {})
        auto_repair = mi_provider.get("AutoRepairConfiguration", {})
        assert auto_repair.get("ActionsStatus") == "ENABLED", (
            "AutoRepairConfiguration.ActionsStatus must be ENABLED"
        )

    def test_capacity_provider_gpu_instance_requirements(self):
        """GPUCapacityProvider must require GPU accelerators."""
        cp = get_resource(self.template, "GPUCapacityProvider")
        props = cp.get("Properties", {})
        mi_provider = props.get("ManagedInstancesProvider", {})
        launch_template = mi_provider.get("InstanceLaunchTemplate", {})
        reqs = launch_template.get("InstanceRequirements", {})
        assert "gpu" in reqs.get("AcceleratorTypes", []), (
            "InstanceRequirements must include AcceleratorTypes: [gpu]"
        )
        assert "nvidia" in reqs.get("AcceleratorManufacturers", []), (
            "InstanceRequirements must include AcceleratorManufacturers: [nvidia]"
        )

    def test_cluster_association_references_cluster_and_cp(self):
        """ClusterCapacityProviderAssociation must reference ECSCluster and GPUCapacityProvider."""
        assoc = get_resource(self.template, "ClusterCapacityProviderAssociation")
        assert assoc is not None, "ClusterCapacityProviderAssociation not found"
        props = assoc.get("Properties", {})
        cluster = props.get("Cluster")
        assert is_ref(cluster, "ECSCluster"), (
            f"Association Cluster should !Ref ECSCluster, got {cluster}"
        )
        cps = props.get("CapacityProviders", [])
        cp_targets = {get_ref_target(c) for c in cps if is_ref(c)}
        assert "GPUCapacityProvider" in cp_targets, (
            f"Association should include GPUCapacityProvider, found {cp_targets}"
        )


class TestInfrastructureRoleCrossReferences:
    """7. Infrastructure Role -> Managed Policy"""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)

    def test_infrastructure_role_exists(self):
        """ECSInfrastructureRole must exist for Managed Instances."""
        role = get_resource(self.template, "ECSInfrastructureRole")
        assert role is not None, "ECSInfrastructureRole not found"

    def test_infrastructure_role_trust_policy(self):
        """ECSInfrastructureRole must trust ecs.amazonaws.com."""
        role = get_resource(self.template, "ECSInfrastructureRole")
        props = role.get("Properties", {})
        trust = props.get("AssumeRolePolicyDocument", {})
        statements = trust.get("Statement", [])
        principals = []
        for stmt in statements:
            principal = stmt.get("Principal", {})
            svc = principal.get("Service")
            if isinstance(svc, str):
                principals.append(svc)
            elif isinstance(svc, list):
                principals.extend(svc)
        assert "ecs.amazonaws.com" in principals, (
            f"Infrastructure role must trust ecs.amazonaws.com, found {principals}"
        )

    def test_infrastructure_role_has_managed_instances_policy(self):
        """ECSInfrastructureRole must have the Managed Instances policy."""
        role = get_resource(self.template, "ECSInfrastructureRole")
        props = role.get("Properties", {})
        policies = props.get("ManagedPolicyArns", [])
        assert any("AmazonECSInfrastructureRolePolicyForManagedInstances" in p for p in policies), (
            f"Infrastructure role must include AmazonECSInfrastructureRolePolicyForManagedInstances, found {policies}"
        )


class TestGlobalReferenceIntegrity:
    """Verify all !Ref and !GetAtt targets resolve to existing resources or parameters."""

    def setup_method(self):
        self.template = load_template()
        self.resources = get_all_resource_ids(self.template)
        self.params = get_all_parameter_names(self.template)
        # Pseudo-parameters that are always valid
        self.pseudo_params = {
            "AWS::StackName", "AWS::Region", "AWS::AccountId",
            "AWS::StackId", "AWS::URLSuffix", "AWS::NoValue",
            "AWS::NotificationARNs", "AWS::Partition",
        }
        self.valid_targets = self.resources | self.params | self.pseudo_params

    def test_all_refs_resolve(self):
        """Every !Ref in the template must point to a defined resource or parameter."""
        refs, _ = collect_refs_recursive(self.template.get("Resources", {}))
        unresolved = refs - self.valid_targets
        assert len(unresolved) == 0, (
            f"Unresolved !Ref targets: {unresolved}. "
            f"These must be defined as Resources or Parameters."
        )

    def test_all_getatts_resolve(self):
        """Every !GetAtt in the template must point to a defined resource."""
        _, getatts = collect_refs_recursive(self.template.get("Resources", {}))
        unresolved = getatts - self.resources
        assert len(unresolved) == 0, (
            f"Unresolved !GetAtt resource targets: {unresolved}. "
            f"These must be defined as Resources."
        )

    def test_all_sub_refs_resolve(self):
        """Every ${Ref} inside !Sub strings must point to a valid target."""
        refs, getatts = collect_refs_recursive(self.template.get("Resources", {}))
        all_unresolved_refs = refs - self.valid_targets
        all_unresolved_getatts = getatts - self.resources
        assert len(all_unresolved_refs) == 0, (
            f"Unresolved references in !Sub: {all_unresolved_refs}"
        )
        assert len(all_unresolved_getatts) == 0, (
            f"Unresolved !GetAtt in !Sub: {all_unresolved_getatts}"
        )

    def test_outputs_reference_existing_resources(self):
        """All Outputs must reference existing resources."""
        outputs = self.template.get("Outputs", {})
        refs, getatts = collect_refs_recursive(outputs)
        unresolved_refs = refs - self.valid_targets
        unresolved_getatts = getatts - self.resources
        assert len(unresolved_refs) == 0, (
            f"Unresolved !Ref in Outputs: {unresolved_refs}"
        )
        assert len(unresolved_getatts) == 0, (
            f"Unresolved !GetAtt in Outputs: {unresolved_getatts}"
        )

    def test_no_orphaned_iam_roles(self):
        """All IAM roles must be referenced by at least one other resource."""
        iam_roles = {
            name for name, res in self.template.get("Resources", {}).items()
            if res.get("Type") == "AWS::IAM::Role"
        }
        # Collect all refs/getatts from non-IAM-role resources
        referenced = set()
        for name, res in self.template.get("Resources", {}).items():
            if res.get("Type") != "AWS::IAM::Role":
                r, g = collect_refs_recursive(res)
                referenced.update(r)
                referenced.update(g)
        # Also check outputs
        r, g = collect_refs_recursive(self.template.get("Outputs", {}))
        referenced.update(r)
        referenced.update(g)

        orphaned = iam_roles - referenced
        assert len(orphaned) == 0, (
            f"Orphaned IAM roles (not referenced by any resource): {orphaned}"
        )
