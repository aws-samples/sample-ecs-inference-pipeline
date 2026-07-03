"""
Property 1: Private subnet enforcement

For any subnet referenced in the CloudFormation template for ECS task placement,
the subnet configuration SHALL have MapPublicIpOnLaunch set to false and the ECS
service network configuration SHALL have AssignPublicIp set to DISABLED.

Validates: Requirements 1.1, 7.1
"""

import os
import yaml
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "infrastructure", "template.yaml"
)


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
        yaml.add_multi_constructor(
            tag, _cfn_constructor, Loader=loader
        )
    return loader


def load_template():
    """Load and parse the CloudFormation template."""
    with open(TEMPLATE_PATH, "r") as f:
        return yaml.load(f, Loader=_get_cfn_loader())


def get_subnet_resources(template):
    """Return all AWS::EC2::Subnet resources from the template."""
    resources = template.get("Resources", {})
    return {
        name: res
        for name, res in resources.items()
        if res.get("Type") == "AWS::EC2::Subnet"
    }


def get_ecs_service_resources(template):
    """Return all AWS::ECS::Service resources from the template."""
    resources = template.get("Resources", {})
    return {
        name: res
        for name, res in resources.items()
        if res.get("Type") == "AWS::ECS::Service"
    }


class TestPrivateSubnetEnforcementUnit:
    """Unit tests for private subnet enforcement."""

    def test_template_loads_successfully(self):
        template = load_template()
        assert template is not None
        assert "Resources" in template

    def test_all_subnets_disable_public_ip(self):
        """Every subnet in the template must have MapPublicIpOnLaunch: false."""
        template = load_template()
        subnets = get_subnet_resources(template)
        assert len(subnets) > 0, "Template must contain at least one subnet"

        for name, subnet in subnets.items():
            props = subnet.get("Properties", {})
            map_public = props.get("MapPublicIpOnLaunch", True)
            assert map_public is False, (
                f"Subnet {name} has MapPublicIpOnLaunch={map_public}, expected false"
            )

    def test_private_subnets_exist(self):
        """Template must contain private subnets for ECS task placement."""
        template = load_template()
        subnets = get_subnet_resources(template)
        private_subnets = {
            name: s
            for name, s in subnets.items()
            if "private" in name.lower() or "Private" in name
        }
        assert len(private_subnets) >= 2, (
            "Template must have at least 2 private subnets for multi-AZ placement"
        )

    def test_ecs_service_assign_public_ip_disabled(self):
        """If an ECS service exists, AssignPublicIp must be DISABLED."""
        template = load_template()
        services = get_ecs_service_resources(template)

        if not services:
            pytest.skip(
                "ECS Service not yet defined in template (will be added in task 2.4)"
            )

        for name, service in services.items():
            props = service.get("Properties", {})
            net_config = props.get("NetworkConfiguration", {})
            awsvpc_config = net_config.get("AwsvpcConfiguration", {})
            assign_public = awsvpc_config.get("AssignPublicIp", "ENABLED")
            assert assign_public == "DISABLED", (
                f"ECS Service {name} has AssignPublicIp={assign_public}, expected DISABLED"
            )


class TestPrivateSubnetEnforcementProperty:
    """
    Property-based test for private subnet enforcement.

    **Validates: Requirements 1.1, 7.1**

    Since the CloudFormation template is a static artifact, we use hypothesis
    to generate random subnet indices and verify the property holds for every
    subnet selected. This ensures the property is checked across all possible
    subnet selections from the template.
    """

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_subnet_selection_always_private(self, data):
        """
        Property 1: Private subnet enforcement

        For any randomly selected subnet from the template, MapPublicIpOnLaunch
        must be false.

        **Validates: Requirements 1.1, 7.1**
        """
        template = load_template()
        subnets = get_subnet_resources(template)
        subnet_names = list(subnets.keys())
        assert len(subnet_names) > 0, "No subnets found in template"

        # Pick a random subnet from the template
        selected_name = data.draw(st.sampled_from(subnet_names))
        subnet = subnets[selected_name]
        props = subnet.get("Properties", {})
        map_public = props.get("MapPublicIpOnLaunch", True)

        assert map_public is False, (
            f"Subnet {selected_name} has MapPublicIpOnLaunch={map_public}, "
            f"expected false. All subnets used for ECS task placement must "
            f"disable public IP assignment."
        )

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_ecs_service_has_public_ip_disabled(self, data):
        """
        Property 1: Private subnet enforcement (ECS Service network config)

        For any ECS service in the template, AssignPublicIp must be DISABLED.

        **Validates: Requirements 1.1, 7.1**
        """
        template = load_template()
        services = get_ecs_service_resources(template)

        if not services:
            pytest.skip(
                "ECS Service not yet defined in template (will be added in task 2.4)"
            )

        service_names = list(services.keys())
        selected_name = data.draw(st.sampled_from(service_names))
        service = services[selected_name]

        props = service.get("Properties", {})
        net_config = props.get("NetworkConfiguration", {})
        awsvpc_config = net_config.get("AwsvpcConfiguration", {})
        assign_public = awsvpc_config.get("AssignPublicIp", "ENABLED")

        assert assign_public == "DISABLED", (
            f"ECS Service {selected_name} has AssignPublicIp={assign_public}, "
            f"expected DISABLED. All ECS services must disable public IP assignment."
        )
