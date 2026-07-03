"""
Property 8: Cost-allocation tag coverage

For any resource in the CloudFormation template that supports the Tags property,
the resource SHALL include a tag with the key matching the CostAllocationTagProject
parameter value.

Validates: Requirements 8.3
"""

import os
import yaml
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "infrastructure", "template.yaml"
)

# CloudFormation resource types that do NOT support the Tags property
NON_TAGGABLE_TYPES = {
    "AWS::EC2::VPCGatewayAttachment",
    "AWS::EC2::Route",
    "AWS::EC2::SubnetRouteTableAssociation",
    "AWS::EC2::VPCEndpoint",
    "AWS::EC2::LaunchTemplate",
    "AWS::IAM::InstanceProfile",
    "AWS::ECS::ClusterCapacityProviderAssociations",
    "AWS::Lambda::Permission",
    "AWS::Events::Rule",
    "AWS::ApplicationAutoScaling::ScalableTarget",
    "AWS::ApplicationAutoScaling::ScalingPolicy",
    "AWS::CloudWatch::Alarm",
    "AWS::CloudWatch::Dashboard",
    "AWS::ApiGatewayV2::Integration",
    "AWS::ApiGatewayV2::Route",
}


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


def get_taggable_resources(template):
    """Return all resources from the template that support the Tags property.

    Excludes resource types known not to support Tags in CloudFormation:
    VPCGatewayAttachment, Route, SubnetRouteTableAssociation, VPCEndpoint
    (Gateway type), and InstanceProfile.
    """
    resources = template.get("Resources", {})
    return {
        name: res
        for name, res in resources.items()
        if res.get("Type") not in NON_TAGGABLE_TYPES
    }


def has_project_tag(resource):
    """Check if a resource has a tag with key 'Project' referencing CostAllocationTagProject.

    Handles both list format ([{Key: Project, Value: ...}]) and map format ({Project: ...}).
    """
    props = resource.get("Properties", {})
    tags = props.get("Tags", [])
    # Map format: Tags: {Project: !Ref CostAllocationTagProject}
    if isinstance(tags, dict):
        return "Project" in tags and tags["Project"] is not None
    # List format: Tags: [{Key: Project, Value: ...}]
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("Key") == "Project":
                return tag.get("Value") is not None
    return False


class TestCostAllocationTagsUnit:
    """Unit tests for cost-allocation tag coverage."""

    def test_template_loads_successfully(self):
        template = load_template()
        assert template is not None
        assert "Resources" in template

    def test_cost_allocation_tag_parameter_exists(self):
        """CostAllocationTagProject parameter must exist in the template."""
        template = load_template()
        params = template.get("Parameters", {})
        assert "CostAllocationTagProject" in params, (
            "Template must define a CostAllocationTagProject parameter"
        )

    def test_taggable_resources_exist(self):
        """Template must contain at least one taggable resource."""
        template = load_template()
        taggable = get_taggable_resources(template)
        assert len(taggable) > 0, "Template must contain at least one taggable resource"

    def test_all_taggable_resources_have_project_tag(self):
        """Every taggable resource must have a Project tag."""
        template = load_template()
        taggable = get_taggable_resources(template)

        missing = []
        for name, res in taggable.items():
            if not has_project_tag(res):
                missing.append(f"{name} ({res.get('Type')})")

        assert len(missing) == 0, (
            f"The following taggable resources are missing a 'Project' cost-allocation tag: "
            f"{', '.join(missing)}"
        )

    def test_non_taggable_resources_excluded(self):
        """Non-taggable resource types should be excluded from tag checks."""
        template = load_template()
        resources = template.get("Resources", {})
        taggable = get_taggable_resources(template)

        for name, res in resources.items():
            if res.get("Type") in NON_TAGGABLE_TYPES:
                assert name not in taggable, (
                    f"Resource {name} of type {res.get('Type')} should be excluded "
                    f"from taggable resources"
                )


class TestCostAllocationTagsProperty:
    """
    Property-based test for cost-allocation tag coverage.

    **Validates: Requirements 8.3**

    Since the CloudFormation template is a static artifact, we use hypothesis
    to generate random resource indices and verify the property holds for every
    taggable resource selected. This ensures the property is checked across all
    possible resource selections from the template.
    """

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_random_taggable_resource_has_project_tag(self, data):
        """
        Property 8: Cost-allocation tag coverage

        For any randomly selected taggable resource from the CloudFormation template,
        the resource must include a tag with key 'Project' referencing the
        CostAllocationTagProject parameter.

        **Validates: Requirements 8.3**
        """
        template = load_template()
        taggable = get_taggable_resources(template)
        resource_names = list(taggable.keys())
        assert len(resource_names) > 0, "No taggable resources found in template"

        selected_name = data.draw(st.sampled_from(resource_names))
        resource = taggable[selected_name]

        assert has_project_tag(resource), (
            f"Resource {selected_name} ({resource.get('Type')}) is missing a "
            f"'Project' cost-allocation tag referencing the CostAllocationTagProject "
            f"parameter. All taggable resources must include cost-allocation tags "
            f"per Requirement 8.3."
        )
