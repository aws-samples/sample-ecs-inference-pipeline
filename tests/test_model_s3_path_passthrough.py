"""
Property 2: Model S3 path passthrough

For any valid S3 URI set in the MODEL_S3_PATH environment variable, the entrypoint
script SHALL invoke the S3 download command targeting that exact URI, and the resulting
local model path SHALL be passed to the vLLM server startup command.

Validates: Requirements 2.3
"""

import os
import re
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

ENTRYPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "container", "entrypoint.sh"
)


def load_entrypoint():
    """Load and return the entrypoint.sh script content."""
    with open(ENTRYPOINT_PATH, "r") as f:
        return f.read()


# --- Hypothesis strategies for valid S3 URIs ---

# S3 bucket names: 3-63 chars, lowercase letters, numbers, hyphens
_bucket_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyz0123456789-"
)

_bucket_name = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1, max_size=1,
).flatmap(
    lambda first: st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
        min_size=1, max_size=30,
    ).map(lambda rest: first + rest)
).filter(lambda s: not s.endswith("-") and "--" not in s)

# S3 key path segments: alphanumeric with hyphens and underscores
_path_segment = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1, max_size=20,
)

_s3_key = st.lists(_path_segment, min_size=1, max_size=5).map(
    lambda parts: "/".join(parts)
)

# Full S3 URI: s3://bucket/key
s3_uri_strategy = st.builds(
    lambda bucket, key: f"s3://{bucket}/{key}",
    _bucket_name,
    _s3_key,
)

# Model name strategy
_model_name = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1, max_size=30,
).filter(lambda s: len(s.strip()) > 0)


def extract_s3_sync_commands(script_content):
    """Extract all aws s3 sync command invocations from the script."""
    return re.findall(r'aws\s+s3\s+sync\s+[^\n]+', script_content)


def extract_vllm_model_flag(script_content):
    """Extract the --model flag value pattern from the script."""
    return re.findall(r'"--model"\s+"([^"]+)"', script_content)


class TestModelS3PathPassthroughUnit:
    """Unit tests for model S3 path passthrough in entrypoint.sh."""

    def test_entrypoint_exists(self):
        assert os.path.isfile(ENTRYPOINT_PATH), (
            f"entrypoint.sh not found at {ENTRYPOINT_PATH}"
        )

    def test_entrypoint_uses_model_s3_path_in_s3_sync(self):
        """The entrypoint must use MODEL_S3_PATH in the aws s3 sync command."""
        script = load_entrypoint()
        s3_commands = extract_s3_sync_commands(script)
        assert len(s3_commands) > 0, "No 'aws s3 sync' command found in entrypoint.sh"

        uses_model_s3_path = any(
            "${MODEL_S3_PATH}" in cmd or "$MODEL_S3_PATH" in cmd
            for cmd in s3_commands
        )
        assert uses_model_s3_path, (
            "aws s3 sync command does not reference MODEL_S3_PATH variable. "
            f"Found commands: {s3_commands}"
        )

    def test_entrypoint_uses_model_name_in_s3_sync_destination(self):
        """The entrypoint must use MODEL_NAME in the s3 sync destination path."""
        script = load_entrypoint()
        s3_commands = extract_s3_sync_commands(script)
        assert len(s3_commands) > 0, "No 'aws s3 sync' command found in entrypoint.sh"

        uses_model_name = any(
            "${MODEL_NAME}" in cmd or "$MODEL_NAME" in cmd
            for cmd in s3_commands
        )
        assert uses_model_name, (
            "aws s3 sync command does not reference MODEL_NAME in destination path. "
            f"Found commands: {s3_commands}"
        )

    def test_entrypoint_uses_model_name_in_vllm_model_flag(self):
        """The vLLM startup must use MODEL_NAME in the --model flag path."""
        script = load_entrypoint()
        # Check that --model flag references a path containing MODEL_NAME
        has_model_flag = (
            "${MODEL_NAME}" in script or "$MODEL_NAME" in script
        )
        assert has_model_flag, "entrypoint.sh does not reference MODEL_NAME"

        # Verify --model flag is present
        assert '"--model"' in script, (
            "entrypoint.sh does not contain --model flag for vLLM"
        )

        # Verify the --model value includes MODEL_NAME or MODEL_PATH (derived from MODEL_NAME)
        model_flag_lines = [
            line for line in script.splitlines()
            if "--model" in line
        ]
        model_name_in_flag = any(
            "MODEL_NAME" in line or "MODEL_PATH" in line for line in model_flag_lines
        )
        assert model_name_in_flag, (
            "vLLM --model flag does not reference MODEL_NAME or MODEL_PATH. "
            f"Found lines: {model_flag_lines}"
        )

    def test_entrypoint_exits_nonzero_on_s3_failure(self):
        """The script must exit non-zero if the S3 download fails."""
        script = load_entrypoint()
        # Check for exit 1 after s3 sync failure
        assert "exit 1" in script, (
            "entrypoint.sh does not contain 'exit 1' for S3 download failure"
        )
        # Verify there's error handling around the s3 sync command
        has_error_check = (
            "if !" in script and "aws s3 sync" in script
        ) or (
            "set -e" in script and "aws s3 sync" in script
        )
        assert has_error_check, (
            "entrypoint.sh does not have error handling for aws s3 sync failure"
        )

    def test_entrypoint_logs_error_on_s3_failure(self):
        """The script must log the failure reason to stdout on S3 download error."""
        script = load_entrypoint()
        # Look for an error message near the s3 sync failure path
        has_error_log = (
            "ERROR" in script and "S3" in script
        ) or (
            "Failed" in script and "S3" in script
        ) or (
            "failed" in script and "s3" in script.lower()
        )
        assert has_error_log, (
            "entrypoint.sh does not log an error message on S3 download failure"
        )


class TestModelS3PathPassthroughProperty:
    """
    Property-based test for model S3 path passthrough.

    **Validates: Requirements 2.3**

    We generate random valid S3 URIs and model names, then verify that the
    entrypoint script structure correctly passes MODEL_S3_PATH to the aws s3 sync
    command and MODEL_NAME to the vLLM --model flag path.
    """

    @given(s3_uri=s3_uri_strategy, model_name=_model_name)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_s3_path_passthrough_to_download_command(self, s3_uri, model_name):
        """
        Property 2: Model S3 path passthrough (download command)

        For any valid S3 URI, the entrypoint script must use MODEL_S3_PATH
        as the source argument to aws s3 sync.

        **Validates: Requirements 2.3**
        """
        script = load_entrypoint()

        # Verify the script uses MODEL_S3_PATH in the s3 sync command
        s3_commands = extract_s3_sync_commands(script)
        assert len(s3_commands) > 0, (
            f"No aws s3 sync command found for S3 URI: {s3_uri}"
        )

        # The s3 sync source must be the MODEL_S3_PATH variable
        source_uses_var = any(
            "${MODEL_S3_PATH}" in cmd or "$MODEL_S3_PATH" in cmd
            for cmd in s3_commands
        )
        assert source_uses_var, (
            f"For S3 URI '{s3_uri}', the aws s3 sync command does not use "
            f"MODEL_S3_PATH as source. Commands: {s3_commands}"
        )

        # The s3 sync destination must include MODEL_NAME
        dest_uses_model_name = any(
            "${MODEL_NAME}" in cmd or "$MODEL_NAME" in cmd
            for cmd in s3_commands
        )
        assert dest_uses_model_name, (
            f"For model '{model_name}', the aws s3 sync destination does not "
            f"include MODEL_NAME. Commands: {s3_commands}"
        )

    @given(model_name=_model_name)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_model_name_passthrough_to_vllm_startup(self, model_name):
        """
        Property 2: Model S3 path passthrough (vLLM startup)

        For any model name, the entrypoint script must pass MODEL_NAME
        in the vLLM --model flag path.

        **Validates: Requirements 2.3**
        """
        script = load_entrypoint()

        # The --model flag must be present
        assert '"--model"' in script, (
            f"For model '{model_name}', no --model flag found in vLLM startup"
        )

        # The --model value must reference a path containing MODEL_NAME
        # In the script, VLLM_ARGS has: "--model" "/models/${MODEL_NAME}"
        vllm_args_section = script[script.find("VLLM_ARGS"):]
        model_flag_uses_name = (
            "${MODEL_NAME}" in vllm_args_section
            or "$MODEL_NAME" in vllm_args_section
        )
        assert model_flag_uses_name, (
            f"For model '{model_name}', the vLLM --model flag does not reference "
            f"MODEL_NAME variable in its path"
        )

    @given(s3_uri=s3_uri_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_s3_download_failure_exits_nonzero(self, s3_uri):
        """
        Property 2: Model S3 path passthrough (failure handling)

        For any S3 URI, if the download fails, the script must exit non-zero.

        **Validates: Requirements 2.3**
        """
        script = load_entrypoint()

        # The script must have error handling for s3 sync
        # Check for explicit error check pattern: if ! aws s3 sync ... ; then exit 1
        has_conditional_exit = (
            "if !" in script
            and "aws s3 sync" in script
            and "exit 1" in script
        )
        # Or set -e which causes any command failure to exit
        has_set_e = "set -e" in script

        assert has_conditional_exit or has_set_e, (
            f"For S3 URI '{s3_uri}', the entrypoint does not exit non-zero "
            f"on S3 download failure. Must have 'if ! aws s3 sync ... exit 1' "
            f"or 'set -e'"
        )

        # Verify explicit exit 1 exists for the s3 failure path
        assert "exit 1" in script, (
            f"For S3 URI '{s3_uri}', entrypoint.sh does not contain 'exit 1' "
            f"for S3 download failure handling"
        )
