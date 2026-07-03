"""Pytest configuration and marker registration."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring a deployed stack (deselect with '-m \"not integration\"')",
    )
