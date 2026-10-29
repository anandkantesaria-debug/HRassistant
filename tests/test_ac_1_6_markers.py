import pytest


def test_ac_1_6_unit_tests_mock_network_and_integration_marker_registered(
    pytestconfig: pytest.Config,
) -> None:
    # Verify the integration marker is registered in pytest ini configuration
    registered_markers = [m.split(":")[0].strip() for m in pytestconfig.getini("markers")]
    assert "integration" in registered_markers

    # Verify addopts skips integration by default
    addopts = pytestconfig.getini("addopts")
    assert "not integration" in addopts

