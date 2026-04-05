"""
Tests for Phase 2 — Vendor Validation in the Orchestrator.

Run with:  pytest tests/test_orchestrator.py -v

These tests verify that the orchestrator correctly checks whether the
declared accelerator_vendor matches what is detectable on the current
machine, and produces the right validation status in each case.

All tests use unittest.mock.patch to control what shutil.which() returns.
This means the tests produce consistent results on any machine regardless
of whether nvidia-smi or rocm-smi is actually installed.
"""
import pytest
from unittest.mock import patch
from scalelab.core.models import Scenario, ClusterConfig, WorkloadConfig, LaunchConfig
from scalelab.core.orchestrator import _validate_vendor


def make_scenario(vendor: str, arch: str = "h100") -> Scenario:
    """
    Build a minimal Scenario for testing vendor validation.
    Only vendor and arch matter here — the rest are placeholders.
    """
    return Scenario(
        name="test",
        cluster=ClusterConfig(accelerator_vendor=vendor, accelerator_arch=arch),
        workload=WorkloadConfig(model="test-model", backend="vllm"),
        launch=LaunchConfig(),
    )


class TestVendorValidation:
    """
    Tests for _validate_vendor() in orchestrator.py.

    _validate_vendor() checks whether the declared accelerator_vendor
    has its corresponding CLI tool (nvidia-smi or rocm-smi) available
    in PATH. It returns a dict with vendor_validation set to one of:
      "passed"  — tool found, hardware matches declaration
      "warning" — tool not found, hardware may not match
      "skipped" — vendor is unknown, no tool to check against
    """

    def test_nvidia_tool_found_returns_passed(self):
        """
        When nvidia-smi is in PATH and vendor is 'nvidia', validation passes.

        This is the happy path for an NVIDIA machine with drivers installed.
        We mock shutil.which to return a path, simulating nvidia-smi being
        present without actually needing an NVIDIA GPU.
        """
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = _validate_vendor(make_scenario("nvidia"))
        assert result["vendor_validation"] == "passed"

    def test_nvidia_tool_missing_returns_warning(self):
        """
        When nvidia-smi is not in PATH but vendor is 'nvidia', result is warning.

        This happens when: running on a head node without GPU drivers,
        nvidia-smi not in PATH despite GPU being present, or the wrong
        vendor was declared in the scenario YAML.
        """
        with patch("shutil.which", return_value=None):
            result = _validate_vendor(make_scenario("nvidia"))
        assert result["vendor_validation"] == "warning"

    def test_amd_tool_found_returns_passed(self):
        """
        When rocm-smi is in PATH and vendor is 'amd', validation passes.
        Happy path for an AMD ROCm machine with drivers installed.
        """
        with patch("shutil.which", return_value="/usr/bin/rocm-smi"):
            result = _validate_vendor(make_scenario("amd"))
        assert result["vendor_validation"] == "passed"

    def test_amd_tool_missing_returns_warning(self):
        """
        When rocm-smi is not in PATH but vendor is 'amd', result is warning.
        Same scenarios as the NVIDIA case — head node, missing PATH, wrong vendor.
        """
        with patch("shutil.which", return_value=None):
            result = _validate_vendor(make_scenario("amd"))
        assert result["vendor_validation"] == "warning"

    def test_unknown_vendor_returns_skipped(self):
        """
        An unrecognised vendor string should produce 'skipped', not an error.

        We have no CLI tool to check against for vendors like 'groq',
        'intel', or 'custom'. Rather than crashing or producing a false
        warning, we skip validation and let the benchmark proceed.
        """
        result = _validate_vendor(make_scenario("groq"))
        assert result["vendor_validation"] == "skipped"

    def test_warning_note_mentions_tool_name(self):
        """
        The warning message must name the specific missing tool.

        A message saying "tool not found" is not actionable. A message
        saying "nvidia-smi not found" tells the user exactly what to
        install or add to their PATH to fix the issue.
        """
        with patch("shutil.which", return_value=None):
            result = _validate_vendor(make_scenario("nvidia"))
        assert "nvidia-smi" in result["vendor_validation_note"]

    def test_warning_cross_vendor_hint(self):
        """
        If nvidia is declared but rocm-smi is present, the warning should
        hint that the user may have the wrong vendor set.

        This is a common mistake when copying a scenario YAML from an
        NVIDIA setup to an AMD machine. Detecting the other vendor's tool
        gives a specific, actionable hint rather than a generic warning.
        """
        def mock_which(tool):
            # Simulate an AMD machine: rocm-smi present, nvidia-smi absent
            return "/usr/bin/rocm-smi" if tool == "rocm-smi" else None

        with patch("shutil.which", side_effect=mock_which):
            result = _validate_vendor(make_scenario("nvidia"))

        # Should still be a warning — the declared vendor doesn't have its tool
        assert result["vendor_validation"] == "warning"
        # Note should mention 'amd' as the likely correct vendor
        assert "amd" in result["vendor_validation_note"].lower()

    def test_passed_note_mentions_vendor(self):
        """
        The success message should confirm which vendor was validated.

        A passing note that says "validation passed" without naming the
        vendor is harder to read in a log file. Including the vendor name
        makes the result unambiguous at a glance.
        """
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = _validate_vendor(make_scenario("nvidia"))
        assert "nvidia" in result["vendor_validation_note"]

    def test_result_always_has_both_keys(self):
        """
        Both vendor_validation and vendor_validation_note must be present
        in the returned dict for every possible vendor string.

        The orchestrator merges this dict into launch_result and saves it
        to the project file. If either key is missing, code that reads
        the project file later will need to handle the absence defensively.
        Guaranteeing both keys are always present avoids that complexity.
        """
        for vendor in ["nvidia", "amd", "groq", "intel"]:
            result = _validate_vendor(make_scenario(vendor))
            assert "vendor_validation" in result, f"Missing key for vendor={vendor}"
            assert "vendor_validation_note" in result, f"Missing note for vendor={vendor}"

    def test_vendor_matching_is_case_insensitive(self):
        """
        NVIDIA, Nvidia, and nvidia in the scenario YAML must all produce
        the same validation result.

        Users write vendor strings inconsistently. Case-sensitive matching
        would silently skip validation for 'NVIDIA' while running it for
        'nvidia', producing confusing differences in saved project files.
        """
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            r1 = _validate_vendor(make_scenario("NVIDIA"))
            r2 = _validate_vendor(make_scenario("Nvidia"))
            r3 = _validate_vendor(make_scenario("nvidia"))
        assert r1["vendor_validation"] == r2["vendor_validation"] == r3["vendor_validation"]