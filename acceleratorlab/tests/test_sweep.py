"""
Tests for Phase 3 — Sweep Automation.

Run with:  pytest tests/test_sweep.py -v

These tests verify that:
  - SweepConfig builds correctly from dicts and has the right defaults
  - generate_sweep_scenarios() produces the correct number of scenarios
    with the correct parameter values in each one
  - The base scenario is never mutated by the generator
  - Scenario names are unique and descriptive
  - Edge cases (empty ranges, single values) are handled safely
  - SweepResult carries the right structure and counts
  - load_sweep_file() correctly parses a sweep YAML into a Scenario + SweepConfig
  - The CLI correctly switches between --scenario and --sweep modes

No real server or GPU is needed — sweep generation is pure Python logic.
The run_sweep() integration tests mock execute_scenario() so they verify
the sweep loop without making real HTTP requests.
"""
import copy
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from scalelab.core.models import Scenario, ClusterConfig, WorkloadConfig, LaunchConfig
from scalelab.core.sweep import SweepConfig, SweepResult, generate_sweep_scenarios, run_sweep
from scalelab.core.io import load_sweep_file


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def base_scenario(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    backend: str = "openai-compat",
    vendor: str = "nvidia",
) -> Scenario:
    """
    Build a minimal base Scenario for sweep tests.
    Only the fields that sweep generation touches are set explicitly —
    everything else uses defaults.
    """
    return Scenario(
        name="base",
        cluster=ClusterConfig(
            accelerator_vendor=vendor,
            accelerator_arch="h100",
        ),
        workload=WorkloadConfig(
            model=model,
            backend=backend,
            concurrency=1,
            prompt_tokens=512,
            output_tokens=128,
        ),
        launch=LaunchConfig(executor="local"),
    )


def minimal_config(**kwargs) -> SweepConfig:
    """
    Build a SweepConfig with the smallest possible grid (1 value per dimension)
    so tests that only care about one dimension don't produce unexpected combinations.
    """
    defaults = dict(
        concurrency_levels=[8],
        prompt_tokens_values=[512],
        output_tokens_values=[128],
        models=[],
        backends=[],
    )
    defaults.update(kwargs)
    return SweepConfig(**defaults)


# ===========================================================================
# SweepConfig
# ===========================================================================

class TestSweepConfig:
    """Tests for the SweepConfig dataclass and its from_dict constructor."""

    def test_default_concurrency_levels(self):
        """
        Default concurrency levels should cover a useful range from 1 to 64.
        A sweep with no configuration should still produce meaningful data.
        """
        config = SweepConfig()
        assert len(config.concurrency_levels) > 0
        assert 1 in config.concurrency_levels
        assert 64 in config.concurrency_levels

    def test_default_prompt_tokens(self):
        """
        Default prompt token values should include at least two sizes
        to exercise both short and long prompt scenarios.
        """
        config = SweepConfig()
        assert len(config.prompt_tokens_values) >= 2

    def test_default_output_tokens(self):
        """
        Default output token values should include at least two sizes
        to capture the difference between short and long generation.
        """
        config = SweepConfig()
        assert len(config.output_tokens_values) >= 2

    def test_default_models_is_empty(self):
        """
        Default models list should be empty — meaning the sweep will use
        the base scenario's model rather than overriding it.
        """
        config = SweepConfig()
        assert config.models == []

    def test_default_backends_is_empty(self):
        """
        Default backends list should be empty — meaning the sweep will use
        the base scenario's backend rather than overriding it.
        """
        config = SweepConfig()
        assert config.backends == []

    def test_from_dict_concurrency(self):
        """
        from_dict() should read concurrency values from the 'concurrency' key.
        """
        config = SweepConfig.from_dict({"concurrency": [1, 8, 32]})
        assert config.concurrency_levels == [1, 8, 32]

    def test_from_dict_prompt_tokens(self):
        """from_dict() should read prompt sizes from 'prompt_tokens'."""
        config = SweepConfig.from_dict({"prompt_tokens": [256, 2048]})
        assert config.prompt_tokens_values == [256, 2048]

    def test_from_dict_output_tokens(self):
        """from_dict() should read output sizes from 'output_tokens'."""
        config = SweepConfig.from_dict({"output_tokens": [64, 512]})
        assert config.output_tokens_values == [64, 512]

    def test_from_dict_models(self):
        """from_dict() should read model overrides from 'models'."""
        models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
        config = SweepConfig.from_dict({"models": models})
        assert config.models == models

    def test_from_dict_backends(self):
        """from_dict() should read backend overrides from 'backends'."""
        config = SweepConfig.from_dict({"backends": ["vllm", "sglang"]})
        assert config.backends == ["vllm", "sglang"]

    def test_from_dict_name(self):
        """from_dict() should carry the name through."""
        config = SweepConfig.from_dict({"name": "my-sweep"})
        assert config.name == "my-sweep"

    def test_from_dict_missing_keys_use_defaults(self):
        """
        from_dict() with an empty dict should produce the same defaults
        as the no-arg constructor. No KeyError should be raised.
        """
        config = SweepConfig.from_dict({})
        assert len(config.concurrency_levels) > 0
        assert len(config.prompt_tokens_values) > 0

    def test_to_dict_roundtrip(self):
        """
        to_dict() then from_dict() should produce an equivalent SweepConfig.
        This verifies the serialisation is lossless.
        """
        original = SweepConfig(
            name="test",
            concurrency_levels=[1, 8],
            prompt_tokens_values=[512],
            output_tokens_values=[128],
            models=["model-a"],
            backends=["vllm"],
        )
        restored = SweepConfig.from_dict(original.to_dict())
        assert restored.concurrency_levels   == original.concurrency_levels
        assert restored.prompt_tokens_values == original.prompt_tokens_values
        assert restored.output_tokens_values == original.output_tokens_values
        assert restored.models               == original.models
        assert restored.backends             == original.backends

    def test_total_combinations_simple(self):
        """
        total_combinations should return the cartesian product size.
        2 concurrency × 3 prompt × 2 output = 12.
        """
        config = SweepConfig(
            concurrency_levels=[1, 8],
            prompt_tokens_values=[128, 512, 1024],
            output_tokens_values=[64, 256],
        )
        assert config.total_combinations == 12

    def test_total_combinations_with_models(self):
        """
        When models list is provided it multiplies the total.
        2 models × 2 concurrency × 1 prompt × 1 output = 4.
        """
        config = SweepConfig(
            concurrency_levels=[1, 8],
            prompt_tokens_values=[512],
            output_tokens_values=[128],
            models=["model-a", "model-b"],
        )
        assert config.total_combinations == 4

    def test_total_combinations_empty_models_counts_as_one(self):
        """
        Empty models list should count as 1 (use base scenario's model),
        not 0, so total_combinations doesn't go to zero.
        """
        config = SweepConfig(
            concurrency_levels=[1, 8],
            prompt_tokens_values=[512],
            output_tokens_values=[128],
            models=[],
        )
        assert config.total_combinations == 2


# ===========================================================================
# generate_sweep_scenarios
# ===========================================================================

class TestGenerateSweepScenarios:
    """
    Tests for the scenario generator that produces the cartesian product
    of all parameter combinations.
    """

    def test_single_value_each_yields_one_scenario(self):
        """
        When every dimension has exactly one value, the generator should
        produce exactly one scenario.
        """
        config = minimal_config()
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert len(scenarios) == 1

    def test_correct_count_two_concurrency(self):
        """
        Two concurrency levels × 1 prompt × 1 output = 2 scenarios.
        """
        config = minimal_config(concurrency_levels=[4, 16])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert len(scenarios) == 2

    def test_correct_count_cartesian_product(self):
        """
        3 concurrency × 2 prompt × 2 output = 12 scenarios.
        Verifies the full cartesian product is generated.
        """
        config = SweepConfig(
            concurrency_levels=[1, 8, 32],
            prompt_tokens_values=[512, 1024],
            output_tokens_values=[128, 256],
        )
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert len(scenarios) == 12

    def test_correct_count_with_model_sweep(self):
        """
        2 models × 2 concurrency × 1 prompt × 1 output = 4 scenarios.
        """
        config = minimal_config(
            concurrency_levels=[1, 8],
            models=["model-a", "model-b"],
        )
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert len(scenarios) == 4

    def test_correct_count_with_backend_sweep(self):
        """
        2 backends × 1 concurrency × 1 prompt × 1 output = 2 scenarios.
        """
        config = minimal_config(backends=["vllm", "sglang"])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert len(scenarios) == 2

    def test_concurrency_values_are_set(self):
        """
        Each generated scenario should have the concurrency value from
        its position in the concurrency_levels list.
        """
        config = minimal_config(concurrency_levels=[4, 16, 64])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        concurrencies = [s.workload.concurrency for s in scenarios]
        assert sorted(concurrencies) == [4, 16, 64]

    def test_prompt_tokens_values_are_set(self):
        """
        Each generated scenario should have the correct prompt_tokens value.
        """
        config = minimal_config(prompt_tokens_values=[256, 1024, 4096])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        prompt_sizes = sorted([s.workload.prompt_tokens for s in scenarios])
        assert prompt_sizes == [256, 1024, 4096]

    def test_output_tokens_values_are_set(self):
        """
        Each generated scenario should have the correct output_tokens value.
        """
        config = minimal_config(output_tokens_values=[64, 512])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        output_sizes = sorted([s.workload.output_tokens for s in scenarios])
        assert output_sizes == [64, 512]

    def test_model_override_applied(self):
        """
        When models is non-empty, the generated scenarios should use those
        model IDs rather than the base scenario's model.
        """
        override_model = "Qwen/Qwen2.5-7B-Instruct"
        config = minimal_config(models=[override_model])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert all(s.workload.model == override_model for s in scenarios)

    def test_empty_models_uses_base_model(self):
        """
        When models is empty the generator should use the base scenario's
        model for every generated scenario.
        """
        base = base_scenario(model="base-model")
        config = minimal_config(models=[])
        scenarios = list(generate_sweep_scenarios(base, config))
        assert all(s.workload.model == "base-model" for s in scenarios)

    def test_backend_override_applied(self):
        """
        When backends is non-empty, the generated scenarios should use those
        backend names rather than the base scenario's backend.
        """
        config = minimal_config(backends=["sglang"])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert all(s.workload.backend == "sglang" for s in scenarios)

    def test_empty_backends_uses_base_backend(self):
        """
        When backends is empty the generator should use the base scenario's
        backend for every generated scenario.
        """
        base = base_scenario(backend="tgi")
        config = minimal_config(backends=[])
        scenarios = list(generate_sweep_scenarios(base, config))
        assert all(s.workload.backend == "tgi" for s in scenarios)

    def test_cluster_config_preserved(self):
        """
        The cluster config (vendor, arch, nodes, etc.) from the base scenario
        must be copied unchanged into every generated scenario.
        Sweep dimensions only affect workload parameters.
        """
        base = base_scenario(vendor="amd")
        base.cluster.accelerator_arch = "mi300x"
        config = minimal_config(concurrency_levels=[1, 8])
        scenarios = list(generate_sweep_scenarios(base, config))
        for s in scenarios:
            assert s.cluster.accelerator_vendor == "amd"
            assert s.cluster.accelerator_arch   == "mi300x"

    def test_launch_config_preserved(self):
        """
        The launch config (executor, tensor_parallel, etc.) from the base
        scenario must be preserved in every generated scenario.
        """
        base = base_scenario()
        base.launch.tensor_parallel = 4
        base.launch.executor = "slurm"
        config = minimal_config(concurrency_levels=[1, 8])
        scenarios = list(generate_sweep_scenarios(base, config))
        for s in scenarios:
            assert s.launch.tensor_parallel == 4
            assert s.launch.executor        == "slurm"

    def test_base_scenario_not_mutated(self):
        """
        The base scenario must not be modified during generation.
        This is critical — if the base is mutated, repeated calls to
        generate_sweep_scenarios() would produce different results.
        """
        base = base_scenario()
        original_concurrency = base.workload.concurrency
        original_model       = base.workload.model

        config = minimal_config(concurrency_levels=[99], models=["override-model"])
        list(generate_sweep_scenarios(base, config))  # consume the generator

        assert base.workload.concurrency == original_concurrency
        assert base.workload.model       == original_model

    def test_scenario_names_are_unique(self):
        """
        Every generated scenario must have a unique name.
        Duplicate names would make it impossible to distinguish runs in saved results.
        """
        config = SweepConfig(
            concurrency_levels=[1, 4, 8],
            prompt_tokens_values=[512, 1024],
            output_tokens_values=[128, 256],
        )
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        names = [s.name for s in scenarios]
        assert len(names) == len(set(names)), "Duplicate scenario names found"

    def test_scenario_names_contain_concurrency(self):
        """
        Scenario names should include the concurrency value so a user can
        identify which parameter combination produced which result.
        """
        config = minimal_config(concurrency_levels=[42])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert "42" in scenarios[0].name

    def test_scenario_names_contain_prompt_tokens(self):
        """
        Scenario names should include the prompt token count.
        """
        config = minimal_config(prompt_tokens_values=[999])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert "999" in scenarios[0].name

    def test_scenario_names_contain_output_tokens(self):
        """
        Scenario names should include the output token count.
        """
        config = minimal_config(output_tokens_values=[777])
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert "777" in scenarios[0].name

    def test_empty_concurrency_yields_nothing(self):
        """
        An empty concurrency_levels list should produce zero scenarios.
        This is a valid edge case — a sweep with no values in one dimension
        has no valid combinations.
        """
        config = SweepConfig(
            concurrency_levels=[],
            prompt_tokens_values=[512],
            output_tokens_values=[128],
        )
        scenarios = list(generate_sweep_scenarios(base_scenario(), config))
        assert len(scenarios) == 0


# ===========================================================================
# SweepResult
# ===========================================================================

class TestSweepResult:
    """Tests for the SweepResult dataclass."""

    def test_to_dict_has_required_keys(self):
        """
        to_dict() must include all keys that downstream code and saved
        project files depend on.
        """
        result = SweepResult(
            name="test",
            config=SweepConfig(),
            results=[],
            total_runs=5,
            completed_runs=4,
            failed_runs=1,
            elapsed_s=12.3,
        )
        d = result.to_dict()
        for key in ["name", "config", "total_runs", "completed_runs",
                    "failed_runs", "elapsed_s", "results"]:
            assert key in d, f"Missing key: {key}"

    def test_summary_contains_name(self):
        """
        The summary string should include the sweep name so it can be
        identified in a log file without additional context.
        """
        result = SweepResult(name="my-sweep", config=SweepConfig())
        assert "my-sweep" in result.summary()

    def test_summary_contains_counts(self):
        """
        The summary should include completed and total run counts so
        the user can see at a glance how many runs succeeded.
        """
        result = SweepResult(
            name="test", config=SweepConfig(),
            total_runs=10, completed_runs=8, failed_runs=2
        )
        summary = result.summary()
        assert "8" in summary
        assert "10" in summary

    def test_elapsed_s_rounded_in_dict(self):
        """
        elapsed_s in to_dict() should be rounded — unrounded float values
        like 12.3456789... make result files harder to read and diff.
        """
        result = SweepResult(
            name="test", config=SweepConfig(), elapsed_s=12.3456789
        )
        d = result.to_dict()
        assert d["elapsed_s"] == round(12.3456789, 2)


# ===========================================================================
# run_sweep — integration (mocked execute_scenario)
# ===========================================================================

class TestRunSweep:
    """
    Integration tests for run_sweep().

    execute_scenario() is mocked so these tests verify the sweep loop
    logic (counting, error handling, callbacks) without making real
    HTTP requests or needing a GPU.
    """

    def _fake_result(self, scenario):
        """Return a minimal fake benchmark result dict."""
        return {
            "scenario":         scenario.to_dict(),
            "launch_result":    {},
            "benchmark_result": {"tok_s": 100.0, "ttft_ms": 200.0},
        }

    def test_completed_runs_counted(self):
        """
        completed_runs in the SweepResult should equal the number of
        scenarios that execute_scenario() returned successfully.
        """
        config = minimal_config(concurrency_levels=[1, 4, 8])
        base   = base_scenario()

        with patch("scalelab.core.sweep.execute_scenario",
                   side_effect=lambda s, **kw: self._fake_result(s)):
            result = run_sweep(base, config)

        assert result.completed_runs == 3
        assert result.failed_runs    == 0
        assert result.total_runs     == 3

    def test_failed_runs_counted(self):
        """
        When execute_scenario() raises an exception, failed_runs should
        increment and the sweep should continue rather than aborting.
        This ensures one bad run doesn't discard all other results.
        """
        config = minimal_config(concurrency_levels=[1, 4, 8])
        base   = base_scenario()

        call_count = [0]
        def flaky(scenario, **kw):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated server timeout")
            return self._fake_result(scenario)

        with patch("scalelab.core.sweep.execute_scenario", side_effect=flaky):
            result = run_sweep(base, config)

        assert result.completed_runs == 2
        assert result.failed_runs    == 1
        assert result.total_runs     == 3

    def test_results_list_length_matches_total(self):
        """
        The results list should always have one entry per run — both
        successful runs and failed runs should be recorded.
        """
        config = minimal_config(concurrency_levels=[1, 4])
        base   = base_scenario()

        with patch("scalelab.core.sweep.execute_scenario",
                   side_effect=lambda s, **kw: self._fake_result(s)):
            result = run_sweep(base, config)

        assert len(result.results) == 2

    def test_on_result_callback_called_for_each_run(self):
        """
        The on_result callback should be called once per run, in order.
        This is used by the CLI to print progress and by the GUI to
        update a progress bar.
        """
        config    = minimal_config(concurrency_levels=[1, 4, 8])
        base      = base_scenario()
        calls     = []

        def capture(run_index, total, result):
            calls.append((run_index, total))

        with patch("scalelab.core.sweep.execute_scenario",
                   side_effect=lambda s, **kw: self._fake_result(s)):
            run_sweep(base, config, on_result=capture)

        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_elapsed_s_is_positive(self):
        """
        The elapsed time should always be a positive number after the sweep
        completes, even if all runs are very fast.
        """
        config = minimal_config()
        base   = base_scenario()

        with patch("scalelab.core.sweep.execute_scenario",
                   side_effect=lambda s, **kw: self._fake_result(s)):
            result = run_sweep(base, config)

        assert result.elapsed_s > 0


# ===========================================================================
# load_sweep_file
# ===========================================================================

class TestLoadSweepFile:
    """
    Tests for io.load_sweep_file() — parses a sweep YAML into a
    (Scenario, SweepConfig) tuple.
    """

    def test_loads_base_scenario(self, tmp_path):
        """
        load_sweep_file() should return a valid Scenario whose fields
        match the base_scenario section of the YAML.
        """
        sweep_yaml = tmp_path / "sweep.yaml"
        sweep_yaml.write_text("""
sweep:
  name: test-sweep
  base_scenario:
    cluster:
      accelerator_vendor: nvidia
      accelerator_arch: h100
    workload:
      model: test-model
      backend: openai-compat
    launch:
      executor: local
  ranges:
    concurrency: [1, 8]
    prompt_tokens: [512]
    output_tokens: [128]
""")
        scenario, config = load_sweep_file(sweep_yaml)
        assert scenario.cluster.accelerator_vendor == "nvidia"
        assert scenario.workload.model == "test-model"

    def test_loads_sweep_config_ranges(self, tmp_path):
        """
        load_sweep_file() should return a SweepConfig whose ranges match
        the 'ranges' section of the YAML.
        """
        sweep_yaml = tmp_path / "sweep.yaml"
        sweep_yaml.write_text("""
sweep:
  name: range-test
  base_scenario:
    cluster:
      accelerator_vendor: nvidia
      accelerator_arch: h100
    workload:
      model: test-model
      backend: openai-compat
    launch:
      executor: local
  ranges:
    concurrency: [1, 4, 16]
    prompt_tokens: [256, 2048]
    output_tokens: [64]
""")
        _, config = load_sweep_file(sweep_yaml)
        assert config.concurrency_levels    == [1, 4, 16]
        assert config.prompt_tokens_values  == [256, 2048]
        assert config.output_tokens_values  == [64]

    def test_sweep_name_carried_to_config(self, tmp_path):
        """
        The sweep name from the YAML should appear in the SweepConfig,
        so the SweepResult is named correctly.
        """
        sweep_yaml = tmp_path / "sweep.yaml"
        sweep_yaml.write_text("""
sweep:
  name: my-named-sweep
  base_scenario:
    cluster:
      accelerator_vendor: nvidia
      accelerator_arch: h100
    workload:
      model: test-model
      backend: openai-compat
    launch:
      executor: local
  ranges:
    concurrency: [1]
    prompt_tokens: [512]
    output_tokens: [128]
""")
        _, config = load_sweep_file(sweep_yaml)
        assert config.name == "my-named-sweep"

    def test_unsupported_extension_raises(self, tmp_path):
        """
        Attempting to load a sweep from an unsupported file format should
        raise a ValueError with a clear message.
        """
        bad_file = tmp_path / "sweep.csv"
        bad_file.write_text("not a sweep")
        with pytest.raises(ValueError, match="Unsupported"):
            load_sweep_file(bad_file)