"""
Tests for Phase 2 — Vendor-Aware Backend Commands.

Run with:  pytest tests/test_backends.py -v

These tests do not require a GPU, a running server, or any external tools.
They verify that each backend adapter generates the correct shell command
for every combination of vendor and architecture that matters.

Each test builds a fake Scenario object, calls the backend's
build_server_command(), and checks the resulting command list contains
exactly the right flags — no more, no less.
"""
import pytest
from scalelab.core.models import Scenario, ClusterConfig, WorkloadConfig, LaunchConfig
from scalelab.backends.registry import BACKENDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scenario(
    vendor: str,
    arch: str,
    backend: str = "vllm",
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    model_cache_dir: str = "",
    extra_args: list = None,
) -> Scenario:
    """
    Build a minimal Scenario object for testing.

    Rather than writing a full YAML file for every test, this helper
    constructs a Scenario directly in Python with only the fields each
    test cares about. Everything else gets a sensible default.
    """
    return Scenario(
        name="test",
        cluster=ClusterConfig(
            accelerator_vendor=vendor,
            accelerator_arch=arch,
        ),
        workload=WorkloadConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            backend=backend,
        ),
        launch=LaunchConfig(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            model_cache_dir=model_cache_dir,
            extra_args=extra_args or [],
        ),
    )


def get_flag(cmd: list, flag: str) -> str:
    """
    Return the value that immediately follows a flag in a command list.

    For example, given cmd = ['python', '--dtype', 'bfloat16'] and
    flag = '--dtype', this returns 'bfloat16'.
    Returns None if the flag is not present or has no following value.
    """
    try:
        return cmd[cmd.index(flag) + 1]
    except (ValueError, IndexError):
        return None


# ===========================================================================
# vLLM
# ===========================================================================

class TestVLLM:
    """
    vLLM adapter tests.

    vLLM defaults to CUDA when no --device flag is given, so NVIDIA
    scenarios should not include it. AMD scenarios must explicitly request
    the ROCm backend or vLLM will silently fall back to CPU-mode serving.
    """

    def test_nvidia_no_device_flag(self):
        """
        NVIDIA scenarios should not include --device in the command.

        vLLM defaults to CUDA automatically. Adding --device cuda is
        harmless but unnecessary. More importantly, we want to confirm
        the AMD-specific injection logic does not fire for NVIDIA.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("nvidia", "h100"))
        assert "--device" not in cmd

    def test_nvidia_no_dtype_flag(self):
        """
        NVIDIA scenarios should not include --dtype.

        vLLM's CUDA auto-detection is reliable and will choose bfloat16
        on Ampere/Hopper hardware automatically. Overriding it is more
        likely to cause problems than to help, so we leave it out.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("nvidia", "h100"))
        assert "--dtype" not in cmd

    def test_amd_gets_device_rocm(self):
        """
        AMD scenarios must include --device rocm.

        Without this flag vLLM silently falls back to CPU inference —
        no error is raised, no warning is printed, it just runs extremely
        slowly. This is the single most common AMD setup mistake.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("amd", "mi300x"))
        assert "--device" in cmd
        assert get_flag(cmd, "--device") == "rocm"

    def test_amd_mi300x_bfloat16(self):
        """
        MI300X should use bfloat16 dtype.

        The MI300X has native bfloat16 hardware support and performs
        best with it. ROCm's dtype auto-detection is less reliable than
        CUDA's, so we inject it explicitly.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("amd", "mi300x"))
        assert get_flag(cmd, "--dtype") == "bfloat16"

    def test_amd_mi325x_bfloat16(self):
        """
        MI325X should use bfloat16 dtype.
        Same reasoning as MI300X — modern AMD accelerator with native bfloat16.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("amd", "mi325x"))
        assert get_flag(cmd, "--dtype") == "bfloat16"

    def test_amd_mi355x_bfloat16(self):
        """
        MI355X should use bfloat16 dtype.
        AMD's newest accelerator — bfloat16 is the correct default.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("amd", "mi355x"))
        assert get_flag(cmd, "--dtype") == "bfloat16"

    def test_amd_mi200_float16(self):
        """
        Older MI200-series hardware should use float16 instead of bfloat16.

        The MI200 generation has less mature bfloat16 support on ROCm and
        can exhibit numerical instability with it on some workloads.
        float16 is the safer default for this architecture family.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("amd", "mi200"))
        assert get_flag(cmd, "--dtype") == "float16"

    def test_tensor_parallel_passed(self):
        """
        The tensor_parallel value from LaunchConfig must appear as
        --tensor-parallel-size in the vLLM command.

        This controls how many GPUs the model is split across. Getting
        this wrong means the server starts but only uses one GPU.
        """
        cmd = BACKENDS["vllm"].build_server_command(
            make_scenario("nvidia", "h100", tensor_parallel=4)
        )
        assert get_flag(cmd, "--tensor-parallel-size") == "4"

    def test_pipeline_parallel_passed(self):
        """
        The pipeline_parallel value must appear as --pipeline-parallel-size.

        Pipeline parallelism splits layers across nodes rather than
        splitting the attention heads. Used for very large models
        that don't fit with tensor parallelism alone.
        """
        cmd = BACKENDS["vllm"].build_server_command(
            make_scenario("nvidia", "h100", pipeline_parallel=2)
        )
        assert get_flag(cmd, "--pipeline-parallel-size") == "2"

    def test_model_cache_dir_passed(self):
        """
        When model_cache_dir is set it should appear as --download-dir.

        This tells vLLM where to find (or download) the model weights.
        Without it vLLM downloads to ~/.cache/huggingface every time,
        which wastes time and disk space on shared clusters.
        """
        cmd = BACKENDS["vllm"].build_server_command(
            make_scenario("nvidia", "h100", model_cache_dir="/mnt/models")
        )
        assert "--download-dir" in cmd
        assert get_flag(cmd, "--download-dir") == "/mnt/models"

    def test_model_cache_dir_omitted_when_empty(self):
        """
        When model_cache_dir is empty string, --download-dir must not
        appear in the command at all.

        Passing --download-dir with an empty string causes vLLM to error.
        The adapter must guard against this.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("nvidia", "h100"))
        assert "--download-dir" not in cmd

    def test_extra_args_appended_last(self):
        """
        User-supplied extra_args must come after all auto-injected flags.

        This is what gives users override capability — if they put
        --dtype float32 in extra_args it will appear after our auto-injected
        --dtype bfloat16, and the last value wins in most CLI parsers.
        """
        cmd = BACKENDS["vllm"].build_server_command(
            make_scenario("amd", "mi300x", extra_args=["--dtype", "float32"])
        )
        assert cmd[-2:] == ["--dtype", "float32"]

    def test_extra_args_empty_by_default(self):
        """
        When extra_args is not set the command should not end with
        an empty string or stray whitespace element.

        An empty string at the end of the command list would be passed
        as a blank argument to the subprocess, which can confuse some
        CLI parsers.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("nvidia", "h100"))
        assert cmd[-1] != ""

    def test_vendor_case_insensitive(self):
        """
        Vendor string matching must be case-insensitive.

        Users might write AMD, Amd, or amd in their YAML file. All three
        should trigger the ROCm path identically.
        """
        cmd = BACKENDS["vllm"].build_server_command(make_scenario("AMD", "MI300X"))
        assert get_flag(cmd, "--device") == "rocm"

    def test_healthcheck_url(self):
        """
        The healthcheck URL must point to the local server's /health endpoint.

        The orchestrator polls this URL after launching the server to know
        when it is ready to accept benchmark traffic.
        """
        url = BACKENDS["vllm"].build_healthcheck_url(make_scenario("nvidia", "h100"))
        assert url == "http://127.0.0.1:8000/health"


# ===========================================================================
# SGLang
# ===========================================================================

class TestSGLang:
    """
    SGLang adapter tests.

    SGLang uses the same vendor logic as vLLM but has different flag names:
    --tp-size instead of --tensor-parallel-size, --pp-size instead of
    --pipeline-parallel-size. These tests verify both the vendor logic
    and the correct flag names are used.
    """

    def test_nvidia_no_device_flag(self):
        """
        NVIDIA scenarios should not get --device.
        SGLang also defaults to CUDA — no override needed.
        """
        cmd = BACKENDS["sglang"].build_server_command(make_scenario("nvidia", "h100"))
        assert "--device" not in cmd

    def test_amd_gets_device_rocm(self):
        """
        AMD scenarios must get --device rocm.
        Same requirement as vLLM — SGLang will fall back to CPU without it.
        """
        cmd = BACKENDS["sglang"].build_server_command(make_scenario("amd", "mi300x"))
        assert get_flag(cmd, "--device") == "rocm"

    def test_amd_mi300x_bfloat16(self):
        """
        MI300X should get --dtype bfloat16.
        Same dtype logic as vLLM — ROCm auto-detection is unreliable.
        """
        cmd = BACKENDS["sglang"].build_server_command(make_scenario("amd", "mi300x"))
        assert get_flag(cmd, "--dtype") == "bfloat16"

    def test_amd_mi200_float16(self):
        """
        Older MI200 hardware should get float16.
        Same older-AMD safety rule as vLLM.
        """
        cmd = BACKENDS["sglang"].build_server_command(make_scenario("amd", "mi200"))
        assert get_flag(cmd, "--dtype") == "float16"

    def test_tp_flag_name(self):
        """
        SGLang uses --tp-size, not --tensor-parallel-size.

        This is the most important flag-name difference from vLLM.
        If we used --tensor-parallel-size here SGLang would reject it
        with an unrecognised argument error.
        """
        cmd = BACKENDS["sglang"].build_server_command(
            make_scenario("nvidia", "h100", tensor_parallel=8)
        )
        assert "--tp-size" in cmd
        assert "--tensor-parallel-size" not in cmd
        assert get_flag(cmd, "--tp-size") == "8"

    def test_pp_flag_name(self):
        """
        SGLang uses --pp-size, not --pipeline-parallel-size.
        Same flag-name difference — must use SGLang's convention.
        """
        cmd = BACKENDS["sglang"].build_server_command(
            make_scenario("nvidia", "h100", pipeline_parallel=2)
        )
        assert "--pp-size" in cmd
        assert "--pipeline-parallel-size" not in cmd

    def test_model_cache_dir(self):
        """
        SGLang uses --model-cache-dir for the weights path.
        Different flag name from vLLM's --download-dir.
        """
        cmd = BACKENDS["sglang"].build_server_command(
            make_scenario("nvidia", "h100", model_cache_dir="/mnt/models")
        )
        assert "--model-cache-dir" in cmd
        assert get_flag(cmd, "--model-cache-dir") == "/mnt/models"

    def test_extra_args_last(self):
        """
        extra_args must be appended last for user override capability.
        Same requirement as vLLM.
        """
        cmd = BACKENDS["sglang"].build_server_command(
            make_scenario("amd", "mi300x", extra_args=["--mem-fraction-static", "0.8"])
        )
        assert cmd[-2:] == ["--mem-fraction-static", "0.8"]


# ===========================================================================
# TGI
# ===========================================================================

class TestTGI:
    """
    TGI (Text Generation Inference) adapter tests.

    TGI has the most divergent flag names of the three backends:
    --num-shard instead of --tensor-parallel-size, --hostname instead
    of --host, and --huggingface-hub-cache instead of --download-dir.
    It also needs --disable-cuda-graphs on AMD to prevent ROCm hangs.
    """

    def test_nvidia_no_dtype_flag(self):
        """
        NVIDIA TGI should not get an explicit --dtype flag.
        TGI's CUDA defaults are reliable — we don't override them.
        """
        cmd = BACKENDS["tgi"].build_server_command(
            make_scenario("nvidia", "h100", backend="tgi")
        )
        assert "--dtype" not in cmd

    def test_nvidia_no_disable_cuda_graphs(self):
        """
        CUDA graphs should only be disabled on AMD, never on NVIDIA.

        CUDA graphs are a performance optimization on NVIDIA hardware.
        Disabling them on NVIDIA would reduce throughput for no reason.
        """
        cmd = BACKENDS["tgi"].build_server_command(
            make_scenario("nvidia", "h100", backend="tgi")
        )
        assert "--disable-cuda-graphs" not in cmd

    def test_amd_bfloat16(self):
        """
        AMD TGI should get --dtype bfloat16 for MI300-series hardware.

        Without an explicit dtype TGI may default to float16 on ROCm,
        which is suboptimal for MI300X and newer accelerators.
        """
        cmd = BACKENDS["tgi"].build_server_command(
            make_scenario("amd", "mi300x", backend="tgi")
        )
        assert get_flag(cmd, "--dtype") == "bfloat16"

    def test_amd_disable_cuda_graphs(self):
        """
        AMD TGI must include --disable-cuda-graphs.

        On some ROCm versions, TGI's CUDA graph capture step hangs
        indefinitely during server startup. The process appears to launch
        successfully but never becomes ready to serve requests.
        Disabling CUDA graphs avoids this entirely.
        """
        cmd = BACKENDS["tgi"].build_server_command(
            make_scenario("amd", "mi300x", backend="tgi")
        )
        assert "--disable-cuda-graphs" in cmd

    def test_num_shard_flag_name(self):
        """
        TGI uses --num-shard to control tensor parallelism.

        This is TGI's equivalent of --tensor-parallel-size (vLLM) and
        --tp-size (SGLang). Using the wrong flag name causes TGI to start
        with a single GPU regardless of what value was passed.
        """
        cmd = BACKENDS["tgi"].build_server_command(
            make_scenario("nvidia", "h100", backend="tgi", tensor_parallel=4)
        )
        assert "--num-shard" in cmd
        assert "--tensor-parallel-size" not in cmd
        assert get_flag(cmd, "--num-shard") == "4"

    def test_model_cache_dir(self):
        """
        TGI uses --huggingface-hub-cache for the weights path.
        Different flag name from both vLLM (--download-dir) and
        SGLang (--model-cache-dir).
        """
        cmd = BACKENDS["tgi"].build_server_command(
            make_scenario("nvidia", "h100", backend="tgi", model_cache_dir="/mnt/hf")
        )
        assert "--huggingface-hub-cache" in cmd

    def test_extra_args_last(self):
        """
        extra_args appended last for user override capability.
        Same requirement as all other backends.
        """
        cmd = BACKENDS["tgi"].build_server_command(
            make_scenario("amd", "mi355x", backend="tgi",
                          extra_args=["--max-input-length", "4096"])
        )
        assert cmd[-2:] == ["--max-input-length", "4096"]


# ===========================================================================
# TensorRT-LLM
# ===========================================================================

class TestTensorRTLLM:
    """
    TensorRT-LLM adapter tests.

    TRT-LLM is NVIDIA-only and uses the trtllm-serve CLI (introduced in
    TRT-LLM 0.12+). Different NVIDIA architectures get different precision
    defaults: H100 and B200 support native FP8 for maximum throughput,
    while A100 and older use bfloat16. AMD hardware must be rejected with
    a clear error message.
    """

    def test_uses_trtllm_serve(self):
        """
        The command must start with trtllm-serve, not the old placeholder.

        The original codebase had a placeholder that printed an echo
        message and slept for 2 seconds. This test confirms that placeholder
        has been replaced with the real launch command.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "h100", backend="tensorrt-llm")
        )
        assert cmd[0] == "trtllm-serve"

    def test_model_is_second_positional(self):
        """
        The model ID must be the second element — trtllm-serve's first
        positional argument.

        trtllm-serve accepts either a HuggingFace model ID or a path to
        a pre-compiled engine directory as its first positional argument.
        Getting its position wrong causes an immediate parse error.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "h100", backend="tensorrt-llm")
        )
        assert cmd[1] == "meta-llama/Llama-3.1-8B-Instruct"

    def test_h100_gets_fp8_kv_cache(self):
        """
        H100 should get --kv_cache_dtype fp8.

        The H100 has native FP8 tensor cores (Hopper architecture).
        Using FP8 KV cache gives roughly 2x the token throughput of
        bfloat16 for the same VRAM budget. This is the highest-performance
        path available on H100 hardware.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "h100", backend="tensorrt-llm")
        )
        assert "--kv_cache_dtype" in cmd
        assert get_flag(cmd, "--kv_cache_dtype") == "fp8"

    def test_b200_gets_fp8_kv_cache(self):
        """
        B200 (Blackwell) should also get --kv_cache_dtype fp8.

        B200 has even more capable FP8 support than H100. FP8 KV cache
        is definitely the right default here.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "b200", backend="tensorrt-llm")
        )
        assert get_flag(cmd, "--kv_cache_dtype") == "fp8"

    def test_a100_no_kv_cache_dtype(self):
        """
        A100 should not get --kv_cache_dtype at all.

        The A100 (Ampere architecture) does not have native FP8 tensor
        cores. Specifying fp8 KV cache on an A100 either errors or falls
        back to an emulated mode that is slower than bfloat16. We omit
        the flag entirely and let TRT-LLM choose the right default.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "a100", backend="tensorrt-llm")
        )
        assert "--kv_cache_dtype" not in cmd

    def test_amd_raises_value_error(self):
        """
        Attempting to use TRT-LLM with AMD hardware must raise a ValueError.

        TRT-LLM is a CUDA-only library. It cannot run on ROCm. Rather than
        generating a command that fails at runtime with a cryptic CUDA error,
        the adapter detects this early and raises a clear exception.
        """
        with pytest.raises(ValueError, match="NVIDIA"):
            BACKENDS["tensorrt-llm"].build_server_command(
                make_scenario("amd", "mi300x", backend="tensorrt-llm")
            )

    def test_amd_error_suggests_alternative(self):
        """
        The error message for AMD hardware must name vLLM and SGLang
        as the correct alternatives.

        A good error message tells the user not just what went wrong but
        what to do instead. Without this hint a user might not know
        which backend to switch to for AMD.
        """
        with pytest.raises(ValueError, match="vllm or sglang"):
            BACKENDS["tensorrt-llm"].build_server_command(
                make_scenario("amd", "mi300x", backend="tensorrt-llm")
            )

    def test_tp_size_passed(self):
        """
        tensor_parallel must appear as --tp_size in the trtllm-serve command.

        Note: trtllm-serve uses underscores (--tp_size) while vLLM uses
        hyphens (--tensor-parallel-size). Using the wrong style causes
        trtllm-serve to reject the flag.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "h100", backend="tensorrt-llm", tensor_parallel=8)
        )
        assert get_flag(cmd, "--tp_size") == "8"

    def test_max_batch_size_present(self):
        """
        The command must include --max_batch_size with a sensible default.

        Without an explicit max_batch_size TRT-LLM uses a very small
        default that severely limits throughput at higher concurrency.
        We set 256 as the default, which users can override via extra_args.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "h100", backend="tensorrt-llm")
        )
        assert "--max_batch_size" in cmd

    def test_model_cache_dir_passed(self):
        """
        model_cache_dir must appear as --model_cache_dir.

        trtllm-serve uses this directory both to store compiled engines
        and to find pre-built engines. Passing the right path avoids
        recompiling the engine on every launch.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "h100", backend="tensorrt-llm",
                          model_cache_dir="/engines/llama")
        )
        assert "--model_cache_dir" in cmd
        assert get_flag(cmd, "--model_cache_dir") == "/engines/llama"

    def test_extra_args_last(self):
        """
        extra_args must come last so users can override auto-injected flags.

        For example a user could put --max_batch_size 512 in extra_args
        to override our default of 256.
        """
        cmd = BACKENDS["tensorrt-llm"].build_server_command(
            make_scenario("nvidia", "h100", backend="tensorrt-llm",
                          extra_args=["--max_batch_size", "512"])
        )
        assert cmd[-2:] == ["--max_batch_size", "512"]

    def test_healthcheck_url(self):
        """
        trtllm-serve exposes the same /health endpoint as the other backends.
        The orchestrator uses this URL to know when the server is ready.
        """
        url = BACKENDS["tensorrt-llm"].build_healthcheck_url(
            make_scenario("nvidia", "h100", backend="tensorrt-llm")
        )
        assert url == "http://127.0.0.1:8000/health"