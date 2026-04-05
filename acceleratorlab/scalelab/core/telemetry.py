"""
GPU telemetry collector — samples hardware metrics during a benchmark window.

Runs a background thread that polls nvidia-smi (NVIDIA) or rocm-smi (AMD)
once per second throughout the benchmark. On stop() it returns mean and peak
values for utilization, VRAM, power draw, and temperature.

Usage
-----
    collector = TelemetryCollector(vendor="nvidia")
    collector.start()
    # ... run benchmark traffic ...
    result = collector.stop()

    print(result.gpu_util_mean_pct)   # e.g. 87.3
    print(result.power_mean_w)        # e.g. 312.4
    print(result.available)           # False if nvidia-smi not found

The collector never raises — if the tool is missing or parsing fails it
returns a TelemetryResult with available=False and all metrics set to 0.
This keeps the benchmark pipeline intact on machines without GPU tools.
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TelemetryResult:
    """All hardware metrics captured during one benchmark window."""

    # Availability
    available: bool = False          # False if tool not found or all samples failed
    vendor: str = "unknown"
    gpu_count: int = 0
    sample_count: int = 0
    error: str = ""                  # populated if something went wrong

    # GPU utilization (%)
    gpu_util_mean_pct: float = 0.0
    gpu_util_peak_pct: float = 0.0

    # VRAM (GB)
    vram_used_mean_gb: float = 0.0
    vram_used_peak_gb: float = 0.0
    vram_total_gb: float = 0.0       # physical capacity — static across the run

    # Power draw (watts, summed across all GPUs)
    power_mean_w: float = 0.0
    power_peak_w: float = 0.0

    # Temperature (°C, mean across all GPUs)
    temp_mean_c: float = 0.0
    temp_peak_c: float = 0.0

    def to_dict(self) -> dict:
        return {
            "telemetry_available":   self.available,
            "telemetry_vendor":      self.vendor,
            "gpu_count":             self.gpu_count,
            "telemetry_samples":     self.sample_count,
            "telemetry_error":       self.error,
            "gpu_util_mean_pct":     self.gpu_util_mean_pct,
            "gpu_util_peak_pct":     self.gpu_util_peak_pct,
            "vram_used_mean_gb":     self.vram_used_mean_gb,
            "vram_used_peak_gb":     self.vram_used_peak_gb,
            "vram_total_gb":         self.vram_total_gb,
            "power_mean_w":          self.power_mean_w,
            "power_peak_w":          self.power_peak_w,
            "temp_mean_c":           self.temp_mean_c,
            "temp_peak_c":           self.temp_peak_c,
        }


# ---------------------------------------------------------------------------
# Per-sample snapshot (internal)
# ---------------------------------------------------------------------------

@dataclass
class _Sample:
    """One poll's worth of data — averaged across all GPUs present."""
    util_pct: float       # mean GPU utilization across all GPUs
    vram_used_gb: float   # mean VRAM used across all GPUs
    vram_total_gb: float  # mean VRAM total (should be constant)
    power_w: float        # TOTAL power across all GPUs
    temp_c: float         # mean temperature across all GPUs
    gpu_count: int


# ---------------------------------------------------------------------------
# NVIDIA — nvidia-smi
# ---------------------------------------------------------------------------

def _nvidia_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _poll_nvidia() -> Optional[_Sample]:
    """
    Query nvidia-smi for per-GPU stats and aggregate into a single _Sample.

    nvidia-smi CSV output (one row per GPU):
        index, utilization.gpu [%], memory.used [MiB], memory.total [MiB],
        power.draw [W], temperature.gpu
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                "power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None

    utils, vram_used, vram_total, powers, temps = [], [], [], [], []

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            utils.append(float(parts[1]))
            vram_used.append(float(parts[2]) / 1024)   # MiB → GB
            vram_total.append(float(parts[3]) / 1024)  # MiB → GB
            # power.draw can be "N/A" on some GPUs (e.g. MIG slices)
            pw = parts[4]
            powers.append(float(pw) if pw.replace(".", "").isdigit() else 0.0)
            temps.append(float(parts[5]))
        except (ValueError, IndexError):
            continue

    if not utils:
        return None

    return _Sample(
        util_pct=statistics.mean(utils),
        vram_used_gb=statistics.mean(vram_used),
        vram_total_gb=statistics.mean(vram_total),
        power_w=sum(powers),                 # total rack power
        temp_c=statistics.mean(temps),
        gpu_count=len(utils),
    )


# ---------------------------------------------------------------------------
# AMD — rocm-smi
# ---------------------------------------------------------------------------

def _amd_available() -> bool:
    return shutil.which("rocm-smi") is not None


def _poll_amd() -> Optional[_Sample]:
    """
    Query rocm-smi for per-GPU stats.

    Tries JSON output first (ROCm 5.x+), falls back to CSV-style text
    (older ROCm).  Both paths aggregate across all GPUs.
    """
    sample = _poll_amd_json()
    if sample is None:
        sample = _poll_amd_text()
    return sample


def _poll_amd_json() -> Optional[_Sample]:
    """rocm-smi --json path (ROCm 5.x / 6.x)."""
    try:
        import json as _json
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram",
             "--showpower", "--showtemp", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        data = _json.loads(out)
    except Exception:
        return None

    utils, vram_used, vram_total, powers, temps = [], [], [], [], []

    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        # GPU utilization — key varies by ROCm version
        for uk in ("GPU use (%)", "GPU Use (%)", "GPU use(%)",
                   "GPU Activity", "gpu_use_pct"):
            if uk in val:
                try:
                    utils.append(float(str(val[uk]).replace("%", "").strip()))
                except ValueError:
                    pass
                break

        # VRAM
        for mk in ("VRAM Total Memory (B)", "vram_total"):
            if mk in val:
                try:
                    vram_total.append(float(val[mk]) / (1024 ** 3))
                except ValueError:
                    pass
                break
        for mk in ("VRAM Total Used Memory (B)", "vram_used"):
            if mk in val:
                try:
                    vram_used.append(float(val[mk]) / (1024 ** 3))
                except ValueError:
                    pass
                break

        # Power
        for pk in ("Average Graphics Package Power (W)",
                   "Current Socket Graphics Package Power (W)",
                   "power_avg_w"):
            if pk in val:
                try:
                    powers.append(float(str(val[pk]).replace("W", "").strip()))
                except ValueError:
                    pass
                break

        # Temperature
        for tk in ("Temperature (Sensor edge) (C)",
                   "Temperature (Sensor junction) (C)",
                   "temp_edge_c"):
            if tk in val:
                try:
                    temps.append(float(str(val[tk]).replace("C", "").replace("°", "").strip()))
                except ValueError:
                    pass
                break

    if not utils:
        return None

    return _Sample(
        util_pct=statistics.mean(utils),
        vram_used_gb=statistics.mean(vram_used) if vram_used else 0.0,
        vram_total_gb=statistics.mean(vram_total) if vram_total else 0.0,
        power_w=sum(powers),
        temp_c=statistics.mean(temps) if temps else 0.0,
        gpu_count=len(utils),
    )


def _poll_amd_text() -> Optional[_Sample]:
    """
    Fallback text parser for older rocm-smi that does not support --json.
    Runs individual rocm-smi queries and parses line-by-line output.
    """
    utils, powers, temps = [], [], []

    # GPU utilization
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.splitlines():
            # Lines look like:  GPU[0]        : GPU use (%): 87
            if "GPU use" in line and ":" in line:
                try:
                    utils.append(float(line.rsplit(":", 1)[-1].strip()))
                except ValueError:
                    pass
    except Exception:
        pass

    # Power
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showpower"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.splitlines():
            if "Power" in line and ":" in line:
                try:
                    pw = line.rsplit(":", 1)[-1].strip().replace("W", "").strip()
                    powers.append(float(pw))
                except ValueError:
                    pass
    except Exception:
        pass

    # Temperature
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showtemp"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.splitlines():
            if "Temperature" in line and "edge" in line.lower() and ":" in line:
                try:
                    t = line.rsplit(":", 1)[-1].strip().replace("C", "").replace("°", "").strip()
                    temps.append(float(t))
                except ValueError:
                    pass
    except Exception:
        pass

    if not utils:
        return None

    return _Sample(
        util_pct=statistics.mean(utils),
        vram_used_gb=0.0,   # not available in text fallback
        vram_total_gb=0.0,
        power_w=sum(powers),
        temp_c=statistics.mean(temps) if temps else 0.0,
        gpu_count=len(utils),
    )


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------

class TelemetryCollector:
    """
    Background-thread GPU telemetry sampler.

    Parameters
    ----------
    vendor : str
        "nvidia" or "amd" — drives which tool is called.
        Any other value disables collection gracefully.
    interval_s : float
        Seconds between samples. Default 1.0.
    """

    def __init__(self, vendor: str = "nvidia", interval_s: float = 1.0) -> None:
        self._vendor = vendor.lower().strip()
        self._interval = interval_s
        self._samples: List[_Sample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: str = ""

        # Select poll function based on vendor — fail early if tool missing
        if self._vendor == "nvidia":
            if _nvidia_available():
                self._poll_fn = _poll_nvidia
            else:
                self._poll_fn = None
                self._error = "nvidia-smi not found in PATH"
        elif self._vendor == "amd":
            if _amd_available():
                self._poll_fn = _poll_amd
            else:
                self._poll_fn = None
                self._error = "rocm-smi not found in PATH"
        else:
            self._poll_fn = None
            self._error = f"Unknown vendor '{vendor}' — telemetry disabled"

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start sampling in a background daemon thread."""
        if self._poll_fn is None:
            return   # tool unavailable — silently skip, stop() returns zeros
        self._stop_event.clear()
        self._samples.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="telemetry-sampler"
        )
        self._thread.start()

    def stop(self) -> TelemetryResult:
        """
        Stop the background thread and return aggregated metrics.
        Safe to call even if start() was never called or the tool was absent.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 3)
            self._thread = None
        return self._aggregate()

    # ── Background thread ────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t_start = time.monotonic()
            try:
                sample = self._poll_fn()
                if sample is not None:
                    self._samples.append(sample)
            except Exception as exc:
                # Never crash the benchmark — record the error and keep going
                if not self._error:
                    self._error = f"Sampling error: {exc}"
            elapsed = time.monotonic() - t_start
            sleep_for = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    # ── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self) -> TelemetryResult:
        if not self._samples:
            return TelemetryResult(
                available=False,
                vendor=self._vendor,
                error=self._error or "No samples collected",
            )

        util_series  = [s.util_pct      for s in self._samples]
        vram_u_series= [s.vram_used_gb  for s in self._samples]
        vram_t_series= [s.vram_total_gb for s in self._samples]
        power_series = [s.power_w       for s in self._samples]
        temp_series  = [s.temp_c        for s in self._samples]
        gpu_count    = self._samples[-1].gpu_count

        def _mean(lst):
            return round(statistics.mean(lst), 2) if lst else 0.0

        def _peak(lst):
            return round(max(lst), 2) if lst else 0.0

        return TelemetryResult(
            available=True,
            vendor=self._vendor,
            gpu_count=gpu_count,
            sample_count=len(self._samples),
            error=self._error,
            gpu_util_mean_pct=_mean(util_series),
            gpu_util_peak_pct=_peak(util_series),
            vram_used_mean_gb=_mean(vram_u_series),
            vram_used_peak_gb=_peak(vram_u_series),
            vram_total_gb=_mean(vram_t_series),
            power_mean_w=_mean(power_series),
            power_peak_w=_peak(power_series),
            temp_mean_c=_mean(temp_series),
            temp_peak_c=_peak(temp_series),
        )


# ---------------------------------------------------------------------------
# Convenience function — used by traffic.py
# ---------------------------------------------------------------------------

def collect_during(vendor: str, fn, *args, interval_s: float = 1.0, **kwargs):
    """
    Run fn(*args, **kwargs) while collecting telemetry, then return both.

    Returns
    -------
    (result, TelemetryResult)
        result         — whatever fn returned
        TelemetryResult — hardware metrics captured during fn's execution
    """
    collector = TelemetryCollector(vendor=vendor, interval_s=interval_s)
    collector.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        telemetry = collector.stop()
    return result, telemetry