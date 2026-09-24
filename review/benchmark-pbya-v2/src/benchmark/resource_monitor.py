"""Lightweight subprocess-based resource monitor.

Polls /proc/{pid}/status and nvidia-smi at 1-second intervals.
Writes a JSON summary when stop() is called.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path


class ResourceMonitor:
    """Track peak RAM and GPU memory for a process tree."""

    def __init__(self, pid, poll_interval=1.0):
        self.pid = pid
        self.poll_interval = poll_interval
        self._peak_rss_kb = 0
        self._peak_gpu_mb = 0.0
        self._gpu_name = None
        self._n_samples = 0
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        wall_time = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "wall_time_s": round(wall_time, 1),
            "peak_rss_mb": round(self._peak_rss_kb / 1024, 1),
            "peak_gpu_mb": round(self._peak_gpu_mb, 1),
            "gpu_name": self._gpu_name,
            "n_samples": self._n_samples,
        }

    def _poll_loop(self):
        while not self._stop_event.is_set():
            self._sample_rss()
            self._sample_gpu()
            self._n_samples += 1
            self._stop_event.wait(self.poll_interval)

    def _get_tree_pids(self):
        """Get all descendant PIDs (including self.pid)."""
        pids = set()
        try:
            result = subprocess.run(
                ["ps", "--no-headers", "-o", "pid", "--ppid", str(self.pid)],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line:
                    pids.add(int(line))
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        pids.add(self.pid)
        return pids

    def _sample_rss(self):
        total_rss = 0
        for pid in self._get_tree_pids():
            status_path = f"/proc/{pid}/status"
            try:
                with open(status_path) as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            total_rss += int(line.split()[1])  # kB
                            break
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        self._peak_rss_kb = max(self._peak_rss_kb, total_rss)

    def _sample_gpu(self):
        if not shutil.which("nvidia-smi"):
            return
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            tree_pids = self._get_tree_pids()
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        gpid, mem = int(parts[0].strip()), float(parts[1].strip())
                        if gpid in tree_pids:
                            self._peak_gpu_mb = max(self._peak_gpu_mb, mem)
                    except ValueError:
                        continue
            # Get GPU name once
            if self._gpu_name is None:
                name_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                name = name_result.stdout.strip().split("\n")[0].strip()
                if name:
                    self._gpu_name = name
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass


def write_resources(resources_dict, output_path):
    """Write resource dict to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(resources_dict, f, indent=2)
