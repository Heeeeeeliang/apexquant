"""
Local compute backend for ApexQuant.

Runs training jobs as subprocesses on the local machine.
Detects GPU availability (CUDA / MPS) and routes accordingly.

Usage::

    from compute.local_backend import LocalBackend

    backend = LocalBackend(config)
    job_id = backend.submit_job("train.py", config)
    status = backend.get_status(job_id)
"""

__all__ = ["LocalBackend"]

import json
import os
import subprocess
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from uuid import uuid4

from loguru import logger

from compute.base import ComputeBackend


class LocalBackend(ComputeBackend):
    """Runs jobs as local subprocesses.

    Attributes:
        device: Detected compute device (``"cuda"``, ``"mps"``, or ``"cpu"``).
        jobs: Dict mapping job_id to job state.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the local backend.

        Args:
            config: Full CONFIG dict.  Reads
                ``config["compute"]["local"]["device"]``.
        """
        self.config = config
        local_cfg = config.get("compute", {}).get("local", {})
        device_pref = local_cfg.get("device", "auto")

        self.device: str = (
            self._detect_device() if device_pref == "auto" else device_pref
        )
        self.jobs: dict[str, dict[str, Any]] = {}

        logger.info("LocalBackend initialised (device={})", self.device)

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available compute device.

        Returns:
            ``"cuda"``, ``"mps"``, or ``"cpu"``.
        """
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info("CUDA device detected: {}", device_name)
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS (Apple Silicon) device detected")
                return "mps"
        except Exception as exc:
            logger.debug("torch not available for device detection: {}", exc)

        logger.info("No GPU detected, using CPU")
        return "cpu"

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def submit_job(
        self, script: str, config: dict[str, Any], job_name: str = ""
    ) -> str:
        """Submit a training job as a local subprocess.

        The config is passed via the ``APEXQUANT_CONFIG`` environment
        variable as a JSON string.

        Args:
            script: Path to the Python script to execute.
            config: Full CONFIG dict.
            job_name: Optional human-readable name.

        Returns:
            Unique job ID.
        """
        job_id = uuid4().hex[:12]
        job_name = job_name or f"local_{job_id}"

        env = os.environ.copy()
        env["APEXQUANT_CONFIG"] = json.dumps(config, default=str)
        env["APEXQUANT_DEVICE"] = self.device

        process = subprocess.Popen(
            ["python", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )

        self.jobs[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "script": script,
            "process": process,
            "status": "running",
            "progress": 0,
            "logs": [],
            "start_time": datetime.now(),
            "end_time": None,
            "return_code": None,
        }

        logger.info(
            "Submitted local job: id={}, name={}, script={}, device={}",
            job_id,
            job_name,
            script,
            self.device,
        )
        return job_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get current job status.

        Polls the subprocess and collects any new stdout lines.

        Args:
            job_id: Job identifier.

        Returns:
            Status dict with ``status``, ``progress``, ``message``.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return {"status": "unknown", "progress": 0, "message": f"Unknown job: {job_id}"}

        # Collect new output lines
        self._collect_output(job)

        process: subprocess.Popen = job["process"]
        return_code = process.poll()

        if return_code is None:
            job["status"] = "running"
            # Estimate progress from log line count (rough heuristic)
            n_lines = len(job["logs"])
            job["progress"] = min(95, n_lines)
            message = f"Running ({n_lines} log lines)"
        elif return_code == 0:
            job["status"] = "done"
            job["progress"] = 100
            job["end_time"] = job.get("end_time") or datetime.now()
            elapsed = (job["end_time"] - job["start_time"]).total_seconds()
            message = f"Completed in {elapsed:.1f}s"
        else:
            job["status"] = "failed"
            job["end_time"] = job.get("end_time") or datetime.now()
            message = f"Failed with return code {return_code}"

        return {
            "status": job["status"],
            "progress": job["progress"],
            "message": message,
            "job_name": job["job_name"],
            "start_time": job["start_time"].isoformat(),
        }

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def get_logs(self, job_id: str) -> Iterator[str]:
        """Stream log lines from the job's stdout.

        Args:
            job_id: Job identifier.

        Yields:
            Log lines.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return

        self._collect_output(job)
        yield from job["logs"]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self, job_id: str) -> dict[str, Any]:
        """Return output paths for a completed job.

        Args:
            job_id: Job identifier.

        Returns:
            Dict with ``output_dir`` and job metadata.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return {"error": f"Unknown job: {job_id}"}

        return {
            "output_dir": "results/runs/latest/",
            "job_id": job_id,
            "job_name": job["job_name"],
            "status": job["status"],
            "device": self.device,
        }

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job by terminating the subprocess.

        Args:
            job_id: Job identifier.

        Returns:
            ``True`` if the process was terminated.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return False

        process: subprocess.Popen = job["process"]
        if process.poll() is not None:
            logger.debug("Job {} already finished, nothing to cancel", job_id)
            return False

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        job["status"] = "failed"
        job["end_time"] = datetime.now()
        logger.info("Cancelled job {}", job_id)
        return True

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all known jobs.

        Returns:
            List of job summary dicts.
        """
        result: list[dict[str, Any]] = []
        for job_id in self.jobs:
            status = self.get_status(job_id)
            result.append(status)
        return result

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    def test_connection(self) -> dict[str, Any]:
        """Test local environment readiness.

        Checks Python availability and device detection.

        Returns:
            Connection test result dict.
        """
        start = time.monotonic()

        try:
            proc = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            python_ok = proc.returncode == 0
            python_version = proc.stdout.strip()
        except Exception as exc:
            python_ok = False
            python_version = str(exc)

        latency = int((time.monotonic() - start) * 1000)

        return {
            "ok": python_ok,
            "latency_ms": latency,
            "message": f"Python: {python_version}, Device: {self.device}",
            "device": self.device,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_output(job: dict[str, Any]) -> None:
        """Read any available stdout lines from the subprocess.

        For a finished process, reads all remaining output.
        For a running process, uses a thread to avoid blocking.

        Args:
            job: Job state dict (mutated in-place).
        """
        process: subprocess.Popen = job["process"]
        if process.stdout is None:
            return

        # If already collected all output, skip
        if job.get("_output_collected"):
            return

        # If process is done, read all remaining output
        if process.poll() is not None:
            try:
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        job["logs"].append(line)
                job["_output_collected"] = True
            except (OSError, ValueError):
                pass
            return

        # For running process, start a background reader thread if not yet started
        if not job.get("_reader_started"):
            import threading

            def _reader() -> None:
                try:
                    for line in process.stdout:
                        job["logs"].append(line.rstrip("\n"))
                except (OSError, ValueError):
                    pass

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            job["_reader_started"] = True

    def __repr__(self) -> str:
        return f"LocalBackend(device={self.device!r}, jobs={len(self.jobs)})"
