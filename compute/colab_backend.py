"""
Google Colab compute backend for ApexQuant.

Async job execution via Google Drive file sync.  The local machine
writes a job file to Drive; a Colab notebook running
:func:`generate_colab_poll_script` polls the queue directory,
executes pending jobs, and writes results back.

Usage::

    from compute.colab_backend import ColabBackend

    backend = ColabBackend(config)
    job_id = backend.submit_job("train.py", config)

    # In Colab notebook, paste the output of:
    print(ColabBackend.generate_colab_poll_script())
"""

__all__ = ["ColabBackend"]

import json
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from compute.base import ComputeBackend


class ColabBackend(ComputeBackend):
    """Google Drive-based async job backend for Colab.

    Communication protocol:

    - ``job_queue/{job_id}.json`` — job submission payload
    - ``job_status/{job_id}.json`` — status updates from Colab
    - ``job_logs/{job_id}.txt`` — streaming log output
    - ``job_results/{job_id}.json`` — output paths and metadata

    Attributes:
        drive_path: Root Google Drive sync path.
        queue_dir: Path to the job queue directory.
        status_dir: Path to job status files.
        logs_dir: Path to job log files.
        results_dir: Path to job result files.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the Colab backend.

        Args:
            config: Full CONFIG dict.  Reads
                ``config["compute"]["colab"]["drive_path"]``.
        """
        self.config = config
        colab_cfg = config.get("compute", {}).get("colab", {})
        self.drive_path = Path(colab_cfg.get("drive_path", "/gdrive/MyDrive/apexquant/"))
        self.poll_interval: int = colab_cfg.get("poll_interval", 30)

        self.queue_dir = self.drive_path / "job_queue"
        self.status_dir = self.drive_path / "job_status"
        self.logs_dir = self.drive_path / "job_logs"
        self.results_dir = self.drive_path / "job_results"

        logger.info("ColabBackend initialised (drive_path={})", self.drive_path)

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def submit_job(
        self, script: str, config: dict[str, Any], job_name: str = ""
    ) -> str:
        """Submit a job by writing to the Drive queue directory.

        Args:
            script: Path to the training script (relative to project root).
            config: Full CONFIG dict.
            job_name: Optional human-readable name.

        Returns:
            Unique job ID.

        Raises:
            OSError: If the queue directory cannot be created.
        """
        job_id = uuid4().hex[:12]
        job_name = job_name or f"colab_{job_id}"

        payload: dict[str, Any] = {
            "job_id": job_id,
            "job_name": job_name,
            "script": script,
            "config": config,
            "status": "pending",
            "submitted_at": datetime.now().isoformat(),
        }

        self.queue_dir.mkdir(parents=True, exist_ok=True)
        job_file = self.queue_dir / f"{job_id}.json"

        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(
            "Submitted Colab job: id={}, name={}, script={}, queue={}",
            job_id,
            job_name,
            script,
            job_file,
        )
        return job_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Read job status from the Drive status directory.

        Args:
            job_id: Job identifier.

        Returns:
            Status dict with ``status``, ``progress``, ``message``.
        """
        status_file = self.status_dir / f"{job_id}.json"

        if not status_file.exists():
            # Check if job is still in queue
            queue_file = self.queue_dir / f"{job_id}.json"
            if queue_file.exists():
                return {
                    "status": "pending",
                    "progress": 0,
                    "message": "Waiting for Colab to pick up job",
                }
            return {
                "status": "unknown",
                "progress": 0,
                "message": f"No status file found for {job_id}",
            }

        try:
            with open(status_file, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "status": data.get("status", "unknown"),
                "progress": data.get("progress", 0),
                "message": data.get("message", ""),
            }
        except (json.JSONDecodeError, OSError) as exc:
            return {
                "status": "unknown",
                "progress": 0,
                "message": f"Error reading status: {exc}",
            }

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def get_logs(self, job_id: str) -> Iterator[str]:
        """Read log lines from the Drive logs directory.

        The Colab executor appends to this file as the job runs.
        Callers should poll this method periodically.

        Args:
            job_id: Job identifier.

        Yields:
            Log lines.
        """
        log_file = self.logs_dir / f"{job_id}.txt"

        if not log_file.exists():
            return

        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    yield line.rstrip("\n")
        except OSError as exc:
            logger.warning("Error reading logs for {}: {}", job_id, exc)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self, job_id: str) -> dict[str, Any]:
        """Read output paths from the Drive results directory.

        Args:
            job_id: Job identifier.

        Returns:
            Dict with output paths, or error info.
        """
        results_file = self.results_dir / f"{job_id}.json"

        if not results_file.exists():
            return {"error": f"No results file for {job_id}"}

        try:
            with open(results_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": f"Error reading results: {exc}"}

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation by writing to the status directory.

        The Colab executor should check for ``cancel_requested``
        status and abort.

        Args:
            job_id: Job identifier.

        Returns:
            ``True`` if the cancellation request was written.
        """
        self.status_dir.mkdir(parents=True, exist_ok=True)
        status_file = self.status_dir / f"{job_id}.json"

        cancel_payload = {
            "status": "cancel_requested",
            "progress": 0,
            "message": "Cancellation requested by user",
            "requested_at": datetime.now().isoformat(),
        }

        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(cancel_payload, f, indent=2)
            logger.info("Cancel requested for Colab job {}", job_id)
            return True
        except OSError as exc:
            logger.warning("Failed to cancel job {}: {}", job_id, exc)
            return False

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs found in the queue and status directories.

        Returns:
            List of job summary dicts.
        """
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Jobs in status directory
        if self.status_dir.exists():
            for f in self.status_dir.glob("*.json"):
                job_id = f.stem
                seen.add(job_id)
                status = self.get_status(job_id)
                status["job_id"] = job_id
                jobs.append(status)

        # Pending jobs still in queue
        if self.queue_dir.exists():
            for f in self.queue_dir.glob("*.json"):
                job_id = f.stem
                if job_id not in seen:
                    jobs.append({
                        "job_id": job_id,
                        "status": "pending",
                        "progress": 0,
                        "message": "In queue",
                    })

        return jobs

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    def test_connection(self) -> dict[str, Any]:
        """Test if the Google Drive path is accessible.

        Returns:
            Connection test result dict.
        """
        start = time.monotonic()
        exists = self.drive_path.exists()
        latency = int((time.monotonic() - start) * 1000)

        if exists:
            return {
                "ok": True,
                "latency_ms": latency,
                "message": f"Drive path accessible: {self.drive_path}",
            }
        return {
            "ok": False,
            "latency_ms": latency,
            "message": f"Drive path not found: {self.drive_path}. "
            f"Mount Google Drive or update config.",
        }

    # ------------------------------------------------------------------
    # Colab poll script generator
    # ------------------------------------------------------------------

    @staticmethod
    def generate_colab_poll_script(
        drive_path: str = "/gdrive/MyDrive/apexquant/",
        poll_interval: int = 30,
    ) -> str:
        """Generate a Python script for a Colab notebook cell.

        The script polls the job queue directory every
        ``poll_interval`` seconds, picks up pending jobs, executes
        them, and writes status/logs/results back.

        Paste the returned string into a Colab code cell and run it.

        Args:
            drive_path: Root Google Drive path.
            poll_interval: Seconds between queue polls.

        Returns:
            Python source code string.
        """
        return f'''"""
ApexQuant Colab Job Executor
Paste this into a Colab code cell and run it.
It will poll for jobs every {poll_interval} seconds.
Press the stop button to halt.
"""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

DRIVE_PATH = Path("{drive_path}")
QUEUE_DIR = DRIVE_PATH / "job_queue"
STATUS_DIR = DRIVE_PATH / "job_status"
LOGS_DIR = DRIVE_PATH / "job_logs"
RESULTS_DIR = DRIVE_PATH / "job_results"

for d in [QUEUE_DIR, STATUS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"ApexQuant Colab Executor started")
print(f"Watching: {{QUEUE_DIR}}")
print(f"Poll interval: {poll_interval}s")
print("Press stop to halt.\\n")

while True:
    job_files = sorted(QUEUE_DIR.glob("*.json"))

    for job_file in job_files:
        try:
            with open(job_file) as f:
                job = json.load(f)
        except Exception as e:
            print(f"Error reading {{job_file}}: {{e}}")
            continue

        job_id = job["job_id"]
        script = job.get("script", "")
        config = job.get("config", {{}})

        # Check for cancellation
        status_file = STATUS_DIR / f"{{job_id}}.json"
        if status_file.exists():
            with open(status_file) as f:
                st = json.load(f)
            if st.get("status") == "cancel_requested":
                print(f"Job {{job_id}} cancelled, skipping")
                job_file.unlink(missing_ok=True)
                continue

        print(f"Executing job {{job_id}}: {{script}}")

        # Update status to running
        with open(status_file, "w") as f:
            json.dump({{"status": "running", "progress": 10,
                       "message": "Executing on Colab",
                       "started_at": datetime.now().isoformat()}}, f)

        # Write config for the script
        env = os.environ.copy()
        env["APEXQUANT_CONFIG"] = json.dumps(config, default=str)

        log_file = LOGS_DIR / f"{{job_id}}.txt"

        try:
            with open(log_file, "w") as lf:
                proc = subprocess.run(
                    ["python", script],
                    stdout=lf, stderr=subprocess.STDOUT,
                    env=env, timeout=3600,
                )

            if proc.returncode == 0:
                status = "done"
                msg = "Completed successfully"
            else:
                status = "failed"
                msg = f"Exit code {{proc.returncode}}"

        except subprocess.TimeoutExpired:
            status = "failed"
            msg = "Timed out after 3600s"
        except Exception as e:
            status = "failed"
            msg = str(e)

        # Write final status
        with open(status_file, "w") as f:
            json.dump({{"status": status, "progress": 100,
                       "message": msg,
                       "finished_at": datetime.now().isoformat()}}, f)

        # Write results
        results_file = RESULTS_DIR / f"{{job_id}}.json"
        with open(results_file, "w") as f:
            json.dump({{"job_id": job_id, "status": status,
                       "output_dir": str(DRIVE_PATH / "results"),
                       "finished_at": datetime.now().isoformat()}}, f)

        # Remove from queue
        job_file.unlink(missing_ok=True)
        print(f"Job {{job_id}}: {{status}} - {{msg}}")

    time.sleep({poll_interval})
'''

    def __repr__(self) -> str:
        return f"ColabBackend(drive_path={self.drive_path!r})"
