"""
Abstract compute backend for ApexQuant.

Defines the interface that all compute backends (local, Colab,
GCloud, AWS) must implement for job submission, monitoring, and
result retrieval.

Usage::

    from compute.base import ComputeBackend

    class MyBackend(ComputeBackend):
        def submit_job(self, script, config, job_name=""): ...
        ...
"""

__all__ = ["ComputeBackend"]

import abc
from collections.abc import Iterator
from typing import Any


class ComputeBackend(abc.ABC):
    """Abstract base class for compute backends.

    All backends must implement job submission, status polling,
    log streaming, result retrieval, cancellation, listing, and
    connection testing.
    """

    @abc.abstractmethod
    def submit_job(
        self, script: str, config: dict[str, Any], job_name: str = ""
    ) -> str:
        """Submit a training job.

        Args:
            script: Path to the Python script to execute.
            config: Full CONFIG dict to pass to the job.
            job_name: Optional human-readable job name.

        Returns:
            Unique job ID string.
        """
        ...

    @abc.abstractmethod
    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get the current status of a job.

        Args:
            job_id: Job identifier returned by :meth:`submit_job`.

        Returns:
            Dict with keys:

            - ``status``: ``"pending"`` | ``"running"`` | ``"done"`` | ``"failed"``
            - ``progress``: Integer 0--100
            - ``message``: Human-readable status message
        """
        ...

    @abc.abstractmethod
    def get_logs(self, job_id: str) -> Iterator[str]:
        """Stream log lines for a job.

        Args:
            job_id: Job identifier.

        Yields:
            Log lines as strings.
        """
        ...

    @abc.abstractmethod
    def get_results(self, job_id: str) -> dict[str, Any]:
        """Return output file paths for a completed job.

        Args:
            job_id: Job identifier.

        Returns:
            Dict with output paths (e.g. ``{"output_dir": "..."}``).
        """
        ...

    @abc.abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job.

        Args:
            job_id: Job identifier.

        Returns:
            ``True`` if the cancellation was accepted.
        """
        ...

    @abc.abstractmethod
    def list_jobs(self) -> list[dict[str, Any]]:
        """List all known jobs.

        Returns:
            List of dicts, each with at least ``job_id`` and ``status``.
        """
        ...

    @abc.abstractmethod
    def test_connection(self) -> dict[str, Any]:
        """Test the backend connection.

        Returns:
            Dict with keys:

            - ``ok``: ``True`` if the backend is reachable
            - ``latency_ms``: Round-trip latency in milliseconds
            - ``message``: Human-readable status
        """
        ...
