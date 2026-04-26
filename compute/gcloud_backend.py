"""
Google Cloud compute backend for ApexQuant (stub).

Would use Vertex AI Training or GCE for job execution.
All methods raise :class:`NotImplementedError` with SDK hints.

Usage::

    from compute.gcloud_backend import GCloudBackend
    backend = GCloudBackend(config)  # stub
"""

__all__ = ["GCloudBackend"]

from collections.abc import Iterator
from typing import Any

from loguru import logger

from compute.base import ComputeBackend


class GCloudBackend(ComputeBackend):
    """Google Cloud backend (stub).

    Attributes:
        project: GCP project ID.
        region: GCP region.
        machine_type: GCE machine type.
        accelerator: GPU accelerator type.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gcloud_cfg = config.get("compute", {}).get("gcloud", {})
        self.project: str = gcloud_cfg.get("project", "")
        self.region: str = gcloud_cfg.get("region", "us-central1")
        self.machine_type: str = gcloud_cfg.get("machine_type", "n1-standard-8")
        self.accelerator: str = gcloud_cfg.get("accelerator", "NVIDIA_TESLA_T4")

        logger.info(
            "GCloudBackend initialised (project={}, region={})",
            self.project,
            self.region,
        )

    def submit_job(
        self, script: str, config: dict[str, Any], job_name: str = ""
    ) -> str:
        # TODO: google.cloud.aiplatform.CustomTrainingJob.run()
        raise NotImplementedError(
            "GCloud backend not implemented. "
            "Would use google.cloud.aiplatform.CustomTrainingJob.run()"
        )

    def get_status(self, job_id: str) -> dict[str, Any]:
        # TODO: google.cloud.aiplatform.CustomJob.get(job_id).state
        raise NotImplementedError(
            "GCloud backend not implemented. "
            "Would use google.cloud.aiplatform.CustomJob.get().state"
        )

    def get_logs(self, job_id: str) -> Iterator[str]:
        # TODO: google.cloud.logging.Client().list_entries(filter_=...)
        raise NotImplementedError(
            "GCloud backend not implemented. "
            "Would use google.cloud.logging.Client().list_entries()"
        )

    def get_results(self, job_id: str) -> dict[str, Any]:
        # TODO: google.cloud.storage.Client().bucket().list_blobs(prefix=job_id)
        raise NotImplementedError(
            "GCloud backend not implemented. "
            "Would use google.cloud.storage.Client().bucket().list_blobs()"
        )

    def cancel_job(self, job_id: str) -> bool:
        # TODO: google.cloud.aiplatform.CustomJob.get(job_id).cancel()
        raise NotImplementedError(
            "GCloud backend not implemented. "
            "Would use google.cloud.aiplatform.CustomJob.cancel()"
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        # TODO: google.cloud.aiplatform.CustomJob.list()
        raise NotImplementedError(
            "GCloud backend not implemented. "
            "Would use google.cloud.aiplatform.CustomJob.list()"
        )

    def test_connection(self) -> dict[str, Any]:
        # TODO: google.auth.default() + google.cloud.storage.Client().get_bucket()
        raise NotImplementedError(
            "GCloud backend not implemented. "
            "Would use google.auth.default() + storage client"
        )

    def __repr__(self) -> str:
        return f"GCloudBackend(project={self.project!r}, region={self.region!r})"
