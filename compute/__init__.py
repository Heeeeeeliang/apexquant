"""
ApexQuant Compute Module
=========================

Routes training jobs to the correct compute backend based on
``config["compute"]["backend"]``.

Supported backends:

- ``"local"`` — local subprocess execution with GPU detection
- ``"colab"`` — async execution via Google Drive file sync
- ``"gcloud"`` — Google Cloud Vertex AI (stub)
- ``"aws"`` — AWS SageMaker (stub)

Usage::

    from compute import get_backend
    from config.default import CONFIG

    backend = get_backend(CONFIG)
    job_id = backend.submit_job("train.py", CONFIG)
    status = backend.get_status(job_id)
"""

__all__ = [
    "get_backend",
    "ComputeBackend",
    "LocalBackend",
    "ColabBackend",
    "GCloudBackend",
    "AWSBackend",
]

from typing import Any

from loguru import logger

from compute.base import ComputeBackend
from compute.local_backend import LocalBackend
from compute.colab_backend import ColabBackend
from compute.gcloud_backend import GCloudBackend
from compute.aws_backend import AWSBackend


def get_backend(config: dict[str, Any]) -> ComputeBackend:
    """Instantiate the correct compute backend from config.

    Args:
        config: Full CONFIG dict.  Reads
            ``config["compute"]["backend"]``.

    Returns:
        A :class:`ComputeBackend` instance.

    Raises:
        ValueError: If the backend name is not recognised.
    """
    backend_name = config.get("compute", {}).get("backend", "local")

    if backend_name == "local":
        return LocalBackend(config)
    if backend_name == "colab":
        return ColabBackend(config)
    if backend_name == "gcloud":
        return GCloudBackend(config)
    if backend_name == "aws":
        return AWSBackend(config)

    raise ValueError(
        f"Unknown compute backend: {backend_name!r}. "
        f"Valid: 'local', 'colab', 'gcloud', 'aws'"
    )
