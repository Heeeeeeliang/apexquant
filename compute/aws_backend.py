"""
AWS SageMaker compute backend for ApexQuant (stub).

Would use SageMaker Training Jobs for execution.
All methods raise :class:`NotImplementedError` with SDK hints.

Usage::

    from compute.aws_backend import AWSBackend
    backend = AWSBackend(config)  # stub
"""

__all__ = ["AWSBackend"]

from collections.abc import Iterator
from typing import Any

from loguru import logger

from compute.base import ComputeBackend


class AWSBackend(ComputeBackend):
    """AWS SageMaker backend (stub).

    Attributes:
        instance_type: SageMaker instance type.
        role_arn: IAM role ARN for SageMaker.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        aws_cfg = config.get("compute", {}).get("aws", {})
        self.instance_type: str = aws_cfg.get("instance_type", "ml.p3.2xlarge")
        self.role_arn: str = aws_cfg.get("role_arn", "")

        logger.info(
            "AWSBackend initialised (instance_type={})",
            self.instance_type,
        )

    def submit_job(
        self, script: str, config: dict[str, Any], job_name: str = ""
    ) -> str:
        # TODO: sagemaker.estimator.Estimator.fit()
        raise NotImplementedError(
            "AWS backend not implemented. "
            "Would use sagemaker.estimator.Estimator(instance_type=...).fit()"
        )

    def get_status(self, job_id: str) -> dict[str, Any]:
        # TODO: boto3.client('sagemaker').describe_training_job(TrainingJobName=job_id)
        raise NotImplementedError(
            "AWS backend not implemented. "
            "Would use boto3.client('sagemaker').describe_training_job()"
        )

    def get_logs(self, job_id: str) -> Iterator[str]:
        # TODO: boto3.client('logs').get_log_events(logGroupName=..., logStreamName=job_id)
        raise NotImplementedError(
            "AWS backend not implemented. "
            "Would use boto3.client('logs').get_log_events()"
        )

    def get_results(self, job_id: str) -> dict[str, Any]:
        # TODO: boto3.client('s3').list_objects_v2(Bucket=..., Prefix=job_id)
        raise NotImplementedError(
            "AWS backend not implemented. "
            "Would use boto3.client('s3').list_objects_v2()"
        )

    def cancel_job(self, job_id: str) -> bool:
        # TODO: boto3.client('sagemaker').stop_training_job(TrainingJobName=job_id)
        raise NotImplementedError(
            "AWS backend not implemented. "
            "Would use boto3.client('sagemaker').stop_training_job()"
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        # TODO: boto3.client('sagemaker').list_training_jobs()
        raise NotImplementedError(
            "AWS backend not implemented. "
            "Would use boto3.client('sagemaker').list_training_jobs()"
        )

    def test_connection(self) -> dict[str, Any]:
        # TODO: boto3.client('sts').get_caller_identity()
        raise NotImplementedError(
            "AWS backend not implemented. "
            "Would use boto3.client('sts').get_caller_identity()"
        )

    def __repr__(self) -> str:
        return f"AWSBackend(instance_type={self.instance_type!r})"
