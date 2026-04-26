"""InferenceStack — batch prediction Lambda, triggered after feature build.

Components:
    * DockerImageFunction ``InferenceFunction`` (numpy/pandas/sklearn/lightgbm image)
    * EventBridge rule matching the FeatureStack Lambda's async invocation
      success → invokes the inference Lambda with ``{"mode": "both"}``
    * IAM grant: read/write on the existing FPIngestStack data bucket

The trigger relies on Lambda async-invoke success events emitted by the
feature function (configured in FeatureStack) — there's no cross-stack
mutation, both stacks just route through the default EventBridge bus.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as targets,
)
from aws_cdk import (
    aws_lambda as _lambda,
)
from aws_cdk import (
    aws_s3 as s3,
)

if TYPE_CHECKING:
    from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]


class InferenceStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        data_bucket_name: str,
        feature_function_arn: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        data_bucket = s3.Bucket.from_bucket_name(self, "DataBucket", data_bucket_name)

        inference_fn = _lambda.DockerImageFunction(
            self,
            "InferenceFunction",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=str(REPO_ROOT),
                file="Dockerfile.inference",
            ),
            memory_size=3008,
            timeout=Duration.minutes(5),
            environment={
                "DATA_BUCKET": data_bucket_name,
            },
        )
        data_bucket.grant_read_write(inference_fn)

        events.Rule(
            self,
            "OnFeatureSucceeded",
            description="Run batch inference after the feature pipeline succeeds",
            event_pattern=events.EventPattern(
                source=["lambda"],
                detail_type=["Lambda Function Invocation Result - Success"],
                detail={
                    "requestContext": {
                        "functionArn": [{"prefix": feature_function_arn}],
                    },
                },
            ),
            targets=[
                targets.LambdaFunction(
                    inference_fn,
                    event=events.RuleTargetInput.from_object({"mode": "both"}),
                )
            ],
        )

        self.inference_function = inference_fn
        CfnOutput(self, "InferenceFunctionName", value=inference_fn.function_name)
