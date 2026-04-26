"""HostingStack — S3 + CloudFront for the static Angular site.

Components:
    * Private S3 bucket ``WebBucket`` for the Angular dist
    * CloudFront distribution with two behaviours:
        - default ``/*``       → WebBucket (SPA, with index.html fallback on 404/403)
        - path ``/data/*``     → existing data bucket, mapped to the
                                 ``web`` key prefix (so CloudFront request
                                 ``/data/competitions.json`` reads
                                 ``s3://data/web/data/competitions.json``)
    * Origin Access Control on both origins (no public-read on either bucket)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_cloudfront as cloudfront,
)
from aws_cdk import (
    aws_cloudfront_origins as origins,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_s3 as s3,
)

if TYPE_CHECKING:
    from constructs import Construct


class HostingStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        data_bucket_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        web_bucket = s3.Bucket(
            self,
            "WebBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        data_bucket = s3.Bucket.from_bucket_name(self, "DataBucket", data_bucket_name)

        web_origin = origins.S3BucketOrigin.with_origin_access_control(web_bucket)
        data_origin = origins.S3BucketOrigin.with_origin_access_control(
            data_bucket,
            origin_path="/web",
        )

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=web_origin,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                compress=True,
            ),
            additional_behaviors={
                "/data/*": cloudfront.BehaviorOptions(
                    origin=data_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    cache_policy=cloudfront.CachePolicy(
                        self,
                        "DataCachePolicy",
                        default_ttl=Duration.minutes(1),
                        max_ttl=Duration.minutes(5),
                        min_ttl=Duration.seconds(0),
                    ),
                    compress=True,
                ),
            },
            default_root_object="index.html",
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.minutes(0),
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.minutes(0),
                ),
            ],
        )

        data_bucket_arn = f"arn:aws:s3:::{data_bucket_name}"
        data_bucket_policy_statement = iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[f"{data_bucket_arn}/web/*"],
            principals=[iam.ServicePrincipal("cloudfront.amazonaws.com")],
            conditions={
                "StringEquals": {
                    "AWS:SourceArn": (
                        f"arn:aws:cloudfront::{self.account}:distribution/"
                        f"{distribution.distribution_id}"
                    ),
                },
            },
        )

        self.web_bucket = web_bucket
        self.distribution = distribution
        self.data_bucket_policy_statement = data_bucket_policy_statement

        CfnOutput(self, "WebBucketName", value=web_bucket.bucket_name)
        CfnOutput(self, "DistributionDomainName", value=distribution.distribution_domain_name)
        CfnOutput(self, "DistributionId", value=distribution.distribution_id)
        CfnOutput(
            self,
            "DataBucketPolicyStatement",
            description=(
                "Add this statement to the data bucket's bucket policy "
                "for CloudFront read access"
            ),
            value=str({
                "Sid": "AllowCloudFrontReadWeb",
                "Effect": "Allow",
                "Principal": {"Service": "cloudfront.amazonaws.com"},
                "Action": "s3:GetObject",
                "Resource": f"{data_bucket_arn}/web/*",
                "Condition": {
                    "StringEquals": {
                        "AWS:SourceArn": (
                            f"arn:aws:cloudfront::{self.account}:distribution/"
                            f"{distribution.distribution_id}"
                        ),
                    },
                },
            }),
        )
