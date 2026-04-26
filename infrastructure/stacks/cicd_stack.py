"""CICDStack — GitHub OIDC trust for keyless deploys from GitHub Actions.

Phase A creates the foundation only:
    * GitHub OIDC identity provider in the account
    * One IAM role (``fp-github-actions``) trusted by this repo
    * No managed/inline policies attached yet — Phase B (per-subsystem
      auto-deploy) will add deploy permissions, scoped narrowly per subsystem.

The PR-gate workflow (``.github/workflows/pr-gate.yml``) doesn't need
this role — it runs lint, tests, and ``cdk synth`` without AWS access.
The role is set up now so Phase B can attach policies without having
to bootstrap OIDC from scratch.
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_iam as iam,
)
from constructs import Construct


_GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
_GITHUB_OIDC_AUDIENCE = "sts.amazonaws.com"


class CICDStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        github_repo: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        provider = iam.OpenIdConnectProvider(
            self,
            "GitHubOidcProvider",
            url=_GITHUB_OIDC_ISSUER,
            client_ids=[_GITHUB_OIDC_AUDIENCE],
        )

        condition_repo = f"repo:{github_repo}:*"
        issuer_host = "token.actions.githubusercontent.com"

        role = iam.Role(
            self,
            "GitHubActionsRole",
            role_name="fp-github-actions",
            description=f"Assumed by GitHub Actions runs in {github_repo}",
            max_session_duration=Duration.hours(1),
            assumed_by=iam.FederatedPrincipal(
                federated=provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        f"{issuer_host}:aud": _GITHUB_OIDC_AUDIENCE,
                    },
                    "StringLike": {
                        f"{issuer_host}:sub": condition_repo,
                    },
                },
                assume_role_action="sts:AssumeRoleWithWebIdentity",
            ),
        )

        self.oidc_provider = provider
        self.github_role = role

        CfnOutput(self, "OidcProviderArn", value=provider.open_id_connect_provider_arn)
        CfnOutput(self, "GitHubActionsRoleArn", value=role.role_arn)
