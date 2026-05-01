# AWS Deployment Architecture

> Infrastructure, deployment flow, and cost management for the football predictions platform.

---

## Services

| Component | AWS Service | Notes |
|---|---|---|
| ML model API | ECS Fargate + ALB | FastAPI container, auto-scaling |
| BFF layer | ECS Fargate (or Lambda) | Node.js Express proxy, SSR-ready |
| Frontend hosting | S3 + CloudFront | Angular static build, HTTPS |
| Database | RDS PostgreSQL (db.t3.micro) | Fixtures, teams, features, predictions |
| Raw data / artefacts | S3 | API responses, model files, SHAP outputs |
| Secrets | Secrets Manager | API keys, DB credentials |
| Data pipeline | Step Functions + Lambda | Scheduled data refresh (daily during tournament) |
| Monitoring | CloudWatch | API latency, error rates, model drift |
| DNS | Route 53 | Custom domain (optional) |

---

## Infrastructure as Code

Use **AWS CDK (Python)** for all resource provisioning. No manual console clicks.

### CDK Stack Layout

```
infrastructure/
├── app.py
├── config.py
└── stacks/
    ├── database.py         # RDS PostgreSQL
    ├── storage.py          # S3 buckets (raw data, artefacts, frontend)
    ├── api.py              # ECS Fargate + ALB for FastAPI
    ├── bff.py              # ECS Fargate for Node.js BFF
    ├── frontend.py         # S3 + CloudFront for Angular
    └── pipeline.py         # Step Functions for data refresh
```

---

## Deployment Flow

1. `git push` triggers GitHub Actions CI
2. CI runs linting, type checking, and tests (see `testing-and-security.md`)
3. On merge to `main`:
   - Build FastAPI Docker image, push to ECR, update ECS service
   - Build Node.js BFF Docker image, push to ECR, update ECS service
4. Frontend: `ng build` Angular app, sync to S3, invalidate CloudFront cache

---

## Environment Variables

Never commit values. Store in AWS Secrets Manager or pass via ECS task definitions.

```
API_FOOTBALL_KEY=
DATABASE_URL=
AWS_REGION=
S3_BUCKET_ARTEFACTS=
S3_BUCKET_RAW_DATA=
MODEL_VERSION=
FASTAPI_URL=              # Internal URL for BFF → FastAPI communication
```

---

## Cost Awareness

| Resource | Dev Tier | Tournament Tier |
|---|---|---|
| RDS | db.t3.micro | db.t3.small |
| ECS Fargate (FastAPI) | 0.25 vCPU / 0.5 GB | 0.5 vCPU / 1 GB |
| ECS Fargate (BFF) | 0.25 vCPU / 0.5 GB | 0.25 vCPU / 0.5 GB |
| S3 | Negligible | Negligible |
| CloudFront | Free tier | Free tier |

Scale up only during World Cup 2026 (June–July). Scale back down after.

---

## Monitoring & Alerts

- **CloudWatch Alarms:** API 5xx rate > 1%, p99 latency > 2s, ECS task restarts
- **Model drift:** Track prediction accuracy daily during tournament against actual results. Alert if rolling accuracy drops below baseline threshold.
- **Data pipeline:** Alert on failed Step Function executions or API-Football quota exhaustion.
