---
layout: post
title: "Shipping a LangGraph agent stack to Cloud Run with keyless GitHub-OIDC and seven failed deploys"
date: 2026-05-05 20:00:00 +0800
categories: [DevOps, Engineering]
tags: [terraform, cloudrun, gcp, github-actions, oidc, wif, langgraph, devops]
---

> Terraform-managed Cloud Run × 3 (api / web / reranker), Workload Identity Federation for keyless GitHub-OIDC deploys, Artifact Registry, Secret Manager. From `gcloud projects create` to a green deploy in seven iterations. Here's what broke and how I fixed it.

## The target architecture

```text
   browser
      │
      ▼
[Cloud Run / web]   Next.js 14 standalone, port 3000
      │ NEXT_PUBLIC_API_URL
      ▼
[Cloud Run / api]   FastAPI + LangGraph, port 8000
   │ │ │
   │ │ └── Supabase Postgres + pgvector  (external)
   │ └──── Cloud Run / reranker          (bge-reranker, port 8001)
   ├────── Secret Manager                (10 secrets)
   ├────── Cloud Storage                 (reports bucket)
   └────── Cloud Tasks queue             (M4 ingestion jobs)
```

All three services scale-to-zero. Cold start is fast on api/web (~3 s) and acceptable on reranker (~5 s — the model is baked into the image).

## Why keyless OIDC

Storing a `service-account.json` as a GitHub secret means:

- It can't be rotated without a code change.
- Anybody with read access to Actions logs can leak it accidentally.
- The principle of least privilege evaporates because the JSON has all the SA's roles forever.

Workload Identity Federation does the dance properly:

```text
GitHub Actions emits an OIDC token (signed by GitHub)
         │
         ▼
google-github-actions/auth@v2
         │
         ▼
STS exchanges the OIDC token for a short-lived GCP access token,
constrained by:
  - WIF pool / provider configuration
  - attribute_condition: assertion.repository == "box1401/stock-rag-agentic"
  - principalSet binding to the deployer SA
         │
         ▼
deployer SA: 5-minute access token
```

No JSON ever exists. Tokens expire in 5 minutes. The only thing GitHub knows is its own OIDC private key.

Setup is a one-time `bootstrap.sh`:

```bash
gcloud iam workload-identity-pools create github-pool --location=global
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition='assertion.repository == "box1401/stock-rag-agentic"' \
  --issuer-uri="https://token.actions.githubusercontent.com"
gcloud iam service-accounts add-iam-policy-binding compass-deployer@... \
  --role=roles/iam.workloadIdentityUser \
  --member=principalSet://iam.googleapis.com/$POOL_ID/attribute.repository/$REPO
```

GitHub side: two secrets (`WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`), three vars (`GCP_PROJECT_ID`, `GCP_PROJECT_NUMBER`, `GCP_REGION`).

## The seven things that broke

### 1. `web/Dockerfile` failed because `public/` didn't exist

The Next.js standalone build wanted `COPY /app/public ./public`. The repo had no `public/`. Fixed by creating an empty `apps/web/public/.gitkeep` and adding `target: dev` to docker-compose so local dev uses the dev stage.

### 2. The `gemini-2.0-flash` quota was 0 on this API key

```text
429 RESOURCE_EXHAUSTED ... limit: 0, model: gemini-2.0-flash
```

The free-tier quota is **per model**, and 2.0-flash had been migrated off free tier on this billing project. Probed available models with a list-models call, found `gemini-2.5-flash` was free and capable. Changed primary + fallback to 2.5 Flash + 2.5 Flash Lite.

### 3. `gemini-1.5-pro` was returning 404

It had been retired entirely. The model-listing probe is now a documented step in `SETUP.md`.

### 4. AR repo creation order was wrong

The deploy workflow built and pushed images **before** running `terraform apply`. But Terraform was responsible for creating the Artifact Registry repo. Chicken-and-egg. Fixed by:

- Creating the AR repo in `bootstrap.sh` (one-time, idempotent).
- Switching `terraform/artifact_registry.tf` from `resource` to `data` source.

The same pattern applied to the WIF pool: bootstrap creates it, Terraform doesn't manage it.

### 5. Deployer SA missing two IAM roles

```text
Error 403: Permission 'iam.serviceAccounts.create' denied
```

The bootstrap script granted `roles/iam.serviceAccountUser` (lets you ACT AS an SA) but not `roles/iam.serviceAccountAdmin` (lets you CREATE one). Same for `roles/resourcemanager.projectIamAdmin` (needed for the project-level role bindings Terraform was creating for runtime SAs). Added both.

### 6. Cloud Run refused to start because referenced secrets had no versions

```text
Error code 9: Secret projects/.../secrets/langfuse-public-key/versions/latest was not found
```

When you reference a Secret Manager secret in a Cloud Run env-var via `secret_key_ref.version = "latest"`, the secret must have **at least one version**. Terraform created the secret containers but no versions. Fixed by:

```hcl
resource "google_secret_manager_secret_version" "placeholder" {
  for_each    = local.secret_ids
  secret      = google_secret_manager_secret.all[each.key].id
  secret_data = local.placeholder_values[each.key]
  lifecycle { ignore_changes = [secret_data, enabled] }
}
```

`ignore_changes` matters: once `seed-secrets.sh` adds real values as new versions, Terraform must not "fix" them back to placeholder.

### 7. `DATABASE_URL` placeholder wasn't a valid URL

The api container crashed at import time because `create_async_engine("PLACEHOLDER")` couldn't parse the URL. Fixed by per-secret placeholder values:

```hcl
locals {
  placeholder_values = {
    database_url = "postgresql+asyncpg://placeholder:placeholder@127.0.0.1:5432/placeholder"
    supabase_url = "https://placeholder.supabase.co"
    # ...others get a plain string
  }
}
```

Plus a `depends_on` chain so Cloud Run waits for both the placeholder version and the IAM binding before being created.

### 8 (bonus). Resilience for "DB unreachable in prod"

Even with a parseable URL, the placeholder points at `127.0.0.1:5432` which has no Postgres listening. The RAG `rag_search` tool was opening sessions to that, getting `ConnectionRefusedError`, and crashing the entire LangGraph pipeline.

The fix is a one-line philosophical decision:

```python
try:
    async with sessionmaker() as session:
        results = await hybrid_search(session, ...)
except Exception as e:
    log.warning("rag_search_failed err=%s", e)
    return f"(rag_search unavailable: {type(e).__name__})", []
```

Tools should never crash agents; they should return a string the agent can reason about. Now the production demo runs even before Supabase is wired in — the analyst gracefully reports "I have market data but no thesis context" instead of 500-ing.

## What the final `deploy.yml` does

```yaml
on:
  push:
    branches: [main]
    paths: ["apps/**", "infra/**", ".github/workflows/deploy.yml"]
  workflow_dispatch:

permissions:
  contents: read
  id-token: write   # OIDC

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    concurrency:
      group: deploy-prod
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${% raw %}{{ secrets.WIF_PROVIDER }}{% endraw %}
          service_account: ${% raw %}{{ secrets.WIF_SERVICE_ACCOUNT }}{% endraw %}
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud auth configure-docker ${% raw %}{{ vars.GCP_REGION }}{% endraw %}-docker.pkg.dev --quiet
      - run: |   # build + push api / reranker / web with sha tags
          docker build -t "$IMAGE" apps/api && docker push "$IMAGE"
          # ...same for reranker, web
      - uses: hashicorp/setup-terraform@v3
      - run: terraform init -backend-config=bucket=${PROJECT_ID}-tfstate
      - run: terraform apply -auto-approve -var "image_tag=${SHA}"
```

`concurrency.cancel-in-progress: false` is important: a hot deploy mid-rollout would race with itself.

## Costs (after the dust settled)

| Service | Free tier I'm in | Real cost |
|---|---|---|
| Cloud Run × 3 | 2M req + 360k vCPU-sec / mo | $0 (scale-to-zero) |
| Artifact Registry | 0.5 GB free, 0.10/GB after | ~$0.20/mo (reranker image is 1.6 GB) |
| Secret Manager | 6 active versions / secret | $0 |
| Cloud Storage | 5 GB free | $0 |
| Cloud Logging | 50 GB ingest / mo | $0 |
| GCS for tfstate | 5 GB free | $0 |

So the whole production stack runs at ~$0.20/mo as long as I keep `min_instances = 0` and don't hammer the Gemini API beyond 1500 requests/day.

## Repo

- [`infra/scripts/bootstrap.sh`](https://github.com/box1401/stock-rag-agentic/blob/main/infra/scripts/bootstrap.sh) — one-time GCP setup
- [`infra/terraform/`](https://github.com/box1401/stock-rag-agentic/tree/main/infra/terraform) — 11 .tf files, all validated in CI
- [`.github/workflows/deploy.yml`](https://github.com/box1401/stock-rag-agentic/blob/main/.github/workflows/deploy.yml) — the actual deploy
- [`docs/DEPLOY.md`](https://github.com/box1401/stock-rag-agentic/blob/main/docs/DEPLOY.md) — step-by-step walkthrough

## What the seven failures actually tell you

If you've shipped this kind of thing before, none of the individual failures are surprising. What's interesting is the order: **Cloud Run / Terraform / Workload Identity / Secret Manager / async Python all have their own failure modes that compose**. Each layer's documentation, in isolation, is good. The cross-layer interactions — secrets must have versions before referenced, IAM bindings have eventual-consistency lag, AR repo must exist before docker push, terraform `ignore_changes` interacts with secret rotation — are exactly the things you find out on deploy 6.

Worth it for the architecture signal: WIF + Terraform + Cloud Run × 3 + LangGraph + hybrid RAG. Anyone reading this stack knows the author has shipped something more substantial than a Streamlit demo.

Series wrap-up complete. Code: <https://github.com/box1401/stock-rag-agentic>. Live: <https://compass-web-aujzogkiva-de.a.run.app>.
